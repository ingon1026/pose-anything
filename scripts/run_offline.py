#!/usr/bin/env python3
"""Run the full perception pipeline on an mcap rosbag without ROS.

Reads color + aligned depth + camera_info via `rosbags` as a stream (bounded
memory — bag length doesn't matter), runs SAM3 -> tracker -> OBB, writes an
overlay mp4 at the bag's real average fps, a per-frame CSV and prints
per-track stability metrics.

Usage:
  python3 scripts/run_offline.py --bag bags/test2 --prompts "물통,노트북"
  python3 scripts/run_offline.py --bag bags/test3 --prompts 책 --show
"""
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "roboworld_perception"))

from rosbags.highlevel import AnyReader  # noqa: E402

from roboworld_perception.pipeline import (CSV_HEADER, csv_row,  # noqa: E402
                                           img_to_np, status_text)

COLOR = "/camera/camera/color/image_raw"
DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"
INFO = "/camera/camera/color/camera_info"


def bag_color_fps(bag_path):
    """색상 토픽 평균 fps (타임스탬프만 스캔 — 디코드 없음, 수 초 내)."""
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == COLOR]
        ts = [t for _, t, _ in reader.messages(connections=conns)]
    if len(ts) < 2:
        return 15.0
    return (len(ts) - 1) / ((ts[-1] - ts[0]) * 1e-9)


def read_bag(bag_path, max_frames=None):
    """(stamp_s, rgb, depth_mm, K)를 스트리밍으로 yield — 메모리 O(1).

    bag 메시지는 시간순이므로, 최근 depth만 deque로 들고 있다가 각 색상
    프레임을 가장 가까운 depth와 짝짓는다. 색상 디코드는 짝이 확정된
    프레임에만 수행한다.
    """
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in (COLOR, DEPTH, INFO)]
        K = None
        depths = deque(maxlen=60)   # (t, 디코드된 depth) — 약 2초 분량
        pending = deque()           # 아직 더 늦은 depth를 기다리는 색상 raw
        count = 0

        def match(ct, craw, cconn):
            nonlocal count
            if K is None or not depths:
                return None
            dt, dimg = min(depths, key=lambda d: abs(d[0] - ct))
            if abs(dt - ct) > 50e6:  # >50ms: 짝 없음
                return None
            cmsg = reader.deserialize(craw, cconn.msgtype)
            rgb = img_to_np(cmsg)
            if cmsg.encoding == "bgr8":
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            count += 1
            return ct * 1e-9, rgb, dimg, K

        for conn, t, raw in reader.messages(connections=conns):
            if conn.topic == INFO:
                if K is None:
                    K = np.array(reader.deserialize(raw, conn.msgtype).k).reshape(3, 3)
            elif conn.topic == DEPTH:
                depths.append((t, img_to_np(reader.deserialize(raw, conn.msgtype))))
            else:
                pending.append((t, raw, conn))
            while pending and depths and depths[-1][0] >= pending[0][0]:
                item = match(*pending.popleft())
                if item is not None:
                    yield item
                    if max_frames and count >= max_frames:
                        return
        for args in pending:  # bag 끝: 남은 색상은 마지막 depth들과 매칭
            item = match(*args)
            if item is not None:
                yield item
                if max_frames and count >= max_frames:
                    return


def summarize(rows):
    """Per-track stability metrics from CSV rows."""
    by_track = {}
    for r in rows:
        by_track.setdefault((r["label"], r["track_id"]), []).append(r)
    print("\n=== per-track stability ===")
    for (label, tid), rs in sorted(by_track.items()):
        if len(rs) < 2:
            continue
        c = np.array([[r["x"], r["y"], r["z"]] for r in rs])
        e = np.array([[r["w"], r["d"], r["h"]] for r in rs])
        rpy = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in rs])
        drpy = np.abs(np.diff(rpy, axis=0))
        drpy = np.minimum(drpy, 360 - drpy)  # wrap
        print(f"{label}#{tid}: frames={len(rs)}"
              f" center_std={1000 * c.std(axis=0).round(4)}mm"
              f" size_mean={e.mean(axis=0).round(3)}m size_std={1000 * e.std(axis=0).round(4)}mm"
              f" |dRPY|/frame={drpy.mean(axis=0).round(2)}deg"
              f" flips={rs[-1]['flips']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--prompts", required=True, help="comma-separated object names")
    ap.add_argument("--out", default="output")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--show", action="store_true", help="처리 중 오버레이 창 표시")
    ap.add_argument("--max-per-prompt", type=int, default=1,
                    help="프롬프트당 유지할 인스턴스 수 (0=제한 없음)")
    ap.add_argument("--image-size", type=int, default=0,
                    help="SAM3 입력 해상도 (0=기본 1008)")
    ap.add_argument("--detect-interval", type=int, default=5,
                    help="SAM 검출 주기 (1=매 프레임, N=키프레임+광학흐름 추적)")
    args = ap.parse_args()

    from roboworld_perception.overlay import draw_objects, draw_status, show_window
    from roboworld_perception.pipeline import PerceptionPipeline
    from roboworld_perception.sam3_detector import Sam3Detector, parse_prompts

    prompts = parse_prompts(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = f"{Path(args.bag).name}_{'-'.join(prompts)}"
    fps = bag_color_fps(args.bag)

    print("loading SAM3...")
    pipeline = PerceptionPipeline(
        Sam3Detector(threshold=args.threshold, image_size=args.image_size),
        detect_interval=args.detect_interval,
        max_per_prompt=args.max_per_prompt)

    writer = None
    rows = []
    csv_path = out_dir / f"{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(CSV_HEADER)
        t_start = time.perf_counter()
        n = 0
        for stamp, rgb, depth, K in read_bag(args.bag, args.max_frames):
            t0 = time.perf_counter()
            objects = pipeline.process(rgb, depth, K, prompts)
            proc_ms = (time.perf_counter() - t0) * 1000
            bgr = draw_objects(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), objects, K)
            draw_status(bgr, status_text(1000 / max(proc_ms, 1e-3), pipeline, objects))
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{tag}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         round(fps, 2),
                                         (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
            if args.show and not show_window(bgr):
                args.show = False  # 디스플레이 없는 환경이면 창 없이 계속
            n += 1
            for o in objects:
                if not o.publishable:  # 가림 중이거나 obb 없음 — 기록 제외
                    continue
                cw.writerow(csv_row(o, stamp, proc_ms))
                rows.append(dict(stamp=stamp, track_id=o.track_id, label=o.label,
                                 x=o.obb.center[0], y=o.obb.center[1],
                                 z=o.obb.center[2], w=o.obb.extent[0],
                                 d=o.obb.extent[1], h=o.obb.extent[2],
                                 roll=o.obb.rpy[0], pitch=o.obb.rpy[1],
                                 yaw=o.obb.rpy[2], flips=o.flip_count))
            if n % 20 == 0:
                print(f"frame {n}: {len(objects)} objects, {proc_ms:.0f}ms")
        total = time.perf_counter() - t_start
    if writer:
        writer.release()
    print(f"\n{n} frames in {total:.1f}s -> {n / max(total, 1e-9):.2f} FPS")
    print(f"video: {out_dir / f'{tag}.mp4'} ({fps:.1f}fps)\ncsv:   {csv_path}")
    summarize(rows)


if __name__ == "__main__":
    main()

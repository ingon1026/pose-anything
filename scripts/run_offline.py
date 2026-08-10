#!/usr/bin/env python3
"""Run the full perception pipeline on an mcap rosbag without ROS.

Reads color + aligned depth + camera_info via `rosbags`, pairs frames by
nearest timestamp, runs SAM3 -> tracker -> OBB, writes an overlay mp4,
a per-frame CSV and prints per-track stability metrics.

Usage:
  python3 scripts/run_offline.py --bag bags/test2 --prompts "물통,마우스,필통"
  python3 scripts/run_offline.py --bag bags/test3 --prompts 물통 --max-frames 50
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "roboworld_perception"))

from rosbags.highlevel import AnyReader  # noqa: E402

COLOR = "/camera/camera/color/image_raw"
DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"
INFO = "/camera/camera/color/camera_info"


def read_bag(bag_path, max_frames=None):
    """Yield (stamp_s, rgb, depth_mm, K) with depth matched to each color frame."""
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in (COLOR, DEPTH, INFO)]
        K = None
        depths = []  # (t, msg) buffer; bag is time-ordered so keep it small
        colors = []
        for conn, t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == INFO:
                if K is None:
                    K = np.array(msg.k).reshape(3, 3)
            elif conn.topic == DEPTH:
                depths.append((t, msg))
            else:
                colors.append((t, msg))
        colors.sort(key=lambda x: x[0])
        depth_ts = np.array([t for t, _ in depths])
        count = 0
        for t, cmsg in colors:
            if K is None or len(depths) == 0:
                break
            i = int(np.argmin(np.abs(depth_ts - t)))
            if abs(depth_ts[i] - t) > 50e6:  # >50ms apart: no matching depth
                continue
            dmsg = depths[i][1]
            rgb = np.frombuffer(cmsg.data, np.uint8).reshape(cmsg.height, cmsg.width, 3)
            if cmsg.encoding == "bgr8":
                rgb = rgb[:, :, ::-1]
            depth = np.frombuffer(dmsg.data, np.uint16).reshape(dmsg.height, dmsg.width)
            yield t * 1e-9, rgb.copy(), depth, K
            count += 1
            if max_frames and count >= max_frames:
                break


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

    from roboworld_perception.overlay import draw_objects
    from roboworld_perception.pipeline import PerceptionPipeline
    from roboworld_perception.sam3_detector import Sam3Detector

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    tag = f"{Path(args.bag).name}_{'-'.join(prompts)}"

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
        cw.writerow(["stamp", "track_id", "label", "score", "x", "y", "z",
                     "distance", "w", "d", "h", "roll", "pitch", "yaw",
                     "flips", "proc_ms"])
        t_start = time.perf_counter()
        n = 0
        stamps = []
        tmp_path = out_dir / f"{tag}.tmp.mp4"
        for stamp, rgb, depth, K in read_bag(args.bag, args.max_frames):
            t0 = time.perf_counter()
            objects = pipeline.process(rgb, depth, K, prompts)
            proc_ms = (time.perf_counter() - t0) * 1000
            bgr = draw_objects(rgb[:, :, ::-1].copy(), objects, K)
            stamps.append(stamp)
            if writer is None:
                writer = cv2.VideoWriter(str(tmp_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                         (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
            if args.show:
                try:
                    cv2.imshow("roboworld perception", bgr)
                    cv2.waitKey(1)
                except cv2.error:
                    args.show = False  # 디스플레이 없는 환경이면 창 없이 계속
            n += 1
            for o in objects:
                if o.obb is None:
                    continue
                r, p, y = o.obb.rpy
                row = dict(stamp=stamp, track_id=o.track_id, label=o.label,
                           score=o.score, x=o.obb.center[0], y=o.obb.center[1],
                           z=o.obb.center[2], distance=o.obb.distance,
                           w=o.obb.extent[0], d=o.obb.extent[1], h=o.obb.extent[2],
                           roll=r, pitch=p, yaw=y, flips=o.flip_count,
                           proc_ms=proc_ms)
                rows.append(row)
                cw.writerow([f"{v:.4f}" if isinstance(v, float) else v
                             for v in row.values()])
            if n % 20 == 0:
                print(f"frame {n}: {len(objects)} objects, {proc_ms:.0f}ms")
        total = time.perf_counter() - t_start
    if writer:
        writer.release()
        # x1 재생 보장: 전체 구간 평균 fps로 재인코딩 (카메라 30fps 설정이지만
        # 녹화 드랍으로 실제 ~18fps인 bag에서 프레임 간격 기반 추정은 가속됨)
        fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-3) if len(stamps) > 1 else 15
        cap = cv2.VideoCapture(str(tmp_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(out_dir / f"{tag}.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), round(fps, 2), (w, h))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
        cap.release()
        out.release()
        tmp_path.unlink()
    print(f"\n{n} frames in {total:.1f}s -> {n / max(total, 1e-9):.2f} FPS")
    print(f"video: {out_dir / f'{tag}.mp4'}\ncsv:   {csv_path}")
    summarize(rows)


if __name__ == "__main__":
    main()

"""SAM3 키프레임 시간을 서브모듈별로 쪼갠다 — TensorRT 로 태울 값어치가
어디에 있는지 정하기 위한 계측. bag 은 첫 프레임 한 장만 읽는다(재생 아님).

사용: python3 scripts/bench_sam3_parts.py [bag경로] [반복]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "src" / "roboworld_perception"))
from roboworld_perception.pipeline import img_to_np  # noqa: E402
from roboworld_perception.sam3_detector import Sam3Detector  # noqa: E402

BAG = sys.argv[1] if len(sys.argv) > 1 else "/home/ingon/roboworld/bags/test4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
PROMPTS = ["black bag", "keyboard", "book"]


def first_rgb(bag):
    from rosbags.highlevel import AnyReader
    with AnyReader([Path(bag)]) as r:
        for c, t, raw in r.messages():
            if c.msgtype.endswith("Image") and "depth" not in c.topic:
                return img_to_np(r.deserialize(raw, c.msgtype))
    raise RuntimeError("컬러 프레임 없음")


def timed(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    rgb = first_rgb(BAG)
    print(f"프레임 {rgb.shape}, 프롬프트 {len(PROMPTS)}개, 반복 {N}")
    det = Sam3Detector(threshold=0.4)
    m, proc = det.model, det.processor
    image = Image.fromarray(rgb)

    t_pre = timed(lambda: proc(images=image, return_tensors="pt"), N)
    img_inputs = proc(images=image, return_tensors="pt").to(det.device)
    px = img_inputs.pixel_values.to(det.dtype)
    target_sizes = img_inputs.get("original_sizes").tolist()
    print(f"  전처리(CPU, 리사이즈·정규화)         {t_pre:7.1f} ms")

    with torch.no_grad():
        t_vis = timed(lambda: m.get_vision_features(pixel_values=px), N)
        vis = m.get_vision_features(pixel_values=px)
        print(f"  vision_encoder (프레임당 1회)        {t_vis:7.1f} ms")

        texts = [det._text_inputs(p) for p in PROMPTS]
        t_txt = timed(lambda: [m.get_text_features(**t) for t in texts], N)
        print(f"  text_encoder x{len(PROMPTS)} (매번 재계산 중)     {t_txt:7.1f} ms")

        def heads():
            for t in texts:
                m(vision_embeds=vis, **t)
        t_all = timed(heads, N)
        print(f"  forward(text+DETR+mask) x{len(PROMPTS)}         {t_all:7.1f} ms")
        print(f"    = 그중 디코더류만                  {t_all - t_txt:7.1f} ms")

        outs = [m(vision_embeds=vis, **t) for t in texts]

        def post():
            for o in outs:
                proc.post_process_instance_segmentation(
                    o, threshold=0.1, mask_threshold=0.5,
                    target_sizes=target_sizes)
        t_post = timed(post, N)
        print(f"  후처리 x{len(PROMPTS)}                          {t_post:7.1f} ms")

    total = t_pre + t_vis + t_all + t_post
    print(f"  -- 합계 {total:.1f} ms")
    print(f"\n  vision {100 * t_vis / total:.0f}% | 디코더류 "
          f"{100 * (t_all - t_txt) / total:.0f}% | text {100 * t_txt / total:.0f}%"
          f" | 전처리 {100 * t_pre / total:.0f}% | 후처리 {100 * t_post / total:.0f}%")


if __name__ == "__main__":
    main()

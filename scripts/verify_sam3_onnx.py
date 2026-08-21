"""내보낸 ONNX vision_encoder 가 원본과 같은 값을 내는지 확인.

export 가 성공해도 출력이 틀리면 소용없다. onnxruntime(CPU) 로 돌려 torch
출력과 비교한다 — 느리지만 관문 확인에는 1회면 충분하다.

사용: python3 scripts/verify_sam3_onnx.py [onnx경로]
"""
import sys
from pathlib import Path

import numpy as np
import torch

ONNX = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sam3_vision.onnx"


def main():
    from transformers import Sam3Model
    torch.manual_seed(0)
    px = torch.randn(1, 3, 1008, 1008, dtype=torch.float32)

    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = model.vision_encoder.to(dev)
    with torch.no_grad():
        o = enc(px.to(dev))
    ref = [o.last_hidden_state, *o.fpn_hidden_states, *o.fpn_position_encoding]
    ref = [t.float().cpu().numpy() for t in ref]
    del model, enc, o
    torch.cuda.empty_cache()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"pixel_values": px.numpy()})

    print(f"{'출력':22s} {'shape':22s} {'최대오차':>10s} {'상대오차':>10s}")
    worst = 0.0
    for meta, a, b in zip(sess.get_outputs(), ref, outs):
        d = float(np.abs(a - b).max())
        scale = float(np.abs(a).max()) + 1e-9
        worst = max(worst, d / scale)
        print(f"{meta.name:22s} {str(a.shape):22s} {d:10.3e} {d / scale:10.3e}")
    print(f"\n최대 상대오차 {worst:.3e} -> "
          + ("일치 (fp32 수치오차 수준)" if worst < 1e-3 else "불일치 — 조사 필요"))
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())

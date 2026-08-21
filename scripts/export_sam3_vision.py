"""SAM3 vision_encoder 를 ONNX 로 내보낸다 (TensorRT 관문 1단계).

vision_encoder 만 태우는 이유: 키프레임 시간의 57% 이고(bench_sam3_parts.py),
텍스트와 무관해 입력이 이미지 하나뿐이라 shape 이 완전히 고정된다. 디코더류는
프롬프트 개수에 따라 shape 이 변해 TensorRT 와 궁합이 나쁘다.

출력이 dataclass + 튜플이라 ONNX 가 못 받는다 -> 평탄한 튜플로 래핑한다.

사용: python3 scripts/export_sam3_vision.py [출력경로] [--fp16] [--cpu]
"""
import sys
import traceback
from pathlib import Path

import torch
from torch import nn

OUT = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
           else "/tmp/sam3_vision.onnx")
FP16 = "--fp16" in sys.argv
CPU = "--cpu" in sys.argv


class VisionWrapper(nn.Module):
    """Sam3VisionModel -> 평탄한 텐서 튜플. ONNX 는 dataclass·중첩튜플을 못 받는다."""

    def __init__(self, vision_encoder):
        super().__init__()
        self.enc = vision_encoder

    def forward(self, pixel_values):
        o = self.enc(pixel_values)
        return (o.last_hidden_state, *o.fpn_hidden_states,
                *o.fpn_position_encoding)


def main():
    from transformers import Sam3Model
    device = "cpu" if CPU else "cuda"
    dtype = torch.float16 if FP16 else torch.float32
    print(f"로드: device={device} dtype={dtype}")
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=dtype).eval()
    enc = model.vision_encoder.to(device)
    del model
    torch.cuda.empty_cache()

    size = 1008
    px = torch.randn(1, 3, size, size, dtype=dtype, device=device)
    w = VisionWrapper(enc).eval()
    with torch.no_grad():
        ref = w(px)
    print(f"참조 출력 {len(ref)}개: "
          + ", ".join(str(tuple(t.shape)) for t in ref))

    names = (["last_hidden_state"]
             + [f"fpn_{i}" for i in range(len(ref) // 2)]
             + [f"pos_{i}" for i in range(len(ref) - 1 - len(ref) // 2)])

    for dynamo in (True, False):
        print(f"\n=== torch.onnx.export(dynamo={dynamo}) ===")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    w, (px,), str(OUT), dynamo=dynamo,
                    input_names=["pixel_values"],
                    output_names=names[:len(ref)],
                    opset_version=18)
            mb = OUT.stat().st_size / 1e6
            print(f"성공 -> {OUT} ({mb:.0f} MB)")
            return 0
        except Exception as e:
            print(f"실패: {type(e).__name__}")
            tb = traceback.format_exc()
            # 핵심 줄만 (op 이름이 들어 있는 줄 위주)
            for line in tb.splitlines():
                low = line.lower()
                if any(k in low for k in ("unsupported", "not support", "aten::",
                                          "no symbolic", "error:", "failed",
                                          "torch.onnx", "opset")):
                    print("  | " + line.strip()[:200])
            print("  | (마지막) " + tb.strip().splitlines()[-1][:300])
    return 1


if __name__ == "__main__":
    sys.exit(main())

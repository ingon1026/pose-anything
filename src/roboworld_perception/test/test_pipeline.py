import cv2
import numpy as np

from roboworld_perception.pipeline import PerceptionPipeline, propagate_mask


def textured_frame(shift=0):
    """노이즈 텍스처 사각형이 (100+shift, 100)에 있는 프레임."""
    rng = np.random.default_rng(0)
    frame = np.full((240, 320), 60, np.uint8)
    patch = rng.integers(0, 255, (60, 80), dtype=np.uint8)
    frame[100:160, 100 + shift:180 + shift] = patch
    return frame


def test_propagate_mask_follows_motion():
    prev, cur = textured_frame(0), textured_frame(7)
    mask = np.zeros((240, 320), bool)
    mask[100:160, 100:180] = True
    dx, dy = propagate_mask(prev, cur, mask)
    assert abs(dx - 7) < 1.5
    assert abs(dy) < 1.5


def test_propagate_mask_rejects_empty():
    prev, cur = textured_frame(0), textured_frame(5)
    assert propagate_mask(prev, cur, np.zeros((240, 320), bool)) is None


class StubDetector:
    """호출 횟수를 세는 가짜 검출기 — 키프레임에만 불려야 한다."""
    def __init__(self):
        self.calls = 0

    def detect(self, rgb, prompts):
        self.calls += 1
        mask = np.zeros(rgb.shape[:2], bool)
        mask[100:160, 100:180] = True
        return [{"label": "obj", "mask": mask,
                 "box": np.array([100, 100, 180, 160], float), "score": 0.9}]


def test_hybrid_calls_sam_only_on_keyframes():
    det = StubDetector()
    pipe = PerceptionPipeline(det, detect_interval=5)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    depth = np.full((240, 320), 1000, np.uint16)
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    for i in range(10):
        out = pipe.process(rgb, depth, K, ["obj"])
        assert len(out) == 1
        assert out[0].track_id == 1  # 중간 프레임에도 ID 유지
    assert det.calls == 2  # 프레임 0, 5에만 SAM 호출

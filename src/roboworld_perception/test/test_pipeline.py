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


def test_plane_fit_is_attempted_only_once(monkeypatch):
    """지지면 추정은 검출이 처음 생긴 키프레임에서 한 번만 시도한다.

    매 키프레임 재시도하면, 물체가 이동해 링이 달라진 어느 프레임에서 우연히
    인라이어 문턱을 넘고 그 나쁜 평면이 영구 캐시된다 — test3 실측: 18회 실패
    후 19번째에 "성공"해 pink block 발행 336→168프레임, size_std 6.7→104.9mm.
    고정 카메라에서 지지면은 상수이므로, 첫 시도에 못 잡으면 그 씬은 평면
    가정이 안 맞는 것이다 (docs/belt_plane_2026-08-21.md).
    """
    from roboworld_perception import pipeline as P
    tries = []
    monkeypatch.setattr(P, "_fit_support_plane",
                        lambda *a, **kw: tries.append(1) or None)
    pipe = P.PerceptionPipeline(StubDetector(), detect_interval=5,
                                use_belt_plane=True)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    depth = np.full((240, 320), 1000, np.uint16)
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    for _ in range(20):  # 키프레임 4회
        pipe.process(rgb, depth, K, ["obj"])
    assert pipe.belt_plane is None   # 실패 → 무구속 폴백
    assert len(tries) == 1           # 재시도 없음

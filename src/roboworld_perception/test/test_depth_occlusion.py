import numpy as np

from roboworld_perception.geometry import masked_depth_median
from roboworld_perception.pipeline import PerceptionPipeline
from roboworld_perception.tracker import Track

K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])


def make(depth_mm):
    depth = np.zeros((240, 320), np.uint16)
    mask = np.zeros((240, 320), bool)
    mask[100:160, 100:180] = True
    depth[100:160, 100:180] = depth_mm
    return mask, depth


class NoDetector:
    def detect(self, rgb, prompts):
        return []


def test_masked_depth_median():
    mask, depth = make(1000)
    assert abs(masked_depth_median(mask, depth) - 1.0) < 1e-6
    assert masked_depth_median(np.zeros((240, 320), bool), depth) is None


def test_depth_intrusion_rejected_before_commit():
    pipe = PerceptionPipeline(NoDetector())
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    mask, depth = make(1000)  # 정상: 1.0m
    t.mask = mask
    for _ in range(3):  # depth 기준선 형성
        pipe._update_geometry(t, depth, K)
    assert pipe._depth_ok(t, mask, depth)
    _, depth_occ = make(500)  # 가리개 침입: 0.5m (기준의 50%)
    assert not pipe._depth_ok(t, mask, depth_occ)   # 수락 전 거부


def test_gradual_depth_change_not_flagged():
    pipe = PerceptionPipeline(NoDetector())
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    for mm in range(1000, 900, -10):  # 서서히 접근 (컨베이어가 다가오는 상황)
        t.mask, depth = make(mm)
        assert pipe._depth_ok(t, t.mask, depth)  # 점진 변화는 EMA가 따라감
        pipe._update_geometry(t, depth, K)


def test_size_jump_rejected():
    """크기 급변(오염 blob)은 pose 커밋을 거부한다 — 제3 신호."""
    import numpy as np
    from roboworld_perception.geometry import ObbResult
    pipe = PerceptionPipeline(NoDetector())
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    t.mask, depth = make(1000)
    for _ in range(3):
        pipe._update_geometry(t, depth, K)
    assert not t.occluded
    obb_before = t.obb
    # 마스크를 두 배 영역으로 확장 → 관측 크기 급변 (blob 시뮬레이션)
    big = np.zeros((240, 320), bool)
    big[60:200, 40:240] = True
    d2 = np.zeros((240, 320), np.uint16)
    d2[big] = 1000
    t.mask = big
    pipe._update_geometry(t, d2, K)
    assert t.occluded              # 크기 신호로 오염 판정
    assert t.obb is obb_before     # pose 미갱신


def test_size_gate_escapes_deadlock():
    """크기 거부가 연속되면 실제 변화로 보고 재적응 — 영구 OCCLUDED 방지."""
    import numpy as np
    pipe = PerceptionPipeline(NoDetector())
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    t.mask, depth = make(1000)
    for _ in range(3):
        pipe._update_geometry(t, depth, K)
    big = np.zeros((240, 320), bool)
    big[60:200, 40:240] = True
    d2 = np.zeros((240, 320), np.uint16)
    d2[big] = 1000
    t.mask = big
    for i in range(3):  # SIZE_REJECT_LIMIT 동안은 거부
        t.occluded = False
        pipe._update_geometry(t, d2, K)
        assert t.occluded, f"reject {i+1}"
    t.occluded = False   # 4번째 — 지속되는 변화는 새 현실로 수락
    pipe._update_geometry(t, d2, K)
    assert not t.occluded
    assert t.size_rejects == 0

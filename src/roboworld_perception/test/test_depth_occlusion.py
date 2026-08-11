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

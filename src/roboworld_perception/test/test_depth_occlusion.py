import numpy as np

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
    max_per_prompt = 0

    def detect(self, rgb, prompts):
        return []


def test_depth_intrusion_flags_and_protects_obb():
    pipe = PerceptionPipeline(NoDetector())
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    mask, depth = make(1000)  # 정상: 1.0m
    t.mask = mask
    for _ in range(3):  # depth 기준선 형성
        pipe._update_geometry(t, depth, K)
    assert not t.occluded
    obb_before = t.obb
    _, depth_occ = make(500)  # 가리개 침입: 0.5m (기준의 50%)
    pipe._update_geometry(t, depth_occ, K)
    assert t.occluded              # depth 신호로 가림 판정
    assert t.obb is obb_before     # OBB 는 갱신되지 않음


def test_gradual_depth_change_not_flagged():
    pipe = PerceptionPipeline(NoDetector())
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    for mm in range(1000, 900, -10):  # 서서히 접근 (컨베이어가 다가오는 상황)
        t.mask, depth = make(mm)[0], make(mm)[1]
        pipe._update_geometry(t, depth, K)
        assert not t.occluded  # 점진 변화는 EMA 가 따라가므로 오탐 없음

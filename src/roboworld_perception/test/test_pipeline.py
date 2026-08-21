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


def two_motion_frame(shift_a=0, shift_b=0):
    """서로 다르게 움직이는 텍스처 두 장 — 마스크가 물체와 가리개에 걸친 상황."""
    rng = np.random.default_rng(1)
    frame = np.full((240, 320), 60, np.uint8)
    frame[100:160, 100 + shift_a:140 + shift_a] = rng.integers(
        0, 255, (60, 40), dtype=np.uint8)
    frame[100:160, 140 + shift_b:180 + shift_b] = rng.integers(
        0, 255, (60, 40), dtype=np.uint8)
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


def test_flow_suppressed_when_mask_straddles_two_motions():
    """마스크 절반이 가리개를 따라 움직이면 전파하지 않는다.

    중앙값만 쓰면 큰 쪽으로 조용히 끌려가 마스크가 물체를 떠나고, 떠난
    마스크가 다음 프레임의 흐름을 다시 정하는 자기강화가 시작된다
    (test4 book 실측: 잔차 171 -> 290mm 단조 증가, 발행 9.5초 단절).
    depth 도 score 도 이 상황을 못 잡는다 — 가리개가 같은 높이고 점수는 높다.
    """
    mask = np.zeros((240, 320), bool)
    mask[100:160, 100:180] = True
    prev, cur = two_motion_frame(0, 0), two_motion_frame(0, 12)
    assert propagate_mask(prev, cur, mask) is None


def test_fast_coherent_motion_still_propagates():
    """빠른 물체를 여러 물체로 오인하면 안 된다 — 허용 반경이 변위 크기에
    비례하는 이유 (test3 손밀기 137mm/s 구간)."""
    mask = np.zeros((240, 320), bool)
    mask[100:160, 100:180] = True
    dx, dy = propagate_mask(textured_frame(0), textured_frame(14), mask)
    assert abs(dx - 14) < 2.0
    assert abs(dy) < 2.0


def test_flow_suppression_carries_no_state():
    """보류는 그 프레임의 영상만 보고 판정한다 — 한 번 걸렸다고 다음 프레임의
    정상 흐름까지 막으면 교착이 된다."""
    mask = np.zeros((240, 320), bool)
    mask[100:160, 100:180] = True
    assert propagate_mask(two_motion_frame(0, 0), two_motion_frame(0, 12),
                          mask) is None
    dx, _ = propagate_mask(textured_frame(0), textured_frame(7), mask)
    assert abs(dx - 7) < 1.5


def test_suppressed_flow_does_not_withhold_observations():
    """전파를 보류해도 관측·융합은 계속한다 (교착 불가능성·신선도 보존).

    보류가 관측까지 끊으면 T_STALE 이 흘러 발행이 멈추고, P 만 자라는 구간이
    생겨 fusion 설계의 '거부 중에도 게이트가 스스로 열린다'가 무의미해진다.
    """
    pipe = PerceptionPipeline(StubDetector(), detect_interval=5)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    depth = np.full((240, 320), 1000, np.uint16)
    for i in range(4):  # 0=키프레임, 1~3=흐름이 갈라지는 프레임
        rgb = cv2.cvtColor(two_motion_frame(0, 12 * i), cv2.COLOR_GRAY2RGB)
        pipe.process(rgb, depth, K, ["obj"], stamp_s=i / 15)
    t = pipe.tracker.tracks[0]
    assert t.fresh                     # 관측이 계속 들어왔다
    assert t.last_accept_t == 3 / 15   # 마지막 프레임까지 수락
    assert np.allclose(t.box, [100, 100, 180, 160])  # 마스크는 안 밀렸다
def _plane_pipe():
    """벨트를 z=1.0 에 고정한 평면 구속 파이프라인 (캘리브 노브로 직접 주입)."""
    from roboworld_perception.pipeline import PerceptionPipeline
    return PerceptionPipeline(StubDetector(), detect_interval=5,
                              use_belt_plane=True,
                              belt_plane=(np.array([0.0, 0.0, -1.0]), 1.0))


def _run(pipe, obj_depth_mm):
    """마스크 영역만 obj_depth_mm 인 장면을 여러 프레임 흘린다."""
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    depth = np.full((240, 320), 1000, np.uint16)   # 벨트 z=1.0m
    depth[100:160, 100:180] = obj_depth_mm
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    for _ in range(6):
        pipe.process(rgb, depth, K, ["obj"])
    return pipe.tracker.tracks[0]


def test_impossible_thickness_never_seeds_a_track():
    """물리적으로 불가능한 두께 관측은 트랙을 시드하지 못한다.

    시드 시점에 들어온 값은 χ² 가 기각할 점프가 없어서 그대로 트랙의 진실이
    된다 — test5 실측: gray notebook 쓰레기 트랙이 두께 530mm 로 시드돼
    중심이 265mm 틀린 채 8프레임 발행됐다. publishable 은 pos_std 만 보고
    크기를 안 보므로 발행 단계에서는 못 막는다.
    """
    track = _run(_plane_pipe(), 470)      # 벨트 1000mm - 470mm = 두께 530mm
    assert track.filter is None           # 시드 자체가 안 됨
    assert not track.publishable


def test_normal_thickness_still_seeds():
    """대조군 — 같은 경로로 정상 두께(50mm)는 그대로 시드·발행된다."""
    track = _run(_plane_pipe(), 950)      # 두께 50mm
    assert track.filter is not None
    assert abs(float(track.filter.extent_sorted[2]) - 0.050) < 0.005

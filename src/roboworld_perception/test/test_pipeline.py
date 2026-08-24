import cv2
import numpy as np

from conftest import K
from roboworld_perception.pipeline import (PerceptionPipeline,
                                           STAMP_RESET_BACKWARD_S,
                                           STAMP_RESET_NEW_RUN_MAX_S,
                                           propagate_mask)


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
    # 반환은 (plane, 쓰인 dist_thresh) 다 — 폴백이 3mm 로 붙었는지 6mm 로
    # 떨어졌는지가 두께를 ~1.3mm 바꾸는데 실행 간 산포(±0.40mm)보다 커서,
    # 로그에 남기지 않으면 두 실행이 같은 문턱을 썼는지 확인할 방법이 없다.
    monkeypatch.setattr(P, "_fit_support_plane",
                        lambda *a, **kw: (tries.append(1), (None, None))[1])
    pipe = P.PerceptionPipeline(StubDetector(), detect_interval=5,
                                use_belt_plane=True)
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


def test_suppressed_flow_does_not_refresh_pose_and_keeps_display(monkeypatch):
    """일관성 없는 흐름은 이전 mask를 새 3D 관측으로 재사용하지 않는다.

    마지막 안전 pose는 keyframe 재검출 전에도 표시하지만, 수락 시각을
    갱신하면 가려진 물체의 stale pose가 로봇으로 나간다. 따라서
    T_STALE 뒤 발행은 멈추고 track만 표시용으로 반환해야 한다.
    """
    from roboworld_perception import pipeline as P

    pipe = _plane_pipe()
    depth = np.full((240, 320), 1000, np.uint16)
    depth[100:160, 100:180] = 950
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    # 이 합성 장면은 keyframe에서만 수락되므로 세 keyframe으로 publishable
    # pose를 만든 뒤, 다음 네 flow 프레임을 실패시킨다.
    for frame_idx in range(11):
        pipe.process(rgb, depth, K, ["obj"], stamp_s=10 + frame_idx / 15)
    t = pipe.tracker.tracks[0]
    last_accept_t, box_before = t.last_accept_t, t.box.copy()
    assert t.publishable

    monkeypatch.setattr(P, "propagate_mask", lambda *args: None)
    for stamp in (10.9, 11.1, 11.3, 11.5):
        out = pipe.process(rgb, depth, K, ["obj"], stamp_s=stamp)
        assert out == [t]               # pose는 화면 표시용으로 유지

    assert t.last_accept_t == last_accept_t
    assert t.n_accepted == 3            # 새 3D 관측으로 세지 않음
    assert np.array_equal(t.box, box_before)
    assert not t.fresh and not t.publishable
def _plane_pipe():
    """벨트를 z=1.0 에 고정한 평면 구속 파이프라인 (캘리브 노브로 직접 주입)."""
    return PerceptionPipeline(StubDetector(), detect_interval=5,
                              use_belt_plane=True,
                              belt_plane=(np.array([0.0, 0.0, -1.0]), 1.0))


def _run(pipe, obj_depth_mm):
    """마스크 영역만 obj_depth_mm 인 장면을 여러 프레임 흘린다."""
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


def test_reset_restores_run_state_without_losing_settings():
    """reset() 은 런 상태만 지우고 설정은 남긴다.

    이 함수는 오랫동안 **어떤 테스트도 부르지 않아** 무검증이었고, 2026-08-21 에
    생성자 인자 대입이 복제돼 NameError 로 죽은 채 90개 테스트가 전부 통과했다.
    유일한 호출처가 런타임 프롬프트 교체(/perception/prompt)라 제품에서만
    드러난다. 설정과 런 상태를 양쪽으로 고정한다.
    """
    pipe = PerceptionPipeline(StubDetector(), detect_interval=5,
                              enable_footprint_gate=False, use_belt_plane=False)
    depth = np.full((240, 320), 1000, np.uint16)
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    for _ in range(7):
        pipe.process(rgb, depth, K, ["obj"])
    assert pipe.tracker.tracks and pipe._frame_idx == 7

    pipe.reset()

    assert pipe._frame_idx == 0 and pipe._prev_gray is None   # 런 상태는 지워지고
    assert pipe._last_stamp is None and not pipe.tracker.tracks
    assert pipe.enable_footprint_gate is False                # 설정은 남는다
    assert pipe.use_belt_plane is False
    assert pipe.detect_interval == 5
    pipe.process(rgb, depth, K, ["obj"])                      # 리셋 후에도 돈다


def test_reset_keeps_a_pinned_belt_plane():
    """직접 준 평면(캘리브 값)은 reset 이 버리지 않는다 — _plane_fixed 계약."""
    plane = (np.array([0.0, 0.0, -1.0]), 1.0)
    pipe = PerceptionPipeline(StubDetector(), belt_plane=plane)
    pipe.reset()
    assert pipe.belt_plane is plane


def test_material_time_reversal_drops_old_publishable_pose():
    """새 run의 시간축에서 이전 run pose를 fresh로 되살리지 않는다.

    Track.fresh 는 ``now - last_accept_t`` 를 보므로, /clock 이 0으로
    되감긴 뒤 기존 트랙을 유지하면 이전에 확정된 pose가 수백 초 동안
    publishable 로 남는다. 역행 프레임은 먼저 reset한 뒤 새 관측으로만
    시드해야 하며, M-of-N 확정을 다시 충족할 때까지 발행하면 안 된다.
    """
    pipe = _plane_pipe()
    depth = np.full((240, 320), 1000, np.uint16)
    depth[100:160, 100:180] = 950
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    # 이 합성 장면은 keyframe에서만 pose가 수락되므로, CONFIRM_N=3을
    # 만족할 때까지 세 keyframe을 지난다.
    for frame_idx in range(11):
        stamp = 10.0 + frame_idx / 15
        old = pipe.process(rgb, depth, K, ["obj"], stamp_s=stamp)[0]
    assert old.publishable
    assert old.last_accept_t == 10.0 + 10 / 15

    out = pipe.process(rgb, depth, K, ["obj"], stamp_s=0.0)

    assert pipe._last_stamp == 0.0
    assert pipe._frame_idx == 1
    assert len(pipe.tracker.tracks) == 1
    assert pipe.tracker.tracks[0] is not old
    assert pipe.tracker.tracks[0].n_accepted == 1
    assert not any(track.publishable for track in out)


def test_small_stamp_reversal_does_not_reset_run_or_pinned_plane():
    """clock 샘플의 작은 순서 뒤바뀜은 run reset으로 과민 반응하지 않는다."""
    plane = (np.array([0.0, 0.0, -1.0]), 1.0)
    pipe = PerceptionPipeline(StubDetector(), belt_plane=plane,
                              detect_interval=5)
    depth = np.full((240, 320), 1000, np.uint16)
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    pipe.process(rgb, depth, K, ["obj"], stamp_s=10.0)
    original = pipe.tracker.tracks[0]

    pipe.process(rgb, depth, K, ["obj"],
                 stamp_s=10.0 - STAMP_RESET_BACKWARD_S / 2)

    assert pipe.tracker.tracks[0] is original
    assert pipe.belt_plane is plane


def test_late_nonzero_stamp_is_not_mistaken_for_clock_reset():
    """지연 전달된 과거 RGB-D 쌍은 새 run이 아니라 버릴 수 있는 입력이다."""
    pipe = PerceptionPipeline(StubDetector(), detect_interval=5)
    depth = np.full((240, 320), 1000, np.uint16)
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    pipe.process(rgb, depth, K, ["obj"], stamp_s=12.0)
    original = pipe.tracker.tracks[0]

    assert not pipe.time_reset_required(8.0)
    assert pipe.late_frame_drop_required(8.0)
    assert pipe.process(rgb, depth, K, ["obj"], stamp_s=8.0) == []
    assert pipe.tracker.tracks[0] is original


def test_clock_return_to_new_run_origin_requires_reset():
    """Isaac resetOnStop의 새 clock origin은 기존 pose를 즉시 무효화한다."""
    pipe = PerceptionPipeline(StubDetector(), detect_interval=5)
    depth = np.full((240, 320), 1000, np.uint16)
    rgb = cv2.cvtColor(textured_frame(0), cv2.COLOR_GRAY2RGB)
    pipe.process(rgb, depth, K, ["obj"], stamp_s=12.0)

    assert pipe.time_reset_required(STAMP_RESET_NEW_RUN_MAX_S)

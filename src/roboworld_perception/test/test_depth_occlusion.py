"""파이프라인 수준 융합 게이트 동작 — 이전 이진 게이트(depth·size)가 막던
시나리오를 필터가 같은 수준으로 막고, 이진 게이트의 결함(교착·stale 발행)은
구조적으로 사라졌는지 검증한다."""
import numpy as np

from conftest import K
from roboworld_perception.geometry import masked_depth_median
from roboworld_perception.pipeline import PerceptionPipeline
from roboworld_perception.tracker import Track


def _rect_scene(depth_mm, u0, u1, v0, v1):
    """직사각 마스크 + 균일 depth 합성 장면 (이 파일의 장면 생성 단일 정의)."""
    depth = np.zeros((240, 320), np.uint16)
    mask = np.zeros((240, 320), bool)
    sl = (slice(v0, v1), slice(u0, u1))
    mask[sl] = True
    depth[sl] = depth_mm
    return mask, depth


def make(depth_mm):
    """기본 장면 — 픽셀 고정이라 거리가 변하면 미터 크기가 z² 로 줄어든다.
    그 성질이 문제되는 테스트는 make_rigid 를 쓴다."""
    return _rect_scene(depth_mm, 100, 180, 100, 160)


class NoDetector:
    def detect(self, rgb, prompts):
        return []


def step(pipe, t, depth, dt=1 / 15):
    """process()가 하는 시간 전진(predict) + 기하 융합을 트랙 하나에 재현.
    승격 카운트가 키프레임 수락만 세므로 키프레임으로 취급한다."""
    pipe.last_was_keyframe = True
    t.now += dt
    if t.filter is not None:
        t.filter.predict(dt)
    pipe._update_geometry(t, depth, K)


def seeded_track(pipe, depth_mm=1000, frames=5):
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    t.mask, depth = make(depth_mm)
    for _ in range(frames):
        step(pipe, t, depth)
    return t


def test_masked_depth_median():
    mask, depth = make(1000)
    assert abs(masked_depth_median(mask, depth) - 1.0) < 1e-6
    assert masked_depth_median(np.zeros((240, 320), bool), depth) is None


def test_depth_intrusion_rejected():
    """가리개 침입(depth 급변)은 게이트가 거부 — pose·상태 미오염."""
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    assert t.publishable
    z_before = t.obb.center[2]
    _, depth_occ = make(500)  # 가리개: 0.5m (기준 1.0m의 50%)
    step(pipe, t, depth_occ)
    assert abs(t.obb.center[2] - z_before) < 0.01  # 상태에 침입 반영 안 됨


def test_stale_pose_stops_publishing():
    """관측이 거부되거나 아예 없으면(T_STALE 경과) 발행이 멈춘다 —
    이전 코드의 'depth 소실 후 stale pose 영구 발행' 버그 방어."""
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    assert t.publishable
    _, depth_occ = make(500)
    for _ in range(10):  # 0.67s > T_STALE 동안 침입 관측만
        step(pipe, t, depth_occ)
    assert not t.publishable
    assert t.occluded  # 표시도 가림(회색)으로 전환
    # depth 완전 소실(obb None)도 같은 경로로 멈춘다
    t2 = seeded_track(pipe)
    for _ in range(10):
        step(pipe, t2, np.zeros((240, 320), np.uint16))
    assert not t2.publishable


def test_flow_paused_on_intrusion():
    """가리개가 마스크 위에 오면 flow 전파·융합을 보류한다 — LK가 가리개를
    따라가며 마스크가 물체를 떠나는 것 차단 (라이브 실측: 서류 파일에
    마스크가 붙어 따라감). 보류 중에도 P는 자라 유한 시간 뒤 풀린다."""
    from roboworld_perception.tracker import depth_intrusion
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    _, depth_occ = make(500)
    z_occ = masked_depth_median(t.mask, depth_occ)
    assert depth_intrusion(t, z_occ)          # 침입 판정 → _track_frame 보류
    _, depth_ok = make(1000)
    assert not depth_intrusion(t, masked_depth_median(t.mask, depth_ok))
    for _ in range(200):                       # 보류가 이어지면 σ가 자라서
        t.now += 0.1
        t.filter.predict(0.1)
    assert not depth_intrusion(t, z_occ)       # 유한 시간 안에 해제 (교착 없음)


def make_rigid(depth_mm, w_m=0.2667, h_m=0.2):
    """거리가 변해도 **미터 크기가 같은** 강체의 마스크 — 가까워지면 픽셀이
    커진다. make()는 픽셀을 고정해서 물체가 다가올수록 미터 크기가 z²로
    줄어드는데, 그건 실제 검출기가 내는 마스크가 아니다(그 상황은 부분 가림과
    구분이 안 된다 — pipeline._footprint_deviation 참고)."""
    z = depth_mm / 1000.0
    hw = int(round(w_m / 2 * K[0, 0] / z))
    hh = int(round(h_m / 2 * K[1, 1] / z))
    cu, cv = int(K[0, 2] - 20), int(K[1, 2] + 10)
    return _rect_scene(depth_mm, max(0, cu - hw), cu + hw,
                       max(0, cv - hh), cv + hh)


def test_gradual_depth_change_tracked():
    """벨트 접근 속도(~45mm/s = 3mm/frame)의 점진 변화는 계속 수락된다."""
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    for mm in range(1000, 910, -3):
        t.mask, depth = make_rigid(mm)
        step(pipe, t, depth)
    assert t.publishable
    assert abs(t.obb.center[2] - 0.91) < 0.01  # 상태가 변화를 따라감


def test_blob_size_jump_rejected_but_position_survives():
    """오염 blob(가리개와 병합된 마스크)의 크기 급변은 extent 게이트가
    거부하되, 중심이 멀쩡하면 위치는 계속 서비스된다 — 이전의 '트랙 전체
    OCCLUDED' 대비 부분 수락."""
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    ext_before = np.sort(t.obb.extent)[::-1]
    big = np.zeros((240, 320), bool)
    big[60:200, 40:240] = True  # 두 배 이상 영역 (중심은 동일)
    d2 = np.zeros((240, 320), np.uint16)
    d2[big] = 1000
    t.mask = big
    step(pipe, t, d2)
    assert np.allclose(np.sort(t.obb.extent)[::-1], ext_before, atol=1e-6)
    assert t.publishable  # 위치는 수락 유지


def test_persistent_size_change_eventually_adapts():
    """크기 변화가 지속되면(실제 변화) 유한 시간 안에 재적응한다 —
    거부 중 불확실성이 자라 게이트가 스스로 열린다 (교착 불가)."""
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    e1_before = float(np.exp(t.filter.le[0]))
    big = np.zeros((240, 320), bool)
    big[60:200, 40:240] = True
    d2 = np.zeros((240, 320), np.uint16)
    d2[big] = 1000
    t.mask = big
    # 2배 급변의 탈출은 ~35s(극단 편차일수록 오래 거부 — 설계 의도),
    # 1.5× 수준의 실제 변화는 ~3s에 적응한다
    for _ in range(600):  # dt=0.1 → 60s 상당
        step(pipe, t, d2, dt=0.1)
        if np.exp(t.filter.le[0]) > 1.5 * e1_before:
            break
    assert np.exp(t.filter.le[0]) > 1.5 * e1_before  # 새 크기로 수렴 시작


def test_thick_object_top_surface_is_not_intrusion():
    """침입 판정의 기준선은 물체 중심이 아니라 **상면**이어야 한다.

    비교 대상 z 는 masked_depth_median, 즉 보이는 면의 depth 다. 벨트 평면
    구속 이후 filter.center 는 상면보다 h/2 만큼 멀어져서, 중심과 비교하면
    정상 검출이 매 프레임 "가까운 것"으로 보인다. 통계 절(3.29σ, σ≈2mm)은
    h/2≈15mm 면 이미 통과하므로 물리 절(0.9·z) 하나만 남는데, 실측
    black bag 이 h/2 = 0.083~0.097·z 로 그 문턱에 붙어 있다 — 두께 20cm 면
    넘는다. 넘으면 영구 가림 판정 → flow 보류 → 발행 중단이다.
    """
    from conftest import make_filtered_track
    from roboworld_perception.geometry import ObbResult
    from roboworld_perception.tracker import depth_intrusion
    z_center, thk = 0.9, 0.20
    t = make_filtered_track(z=z_center, extent=(0.44, 0.30, thk))
    t.obb = ObbResult(center=t.filter.center.copy(),
                      extent=np.array([0.44, 0.30, thk]),
                      R=np.eye(3), num_points=100)
    assert abs(t.surface_z - (z_center - thk / 2)) < 1e-6
    assert not depth_intrusion(t, t.surface_z)        # 정상 검출
    assert depth_intrusion(t, t.surface_z - 0.2)      # 진짜 가리개


def make_partial(depth_mm, cover_px):
    """가리개가 마스크의 왼쪽 cover_px 만큼을 덮은 관측 — 보이는 영역이 줄고
    그 중심이 오른쪽으로 밀린다. 부분 가림의 최소 재현."""
    return _rect_scene(depth_mm, 100 + cover_px, 180, 100, 160)


def test_partial_occlusion_does_not_move_state():
    """가리개에 잘린 관측은 상태를 옮기지 못한다.

    보이는 부분의 중심은 물체의 중심이 아니다 — 화면 절단(_touches_border)에서
    이미 아는 사실인데 가리개 절단은 아무도 안 보고 있었다. 필터의 extent 로는
    못 잡는다: 프레임당 변화가 작아 χ² 를 매번 통과하고, 통과할 때마다 extent
    상태가 따라 내려가 다음 프레임의 기준이 된다(천 번의 작은 베임).
    실측 test4 book: 풋프린트 290x241 -> 267x170mm 동안 중심이 14 -> 95mm.
    """
    pipe = PerceptionPipeline(NoDetector(), enable_footprint_gate=True)
    t = seeded_track(pipe)
    x_before = float(t.filter.center[0])
    for _ in range(10):                      # 절반쯤 덮인 관측만 계속
        t.mask, depth = make_partial(1000, 40)
        step(pipe, t, depth)
    assert abs(float(t.filter.center[0]) - x_before) < 0.005   # 중심 안 밀림
    assert not t.publishable                                   # 발행도 멈춤


def test_partial_occlusion_escape_is_bounded():
    """기준을 동결하지 않기 때문에 교착이 구조적으로 불가능하다.

    물체가 실제로 작아졌다면(또는 트랙이 다른 것에 붙었다면) 기준이 기하급수로
    따라가 유한 시간 안에 플래그가 풀린다. 동결 + '연속 N 회 일관하면 채택'
    관용구는 여기서 못 쓴다 — 매끄러운 드리프트는 언제나 자기들끼리 일관해서
    그 탈출구로 그냥 걸어 들어온다(실측: 검출이 72% -> 14% 로 무너졌다).
    """
    pipe = PerceptionPipeline(NoDetector(), enable_footprint_gate=True)
    t = seeded_track(pipe)
    accepted = None
    for i in range(300):
        t.mask, depth = make_partial(1000, 40)
        before = t.last_accept_t
        step(pipe, t, depth)
        if t.last_accept_t != before:
            accepted = i
            break
    assert accepted is not None, "지속되는 새 크기가 영구 기각됐다 (교착)"
    assert accepted < 200          # 상계 — α=0.02 의 기하급수 수렴


def test_small_footprint_wobble_still_accepted():
    """TAU 안의 정상 크기 요동은 그대로 수락된다 (게이트가 과민하지 않다)."""
    pipe = PerceptionPipeline(NoDetector(), enable_footprint_gate=True)
    t = seeded_track(pipe)
    for cover in (0, 3, 0, 4, 0, 3):        # 면적 ~5% 요동 (TAU=0.14 안)
        t.mask, depth = make_partial(1000, cover)
        step(pipe, t, depth)
    assert t.publishable

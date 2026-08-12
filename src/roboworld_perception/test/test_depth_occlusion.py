"""파이프라인 수준 융합 게이트 동작 — 이전 이진 게이트(depth·size)가 막던
시나리오를 필터가 같은 수준으로 막고, 이진 게이트의 결함(교착·stale 발행)은
구조적으로 사라졌는지 검증한다."""
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


def test_gradual_depth_change_tracked():
    """벨트 접근 속도(~45mm/s = 3mm/frame)의 점진 변화는 계속 수락된다."""
    pipe = PerceptionPipeline(NoDetector())
    t = seeded_track(pipe)
    for mm in range(1000, 910, -3):
        t.mask, depth = make(mm)
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

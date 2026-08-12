"""융합 필터 불변식 property test — 설계의 '장면 독립' 주장을 고정한다.

여기의 상수(2.6s, 0.554m, 46mm/s)는 test4/5 실측이지만, 테스트가 검증하는
것은 그 값에 맞춘 튜닝이 아니라 구조적 성질(단조성·유한성)이다.
"""
import numpy as np
import pytest

from roboworld_perception.fusion import CHI2_3DOF, TrackFilter

DT = 1 / 15  # 공칭 프레임 주기


def make(center=(0.0, 0.0, 0.9), extent=(0.3, 0.2, 0.05)):
    f = TrackFilter(center, np.log(extent))
    for _ in range(30):  # 정상 관측으로 수렴시켜 정상상태에서 시작
        f.predict(DT)
        f.fuse_pos(np.array(center))
        f.fuse_extent(np.log(extent))
    return f


def escape_frames(f, delta, axis=2, limit=3000):
    """일정 편차 delta를 계속 관측시켰을 때 게이트가 열릴 때까지 프레임 수."""
    z = f.center.copy()
    z[axis] += delta
    for i in range(1, limit + 1):
        f.predict(DT)
        if f.fuse_pos(z):
            return i
    return limit + 1


def test_rejection_grows_acceptance_region():
    """불변식 1: 거부(미수락)가 이어지는 동안 수락 영역은 단조 증가."""
    f = make()
    prev = np.inf
    widths = []
    for _ in range(50):
        f.predict(DT)
        widths.append(f.P[:, 0, 0].sum())
    assert all(b > a for a, b in zip(widths, widths[1:]))


def test_escape_monotonic_in_delta_and_finite():
    """불변식 3: 탈출 시간은 편차에 단조 증가, 그리고 항상 유한(교착 불가)."""
    t = [escape_frames(make(), d) for d in (0.05, 0.2, 0.554)]
    assert t[0] < t[1] < t[2]
    assert t[2] < 3000  # 최대 편차도 유한 시간 안에 탈출


def test_occluder_held_out_long_enough():
    """F1/F2: 554mm 침입(test4 실측)은 실측 최대 가림 2.6s의 2배 이상 거부."""
    assert escape_frames(make(), 0.554) * DT >= 5.2


def test_partial_occlusion_deviation_outlasts_occlusion():
    """부분 가림의 중간 편차(~100mm)도 가림 지속(최대 2.6s)보다 오래 거부 —
    이게 σ_a의 지배 요구다. 못 지키면 가림이 끝나기 전에 게이트가 열려
    가리개 꼬리를 채택한다 (test4/5 회귀 실측: keyboard z-std 1.7→69mm)."""
    assert escape_frames(make(), 0.1) * DT >= 2.6 * 1.4


def test_honest_change_escapes_quickly():
    """F3: 정직한 5cm 변화(물체 이동·재배치)는 유한 시간(~2s대)에 수락된다.
    부분 가림 방어(위 테스트)와 상충하는 요구라 ~1s까지 줄이지 않는다 —
    교착(무한 거부)만 없으면 F3는 해결이다."""
    assert escape_frames(make(), 0.05) * DT <= 3.0


def test_tracks_conveyor_ramp_without_gating():
    """CV 상태의 존재 이유: 공칭 벨트 46mm/s 램프를 게이트 거부 없이 추적.
    (정지 모델+EMA는 이 입력에서 d²=40 — 게이트 영구 폐쇄였다)"""
    f = make()
    z = f.center.copy()
    rejects = 0
    for _ in range(150):
        z[0] += 0.046 * DT
        f.predict(DT)
        rejects += 0 if f.fuse_pos(z) else 1
    assert rejects == 0
    assert abs(f.center[0] - z[0]) < 0.005  # 잔여 lag < 5mm


def test_static_jitter_not_amplified():
    """정지 물체 + 측정 잡음에서 출력 지터가 입력 잡음을 넘지 않는다."""
    rng = np.random.default_rng(0)
    f = make()
    outs = []
    for _ in range(300):
        f.predict(DT)
        f.fuse_pos(np.array([0, 0, 0.9]) + rng.normal(0, 2e-3, 3))
        outs.append(f.center.copy())
    assert np.array(outs)[50:].std(axis=0).max() < 2e-3


def test_rhat_safeguards():
    """불변식 2: R̂은 하한(물리 모델) 밑으로 안 내려가고, 거부 관측으로는
    갱신되지 않으며, 상한이 있다."""
    from roboworld_perception.fusion import R_POS_FLOOR, RHAT_CAP
    f = make()
    assert (f.rhat_pos >= R_POS_FLOOR ** 2 - 1e-15).all()
    before = f.rhat_pos.copy()
    f.predict(DT)
    accepted = f.fuse_pos(f.center + np.array([0, 0, 0.554]))  # 침입 → 거부
    assert not accepted
    assert np.array_equal(f.rhat_pos, before)  # 거부는 R̂을 못 건드린다
    assert (f.rhat_pos <= RHAT_CAP * R_POS_FLOOR ** 2 + 1e-15).all()


def test_noisy_object_adapts_instead_of_deadlocking():
    """실측 근거(검은 가방: 깨끗한 물체 대비 20×+ 잡음): 물리 하한만으로는
    만성 기각될 잡음 규모에서도 R̂ 적응으로 수락률이 회복된다."""
    rng = np.random.default_rng(1)
    f = make()
    acc = []
    for i in range(400):
        f.predict(DT)
        acc.append(f.fuse_pos(np.array([0, 0, 0.9]) + rng.normal(0, 0.02, 3)))
    assert np.mean(acc[200:]) > 0.7  # 후반부: 잡음 규모를 학습해 대부분 수락


def test_occluder_rejected_even_at_rhat_cap():
    """R̂이 상한(최다잡음 물체 — 검은 가방류)까지 부풀어도 554mm 침입은
    기각된다 — 상한 값의 존재 이유이자 F1/F2의 최후 방어선."""
    from roboworld_perception.fusion import R_POS_FLOOR, RHAT_CAP
    f = make()
    f.rhat_pos[:] = RHAT_CAP * R_POS_FLOOR ** 2
    f.predict(DT)
    assert not f.fuse_pos(f.center + np.array([0, 0, 0.554]))


def test_extent_shrink_not_absorbing():
    """B-2 회귀 방지: 크기 '축소' 관측도 정상적으로 기각·수락된다 —
    축소 방향 흡수상태(항상 수락) 금지."""
    f = make(extent=(0.3, 0.2, 0.05))
    f.predict(DT)
    # 절반 크기 급변(가리개로 반쪽만 보임) → 기각되어야 함
    assert not f.fuse_extent(np.log((0.15, 0.1, 0.025)))
    # 미세 축소(정상 잡음)는 수락
    f.predict(DT)
    assert f.fuse_extent(np.log((0.299, 0.199, 0.0499)))

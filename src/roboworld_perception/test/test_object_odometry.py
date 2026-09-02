"""ROS-independent tests for the body-frame twist contract (object_odometry.py)."""
import numpy as np
import pytest

from roboworld_perception.object_odometry import ANGULAR_VEL_UNKNOWN, body_twist


def test_identity_rotation_is_a_no_op_on_linear_terms():
    """R=단위행렬이면 물체 프레임 = 카메라 프레임이라 값이 그대로 통과해야 한다."""
    v = np.array([0.3, -0.1, 0.05])
    vel_var = np.array([1e-3, 2e-3, 3e-3])

    v_body, cov = body_twist(np.eye(3), v, vel_var)

    assert v_body == pytest.approx(v)
    assert cov.shape == (36,)
    cov6 = cov.reshape(6, 6)
    assert np.diag(cov6)[:3] == pytest.approx(vel_var)
    assert np.diag(cov6)[3:] == pytest.approx([ANGULAR_VEL_UNKNOWN] * 3)
    # 대각 6개를 뺀 나머지는 전부 0 — 선형-각 교차항도, 축간 교차항도 없다.
    off_diag_mask = ~np.eye(6, dtype=bool)
    assert cov6[off_diag_mask] == pytest.approx(np.zeros(30))


def test_z_axis_rotation_matches_hand_computed_body_frame_velocity():
    """R.T @ v 가 실제로 물체 좌표축 기준으로 옮기는 변환인지 손계산으로 확인."""
    # z축 +90도 회전: 열이 물체 로컬 X/Y/Z 축(카메라 좌표 표현)
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    v = np.array([1.0, 0.0, 0.0])
    vel_var = np.array([1e-3, 2e-3, 3e-3])

    v_body, cov = body_twist(R, v, vel_var)

    assert v_body == pytest.approx(R.T @ v)
    assert v_body == pytest.approx([0.0, -1.0, 0.0])
    linear_cov = cov.reshape(6, 6)[:3, :3]
    assert linear_cov == pytest.approx(R.T @ np.diag(vel_var) @ R)


def test_body_frame_velocity_round_trips_back_to_optical_frame():
    """docstring이 약속한 복원식(R @ v_body == v)이 실제로 성립하는지 — 소비자 계약."""
    rng = np.random.default_rng(0)
    # 임의의 회전행렬(QR 분해로 정규직교화, det=+1로 정렬)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    v = np.array([0.2, -0.4, 0.1])

    v_body, _ = body_twist(q, v, np.array([1e-3, 1e-3, 1e-3]))

    assert q @ v_body == pytest.approx(v)

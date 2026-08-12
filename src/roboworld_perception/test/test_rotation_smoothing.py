import numpy as np
from scipy.spatial.transform import Rotation

from roboworld_perception.fusion import TrackFilter
from roboworld_perception.geometry import ObbResult
from roboworld_perception.tracker import Track


def obb_with_yaw(deg):
    return ObbResult(center=np.zeros(3), extent=np.array([0.2, 0.1, 0.05]),
                     R=Rotation.from_euler("z", deg, degrees=True).as_matrix(),
                     num_points=100)


def make_track():
    t = Track(1, "obj", np.zeros(4), 0.9)
    t.filter = TrackFilter(np.zeros(3), np.log([0.2, 0.1, 0.05]))
    return t


def yaw_of(track):
    return Rotation.from_matrix(track.obb.R).as_euler("ZYX", degrees=True)[0]


def test_deadband_freezes_small_jitter():
    t = make_track()
    t.update_obb(obb_with_yaw(0))
    for jitter in [1.5, -1.2, 0.8, -1.9]:  # 2도 미만 노이즈
        t.update_obb(obb_with_yaw(jitter))
    assert abs(yaw_of(t)) < 1e-6  # 완전히 고정


def test_real_rotation_still_tracks():
    t = make_track()
    t.update_obb(obb_with_yaw(0))
    for _ in range(40):  # 물체가 실제로 30도 돌아간 상태 유지
        t.update_obb(obb_with_yaw(30))
    assert yaw_of(t) > 25  # 스무딩이 강해도 수십 프레임 내 수렴

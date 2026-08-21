import numpy as np
from scipy.spatial.transform import Rotation

from roboworld_perception.geometry import (MIN_THICKNESS, compute_obb,
                                           fit_plane, mask_depth_to_points,
                                           match_axes, obb_on_plane)

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def make_box_scene(center=(0.0, 0.0, 1.0), size=(0.20, 0.10, 0.05)):
    """Synthetic top-down view of a box: depth image + mask."""
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    cx, cy, cz = center
    w, h, thickness = size
    z_top = cz - thickness  # top face closer to camera
    u0 = int(K[0, 2] + (cx - w / 2) / z_top * K[0, 0])
    u1 = int(K[0, 2] + (cx + w / 2) / z_top * K[0, 0])
    v0 = int(K[1, 2] + (cy - h / 2) / z_top * K[1, 1])
    v1 = int(K[1, 2] + (cy + h / 2) / z_top * K[1, 1])
    depth[v0:v1, u0:u1] = int(z_top * 1000)
    mask[v0:v1, u0:u1] = True
    return mask, depth


def test_backprojection_center_and_size():
    mask, depth = make_box_scene()
    pts = mask_depth_to_points(mask, depth, K, stride=1, erode_px=0)
    assert len(pts) > 1000
    assert abs(np.median(pts[:, 2]) - 0.95) < 0.01  # top face at 1.0 - 0.05
    assert abs(pts[:, 0].max() - pts[:, 0].min() - 0.20) < 0.01
    assert abs(pts[:, 1].max() - pts[:, 1].min() - 0.10) < 0.01


def test_obb_recovers_extent():
    mask, depth = make_box_scene()
    pts = mask_depth_to_points(mask, depth, K, stride=1, erode_px=0)
    obb = compute_obb(pts, voxel=0.003)
    assert obb is not None
    ext = np.sort(obb.extent)[::-1]
    assert abs(ext[0] - 0.20) < 0.02
    assert abs(ext[1] - 0.10) < 0.02
    assert abs(obb.center[2] - 0.95) < 0.02
    assert 0.9 < obb.distance < 1.0


def test_outlier_rejection():
    mask, depth = make_box_scene()
    depth[mask][:5] = 3000  # a few far outliers inside the mask
    ys, xs = np.nonzero(mask)
    depth[ys[:20], xs[:20]] = 2900
    pts = mask_depth_to_points(mask, depth, K, stride=1, erode_px=0)
    assert pts[:, 2].max() < 1.1  # outliers clipped by MAD filter


def test_match_axes_fixes_flip_and_permutation():
    R_prev = np.eye(3)
    ext = np.array([0.2, 0.1, 0.05])
    # permute (swap x/y) and flip one axis, keep right-handed
    perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]).T
    R_new = R_prev @ perm
    ext_new = ext[[1, 0, 2]]
    R_fixed, ext_fixed = match_axes(R_new, ext_new, R_prev)
    assert np.allclose(np.abs(np.diag(R_prev.T @ R_fixed)), 1, atol=1e-6)
    assert np.all(np.diag(R_prev.T @ R_fixed) > 0.99)  # signs aligned to prev
    assert np.allclose(ext_fixed, ext)
    assert np.linalg.det(R_fixed) > 0


def test_match_axes_small_rotation_stays():
    R_prev = np.eye(3)
    R_new = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    R_fixed, _ = match_axes(R_new, np.array([0.2, 0.1, 0.05]), R_prev)
    assert np.allclose(R_fixed, R_new)  # small rotations must pass through


# ── 벨트 평면 구속 ────────────────────────────────────────
# 벨트를 z=1.0 에 둔다: n·p + d = -z + 1 → 상면 z=0.95 의 높이 = 0.05
BELT = (np.array([0.0, 0.0, -1.0]), 1.0)


def test_obb_on_plane_recovers_thickness_and_center():
    """하방 카메라는 상면만 본다 — 무구속 OBB 는 두께가 0 이고 중심이 상면에
    얹힌다(광축 계통 편향). 평면을 알면 둘 다 복원된다."""
    mask, depth = make_box_scene(size=(0.20, 0.10, 0.05))
    pts = mask_depth_to_points(mask, depth, K, stride=1, erode_px=0)

    free = compute_obb(pts, voxel=0.003)
    assert np.sort(free.extent)[0] < 0.005      # 두께 붕괴
    assert abs(free.center[2] - 0.95) < 0.005   # 중심이 상면

    obb = obb_on_plane(pts, BELT)
    assert obb is not None
    ext = np.sort(obb.extent)[::-1]
    assert abs(ext[0] - 0.20) < 0.01
    assert abs(ext[1] - 0.10) < 0.01
    assert abs(ext[2] - 0.05) < 0.005           # 진짜 두께
    assert abs(obb.center[2] - 0.975) < 0.005   # 상면과 벨트의 중간
    assert abs(float(obb.R[:, 2] @ BELT[0])) > 0.999  # 세 번째 축 = 벨트 법선
    assert abs(np.linalg.det(obb.R) - 1.0) < 1e-6


def test_obb_on_plane_thickness_has_floor():
    """평면 추정 오차로 두께가 0/음수가 되면 log 가 폭발해 필터가 수천 초
    동결된다(2026-08-20). log 앞에서 클램프한다."""
    mask, depth = make_box_scene(size=(0.20, 0.10, 0.05))
    pts = mask_depth_to_points(mask, depth, K, stride=1, erode_px=0)
    # 벨트를 물체 상면보다 카메라 쪽(0.90)으로 잘못 추정 → 높이가 음수
    obb = obb_on_plane(pts, (np.array([0.0, 0.0, -1.0]), 0.90))
    assert obb is not None
    assert np.sort(obb.extent)[0] >= MIN_THICKNESS
    assert np.isfinite(np.log(obb.extent)).all()


def test_obb_on_plane_rejects_tilted_object():
    """기울어 놓인 물체를 평평하다고 보고하면 안 된다 — 무구속 폴백."""
    mask, depth = make_box_scene(size=(0.20, 0.10, 0.05))
    pts = mask_depth_to_points(mask, depth, K, stride=1, erode_px=0)
    c = pts.mean(0)
    R = Rotation.from_euler("x", 40, degrees=True).as_matrix()
    assert obb_on_plane((pts - c) @ R.T + c, BELT) is None


def test_fit_plane_finds_belt_under_object():
    depth = np.full((480, 640), 1000, dtype=np.uint16)  # 벨트 z=1.0
    mask, block = make_box_scene(size=(0.20, 0.10, 0.05))
    depth[mask] = block[mask]
    ring = np.zeros_like(mask)
    ring[100:400, 100:550] = True
    ring &= ~mask
    n, d = fit_plane(depth, K, mask=ring)
    assert abs(float(n @ BELT[0]) - 1.0) < 1e-3
    assert abs(d - 1.0) < 0.005

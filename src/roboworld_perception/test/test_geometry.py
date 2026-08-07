import numpy as np
from scipy.spatial.transform import Rotation

from roboworld_perception.geometry import (compute_obb, mask_depth_to_points,
                                           match_axes)

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

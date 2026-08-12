"""Mask + aligned depth -> point cloud -> Open3D OBB pose/size."""
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class ObbResult:
    center: np.ndarray  # (3,) m, camera optical frame
    extent: np.ndarray  # (3,) m, along R columns
    R: np.ndarray       # (3,3) rotation, columns = local X/Y/Z axes
    num_points: int

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.center))

    @property
    def rpy(self) -> np.ndarray:
        """Roll, pitch, yaw in degrees (ZYX convention)."""
        return Rotation.from_matrix(self.R).as_euler("ZYX", degrees=True)[::-1]

    @property
    def quat_xyzw(self) -> np.ndarray:
        return Rotation.from_matrix(self.R).as_quat()


def mask_depth_to_points(mask, depth, K, depth_scale=0.001, stride=2,
                         z_range=(0.10, 3.0), erode_px=3):
    """Back-project masked depth pixels to 3D points (camera optical frame).

    depth: uint16 (mm) or float (m) aligned-to-color depth image.
    Erodes the mask and clips depth to median +- 3*MAD to drop edge bleed
    onto the conveyor/background.
    """
    m = mask.astype(np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    ys, xs = np.nonzero(m)
    if stride > 1:
        ys, xs = ys[::stride], xs[::stride]
    if len(ys) == 0:
        return np.empty((0, 3))

    z = depth[ys, xs].astype(np.float64)
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        z *= depth_scale
    valid = (z > z_range[0]) & (z < z_range[1])
    z, ys, xs = z[valid], ys[valid], xs[valid]
    if len(z) < 10:
        return np.empty((0, 3))

    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-6
    keep = np.abs(z - med) < 3.0 * 1.4826 * mad
    z, ys, xs = z[keep], ys[keep], xs[keep]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.column_stack([x, y, z])


def masked_depth_median(mask, depth, depth_scale=0.001, box=None,
                        z_range=(0.10, 3.0)):
    """마스크 영역 depth 중앙값(m). "물체 깊이"의 단일 정의 —
    가리개 침입 판정(tracker.depth_intrusion)과 매칭 비용이 모두 이것을 쓴다.

    box(xyxy)가 있으면 그 ROI만 스캔한다. 유효 픽셀이 부족하면 None.
    """
    if box is not None:
        y0, x0 = max(0, int(box[1])), max(0, int(box[0]))
        y1, x1 = int(box[3]) + 1, int(box[2]) + 1
        z = depth[y0:y1, x0:x1][mask[y0:y1, x0:x1]].astype(np.float64)
    else:
        z = depth[mask].astype(np.float64)
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        z *= depth_scale
    z = z[(z > z_range[0]) & (z < z_range[1])]
    if len(z) < 50:
        return None
    return float(np.median(z))


def compute_obb(points, voxel=0.005):
    """PCA-based oriented bounding box via Open3D. Returns None if degenerate."""
    if len(points) < 30:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) < 10:
        return None
    try:
        obb = pcd.get_oriented_bounding_box(robust=True)
    except RuntimeError:
        return None
    R = np.asarray(obb.R).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return ObbResult(center=np.asarray(obb.center).copy(),
                     extent=np.asarray(obb.extent).copy(),
                     R=R, num_points=len(pcd.points))


def match_axes(R_new, extent_new, R_prev):
    """Reorder/flip columns of R_new to best align with R_prev.

    OBB axes are arbitrary up to permutation and sign; this keeps them
    temporally consistent so RPY does not jump 90/180 deg between frames.
    Extent is permuted accordingly.
    """
    dots = R_prev.T @ R_new  # dots[i, j] = prev_i . new_j
    R_out = np.zeros((3, 3))
    ext_out = np.zeros(3)
    used = set()
    for i in range(3):
        order = np.argsort(-np.abs(dots[i]))
        j = next(j for j in order if j not in used)
        used.add(j)
        s = np.sign(dots[i, j]) or 1.0
        R_out[:, i] = s * R_new[:, j]
        ext_out[i] = extent_new[j]
    if np.linalg.det(R_out) < 0:
        k = int(np.argmin(ext_out))  # flip the least significant axis
        R_out[:, k] = -R_out[:, k]
    return R_out, ext_out


def smooth_rotation(R_prev, R_new, alpha=0.5):
    """Slerp between previous and new rotation (alpha=1 -> all new)."""
    slerp = Slerp([0, 1], Rotation.from_matrix([R_prev, R_new]))
    return slerp([alpha])[0].as_matrix()

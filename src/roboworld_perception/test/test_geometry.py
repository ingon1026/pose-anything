import numpy as np
from scipy.spatial.transform import Rotation

from roboworld_perception.geometry import (MAX_THICKNESS, MIN_THICKNESS,
                                           compute_obb, fit_plane,
                                           mask_depth_to_points, match_axes,
                                           obb_on_plane)

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


# 실측 두께 (mm) — 상한을 건드릴 때 실제 물체가 죽는지 여기서 걸린다
MEASURED_THICKNESS_MM = {
    "isaac block": 58,      # 정렬 extent [193, 58, 50] — 두께가 폭보다 크다
    "black bag": 217,       # [520, 377, 217] — 실측 정상 최대
    "book": 43,             # [243, 190, 43]
    "keyboard": 38,         # [402, 126, 38]
    "gray notebook": 35,    # [180, 121, 35]
    "thermos (누움)": 79,   # [273, 90, 79]
    "thermos (세움)": 273,  # 가상 — 세우면 [79, 79, 273], 두께가 최장축이 된다
}
GARBAGE_THICKNESS_MM = 530  # test5 gray notebook 오검출


def test_max_thickness_admits_every_measured_object():
    """상한이 실측 물체를 하나도 죽이지 않는다.

    "세운 thermos"가 이 표에 있는 이유: 비율 규칙(두께 ≤ 장축)으로 바꾸면
    이 케이스가 곧바로 깨진다 — 이 프로젝트의 대표 프롬프트("물통")가 정확히
    그 형상이다. 상한을 비율로 되돌리려는 리팩터는 여기서 실패해야 한다.
    """
    for name, mm in MEASURED_THICKNESS_MM.items():
        assert mm / 1000 < MAX_THICKNESS, f"{name} 두께 {mm}mm 가 상한에 걸린다"


def test_max_thickness_rejects_measured_garbage():
    assert GARBAGE_THICKNESS_MM / 1000 > MAX_THICKNESS


# ── stride 패리티 아티팩트 ────────────────────────────────
def _slab(w_px, h_px=20, z=1.0):
    """z m 앞의 fronto-parallel 슬랩. fx=300 이라 1 px = 3.33 mm(z=1m)."""
    depth = np.zeros((480, 640), dtype=np.uint16)
    mask = np.zeros((480, 640), dtype=bool)
    depth[200:200 + h_px, 300:300 + w_px] = int(z * 1000)
    mask[200:200 + h_px, 300:300 + w_px] = True
    return mask, depth


K_WIDE = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])


def test_stride_does_not_lose_extreme_columns():
    """폭 17~24 px 전 범위에서 stride 감축이 폭을 깎지 않는다.

    구 구현(nonzero 평탄 나열을 [::stride])은 **짝수 폭에서 마지막 열이
    통째로 탈락**해 정확히 1 px(= 3.33 mm) 짧게 나왔다 — 홀수 폭에서는 행마다
    위상이 번갈아 양 끝이 다 살아 손실 0. 즉 폭 편향이 홀짝에 따라 켜졌다
    꺼졌다 했다. 이 테스트는 그 구현에서 짝수 폭 전부 실패한다.
    """
    for w in range(17, 25):
        mask, depth = _slab(w)
        ref = mask_depth_to_points(mask, depth, K_WIDE, stride=1, erode_px=0)
        exact = ref[:, 0].max() - ref[:, 0].min()
        for stride in (2, 3):
            pts = mask_depth_to_points(mask, depth, K_WIDE, stride=stride,
                                       erode_px=0)
            got = pts[:, 0].max() - pts[:, 0].min()
            assert abs(got - exact) < 1e-9, f"폭 {w}px, stride {stride}"
            got_v = pts[:, 1].max() - pts[:, 1].min()
            assert abs(got_v - (ref[:, 1].max() - ref[:, 1].min())) < 1e-9


def test_stride_one_is_unchanged():
    """stride=1 은 극단 복원이 무연산 — 변경이 감축 경로에만 갇혀 있다."""
    mask, depth = _slab(20)
    pts = mask_depth_to_points(mask, depth, K_WIDE, stride=1, erode_px=0)
    ys, xs = np.nonzero(mask)
    assert len(pts) == len(ys)


def test_stride_point_count_stays_near_reduction_target():
    """감축 의도가 유지된다 — 극단 복원은 대략 행 수(H)만큼만 더한다."""
    mask, depth = _slab(200, 100)
    full = len(mask_depth_to_points(mask, depth, K_WIDE, stride=1, erode_px=0))
    pts = len(mask_depth_to_points(mask, depth, K_WIDE, stride=2, erode_px=0))
    assert pts < 0.52 * full  # 1/2 + H/(H*W/2) = 0.505


def test_rotated_mask_footprint_is_stride_invariant():
    """회전해도 풋프린트가 stride 에 안 흔들린다 — 이 수정의 동기다.

    회전하면 마스크 폭이 프레임마다 바뀌어 구 구현의 1 px 손실이 켜졌다
    꺼졌다 했다. 55x50 mm 블록을 0~90° 5° 씩 돌린 합성 실측(z=0.95m,
    fx=600 → 1px=1.58mm)에서 구 구현은 stride=1 대비 오차가 각도에 따라
    0 ~ 1.58 mm 로 진동했다(0°/90° 에서 최대). 새 구현은 전 각도 0 이다.
    """
    import cv2
    z_top, belt = 0.95, 1.0
    plane = (np.array([0.0, 0.0, -1.0]), belt)
    for ang in range(0, 95, 5):
        w = 0.055 / z_top * K[0, 0]
        h = 0.050 / z_top * K[1, 1]
        poly = cv2.boxPoints(((320.0, 240.0), (w, h), ang)).astype(np.int32)
        m = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(m, [poly], 1)
        mask = m.astype(bool)
        depth = np.zeros((480, 640), np.uint16)
        depth[mask] = int(z_top * 1000)

        def footprint(stride):
            pts = mask_depth_to_points(mask, depth, K, stride=stride,
                                       erode_px=0)
            return np.sort(obb_on_plane(pts, plane).extent[:2])[::-1]
        assert np.allclose(footprint(2), footprint(1), atol=1e-9), f"{ang}도"

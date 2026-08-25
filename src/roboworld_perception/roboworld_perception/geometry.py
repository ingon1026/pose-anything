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

    @property
    def extent_sorted(self) -> np.ndarray:
        """내림차순 정렬 extent — fusion.TrackFilter.extent_sorted 와 같은 규약."""
        return np.sort(np.asarray(self.extent))[::-1]


def mask_depth_to_points(mask, depth, K, depth_scale=0.001, stride=2,
                         z_range=(0.10, 3.0), erode_px=3,
                         near_band=0.040, far_band=0.010):
    """Back-project masked depth pixels to 3D points (camera optical frame).

    depth: uint16 (mm) or float (m) aligned-to-color depth image.
    Erodes the mask and clips depth around the median to drop edge bleed
    onto the conveyor/background. The clip half-width is 3*MAD, floored at
    near_band toward the camera and far_band away from it (see below).
    """
    m = mask.astype(np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    # 역투영과 depth 인코딩 정규화는 _backproject 가 단일 정의로 갖는다 —
    # RealSense uint16 mm 와 Isaac float32 m 를 가르는 규약이 두 곳에 흩어지면
    # 어긋났을 때 1000 배 오차가 조용히 난다.
    pts = _backproject(depth, K, depth_scale, stride, z_range, m.astype(bool))
    if len(pts) < 10:
        return np.empty((0, 3))

    z = pts[:, 2]
    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-6
    # MAD 만으로 자르면 합성(Isaac) depth 에서 무너진다. 렌더면은 잡음이 없어
    # MAD 가 0.015~0.020 mm 까지 떨어지고(실측 2026-08-19, 블록 7 개), 클립 폭이
    # ±0.1 mm 가 되어 "중앙값 평면 하나만 남기는 필터"로 퇴화한다. 그러면 물체
    # 자신의 단차가 통째로 잘린다 — 우측 블록의 30 mm 끝단이 점의 36~39 % 와
    # 함께 사라져 길이가 200 -> 152 mm, 중심이 한쪽으로 24 mm(결손의 절반)
    # 밀렸다. 실 RealSense 는 MAD 가 수 mm 라 이 경로로는 드러나지 않는다.
    #
    # 밴드를 비대칭으로 두는 것이 핵심이다. 클립이 막으려는 edge bleed(컨베이어·
    # 배경)는 위에서 내려다보는 한 언제나 물체보다 *멀고*, 살려야 하는 물체의
    # 단차는 *가깝다*. 대칭으로 넓히면 둘을 같이 들인다 — 우측 블록은 벨트가
    # 본체면에서 +24 mm 뿐이라 ±40 mm 로 열면 벨트가 그대로 딸려 들어온다
    # (실측: 두께 0 -> 41~56 mm, 길이 244 mm 까지 부풀었다).
    # far_band 를 24 mm 보다 작게 유지하는 것이 이 씬의 상한이다.
    half = 3.0 * 1.4826 * mad
    keep = (z > med - max(half, near_band)) & (z < med + max(half, far_band))
    return pts[keep]


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
    """Open3D OBB (무구속 폴백). 퇴화하면 None.

    **점군 PCA 가 아니다** — Open3D 의 `get_oriented_bounding_box` 는
    Qhull 볼록껍질을 만든 뒤 그 **정점**(2천 점 입력에서 40~53개)에 PCA 한다.
    이 희소 표본이 면내 yaw 를 못 정하고, yaw 오차 θ 가 지지폭
    `a·cosθ + b·sinθ` 로 extent 를 부풀린다.

    **지배 변수는 잡음이 아니라 종횡비다** (55x50mm 상면 한 장, seed 20):
        σ=0      장축 63.3±4.7  단축 60.0±5.8   (참값 55/50)
        σ=0.02mm      61.1±5.7       57.5±6.6
        σ=1mm         64.2±5.0       61.0±6.3
    **잡음 0 에서 이미 만폭으로 흔들린다.** σ 는 두께축만 바꾼다.
    종횡비별 yaw 표준편차: 1.0 → 30.5° / 1.45 → 5.8° / 3.64 → 1.4°.

    → **근사정사각 물체에서 extent 가 계통적으로 +10~23% 부푼다.** 일방향
    편향이라 필터가 못 지운다. 실기 test3(상시 무구속)에서 종횡비 관계가
    5개 런 전부 재현된다 — 책(1.23) sd 9.6/12.9mm, 블록(5.35) sd 1.1°.

    **그래도 지금 고치지 않는다**: (a) 이 경로는 계약상 "잘못된 평면을 쓰느니
    무구속이 낫다" 의 차악이고 아래 벨트 평면 주석이 이미 열등하다고 선언한다,
    (b) test3 전 트랙에서 flips=0·축 교체율 0.00% — match_axes+데드밴드+slerp
    가 20° 급 yaw 흔들림을 흡수해 하류로 안 샌다, (c) 후보 처방 둘이 서로
    반대 조건에서 깨진다 — 전점 PCA 는 장방형에 최강이나 정사각에서 똑같이
    퇴화하고, minimal OBB 는 정사각을 살리나 저잡음 장방형(= Isaac 렌더면)에서
    더 나쁘다. (d) extent 정의가 바뀌면 MAX_THICKNESS·log-extent 필터 R̂·
    belt_plane A/B 회귀 수치가 전부 재측정 대상이 된다.

    **뒤집을 조건**: 위 +10~23% 과대가 로봇 파지 폭에 쓰이게 되면.
    voxel_down_sample 은 무죄다 — 꺼도 분산이 안 준다(오히려 미세 증가).
    """
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


# ── 벨트 평면 구속 ────────────────────────────────────────
# 하방 카메라는 물체의 상면만 본다. 그래서 무구속 OBB 는 두께가 0 으로
# 붕괴하고(점군이 사실상 한 장의 면), 중심이 물체 중심이 아니라 상면에
# 얹힌다 — 광축 방향 계통 편향 -3.69 mm 의 정체다(2026-08-20 축 분해 실측:
# 위치 오차 에너지의 51 % 가 광축). 벨트 평면을 알면 두께 = 상면 높이,
# 중심 = 상면과 벨트의 중간으로 두 결함이 동시에 풀린다.
TILT_MAX_DEG = 25.0    # 상면 법선이 벨트 법선에서 이만큼 벌어지면 구속 포기
PLANAR_RATIO = 0.05    # λ0/λ1 이 이보다 작으면 "면 한 장만 보이는" 점군
MIN_THICKNESS = 0.003  # m — log-extent 하한. 평면 추정 오차로 두께가 0/음수가
                       # 되면 log 가 폭발해 필터가 수천 초 동결된다(2026-08-20).
MAX_THICKNESS = 0.35   # m — 두께 상한. "이 벨트에 놓일 수 있는 물체 높이"의
                       # 상계다. 실측 정상 최대는 black bag 217mm, 실측 쓰레기는
                       # 530mm(test5 gray notebook 오검출 — 평면보다 530mm 나
                       # 카메라 쪽에 점이 있었다). 둘의 로그 중점 339mm 를
                       # 반올림했다. 하한과 달리 이건 **씬 파라미터**다 —
                       # 더 높은 물체를 다루면 올려야 한다.
                       #
                       # 비율 규칙(두께 ≤ 장축 등)을 쓰지 않은 이유: 세워 놓은
                       # 원통이 곧바로 깨진다. test2 thermos 는 누워 있어
                       # [273, 90, 79] 지만, 세우면 [79, 79, 273] 이라 두께가
                       # 최장축이 된다 — 이 프로젝트의 대표 프롬프트("물통")가
                       # 정확히 그 형상이다. isaac 블록도 두께 58 > 폭 50 이라
                       # 중간축 대비 배율은 이미 성립하지 않는다.


def _row_extreme_flat(mask):
    """행별 최좌·최우 픽셀의 flat(row-major) 인덱스.

    이것만으로 **픽셀 집합의 볼록껍질 정점이 전부 보존된다**: 어떤 픽셀이 제
    행의 최좌도 최우도 아니면 같은 행에 좌·우 이웃이 있어 두 점을 잇는 선분
    내부에 놓이므로 껍질 정점이 될 수 없다. 대우로 껍질 정점은 반드시 어떤
    행의 좌·우 끝이다. 열 방향 극단도 그 행의 좌우 끝에 포함된다.
    minAreaRect 는 껍질만 보므로 풋프린트는 이 집합만 살려도 stride 와
    무관해진다. (열별 극단을 따로 더하는 것은 껍질 보존에 중복이다.)
    """
    # argmax 는 "첫 nonzero" 가 아니라 "첫 최대값" 이다 — 1 과 255 가 섞인
    # uint8 마스크가 들어오면 끝 픽셀이 틀린다. bool 로 못박고 시작한다.
    mask = np.asarray(mask, bool)
    w = mask.shape[1]
    rows = np.flatnonzero(mask.any(1))
    first = mask[rows].argmax(1)
    last = w - 1 - mask[rows, ::-1].argmax(1)
    return np.concatenate([rows * w + first, rows * w + last])


def _backproject(depth, K, depth_scale, stride, z_range, mask=None):
    if mask is None:
        h, w = depth.shape
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        ys, xs = ys.ravel(), xs.ravel()
    else:
        # nonzero 는 row-major 1 차원 나열이라 그걸 stride 로 솎으면 뱀 모양
        # (serpentine) 샘플이 된다 — 행이 바뀔 때 위상이 이어지므로 **마스크
        # 폭이 짝수면 모든 행의 시작 위상이 같아져 마지막 열이 통째로 탈락**
        # 하고, 홀수면 행마다 위상이 번갈아 양 끝이 다 산다. 즉 폭 측정에
        # 1 px 계통 편향이 폭의 홀짝에 따라 켜졌다 꺼졌다 한다(합성 실측,
        # z=1.0m fx=300 → 1px=3.33mm, erode_px=0):
        #     폭 19/21px(홀) stride2 손실 0.00mm
        #     폭 20/22px(짝) stride2 손실 3.33mm (= 1px)
        # Isaac 블록 폭 55mm 는 약 19px 라 평상시엔 안 걸리지만, 회전하면
        # 마스크 폭이 프레임마다 바뀌어 0 과 2.93mm 를 오간다.
        #
        # 격자 샘플(mask[::stride, ::stride])은 홀짝 의존은 없애지만 감축이
        # 1/stride² 라 점이 절반으로 줄어 fit_plane 의 len(pts)<500 게이트를
        # 새로 밟는다. 그래서 기존 감축은 그대로 두고 **행별 극단만 되돌려
        # 넣는다** — 빼는 것이 없으므로 점 개수는 늘기만 하고, 껍질 손실은
        # 폭의 홀짝과 무관하게 0 이 된다(픽셀 껍질 기준. 역투영은 z 가 행마다
        # 다르면 아핀이 아니라 3D 껍질까지 동일하다고는 말할 수 없다).
        # 추가 후보는 행별 2 개(2H)지만 절반쯤은 이미 뽑힌 점이라 실제 증가는
        # 대략 H 개, 즉 직사각 마스크에서 stride/W 다(합성 실측, stride=2:
        # 19x19px +9.9%, 55x30px +3.6%, 200x100px +1.0%, 링 +1.7%).
        # 계산량이 실제로 문제되는 큰 마스크일수록 증가가 사라진다.
        # union1d 가 정렬·중복 제거를 겸한다.
        idx = np.flatnonzero(mask)
        idx = np.union1d(idx[::stride], _row_extreme_flat(mask))
        ys, xs = np.unravel_index(idx, mask.shape)
    z = depth[ys, xs].astype(np.float64)
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        z *= depth_scale
    ok = (z > z_range[0]) & (z < z_range[1])
    z, ys, xs = z[ok], ys[ok], xs[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.column_stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z])


def fit_plane(depth, K, depth_scale=0.001, mask=None, stride=2,
              z_range=(0.10, 3.0), dist_thresh=0.006, min_inlier_ratio=0.35,
              band=0.050, stats=None):
    """지지 평면(= 물체가 놓인 벨트면) → (n, d), n·p + d = 0, |n| = 1.

    n 은 카메라 쪽을 향한다(광학 좌표계에서 n[2] < 0). 인라이어가
    min_inlier_ratio 미만이면 None — 잘못된 평면을 쓰느니 무구속이 낫다.

    **문턱은 0.30 -> 0.35 (2026-08-25).** 실기 4개 bag 113 키프레임을 반복
    21회씩 적합해 "통과한 평면의 최대 오차" 로 고른 값이다(오차 대리값은
    링을 절반씩 나눠 따로 적합했을 때의 불일치 — 참값 있는 합성에서 먼저
    검증했다: 최저 사분위 최대 2.82mm, >5mm 0%):

        문턱    isaac        test2        test4          test3
        0.30   40/40 2.8mm  16/16 58mm  30/30 **598mm**  7/7 **212mm**
        0.35   40/40 2.8mm  16/16 58mm  27/30 **34mm**   **통과 0**
        0.40   변화 없음     변화 없음    변화 없음        통과 0
        0.45   변화 없음     7/16  27mm   7/30  32mm     통과 0

    **test4 의 최악 통과 오차가 598 -> 34mm 로 한 자릿수대까지 떨어진다.**
    단 **34 를 절대값으로 읽지 말 것.** 이 오차 대리값은 같은 프레임에서도
    추첨마다 1.5~2배 흔들리고(test2 한 프레임 6회: 21.0~37.6mm), 참값 있는
    합성에서 최대 10배까지 과대보고했다(참 3.67 -> 보고 36.70mm). 독립
    재측정에서는 같은 무게이트 최댓값이 91mm 로 나왔다. **598 과 34 의 자릿수
    차이는 강건하지만 그 비율은 이 계측기의 분해능 밖이다.** 그리고 test3 이
    완전히 걸러진다 — 이 docstring 아래가 "여기서 걸러지는 것이 의도한
    동작" 이라고 적어 둔 그 씬인데, 0.30 에서는 26프레임 중 7번 통과했고
    그 7개가 73~212mm 오차였다. 0.40 이상은 추가 이득이 없고 0.45 는 test2 를
    반토막 낸다.

    **결정 프레임에서 아무것도 안 잃는다.** 평면은 _plane_tried 로 실행당
    한 번, 첫 강검출 키프레임에서만 맞춘다. 그 프레임의 인라이어 비율을
    41회 추첨해 재면:

        isaac  min 0.471 (여유 +0.121)
        test2  min 0.405 (여유 +0.055)
        test4  min 0.386 (여유 +0.036)   <- 가장 얇다

    **단 이 여유는 낙관적이다.** 위 캡처는 --stride 15 로 뽑아 파이프라인
    (detect_interval=5)이 실제로 고르는 첫 키프레임과 다르다. 실제 실행에서
    재면 test2 가 **0.371** 로 여유가 **+0.021** 까지 좁아졌다(isaac 0.499,
    test4 0.426). 41회 추첨 최소 0.405 보다 낮다.

    → **다른 씬에서 이 문턱이 결정 프레임을 떨어뜨릴 수 있다.** 떨어지면
    무구속 폴백이고 그건 계약상 차악이지 실패가 아니다(compute_obb docstring).
    다만 그 폴백은 근사정사각 물체에서 extent 를 +10~23% 부풀리므로 공짜가
    아니다 — 새 씬을 넣을 때 [belt_plane] 로그의 inlier 를 확인할 것.
    고정 카메라에서는 사실상 상수라 파이프라인이 한 번 맞추고 캐시한다.

    mask 로 픽셀을 한정할 것. 화면 전체를 넣으면 RANSAC 이 "지배 평면"을
    고르는데 그건 벨트가 아니라 바닥이다 — 실측(isaac_belt_moving): 벨트가
    화면의 일부뿐이라 바닥이 이겨 두께가 24 mm 대신 569 mm 로 나왔다.
    물체 마스크 바로 바깥의 링을 쓰면 "물체가 실제로 얹힌 면"을 잰다.

    링이라도 물체 상면·배경이 섞여 들어온다. depth 중앙값 ±band 로 먼저
    쳐내야 인라이어 비율이 "지지면 안에서의 비율"이 된다 — 안 쳐내면 실측
    test2 가 0.28 로 문턱 바로 아래에서 탈락했다(평면 자체는 맞았는데도).
    클립 후 test2 0.42 / test4 0.41 로 통과하고, test3(손밀기 롤러)은 0.27 로
    떨어진다. test3 은 실제로 링이 한 장의 평면이 아니라 추정 파라미터에 따라
    기울기가 2.9~8.2° 로 흔들리는 씬이라, 여기서 걸러지는 것이 의도한 동작이다.

    stats 로 dict 를 주면 진단값을 채워 준다(관측용, 판정에는 안 쓴다) —
    "inlier"(= 게이트가 보는 그 비율), "rms"(인라이어 잔차 RMS, m).
    """
    pts = _backproject(depth, K, depth_scale, stride, z_range, mask)
    if len(pts) < 500:
        return None
    z = pts[:, 2]
    pts = pts[np.abs(z - np.median(z)) < band]
    if len(pts) < 500:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = pcd.segment_plane(dist_thresh, 3, 200)
    if stats is not None:
        # 관측만 한다 — 게이트는 아래 min_inlier_ratio 그대로다.
        #
        # 왜 재나: ~~인라이어 비율은 평면 품질과 사실상 무상관이다. 합성
        # 29표본에서 |편향| 과의 상관이 비율 +0.066(무상관), 잔차 RMS +0.652,
        # 반복 간 불일치 +0.486.~~ 기전 — 6mm 슬랩은 폭 60~100mm 링에서
        # ±3.4~5.7° 기울기를 인라이어 손실 거의 없이 허용한다. ~~비율은
        # "점이 몇 개 붙었나"이지 "평면이 결정됐나"가 아니다.~~ test2 는
        # 0.408 로 문턱 0.35 를 통과하는데 실행 간 d 산포가 ±13.1mm 다
        # (repeats=21). Isaac ±0.40mm, test4 ±2.98mm — 씬마다 자릿수가 다르다.
        #
        # ~~**그런데도 게이트를 RMS 로 안 바꾼 이유**: 표본이 29개(나쁜 클래스
        # 6개)뿐이고 RMS>1.5mm 동작점의 오탈락이 39% 다. 실 bag 에서 이
        # 로그로 분포를 모으기 전에는 문턱을 정할 근거가 없다. 여기 상수를
        # 박기 전에 표본부터 세라.~~
        #
        # ── 정정 (2026-08-25) — 위 취소선 둘은 뒤집혔다 ──────────────
        # **(1) `+0.066` 은 씬을 pooling 한 값이었다.** 씬 **안에서** 재면
        # 오염된 씬에서 **−0.562(test2) / −0.589(test4)** 로, 실기 4 bag
        # 범위에서는 **인라이어가 높을수록 평면이 좋다** — 지표는 정보를
        # 담고 있었고 고칠 대상은 문턱 위치였다(0.30 → 0.35, `55a335f`).
        # 사례는 docs/README.md §3-1 · §5(pooling 함정)에 올라 있다.
        # ⚠ **씬 안에서만 성립한다.** 참값 있는 합성 대조에서는 방향이
        # 반대다 — 유해한 `lip` 의 인라이어가 무해한 `ramp` 보다 **높다**
        # (0.589 vs 0.313). 절대 품질 척도가 아니고, 씬 간에 0.35 를
        # 인용하지 말 것.
        #
        # **(2) RMS 로 안 바꾼 진짜 이유는 표본 수가 아니라 판별력이다.**
        # "자신 있게 틀린" 평면은 잔차가 작다 — RANSAC 이 오염 구조에
        # 붙으면 그 부분집합에 아주 잘 맞는다. 참값 있는 합성에서
        # `lip`(참오차 **15.51mm**)의 rms **1.63mm** 가 무해한 `ramp`
        # (참오차 0.81mm)의 **1.45mm** 와 **0.18mm 차**다. 이것이 rms 가
        # 실기 test2/test4 를 못 가른 이유이고 **표본 수 문제가 아니었다**
        # (docs/README.md §5).
        # → **표본을 더 모으면 답이 나온다고 읽지 말 것.** 저장소 결론은
        # "단발 게이트로 쓸 지표가 남지 않았다" 이고, 더 정확히는 판정이
        # 기대는 오차 대리값(`half_disagree`) 자체가 draw 마다 1.5~2배
        # 흔들려 **이 계측기의 분해능으로는 답할 수 없는 질문**이라는 것이다.
        # **(3) 취소선 문장의 세 수치는 같은 표의 세 열이고 전부 pooling
        # 값이다** — rms 의 씬 안 값은 −0.048 / −0.025 / +0.071 / +0.317 로
        # 0 근처다. `+0.486` 도 이 표에서 pooling 된 값이라는 것까지만 읽을
        # 것 — `시드산포` 계열 수치는 서로 다른 표가 셋이라 바꿔 쓸 수 없다
        # (docs/README.md §5 "이름 주의").
        #
        # (두 기각은 확신 강도가 다르다: 인라이어의 pooling 은 실기 4 bag
        #  로 확정됐고, rms 의 `+0.652` 는 합성 29표본이라 원문도 *"이
        #  함정일 가능성이 크다"* 로 적혀 있다 — 그대로 인용할 것.)
        # ─────────────────────────────────────────────────────────
        #
        # 분모는 클립 후 len(pts) — 게이트가 나누는 그 값이라야 위 수치들과
        # 비교된다. model 은 단위 법선이라 |p·n + d| 가 곧 미터 거리다.
        stats["inlier"] = len(inliers) / len(pts)
        stats["rms"] = float(np.sqrt(np.mean(
            (pts[inliers] @ np.asarray(model[:3], float) + model[3]) ** 2)))
    if len(inliers) < min_inlier_ratio * len(pts):
        return None
    n, d = np.asarray(model[:3], float), float(model[3])
    if n[2] > 0:
        n, d = -n, -d
    return n, d


def obb_on_plane(points, plane, min_thickness=MIN_THICKNESS, top_q=0.98,
                 tilt_max_deg=TILT_MAX_DEG):
    """벨트 평면에 구속한 OBB — 회전 자유도는 법선 둘레 yaw 하나뿐.

    물체가 벨트에 평평하게 놓였다는 가정 위에서만 성립한다. 상면 한 장만
    보이는 점군(λ0/λ1 < PLANAR_RATIO)인데 그 법선이 벨트 법선에서 크게
    벌어졌으면 기울어 놓인 것이므로 None 을 돌려준다 — 호출자가 무구속으로
    폴백해야 한다. 옆면까지 보이는 점군은 이 검사를 건너뛴다(법선 정의가
    상면이 아니게 되므로 판정 자체가 성립하지 않는다).

    **풋프린트는 축마다 1픽셀 짧다 (2026-08-25, 열린 항목).** _backproject 가
    정수 화소 좌표를 역투영하므로 여기 들어오는 점은 전부 **화소 중심**이고,
    아래 minAreaRect 는 그 중심들의 최소 사각형이다. n 화소를 덮는 영역의
    중심들은 n-1 화소만 걸치므로 축마다 정확히 1px 모자란다 — 크기로는
    `1px x z/fx` 다.

    isaac_belt_moving + image_size 1008(이 분해의 캡처 조건)에서 이것이 부족의
    **지배 항**이다 (fx 322.86, z 0.945m -> 2.93mm). **단 아래 표의 5.24mm 는
    그 조건의 상수도 아니다 — 21% 확률 위상이 얼어붙은 값이다**(뒤의 위상 절).
    정답 200x55mm 에 대한 분해(166관측):

        정답                              55.00   200.00
        depth 래스터 -> CC corners        53.73   201.46   (-1.27 / +1.46)
        SAM3 마스크 -> M corners          52.72   201.18   (-1.02 / -0.28)
        화소중심 규약 -> M centers        49.79   198.24   (**-2.93 / -2.94**)
        erode_px=3                        49.79   196.11   (-0.00 / -2.14)

    분해의 양 끝이 고정되어 있다 — 마지막 행이 파이프라인 실측(49.8/195.7)을
    0.01/0.41mm 로 재현하고, CC 팔은 border_margin 의 depth 연결성분 길이
    196.1~199.7mm 를 198.52(IQR 196.36~199.31)로 재현한다.

    **문턱 없는 재확인 — 폭축 화소 행수를 세면 위 표의 depth 항이 갈린다.**
    블록은 축 정렬이고 폭축이 정확히 천저(|v-180| = 0.0px)라 옆면이 안 보인다.
    z=0.945 에서 참 폭은 18.79px 인데:

        depth 연결성분 (마스크 미사용)   18행  (153/162)
        SAM3 마스크                      20행  (144/162)   <- 덮고도 남는다
        마스크 + z밴드                   18행
        + erode_px=3                     18행

    즉 **1008 에서 폭을 정하는 것은 마스크가 아니라 depth 다.** 마스크는 참값을
    1.21px 넘게 덮고 z밴드가 그것을 깎아 마스크 없는 depth 와 같은 18행을 낸다.
    폭축 부족은 두 항으로 닫힌다 — **래스터 양자화 0.79px + 화소중심 규약 1px
    = 1.79px = 5.24mm** (관측 5.2mm). 마스크 몫 0 이다.

    이 계산이 지난 서술 하나를 뒤집는다: z=0.975 군의 "depth 만으로 폭 55.01mm"
    는 depth 가 폭을 안 잃어서가 아니라 **래스터 손실 -0.64mm 와 CC corners 의
    부풀림 +0.65mm 가 상쇄된 값**이다(그 군도 참 18.21px 에 18행이다). corners
    는 가장자리 혼합 화소가 더 먼 z 로 투영돼 부푼다 — 행수에는 그 누출이 없다.

    **두 항의 처방이 같다 — 부분화소 경계다.** 규약 1px 과 래스터 0.79px 둘 다
    "화소를 세는 것" 에서 오므로 각도 인지 보정보다 부분화소 쪽이 본질이다.

    **⚠ 위 분해는 1008 것이고 출하 프리셋은 672 다** (isaac.launch.py:82).
    규약 2.93mm 와 래스터 2.31mm 는 원본 640x360 래스터에서 나와 두 arm 에
    공통이므로, 672 의 관측 부족 8.2mm 에서 빼면 **마스크 몫이 약 3.0mm** 다.
    즉 **출하 조건에서는 마스크가 폭에 기여한다** — 1008 에서 0 인 것을
    운영값으로 읽으면 안 된다. 두 몫을 비로 쓰지 말 것: 분모가 0 으로
    측정된 그 양이라 모수화가 무너진다. 그래서 image_size 672 vs 1008
    결정은 이 절로 닫히지 않는다 — docs/README.md §4.

    **그래도 +1px 를 더하면 안 된다 — 오차가 각도 의존이다.** 합성 래스터
    (위상 17x17 평균)에서:

        회전각      화소중심      화소모서리
          0 deg   -3.03mm(-1px)   -0.10mm
         12 deg   -0.17mm        **+3.30mm**
         31 deg   -0.48mm        **+3.54mm**
         45 deg   -2.04mm         +2.10mm

    **⚠ 위 -3.03 과 실측 -5.21 은 다른 예산이 아니라 같은 것의 두 통계다.**
    0deg 에서 centers 오차는 **값이 딱 두 개**다 — 참 18.79px 에 화소중심이
    18개 들면 span 17px(**-1.79px = -5.24mm**), 19개 들면 18px(**-0.79px =
    -2.31mm**). 위상 확률이 **21% / 79%** 라 평균이 **-1.000px = -2.93mm**
    (위 표의 -3.03). 즉 위 표는 **위상평균**, 실측 -5.21 은 **고정위상** 값이다.
    합성에 래스터 항이 빠진 게 아니라 평균에서 21:79 로 섞였을 뿐이다.

    **그래서 5.24mm 는 상수가 아니라 한 번의 위상 뽑기다.** 이 bag 은 21% 쪽에
    앉았고(162 중 153프레임이 18행) 카메라가 고정이라 얼어붙었다. 79% 쪽이었다면
    관측 부족이 **2.31mm** 였을 것이고 +1.79px 은 **+2.93mm 과보정**이었다.
    **5.24mm 에 상수를 맞추는 것은 그 한 번의 뽑기에 맞추는 것이다** — 아래 각
    분포와 무관한 별개의 이유다.

    **그래도 스칼라 보정은 못 쓴다 — 이유는 각 분포다.** 629관측 실측:

        bag      n    각 중앙값   p95     12~31deg
        isaac  238    0.00deg    0.00    0%
        test2   67   16.99deg   19.92    있음
        test4   67   18.97deg   37.44    있음
        test3  257    6.84deg   27.70    (100% 무구속 - 다른 경로다)

    각 = 축정렬 이탈 min(|th|%90, 90-|th|%90). 종횡비 문턱을 1.0~3.0 으로
    훑어도 판정이 안 뒤집힌다. 물체 하나로 충분하다 - test4 det1 은 종횡비
    3.36, 장축 168px, 각 35.43deg 에 프레임간 sd 0.96deg 다. 2차 모멘트
    주축각과 1.6deg 안에서 일치하니 각은 실재한다.

    **isaac 만 정의역 안이고, 12/31deg 에서 옳은 보정량은 아무도 안 쟀다** -
    회전한 모서리의 화소 피복은 축정렬과 다르므로 각 무관이라 볼 근거가 없고,
    test2/test4 는 물리 실측이 없어 정답 대비 판정 자체가 불가하다.
    필요한 것은 **각도 무관 추정**이고, 출하 크기 전부가 바뀌는 변경이라
    아직 안 넣었다 — docs/README.md §4.

    두께(thk)는 이 영향을 안 받는다. 평면 위 높이의 분위수라 화소 span 이
    아니다.
    """
    if len(points) < 30:
        return None
    n, d = plane
    c = points - points.mean(0)
    lam, V = np.linalg.eigh(c.T @ c)
    if lam[0] <= PLANAR_RATIO * lam[1] and \
            abs(float(V[:, 0] @ n)) < np.cos(np.radians(tilt_max_deg)):
        return None
    thk = max(float(np.quantile(points @ n + d, top_q)), min_thickness)

    # 평면 기저 (u, v, n) 에서 최소면적 사각형 = 풋프린트 + yaw
    u = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    uv = np.column_stack([points @ u, points @ v]).astype(np.float32)
    (cu, cv), (su, sv), ang = cv2.minAreaRect(uv)
    t = np.radians(ang)
    e1 = np.cos(t) * u + np.sin(t) * v
    e2 = -np.sin(t) * u + np.cos(t) * v
    # n·p 는 평면 위 높이 - d 이므로, 중심을 두께의 절반 높이에 놓으려면
    # 법선 성분이 thk/2 - d 여야 한다 (u, v ⊥ n, |n| = 1).
    center = cu * u + cv * v + (thk / 2 - d) * n
    return ObbResult(center=center, extent=np.array([su, sv, thk]),
                     R=np.column_stack([e1, e2, n]), num_points=len(points))


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

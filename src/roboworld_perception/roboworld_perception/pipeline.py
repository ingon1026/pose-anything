"""Per-frame perception pipeline shared by the offline runner and the ROS node.

하이브리드 검출·추적: SAM3는 detect_interval 프레임마다 1번(키프레임),
사이 프레임은 광학흐름으로 마스크를 평행이동해 추적한다. 3D OBB는
어느 쪽이든 그 프레임의 실제 depth로 매번 계산한다.

관측 수락은 트랙별 융합 필터(fusion.TrackFilter)의 χ² 게이트가 판정한다 —
이전의 이진 게이트(score·depth·size)와 달리 거부가 이어지면 불확실성이
자라 게이트가 스스로 열리므로 교착이 없다 (docs/fusion_design_2026-08.md).

offline 스크립트와 ROS 노드가 공유하는 헬퍼(이미지 디코드, CSV 스키마)도
여기에 둔다 — 두 진입점이 따로 복사해 들고 있으면 조용히 어긋난다.
"""
import cv2
import numpy as np

from .fusion import TrackFilter, pos_r_extra
from .geometry import (MAX_THICKNESS, compute_obb, fit_plane,
                       mask_depth_to_points, masked_depth_median, obb_on_plane)
from .tracker import IouTracker, depth_intrusion

NOMINAL_DT = 1 / 15  # s — 공칭 프레임 주기 (스탬프 없을 때의 합성용)

# ── 공유 헬퍼 ──────────────────────────────────────────────

CSV_HEADER = ["stamp", "track_id", "label", "score", "x", "y", "z", "distance",
              "w", "d", "h", "roll", "pitch", "yaw", "flips", "proc_ms"]


def csv_row(obj, stamp_s, proc_ms):
    """CSV_HEADER 순서의 한 행. obj는 obb가 있는 Track."""
    o = obj.obb
    r, p, y = o.rpy
    return [f"{stamp_s:.6f}", obj.track_id, obj.label, f"{obj.score:.3f}",
            *[f"{v:.4f}" for v in o.center], f"{o.distance:.4f}",
            *[f"{v:.4f}" for v in o.extent],
            f"{r:.2f}", f"{p:.2f}", f"{y:.2f}", obj.flip_count, f"{proc_ms:.1f}"]


def parse_plane(text):
    """"a,b,c,d" (n·p+d=0) → (n, d), |n|=1. 빈 문자열이면 None(자동 추정).

    오프라인 러너와 ROS 노드가 같은 문자열 규약을 쓰게 하는 단일 정의.
    """
    if not text:
        return None
    v = np.array([float(x) for x in text.split(",")], dtype=float)
    s = np.linalg.norm(v[:3])
    return v[:3] / s, float(v[3]) / s


def img_to_np(msg):
    """ROS Image/rosbags 메시지(duck-typed) → numpy. step 패딩 처리 포함."""
    if msg.encoding in ("rgb8", "bgr8"):
        ch, dtype = 3, np.uint8
    elif msg.encoding == "16UC1":
        ch, dtype = 1, np.uint16
    elif msg.encoding == "32FC1":
        # Isaac Sim 의 depth. RealSense 는 16UC1 밀리미터인데 Isaac 은
        # float32 미터로 낸다. 아래에서 밀리미터로 환산해 돌려주므로
        # geometry.py 의 depth_scale=0.001 이 그대로 맞는다.
        ch, dtype = 1, np.float32
    else:
        raise ValueError(f"unsupported encoding {msg.encoding}")
    itemsize = np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.step // itemsize)
    arr = arr[:, :msg.width * ch]
    arr = arr.reshape(msg.height, msg.width, ch).squeeze()
    if dtype is np.float32:
        # inf/NaN 은 "값 없음" 이다. RealSense 규약대로 0 으로 둔다.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
    return arr


def status_text(fps, pipeline, objects):
    mode = "KEY" if pipeline.last_was_keyframe else "track"
    return f"{fps:4.1f} FPS | {mode} | objects={len(objects)}"


# ── 광학흐름 추적 ──────────────────────────────────────────
# 흐름 일관성 게이트. 마스크가 물체와 가리개에 걸치면 LK 변위가 두 무리로
# 갈라지는데, 중앙값만 쓰면 큰 쪽으로 조용히 끌려간다 — 그러면 마스크가 물체를
# 떠나고, 떠난 마스크가 다음 프레임의 흐름을 다시 정하는 자기강화가 시작된다
# (test4 book 실측 10.5~20.0s: 잔차가 171 -> 290mm 로 단조 증가, 발행 9.5초 단절.
# 기각 잔차의 z 성분은 -0.6~-4.9mm 로 순수 면내 드리프트였다).
#
# 이 실패는 기존 신호로는 못 잡는다 — 가리개(검은 폴더)가 책과 같은 높이라
# depth_intrusion 이 구조적으로 안 걸리고, SAM 점수는 부분 가림에서도 높다.
# 마스크 안의 공간 구조가 그 둘과 독립인 유일한 신호다. 융합 게이트의 기각을
# 신호로 쓰는 방법도 있지만 그건 증상이 나온 뒤이고, 무엇보다 기본 경로에서는
# 그 관측이 애초에 수락된다(R̂ 이 커져 게이트가 넓다 — 실측 d²=6.1) — 그러면
# 정작 오염 좌표를 발행하는 쪽을 못 고친다.
#
# 허용 반경을 변위 크기에 비례시키는 것이 요점이다. 절대 픽셀로만 재면
# "빠른 물체"를 "여러 물체"로 오인한다 (test3 손밀기 137mm/s).
FLOW_INLIER_PX = 2.0     # px — 정지 물체의 LK 잡음 허용치
FLOW_INLIER_FRAC = 0.25  # 변위 크기에 비례하는 허용 반경
FLOW_MIN_INLIER = 0.7    # 중앙값이 이 비율 이상의 점을 설명해야 전파한다.
                         # 합성 실측(test_pipeline 의 텍스처 프레임): 한 덩어리
                         # 흐름은 0~25px 전 속도에서 1.00, 마스크 절반만 움직이면
                         # 0.60 — [0.60, 1.00] 이 고원이라 그 안이면 값이 둔감하다.
                         # 실기 bag 스윕으로 재확인할 것


def propagate_mask(prev_gray, gray, mask, box=None, max_points=300):
    """광학흐름 중앙값으로 마스크·박스 평행이동량 (dx, dy)를 구한다.

    box(xyxy)가 주어지면 그 ROI 안에서만 마스크 픽셀을 찾는다(전체 프레임
    스캔 회피). 유효 포인트가 부족하거나 변위가 한 덩어리로 일관되지
    않으면 None (이전 위치 유지) — 위 FLOW_* 주석 참고.
    """
    if box is not None:
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = int(box[2]) + 1, int(box[3]) + 1
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        ys, xs = ys + y0, xs + x0
    else:
        ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None
    step = max(1, len(ys) // max_points)
    pts = np.column_stack([xs[::step], ys[::step]]).astype(np.float32).reshape(-1, 1, 2)
    new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return None
    d = (new_pts - pts).reshape(-1, 2)[ok]
    med = np.median(d, axis=0)
    r = max(FLOW_INLIER_PX, FLOW_INLIER_FRAC * float(np.hypot(*med)))
    if np.mean(np.hypot(*(d - med).T) <= r) < FLOW_MIN_INLIER:
        return None  # 마스크가 서로 다르게 움직이는 것들에 걸쳐 있다
    return float(med[0]), float(med[1])


def _shift_mask(mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = mask.shape
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h)) > 0


def _touches_border(mask):
    """마스크가 화면 경계에 닿아 있는가 — 절단 관측(물체 일부만 보임) 판정."""
    return bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any())


# ── 부분 가림 = 화면 절단과 같은 종류의 사건 ────────────────
# 가리개가 물체를 조금씩 덮으면 보이는 영역이 줄고, 그 줄어든 영역의 중심은
# 물체의 중심이 아니다. 화면 경계 절단(_touches_border)에서 이미 알고 있는
# 사실인데, 가리개에 의한 절단은 아무도 안 보고 있었다.
#
# test4 book 실측(5.5~7.9s): 풋프린트가 290x241 -> 267x170mm 로 줄어드는 동안
# 중심이 14 -> 95mm 로 함께 밀렸다. 둘은 한 몸으로 움직인다.
#
# 필터의 extent 로는 못 잡는다. 프레임당 변화가 3mm 라 χ² 를 매번 통과하고,
# 통과할 때마다 extent 상태가 따라 내려가 다음 프레임의 기준이 된다 — 천 번의
# 작은 베임이다. 그래서 **드리프트보다 훨씬 느린 기준**을 따로 둔다.
# **기본 켜짐.** 판정이 비대칭이 된 뒤로는 악화하는 케이스가 없다 —
# 가림 씬 전 물체에서 30mm 초과 발행이 줄고(keyboard 4->0%, black bag
# 74->41%, book 17->1%, gray notebook 19->5%), 정지·이동·등속 대조군은
# 전부 무회귀다(test2/test3 check_accuracy PASS, isaac 트랙 동일).
# 대칭 판정이던 초기 버전은 keyboard 를 4->39% 로 악화시켰다 —
# 그 원인과 실측표는 docs/footprint_gate_2026-08-21.md.
FOOTPRINT_TAU = 0.14     # |log(면적/기준)| 이 넘으면 절단 관측으로 본다.
                         # 실측 고원 [0.10, 0.20] 의 로그 중점(KAPPA_PHYS·
                         # TAU_EXT 와 같은 산출). 0.08 은 정상 프레임 오탐이
                         # 41% 로 폭발하고 0.25 는 검출이 0 이 된다.
FOOTPRINT_ALPHA = 0.02   # 기준 EMA 이득 — 시정수 ~50프레임. 실측 드리프트는
                         # 25프레임(1.5s)이라 기준이 절반도 못 따라간다.


def _footprint_deviation(track, obb):
    """풋프린트 면적의 느린 기준을 갱신하고 log 편차를 돌려준다.

    **항상 불러야 한다** — 게이트가 꺼져 있어도 기준이 따라가고 있어야 켜는
    순간부터 유효하다. 그래서 판정(부호·문턱 비교)은 호출자에게 남긴다:
    갱신과 판정을 한 함수에 묶으면 "노브를 먼저 검사하면 싸다"는 리팩터가
    기준을 조용히 얼려버린다.

    **플래그가 떠도 기준은 계속 따라간다.** 동결이 더 잘 잡을 것 같지만 실측은
    반대였다(드리프트 검출 동일 72%, 정상 프레임 오탐은 7.2% -> 17.8% 로 악화).
    그리고 동결에는 탈출구가 필요한데, "연속 N 회 서로 일관하면 채택" 같은
    관용구는 여기서 안 통한다 — **매끄러운 드리프트는 언제나 자기들끼리
    일관하기 때문**이다(실측: 채택 상계를 붙이자 검출이 72% -> 14% 로 무너졌다).
    창 스프레드로 "정착 vs 이동"을 가르려 해도 분포가 겹친다(드리프트 p50
    0.078 vs 정착 p50 0.049).

    그냥 따라가게 두면 교착이 구조적으로 불가능해진다 — 편차 D 는 기하급수로
    닫히므로 어떤 편차든 ln(TAU/D)/ln(1-α) 프레임 안에 플래그가 풀린다
    (면적 4 배 오염이어도 ~113프레임/7.5초). 상계가 상수 하나에서 나온다.

    미터 면적으로 비교한다 — 강체의 미터 풋프린트는 거리와 무관하게 일정하고,
    각면적(픽셀 수)은 물체가 광축 방향으로 움직이면 그 자체로 변한다.

    **판정이 비대칭인 것이 핵심이다.** 부분 가림은 보이는 영역을 *줄인다*.
    면적이 늘어나는 것은 마스크가 번지거나 인접 물체를 먹은 것인데, 그건
    중심을 밀지 않는다 — test4 raw 실측(프레임별 중심 이동 중앙값):

        면적 감소 < -0.14 : keyboard 134mm · book 87mm · black bag 184mm
        면적 증가 > +0.14 : keyboard 1.5mm · book 2.1mm · black bag 30mm

    대칭(|dev| > TAU)으로 걸면 이 정상 프레임들을 통째로 버린다 —
    keyboard 112프레임, book 203프레임. 실제로 그것이 keyboard 악화
    (>30mm 4% -> 39%)의 원인이었다. mask_depth_to_points 의 near/far 밴드
    비대칭과 같은 계보다: 막으려는 것과 살려야 하는 것이 서로 반대 방향에 있다.
    """
    e = obb.extent_sorted
    la = float(np.log(e[0] * e[1] + 1e-12))
    if track.area_ref is None:
        track.area_ref = la
        return 0.0
    dev = la - track.area_ref
    track.area_ref += FOOTPRINT_ALPHA * dev
    return dev


def _fit_support_plane(detections, depth, K, depth_scale, ring_px=21):
    """검출 마스크 바로 바깥의 링에서 지지면(벨트)을 맞춘다.

    화면 전체로 맞추면 지배 평면 = 바닥이 이긴다(fit_plane 주석 참고).
    링은 정의상 "물체가 얹힌 면"이라 물체가 어디 있든 옳은 면을 고른다.
    """
    union = np.zeros(depth.shape, np.uint8)
    for d in detections:
        union |= d["mask"].astype(np.uint8)
    ring = cv2.dilate(union, np.ones((ring_px, ring_px), np.uint8)) & ~union
    return fit_plane(depth, K, depth_scale, mask=ring.astype(bool))


class PerceptionPipeline:
    def __init__(self, detector, depth_scale=0.001, rot_alpha=0.15,
                 iou_threshold=0.3, max_missed=5, detect_interval=5,
                 max_per_prompt=1, pub_score_min=0.0, enable_merge=True,
                 belt_plane=None, use_belt_plane=True,
                 enable_footprint_gate=True):
        self.detector = detector
        self.depth_scale = depth_scale
        self.rot_alpha = rot_alpha
        # "라벨당 트랙 수" 제한은 트래커의 새 트랙 생성에서만 건다
        # (검출 단계에서 자르면 score 역전 시 ID가 끊김)
        self.tracker = IouTracker(iou_threshold, max_missed,
                                  max_per_label=max_per_prompt,
                                  pub_score_min=pub_score_min,
                                  enable_merge=enable_merge)
        self.detect_interval = max(1, detect_interval)
        # 벨트 평면 (n, d). 고정 카메라라 상수 — 첫 성공 후 캐시한다.
        # 미리 주면(캘리브 노브) 그 값으로 고정하고 다시 맞추지 않는다.
        # **기본 켜짐** (2026-08-21 오후 재평가). 오전에는 꺼두었는데, 그
        # 유일한 근거였던 "test4 가림 중 book 발행 105 -> 8 프레임" 이
        # 풋프린트 게이트로 사라졌다 — 원인이 평면 구속이 아니라 부분 가림에
        # 잘린 관측이었기 때문이다. 게이트를 켠 상태로 다시 재니 평면 구속이
        # 전 물체에서 이긴다(docs/belt_plane_2026-08-21.md 의 재평가 절).
        self.use_belt_plane = use_belt_plane
        self.belt_plane = belt_plane
        self._plane_fixed = belt_plane is not None
        self._plane_tried = False
        # 부분 가림(풋프린트 절단) 게이트 — 위 FOOTPRINT_TAU 주석 참고
        self.enable_footprint_gate = enable_footprint_gate
        self._reset_run_state()

    def _reset_run_state(self):
        """런(run) 단위 상태를 초기화한다 — __init__ 과 reset() 의 단일 정의.

        두 곳에서 손으로 같은 목록을 유지하면 조용히 갈라진다. 2026-08-21 에
        실제로 그랬다: 새 필드를 추가하며 생성자 인자 대입이 reset() 에까지
        복제돼 NameError 가 났고, **어떤 테스트도 reset() 을 부르지 않아**
        90개가 그대로 통과했다. reset() 의 호출처는 런타임 프롬프트 교체
        (/perception/prompt) 하나뿐인데 그게 제로샷의 실사용 형태다.
        """
        if not self._plane_fixed:
            self.belt_plane = None
            self._plane_tried = False
        self._frame_idx = 0
        self._prev_gray = None
        self._last_stamp = None
        self._last_dt = NOMINAL_DT
        self.last_was_keyframe = False  # 상태 표시용

    def reset(self):
        self.tracker.reset()
        self._reset_run_state()

    def process(self, rgb, depth, K, prompts, stamp_s=None):
        """반환: 이 프레임의 트랙 목록(가림 트랙 포함 — 표시용).

        stamp_s: 프레임 시각(초). 실제 스탬프를 쓰는 이유는 프레임 드롭
        (_busy)·bag 재생 속도와 무관하게 필터 dt와 신선도(T_STALE)가
        물리 시간을 따르게 하기 위해서다. 미지정 시 공칭 15fps로 합성.

        pose를 소비(발행·기록)할지는 Track.publishable로 판정할 것 —
        가림 트랙의 obb는 마지막 정상값(stale)이다.
        """
        if stamp_s is None:
            stamp_s = (self._last_stamp or 0.0) + NOMINAL_DT
        dt = (stamp_s - self._last_stamp) if self._last_stamp is not None \
            else NOMINAL_DT
        self._last_stamp = stamp_s
        self._last_dt = float(np.clip(dt, 1e-3, 0.5))  # "타당한 dt"의 단일 정의
        # 관측 유무와 무관하게 모든 트랙의 시간을 전진 — 거부·미관측
        # 프레임에도 P가 자라는 것이 교착 불가능성의 근거다
        for t in self.tracker.tracks:
            t.now = stamp_s
            if t.filter is not None:
                t.filter.predict(self._last_dt)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keyframe = (self._frame_idx % self.detect_interval == 0
                    or self._prev_gray is None)
        self.last_was_keyframe = keyframe
        self._frame_idx += 1

        out = (self._detect_frame(rgb, depth, K, prompts, stamp_s) if keyframe
               else self._track_frame(gray, depth, K))
        self._prev_gray = gray
        return out

    def _detect_frame(self, rgb, depth, K, prompts, stamp_s):
        detections = self.detector.detect(rgb, prompts)
        if self.use_belt_plane and self.belt_plane is None \
                and not self._plane_tried:
            # 링은 반드시 **고점수 검출**로만 만든다. detect()는 연관용
            # 저점수(assoc_threshold=0.1)까지 돌려주는데, 오검출 주변의 링은
            # 물체가 얹힌 면이 아니라 아무 면이나 준다 — test3 실측: 검출
            # 28개로 링을 만들면 depth 가 0.91~1.73m 로 퍼져 RANSAC 인라이어가
            # 0.14 까지 떨어져 추정이 통째로 실패했다.
            thr = getattr(self.detector, "threshold", 0.0)
            strong = [d for d in detections if d["score"] >= thr]
            if strong:
                # 검출이 처음 생긴 키프레임에서 **한 번만** 시도한다. 고정
                # 카메라에서 지지면은 상수이므로 여기서 못 잡으면 그 씬은
                # 평면 가정이 안 맞는 것이다. 계속 재시도하면 물체가 이동해
                # 링이 달라진 어느 프레임에서 우연히 문턱을 넘고, 그 나쁜
                # 평면이 영구 캐시된다 — test3 실측: 18회 실패 후 19번째에
                # 통과한 평면으로 pink block 발행이 336 -> 168 프레임,
                # size_std 가 6.7 -> 104.9mm 로 파탄났다.
                self._plane_tried = True
                self.belt_plane = _fit_support_plane(strong, depth, K,
                                                     self.depth_scale)
                # 단발 추정 후 영구 캐시다. 조용히 틀린 평면은 두께·중심 z 로만
                # 드러나 원인을 평면에서 찾지 않게 된다 — 한 줄이라도 남긴다.
                # 시도한 경우에만 찍는다 — 검출이 늦게 잡히는 씬에서 "실패" 를
                # 매 키프레임 스팸하면 시도조차 안 한 것과 구분이 안 된다.
                # flush 필수 — 없으면 launch 로 띄웠을 때 버퍼에 갇혀
                # 운영 중 진단이 안 된다(2026-08-24 라이브에서 실제로 못 봤다).
                # sam3_detector 의 프롬프트 경고가 같은 이유로 flush 를 쓴다.
                print(f"[belt_plane] {self.belt_plane}" if self.belt_plane
                      else "[belt_plane] 추정 실패 — 무구속 OBB 로 진행",
                      flush=True)
        for d in detections:  # 매칭 비용(depth 충돌 배제)이 쓸 depth를 1회만 계산
            d["z"] = masked_depth_median(d["mask"], depth, self.depth_scale,
                                         d["box"])
        pairs = self.tracker.update(
            detections, high_score=getattr(self.detector, "threshold", None))
        out = []
        for track, det in pairs:
            track.now = stamp_s  # update가 만든 새 트랙 포함 — 여기서 일괄 주입
            track.mask = det["mask"]
            self._update_geometry(track, depth, K)
            out.append(track)
        # 이 프레임 필터가 갱신된 뒤에 중복을 판정한다 — 판정이 두 트랙의
        # 상태·공분산을 보므로 갱신 전에 걸면 한 프레임 낡은 상태로 본다.
        dead = self.tracker.merge_duplicates()
        if dead:
            out = [t for t in out if t.track_id not in dead]
        return self._with_frozen(out)

    def _track_frame(self, gray, depth, K):
        out = []
        for track in self.tracker.tracks:
            # 동결(연속 미검출) 전에는 missed와 무관하게 계속 전파한다 —
            # 이전의 missed>0 중단은 "짧은 미검출 구간 사망" 데드존을 만들었다
            if track.frozen or track.mask is None:
                continue
            # 현재 마스크 위치의 depth가 가리개 침입이면 전파·융합 모두 보류
            # — LK가 가리개(닮은 표면)를 따라가며 마스크·박스가 물체를
            # 떠나는 것을 차단한다 (라이브 실측: 서류 파일에 마스크가 붙어
            # 따라감). 보류 중에도 predict가 P를 키우므로 유한 시간 뒤 해제.
            # filter 가드: 판정 불가 트랙의 depth 중앙값 계산(0.7ms) 생략.
            if track.filter is not None and depth_intrusion(
                    track, masked_depth_median(track.mask, depth,
                                               self.depth_scale, track.box)):
                continue
            flow = propagate_mask(self._prev_gray, gray, track.mask, track.box)
            if flow is not None:
                dx, dy = flow
                track.mask = _shift_mask(track.mask, dx, dy)
                track.box = track.box + np.array([dx, dy, dx, dy])
            self._update_geometry(track, depth, K)
            out.append(track)
        return self._with_frozen(out)

    def _with_frozen(self, out):
        """이 프레임에 관측이 없던 트랙(동결·침입 보류 포함)도 표시용으로
        덧붙인다 — 화면에서 깜빡이며 사라지지 않게. 소비자는
        Track.publishable로 거른다."""
        seen = {t.track_id for t in out}
        return out + [t for t in self.tracker.tracks
                      if t.track_id not in seen
                      and (t.occluded or t.obb is not None)]

    def _update_geometry(self, track, depth, K):
        """관측 → 필터 융합. 게이트 거부 시 상태·표시 OBB 어느 것도 안 바뀌고,
        관측 자체가 없으면(depth 소실 등) 신선도 타이머만 흘러 T_STALE 뒤
        발행이 멈춘다 — stale pose가 로봇에 도달하는 경로가 없다."""
        border = _touches_border(track.mask)
        if track.filter is None and border:
            return  # 절단 관측으로 시드하지 않음 — 점군·OBB 계산(≈10ms)도 생략
        points = mask_depth_to_points(track.mask, depth, K,
                                      depth_scale=self.depth_scale)
        # 벨트 평면 구속이 성립하면 그쪽이 진짜 두께·중심을 준다. 기울어
        # 놓인 물체에서는 obb_on_plane 이 None 을 내고 무구속으로 폴백한다.
        obb = (obb_on_plane(points, self.belt_plane)
               if self.belt_plane is not None else None)
        constrained = obb is not None
        if constrained and obb.extent[2] > MAX_THICKNESS:
            # 물리적으로 불가능한 두께 = 마스크가 물체가 아닌 것을 잡았다.
            # 무구속으로 폴백하지 않고 이 프레임 관측을 통째로 버린다 — 같은
            # 오염 점군의 무구속 OBB 도 쓰레기고, 폴백하면 규약이 뒤집혀
            # 재시드까지 일어난다. 시드 시점에 들어온 값은 χ² 가 기각할 점프가
            # 없어서 그대로 트랙의 진실이 된다(test5 실측: gray notebook 쓰레기
            # 트랙이 두께 530mm 로 시드돼 중심이 265mm 틀린 채 8프레임 발행).
            # 관측이 없으면 신선도 타이머가 흘러 발행이 멈추고, 키프레임 SAM
            # 재검출은 그대로 살아 있어 마스크가 고쳐지면 자연 복구된다.
            return
        if obb is None:
            obb = compute_obb(points)
        if obb is None:
            return
        log_ext = np.log(np.sort(obb.extent)[::-1] + 1e-9)
        if track.filter is None:
            self._seed_filter(track, obb, log_ext, constrained)
            return
        if constrained != track.plane_constrained:
            # 규약이 바뀌면 중심은 h/2, 두께는 log 로 수십 배 점프한다 —
            # 한 트랙 안에서 두 규약을 섞으면 정직한 관측이 영구 기각된다.
            # 키프레임에서만 새 규약으로 재시드하고(승격 3회 다시 벌어야
            # 발행), flow 프레임의 불일치 관측은 버린다.
            if self.last_was_keyframe:
                self._seed_filter(track, obb, log_ext, constrained)
            return
        f = track.filter
        speed = float(np.linalg.norm(f.v))
        steps = (self._frame_idx - 1) % self.detect_interval  # 키프레임 후 경과
        # 이동 물체의 마스크는 프레임 주기 안 어디 시점의 위치인지 불확실 —
        # v·dt 항이 없으면 빠른 구간(test3 실측 137mm/s)에서 정직한 관측이
        # 게이트 폭(~2mm)을 넘어 기각된다. 정지 물체에서는 0이라 무해.
        r_extra = pos_r_extra(speed, self._last_dt, steps)
        # 가리개에 의한 절단도 화면 절단과 같은 사건이다 — 보이는 부분의
        # 중심은 물체의 중심이 아니다. 판정은 화면 경계와 무관하므로 먼저
        # 부르고(기준 EMA 가 매 프레임 돌아야 한다), 처리는 border 와 합친다.
        # 기준 EMA 는 게이트가 꺼져 있어도 매 프레임 돌려야 한다 — 켜는
        # 순간부터 기준이 유효하려면 항상 따라가고 있어야 하고, 부작용이 없다.
        # 기준 EMA 는 게이트가 꺼져 있어도 매 프레임 돌아야 한다 — 켜는 순간
        # 기준이 낡아 있으면 오탐이 터진다. 그래서 갱신을 먼저, 판정은 뒤에.
        area_dev = _footprint_deviation(track, obb)
        if self.enable_footprint_gate and area_dev < -FOOTPRINT_TAU:
            # 가리개에 잘린 관측 — 보이는 부분의 중심은 물체의 중심이 아니다.
            # depth 침입과 같은 처리로 **관측을 통째로 버린다**. 화면 절단처럼
            # r_extra 로 흡수하지 않는 이유: 불확실성을 키우면 게이트가 함께
            # 넓어져 오염 관측이 오히려 더 잘 수락된다(합성 회귀에서 500mm
            # 침입이 통과했다). 원하는 건 "덜 믿는다" 가 아니라 "이 프레임은
            # 안 본다" 이다. 관측이 없으면 신선도 타이머가 흘러 T_STALE 뒤
            # 발행이 멈추고, predict 가 P 를 키우므로 교착도 없다.
            return
        if border:
            # 절단 마스크의 중심은 보이는 쪽으로 편향 — 편향을 불확실성으로
            # 흡수하고 extent는 갱신하지 않는다 (화면 진입/이탈 대응)
            r_extra += (float(f.extent_sorted[0]) / 4) ** 2
        # 위치를 먼저 융합한다 — 위치/extent 는 분리된 블록이라 순서가 결과를
        # 바꾸지 않고, extent 재시드가 "이 프레임 마스크가 진짜 물체 위에
        # 있었는가"를 위치 수락으로 확인할 수 있게 된다.
        ok_pos = f.fuse_pos(obb.center, r_extra)
        ok_ext = (f.fuse_extent(log_ext, self.last_was_keyframe and ok_pos)
                  if not border else False)
        if ok_pos:
            if self.last_was_keyframe:
                # 승격 카운트는 SAM 재검출(키프레임)만 — flow 프레임은 같은
                # 마스크의 전파라 독립 증거가 아니다 (자기 확인 승격 방지)
                track.n_accepted += 1
            track.last_accept_t = track.now
            track.update_obb(obb, self.rot_alpha, allow_rot=ok_ext)
        elif self.last_was_keyframe and not track.confirmed and not border:
            # 미승격 트랙의 기각 관측(키프레임)은 거부 대신 재시드 — 오염
            # blob으로 시드된 필터에 정직한 관측이 수십 초 잠기는 것을 방지.
            # 승격에는 결과적으로 "연속 일관 관측 3회"가 필요해진다 (F4 강화).
            self._seed_filter(track, obb, log_ext, constrained)

    def _seed_filter(self, track, obb, log_ext, constrained=False):
        """필터 (재)시드 — 시드 규약(승격 카운트·신선도·표시 재구성)의 단일 정의."""
        track.filter = TrackFilter(obb.center, log_ext)
        track.plane_constrained = constrained
        # 풋프린트 기준도 함께 버린다 — 재시드는 "이 트랙이 무엇인지" 를 다시
        # 정하는 사건이라 옛 기준을 물려받으면 새 관측이 영구 절단 판정된다.
        track.area_ref = None
        track.n_accepted = 1
        track.last_accept_t = track.now
        track.obb = None  # 이전 시드의 표시 폐기 — update_obb가 새로 구성
        track.update_obb(obb, self.rot_alpha)

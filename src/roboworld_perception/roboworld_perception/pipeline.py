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

from .fusion import FLOW_STEP_STD, SYNC_STD, TrackFilter
from .geometry import compute_obb, mask_depth_to_points, masked_depth_median
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

def propagate_mask(prev_gray, gray, mask, box=None, max_points=300):
    """광학흐름 중앙값으로 마스크·박스 평행이동량 (dx, dy)를 구한다.

    box(xyxy)가 주어지면 그 ROI 안에서만 마스크 픽셀을 찾는다(전체 프레임
    스캔 회피). 유효 포인트가 부족하면 None (이전 위치 유지).
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
    return float(np.median(d[:, 0])), float(np.median(d[:, 1]))


def _shift_mask(mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = mask.shape
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h)) > 0


def _touches_border(mask):
    """마스크가 화면 경계에 닿아 있는가 — 절단 관측(물체 일부만 보임) 판정."""
    return bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any())


class PerceptionPipeline:
    def __init__(self, detector, depth_scale=0.001, rot_alpha=0.15,
                 iou_threshold=0.3, max_missed=5, detect_interval=5,
                 max_per_prompt=1, pub_score_min=0.0):
        self.detector = detector
        self.depth_scale = depth_scale
        self.rot_alpha = rot_alpha
        # "라벨당 트랙 수" 제한은 트래커의 새 트랙 생성에서만 건다
        # (검출 단계에서 자르면 score 역전 시 ID가 끊김)
        self.tracker = IouTracker(iou_threshold, max_missed,
                                  max_per_label=max_per_prompt,
                                  pub_score_min=pub_score_min)
        self.detect_interval = max(1, detect_interval)
        self._frame_idx = 0
        self._prev_gray = None
        self._last_stamp = None
        self._last_dt = NOMINAL_DT
        self.last_was_keyframe = False  # 상태 표시용

    def reset(self):
        self.tracker.reset()
        self._frame_idx = 0
        self._prev_gray = None
        self._last_stamp = None
        self._last_dt = NOMINAL_DT

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
        obb = compute_obb(points)
        if obb is None:
            return
        log_ext = np.log(np.sort(obb.extent)[::-1] + 1e-9)
        if track.filter is None:
            self._seed_filter(track, obb, log_ext)
            return
        f = track.filter
        speed = float(np.linalg.norm(f.v))
        steps = (self._frame_idx - 1) % self.detect_interval  # 키프레임 후 경과
        # 이동 물체의 마스크는 프레임 주기 안 어디 시점의 위치인지 불확실 —
        # v·dt 항이 없으면 빠른 구간(test3 실측 137mm/s)에서 정직한 관측이
        # 게이트 폭(~2mm)을 넘어 기각된다. 정지 물체에서는 0이라 무해.
        r_extra = ((speed * self._last_dt) ** 2 + (speed * SYNC_STD) ** 2
                   + (FLOW_STEP_STD * steps) ** 2)
        if border:
            # 절단 마스크의 중심은 보이는 쪽으로 편향 — 편향을 불확실성으로
            # 흡수하고 extent는 갱신하지 않는다 (화면 진입/이탈 대응)
            r_extra += (float(f.extent_sorted[0]) / 4) ** 2
        ok_ext = f.fuse_extent(log_ext) if not border else False
        if f.fuse_pos(obb.center, r_extra):
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
            self._seed_filter(track, obb, log_ext)

    def _seed_filter(self, track, obb, log_ext):
        """필터 (재)시드 — 시드 규약(승격 카운트·신선도·표시 재구성)의 단일 정의."""
        track.filter = TrackFilter(obb.center, log_ext)
        track.n_accepted = 1
        track.last_accept_t = track.now
        track.obb = None  # 이전 시드의 표시 폐기 — update_obb가 새로 구성
        track.update_obb(obb, self.rot_alpha)

"""Per-frame perception pipeline shared by the offline runner and the ROS node.

하이브리드 검출·추적: SAM3는 detect_interval 프레임마다 1번(키프레임),
사이 프레임은 광학흐름으로 마스크를 평행이동해 추적한다. 3D OBB는
어느 쪽이든 그 프레임의 실제 depth로 매번 계산한다.

offline 스크립트와 ROS 노드가 공유하는 헬퍼(이미지 디코드, CSV 스키마)도
여기에 둔다 — 두 진입점이 따로 복사해 들고 있으면 조용히 어긋난다.
"""
import cv2
import numpy as np

from .geometry import (INTRUSION_RATIO, SIZE_JUMP_ABS, SIZE_JUMP_RATIO,
                       compute_obb, mask_depth_to_points, masked_depth_median)
from .tracker import IouTracker

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
    else:
        raise ValueError(f"unsupported encoding {msg.encoding}")
    itemsize = np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.step // itemsize)
    arr = arr[:, :msg.width * ch]
    return arr.reshape(msg.height, msg.width, ch).squeeze()


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


class PerceptionPipeline:
    def __init__(self, detector, depth_scale=0.001, ema=0.4, rot_alpha=0.15,
                 iou_threshold=0.3, max_missed=5, detect_interval=5,
                 max_per_prompt=1):
        self.detector = detector
        self.depth_scale = depth_scale
        self.ema = ema
        self.rot_alpha = rot_alpha
        # "라벨당 트랙 수" 제한은 트래커의 새 트랙 생성에서만 건다
        # (검출 단계에서 자르면 score 역전 시 ID가 끊김)
        self.tracker = IouTracker(iou_threshold, max_missed,
                                  max_per_label=max_per_prompt)
        self.detect_interval = max(1, detect_interval)
        self._frame_idx = 0
        self._prev_gray = None
        self.last_was_keyframe = False  # 상태 표시용

    def reset(self):
        self.tracker.reset()
        self._frame_idx = 0
        self._prev_gray = None

    def process(self, rgb, depth, K, prompts):
        """반환: 관측된 Track + 표시용 가림(occluded) Track 목록.

        pose를 소비(발행·기록)할지는 Track.publishable로 판정할 것 —
        가림 트랙의 obb는 마지막 정상값(stale)이다.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keyframe = (self._frame_idx % self.detect_interval == 0
                    or self._prev_gray is None)
        self.last_was_keyframe = keyframe
        self._frame_idx += 1

        out = (self._detect_frame(rgb, depth, K, prompts) if keyframe
               else self._track_frame(gray, depth, K))
        self._prev_gray = gray
        return out

    def _z_ok(self, track, z):
        """가리개 침입 검사 — 마스크 depth가 기준선의 INTRUSION_RATIO보다
        가까우면 거부. score와 달리 외형이 닮은 가리개도 못 속인다."""
        return (track.depth_ema == 0 or z is None
                or z >= INTRUSION_RATIO * track.depth_ema)

    def _depth_ok(self, track, mask, depth, box=None):
        if track.depth_ema == 0:
            return True  # 기준선 미형성 — depth 계산 자체를 생략
        return self._z_ok(track, masked_depth_median(mask, depth,
                                                     self.depth_scale, box))

    def _detect_frame(self, rgb, depth, K, prompts):
        detections = self.detector.detect(rgb, prompts)
        for d in detections:  # 매칭 비용·검증이 공유할 depth를 1회만 계산
            d["z"] = masked_depth_median(d["mask"], depth, self.depth_scale,
                                         d["box"])
        pairs = self.tracker.update(
            detections, validate=lambda t, d: self._z_ok(t, d["z"]),
            high_score=getattr(self.detector, "threshold", None))
        out = []
        for track, det in pairs:
            if not track.occluded:  # 검증 통과한 관측만 mask·pose 커밋
                track.mask = det["mask"]
                self._update_geometry(track, depth, K)
            out.append(track)
        return self._with_frozen(out)

    def _track_frame(self, gray, depth, K):
        out = []
        for track in self.tracker.tracks:
            if track.occluded or track.mask is None or track.missed > 0:
                continue
            # 흐름 계산 전에 현재 위치에서 침입부터 확인 — 가리개가 도착한
            # 프레임의 LK·역투영 비용을 건너뛰고, 흐름이 가리개를 따라가는
            # 것도 원천 차단
            if not self._depth_ok(track, track.mask, depth, track.box):
                track.flag_occluded()
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
        """가림 트랙을 표시용으로 덧붙인다 — 소비자는 Track.publishable로 거른다."""
        seen = {t.track_id for t in out}
        return out + [t for t in self.tracker.tracks
                      if t.occluded and t.track_id not in seen]

    def _update_geometry(self, track, depth, K):
        points = mask_depth_to_points(track.mask, depth, K,
                                      depth_scale=self.depth_scale)
        obb = compute_obb(points)
        # 제3 신호 — 크기 일관성: 물체 크기는 프레임 사이에 급변하지 않는다.
        # 닮은 가리개가 같은 높이로 겹치면(score·depth 무력) 관측 blob의
        # 크기가 튀는 것으로 오염을 잡는다.
        if obb is not None and track.obb is not None:
            new_e = np.sort(obb.extent)[::-1]
            old_e = np.sort(track.obb.extent)[::-1]
            diff = np.abs(new_e - old_e)
            jump = (diff / (old_e + 1e-9) > SIZE_JUMP_RATIO) & (diff > SIZE_JUMP_ABS)
            if jump.any():
                track.flag_occluded()
                return  # 오염 관측 — pose·기준선 어느 것도 갱신 안 함
        z = masked_depth_median(track.mask, depth, self.depth_scale, track.box)
        if z is not None:  # 정상 관측만 여기 도달 — 기준선 갱신
            track.depth_ema = z if track.depth_ema == 0 else \
                0.9 * track.depth_ema + 0.1 * z
        track.update_obb(obb, self.ema, self.rot_alpha)

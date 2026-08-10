"""Per-frame perception pipeline shared by the offline runner and the ROS node.

하이브리드 검출·추적: SAM3는 detect_interval 프레임마다 1번(키프레임),
사이 프레임은 광학흐름으로 마스크를 평행이동해 추적한다. 3D OBB는
어느 쪽이든 그 프레임의 실제 depth로 매번 계산한다.

offline 스크립트와 ROS 노드가 공유하는 헬퍼(이미지 디코드, CSV 스키마)도
여기에 둔다 — 두 진입점이 따로 복사해 들고 있으면 조용히 어긋난다.
"""
import cv2
import numpy as np

from .geometry import compute_obb, mask_depth_to_points
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
        """반환: 이번 프레임에 관측된 Track 목록 (mask/box/obb 갱신됨)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keyframe = (self._frame_idx % self.detect_interval == 0
                    or self._prev_gray is None)
        self.last_was_keyframe = keyframe
        self._frame_idx += 1

        out = (self._detect_frame(rgb, depth, K, prompts) if keyframe
               else self._track_frame(gray, depth, K))
        self._prev_gray = gray
        return out

    def _detect_frame(self, rgb, depth, K, prompts):
        detections = self.detector.detect(rgb, prompts)
        pairs = self.tracker.update(detections)
        out = []
        for track, det in pairs:
            track.mask = det["mask"]
            self._update_geometry(track, depth, K)
            out.append(track)
        return out

    def _track_frame(self, gray, depth, K):
        out = []
        for track in self.tracker.tracks:
            if track.mask is None or track.missed > 0:
                continue
            flow = propagate_mask(self._prev_gray, gray, track.mask, track.box)
            if flow is not None:
                dx, dy = flow
                track.mask = _shift_mask(track.mask, dx, dy)
                track.box = track.box + np.array([dx, dy, dx, dy])
            self._update_geometry(track, depth, K)
            out.append(track)
        return out

    def _update_geometry(self, track, depth, K):
        points = mask_depth_to_points(track.mask, depth, K,
                                      depth_scale=self.depth_scale)
        track.update_obb(compute_obb(points), self.ema, self.rot_alpha)

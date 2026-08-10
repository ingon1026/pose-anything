"""Per-frame perception pipeline shared by the offline runner and the ROS node.

하이브리드 검출·추적: SAM3는 detect_interval 프레임마다 1번(키프레임),
사이 프레임은 광학흐름으로 마스크를 평행이동해 추적한다. 3D OBB는
어느 쪽이든 그 프레임의 실제 depth로 매번 계산한다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import ObbResult, compute_obb, mask_depth_to_points
from .tracker import IouTracker


@dataclass
class TrackedObject:
    track_id: int
    label: str
    score: float
    mask: np.ndarray   # HxW bool
    box: np.ndarray    # xyxy px
    obb: ObbResult | None  # smoothed; None if depth was unusable
    flip_count: int


def propagate_mask(prev_gray, gray, mask, max_points=300):
    """광학흐름 중앙값으로 마스크·박스 평행이동량 (dx, dy)를 구한다.

    유효 포인트가 부족하면 None (이동량 신뢰 불가 -> 이전 위치 유지).
    """
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
    def __init__(self, detector, depth_scale=0.001, ema=0.4, rot_alpha=0.5,
                 iou_threshold=0.3, max_missed=5, detect_interval=5,
                 max_per_prompt=1):
        self.detector = detector
        # 검출은 후보 전부 받고(top-1로 자르면 score 역전 시 ID가 끊김),
        # "라벨당 트랙 수" 제한은 트래커의 새 트랙 생성에서만 건다.
        detector.max_per_prompt = 0
        self.depth_scale = depth_scale
        self.ema = ema
        self.rot_alpha = rot_alpha
        self.tracker = IouTracker(iou_threshold, max_missed,
                                  max_per_label=max_per_prompt)
        self.detect_interval = max(1, detect_interval)
        self._frame_idx = 0
        self._prev_gray = None

    def reset(self):
        self.tracker.reset()
        self._frame_idx = 0
        self._prev_gray = None

    def process(self, rgb, depth, K, prompts) -> list[TrackedObject]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keyframe = self._frame_idx % self.detect_interval == 0
        self._frame_idx += 1

        if keyframe or self._prev_gray is None:
            out = self._detect_frame(rgb, depth, K, prompts)
        else:
            out = self._track_frame(gray, depth, K)
        self._prev_gray = gray
        return out

    def _detect_frame(self, rgb, depth, K, prompts):
        detections = self.detector.detect(rgb, prompts)
        pairs = self.tracker.update(detections)
        out = []
        for track, det in pairs:
            track.mask = det["mask"]
            self._update_geometry(track, depth, K)
            out.append(self._to_obj(track))
        return out

    def _track_frame(self, gray, depth, K):
        out = []
        for track in self.tracker.tracks:
            if track.mask is None or track.missed > 0:
                continue
            flow = propagate_mask(self._prev_gray, gray, track.mask)
            if flow is not None:
                dx, dy = flow
                track.mask = _shift_mask(track.mask, dx, dy)
                track.box = track.box + np.array([dx, dy, dx, dy])
            self._update_geometry(track, depth, K)
            out.append(self._to_obj(track))
        return out

    def _update_geometry(self, track, depth, K):
        points = mask_depth_to_points(track.mask, depth, K,
                                      depth_scale=self.depth_scale)
        track.update_obb(compute_obb(points), self.ema, self.rot_alpha)

    @staticmethod
    def _to_obj(track):
        return TrackedObject(track.track_id, track.label, track.score,
                             track.mask, track.box, track.obb, track.flip_count)

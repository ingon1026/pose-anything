"""Per-frame perception pipeline shared by the offline runner and the ROS node."""
from dataclasses import dataclass

import numpy as np

from .geometry import ObbResult, compute_obb, mask_depth_to_points
from .tracker import IouTracker, Track


@dataclass
class TrackedObject:
    track_id: int
    label: str
    score: float
    mask: np.ndarray   # HxW bool
    box: np.ndarray    # xyxy px
    obb: ObbResult | None  # smoothed; None if depth was unusable
    flip_count: int


class PerceptionPipeline:
    def __init__(self, detector, depth_scale=0.001, ema=0.4, rot_alpha=0.5,
                 iou_threshold=0.3, max_missed=5):
        self.detector = detector
        self.depth_scale = depth_scale
        self.ema = ema
        self.rot_alpha = rot_alpha
        self.tracker = IouTracker(iou_threshold, max_missed)

    def reset(self):
        self.tracker.reset()

    def process(self, rgb, depth, K, prompts) -> list[TrackedObject]:
        detections = self.detector.detect(rgb, prompts)
        pairs = self.tracker.update(detections)
        out = []
        for track, det in pairs:
            points = mask_depth_to_points(det["mask"], depth, K,
                                          depth_scale=self.depth_scale)
            track.update_obb(compute_obb(points), self.ema, self.rot_alpha)
            out.append(TrackedObject(track.track_id, track.label, track.score,
                                     det["mask"], track.box, track.obb,
                                     track.flip_count))
        return out

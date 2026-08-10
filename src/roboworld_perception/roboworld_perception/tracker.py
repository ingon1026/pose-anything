"""Greedy 2D-box IoU tracker with EMA smoothing and OBB axis continuity."""
from dataclasses import dataclass, field

import numpy as np

from .geometry import ObbResult, match_axes, smooth_rotation


def box_iou(a, b):
    """IoU of xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class Track:
    track_id: int
    label: str
    box: np.ndarray
    score: float
    obb: ObbResult | None = None
    missed: int = 0
    flip_count: int = 0
    age: int = 0
    mask: np.ndarray | None = field(default=None, repr=False)  # 하이브리드 추적용
    _prev_rpy: np.ndarray | None = field(default=None, repr=False)

    def update_obb(self, obb: ObbResult | None, ema=0.4, rot_alpha=0.15,
                   rot_deadband_deg=2.0):
        if obb is None:
            return
        if self.obb is None:
            self.obb = obb
        else:
            R, ext = match_axes(obb.R, obb.extent, self.obb.R)
            # ponytail: flip metric = >45deg jump of first axis after matching
            if np.dot(R[:, 0], self.obb.R[:, 0]) < np.cos(np.radians(45)):
                self.flip_count += 1
            # 마스크·depth 노이즈로 인한 OBB 방향 떨림 억제:
            # 데드밴드(작은 변화는 무시) + 강한 slerp 스무딩
            rel_trace = np.trace(self.obb.R.T @ R)
            ang = np.degrees(np.arccos(np.clip((rel_trace - 1) / 2, -1, 1)))
            if ang < rot_deadband_deg:
                R_new = self.obb.R
            else:
                R_new = smooth_rotation(self.obb.R, R, rot_alpha)
            self.obb = ObbResult(
                center=ema * obb.center + (1 - ema) * self.obb.center,
                extent=ema * ext + (1 - ema) * self.obb.extent,
                R=R_new,
                num_points=obb.num_points,
            )


class IouTracker:
    def __init__(self, iou_threshold=0.3, max_missed=5, max_per_label=0):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        # 라벨당 트랙 수 제한 (0=무제한). 검출 단계에서 top-1을 자르면
        # score 역전 시 다른 인스턴스로 튀어 ID가 끊기므로, 후보는 전부
        # 받아 기존 트랙과 먼저 매칭하고 "새 트랙 생성"만 제한한다.
        self.max_per_label = max_per_label
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections):
        """detections: list of dicts with keys label, box (xyxy), score.

        Returns list of (track, detection) pairs for this frame.
        Matching is restricted to same-label tracks.
        """
        pairs = []
        n_old = len(self.tracks)  # 이 아래에서 추가되는 새 트랙과 구분
        candidates = [
            (box_iou(t.box, d["box"]), ti, di)
            for ti, t in enumerate(self.tracks)
            for di, d in enumerate(detections)
            if t.label == d["label"]
        ]
        used_t, used_d = set(), set()
        for iou, ti, di in sorted(candidates, key=lambda c: -c[0]):
            if iou < self.iou_threshold or ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t.box = np.asarray(detections[di]["box"], dtype=float)
            t.score = detections[di]["score"]
            t.missed = 0
            t.age += 1
            pairs.append((t, detections[di]))

        unmatched = sorted((d for di, d in enumerate(detections) if di not in used_d),
                           key=lambda d: -d["score"])
        for d in unmatched:
            alive = sum(1 for t in self.tracks
                        if t.label == d["label"] and t.missed <= self.max_missed)
            if self.max_per_label > 0 and alive >= self.max_per_label:
                continue
            t = Track(self._next_id, d["label"], np.asarray(d["box"], dtype=float),
                      d["score"])
            self._next_id += 1
            self.tracks.append(t)
            pairs.append((t, d))

        for ti in range(n_old):
            if ti not in used_t:
                self.tracks[ti].missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return pairs

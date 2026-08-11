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
    score_ema: float = 0.0   # 정상 score 기준선 (부분 가림 판별용)
    depth_ema: float = 0.0   # 정상 depth 기준선(m) — 가리개 침입 판별용
    occluded: bool = False   # 완전/부분 가림 상태 — pose 발행 중단
    _prev_rpy: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.score_ema == 0:
            self.score_ema = self.score  # 생성 score로 기준선 시드

    @property
    def publishable(self):
        """소비자(토픽·CSV)가 이 트랙의 pose를 내보내도 되는가."""
        return self.obb is not None and not self.occluded

    # ── 가림 상태 전이는 아래 세 메서드로만 일어난다 ──────────────
    def observe(self, det):
        """정상 매칭 관측 커밋. score 급락(부분 가림)이면 box·EMA는 보존."""
        self.score = det["score"]
        self.missed = 0
        self.age += 1
        self.occluded = self.score < 0.5 * self.score_ema
        if not self.occluded:
            self.box = np.asarray(det["box"], dtype=float)
            self.score_ema = 0.9 * self.score_ema + 0.1 * self.score

    def flag_occluded(self):
        """가림 판정(depth 침입·완전 소실) — 관측은 있었어도 수락하지 않음."""
        self.occluded = True

    def reactivate(self, det):
        """가림 후 재등장 복귀. depth 기준선은 리셋 — 가림 중 물체가
        들리거나 교체됐을 수 있어 낡은 기준선이 오판을 만든다 (OC-SORT의
        재등장 상태 복구와 같은 취지). 다음 정상 프레임에서 재시드된다."""
        self.box = np.asarray(det["box"], dtype=float)
        self.score = det["score"]
        self.missed = 0
        self.age += 1
        self.occluded = False
        self.depth_ema = 0.0

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
    def __init__(self, iou_threshold=0.3, max_missed=5, max_per_label=0,
                 occlusion_hold=12):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        # 가림 대응: missed가 max_missed를 넘으면 삭제 대신 "동결" —
        # 좌표 발행은 멈추되 트랙을 살려두고, 재등장 시 같은 자리 IoU
        # 매칭으로 ID를 복귀시킨다. 동결까지 합쳐 max_missed+occlusion_hold
        # 회 연속 미검출이면 그때 삭제. (test4 실측: 가림 최대 2.6초 소실)
        self.occlusion_hold = occlusion_hold
        # 라벨당 트랙 수 제한 (0=무제한). 검출 단계에서 top-1을 자르면
        # score 역전 시 다른 인스턴스로 튀어 ID가 끊기므로, 후보는 전부
        # 받아 기존 트랙과 먼저 매칭하고 "새 트랙 생성"만 제한한다.
        self.max_per_label = max_per_label
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def update(self, detections, validate=None):
        """detections: list of dicts with keys label, box (xyxy), score.

        validate(track, det) -> bool: 매칭된 검출을 수락하기 전 외부 검증
        (예: depth 침입). 실패하면 관측을 커밋하지 않고 가림으로 처리한다 —
        "커밋 후 롤백"이 아니라 "수락 전 검증"이라 트랙 상태가 오염될
        틈이 없다. Returns list of (track, detection) pairs.
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
            if validate is not None and not validate(t, detections[di]):
                t.missed = 0       # 자리에 뭔가 있음은 확인됨 — 동결 타이머만 정지
                t.flag_occluded()  # 검증 실패: box·score·EMA 어느 것도 커밋 안 함
            else:
                t.observe(detections[di])
            pairs.append((t, detections[di]))

        unmatched = sorted((d for di, d in enumerate(detections) if di not in used_d),
                           key=lambda d: -d["score"])
        for d in unmatched:
            # 가림 후 재등장 구조(rescue): IoU 매칭에 실패한 검출은 새 트랙을
            # 만들기 전에, 같은 라벨의 동결 트랙에 우선 복귀시킨다 (ID 유지)
            frozen = [t for t in self.tracks
                      if t.label == d["label"] and t.missed > self.max_missed]
            if frozen:
                c = np.asarray(d["box"], dtype=float)
                cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
                t = min(frozen, key=lambda t: (cx - (t.box[0] + t.box[2]) / 2) ** 2
                        + (cy - (t.box[1] + t.box[3]) / 2) ** 2)
                if validate is None or validate(t, d):
                    t.reactivate(d)
                    pairs.append((t, d))
                continue  # 검증 실패면 동결 유지 (가리개를 물체로 오인 방지)
            alive = sum(1 for t in self.tracks if t.label == d["label"])
            if self.max_per_label > 0 and alive >= self.max_per_label:
                continue
            t = Track(self._next_id, d["label"], np.asarray(d["box"], dtype=float),
                      d["score"])
            self._next_id += 1
            self.tracks.append(t)
            pairs.append((t, d))

        for ti in range(n_old):
            if ti not in used_t:
                t = self.tracks[ti]
                t.missed += 1
                if t.missed > self.max_missed:
                    t.flag_occluded()  # 동결: 유지하되 발행·전파 중단
        self.tracks = [t for t in self.tracks
                       if t.missed <= self.max_missed + self.occlusion_hold]
        return pairs

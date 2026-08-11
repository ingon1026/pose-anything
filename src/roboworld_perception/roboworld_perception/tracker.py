"""Greedy 2D-box IoU tracker with EMA smoothing and OBB axis continuity."""
from dataclasses import dataclass, field

import numpy as np

from .geometry import INTRUSION_RATIO, ObbResult, match_axes, smooth_rotation


def box_iou(a, b):
    """IoU of xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _expand(box, k):
    """박스를 중심 기준 k배로 확장 (C-BIoU의 buffered box)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * k, (box[3] - box[1]) / 2 * k
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _depth_conflict(track, det):
    """검출의 depth가 트랙 기준선보다 침입 비율 이상 가까우면 True —
    가리개일 가능성이 높아 매칭 후보에서 제외한다 (PD-SORT의 depth 비용을
    실제 RGB-D로 구현). det["z"]는 pipeline이 채운다; 없으면 판단 안 함."""
    z = det.get("z")
    return (track.depth_ema > 0 and z is not None
            and z < INTRUSION_RATIO * track.depth_ema)


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
                 occlusion_hold=12, rescue_buffer=2.5):
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
        # rescue 매칭 범위 — 박스를 이 배율로 확장한 buffered IoU가 겹쳐야
        # 복귀 허용 (C-BIoU). 무제한 최근접의 원거리 오매칭을 막는다.
        self.rescue_buffer = rescue_buffer
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def _greedy_match(self, det_pool, used_t, exclude_depth_conflict=True):
        """같은 라벨 greedy IoU 매칭 — (track_idx, det) 쌍을 yield."""
        candidates = [
            (box_iou(t.box, d["box"]), ti, d)
            for ti, t in enumerate(self.tracks)
            for d in det_pool
            if t.label == d["label"] and ti not in used_t
            and not (exclude_depth_conflict and _depth_conflict(t, d))
        ]
        used_d = set()
        for iou, ti, d in sorted(candidates, key=lambda c: -c[0]):
            if iou < self.iou_threshold or ti in used_t or id(d) in used_d:
                continue
            used_t.add(ti)
            used_d.add(id(d))
            yield ti, d

    def update(self, detections, validate=None, high_score=None):
        """detections: list of dicts with keys label, box (xyxy), score
        (+ optional "z": 마스크 depth 중앙값 — 매칭 비용에 반영됨).

        validate(track, det) -> bool: 관측 수락 전 외부 검증(depth 침입 등).
        실패 시 커밋 없이 가림 처리 — 트랙 상태가 오염될 틈이 없다.

        high_score: 지정 시 2-pass 매칭(ByteTrack) — 이 값 미만 저점수
        검출은 기존 트랙 유지에만 쓰고(가림 관측), 새 트랙·rescue는 불가.
        Returns list of (track, detection) pairs.
        """
        if high_score is None:
            high, low = list(detections), []
        else:
            high = [d for d in detections if d["score"] >= high_score]
            low = [d for d in detections if d["score"] < high_score]

        pairs = []
        n_old = len(self.tracks)  # 이 아래에서 추가되는 새 트랙과 구분
        used_t = set()

        # 1차: 고점수 검출 — 정상 관측 (depth 충돌 후보는 매칭에서 제외돼
        # 가리개가 진짜 검출의 매칭을 뺏지 못한다)
        matched_high = set()
        for ti, d in self._greedy_match(high, used_t):
            t = self.tracks[ti]
            matched_high.add(id(d))
            if validate is not None and not validate(t, d):
                t.missed = 0       # 자리에 뭔가 있음은 확인됨 — 동결 타이머만 정지
                t.flag_occluded()  # 검증 실패: box·score·EMA 어느 것도 커밋 안 함
            else:
                t.observe(d)
            pairs.append((t, d))

        # 2차: 저점수 검출 — 부분 가림 중인 트랙의 생존 신호로만 사용
        for ti, d in self._greedy_match(low, used_t,
                                        exclude_depth_conflict=False):
            t = self.tracks[ti]
            t.missed = 0
            t.flag_occluded()  # 저점수 = 오염 가능성 — 커밋 없이 유지만
            pairs.append((t, d))

        # rescue·새 트랙은 고점수 검출만 가능
        unmatched = sorted((d for d in high if id(d) not in matched_high),
                           key=lambda d: -d["score"])
        for d in unmatched:
            # 가림 후 재등장 구조(rescue): buffered IoU(C-BIoU)로 범위를
            # 한정해 동결 트랙에 복귀시킨다 — 원거리 오매칭 방지
            frozen = [(ti, t) for ti, t in enumerate(self.tracks)
                      if t.label == d["label"] and t.missed > self.max_missed]
            if frozen:
                eb = _expand(d["box"], self.rescue_buffer)
                ti, t = max(frozen, key=lambda f: box_iou(
                    _expand(f[1].box, self.rescue_buffer), eb))
                if box_iou(_expand(t.box, self.rescue_buffer), eb) > 0 and \
                        (validate is None or validate(t, d)):
                    t.reactivate(d)
                    used_t.add(ti)
                    pairs.append((t, d))
                continue  # 범위 밖·검증 실패면 동결 유지
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

"""Greedy 2D-box IoU tracker with fusion-filter state and OBB axis continuity.

관측 품질 판정(score·depth·size 게이트)은 전부 fusion.TrackFilter의
χ² 게이트로 대체됐다 — 이 파일은 association(2D 매칭·생명주기)과
회전 연속성만 담당한다 (docs/fusion_design_2026-08.md).
"""
from dataclasses import dataclass, field

import numpy as np

from .fusion import TrackFilter
from .geometry import ObbResult, match_axes, smooth_rotation

# ── 생명주기 상수 (근거: docs/fusion_design_2026-08.md §4) ──
T_STALE = 0.5      # s — 그리퍼에 도달할 수 있는 stale pose 최대 나이 (안전 속성)
CONFIRM_N = 3      # 신생 트랙 발행 승격에 필요한 수락 관측 수 (F4 반쪽 재탄생 방어)
PUB_POS_STD_MAX = 0.02  # m — 발행 허용 위치 불확실성 상한 (로봇 사양 확정 전 잠정)
# 조각 판정 포함률. 화면 경계에서 잘린 물체는 SAM3가 부분 조각을 별개
# 인스턴스로 내는데, 조각↔본체 IoU는 구조적으로 작아(~0.1) 매칭 문턱을
# 못 넘고 새 트랙이 태어난다. 포함률로 재면 분리가 깨끗하다 —
# isaac_belt_moving 실측: 서로 다른 블록 12.6만 쌍의 99%분위 0.000·최대
# 0.269 vs 조각의 중앙값 0.61~1.00. 0.3~0.5 전 구간이 고원이라 0.5로 둔다.
FRAGMENT_CONTAIN = 0.5


def box_iou(a, b):
    """IoU of xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def containment(inner, outer):
    """inner 박스가 outer 안에 들어간 비율 (교집합 ÷ inner 면적).

    IoU와 달리 크기 차가 커도 값이 죽지 않는다 — 조각 대 본체는 IoU가
    구조적으로 작아(실측 ~0.1) 매칭 문턱 0.3을 못 넘지만 포함률은 크다.
    """
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((inner[2] - inner[0]) * (inner[3] - inner[1]) + 1e-9)


def _expand(box, k):
    """박스를 중심 기준 k배로 확장 (C-BIoU의 buffered box)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * k, (box[3] - box[1]) / 2 * k
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def depth_intrusion(track, z):
    """관측 depth가 트랙 추정 깊이보다 유의하게 '가까우면' True — 가리개.

    가리개 침입의 단일 술어 (association 매칭 제외·rescue 거부·flow 전파
    보류가 전부 이것을 쓴다). 기준선은 필터의 z 추정 + 불확실성:
    통계 유의(χ²(0.999,1)의 한쪽 꼬리 3.29σ) AND 물리 유의(10% 이상 근접)
    둘 다 만족해야 침입 — 잡음 큰 물체(검은 가방)의 정상 요동을 가리개로
    오인하지 않는다. 거부·미관측 중 P가 자라 σ가 넓어지므로 실제 depth
    변화(물체가 들림 등)는 유한 시간 안에 침입 판정을 벗어난다 (교착 없음)."""
    if z is None or track.filter is None:
        return False
    zt = float(track.filter.center[2])
    sigma = track.filter.innovation_std(2)
    return zt > 0 and z < zt - 3.29 * sigma and z < 0.9 * zt


def _depth_conflict(track, det):
    """검출이 가리개로 보이면 매칭 후보에서 제외 (PD-SORT의 depth 비용을
    실제 RGB-D로 구현). det["z"]는 pipeline이 채운다."""
    return depth_intrusion(track, det.get("z"))


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
    filter: TrackFilter | None = field(default=None, repr=False)  # 위치+크기 상태
    frozen: bool = False       # 연속 미검출로 동결 (association 부기 — 게이트 아님)
    n_accepted: int = 0        # 수락된 pose 관측 수 (M-of-N 승격용)
    last_accept_t: float | None = None  # 마지막 수락 시각 (신선도)
    now: float = 0.0           # 파이프라인이 매 프레임 주입하는 현재 시각
    pub_score_min: float = 0.0  # 발행 하한 (0=끔). 아래 publishable 참고
    _prev_rpy: np.ndarray | None = field(default=None, repr=False)

    @property
    def fresh(self):
        """pose 관측이 신선한가 — stale pose 발행 금지(T_STALE)의 근거."""
        return (self.last_accept_t is not None
                and self.now - self.last_accept_t <= T_STALE)

    @property
    def confirmed(self):
        return self.n_accepted >= CONFIRM_N

    @property
    def occluded(self):
        """표시용 파생 상태 — 동결됐거나, pose가 있는데 신선하지 않으면 가림."""
        return self.frozen or (self.obb is not None and not self.fresh)

    @property
    def publishable(self):
        """소비자(토픽·CSV)가 이 트랙의 pose를 내보내도 되는가.

        pub_score_min 은 기본 0.0(끔)이다. 저점수 관측이 트랙을 살리고
        pose를 갱신하는 것 자체는 의도된 설계이므로(docs/fusion_design_2026-08.md,
        test_occlusion.py의 저점수 2차 매칭 회귀) 여기를 기본으로 막으면 안 된다.

        다만 계속 저점수인 허수 조각(물체의 1/4 크기)이 발행까지 가는 경로가
        열려 있다. 조각의 기하는 자기 자신과 일관되므로 융합 필터의 χ² 게이트를
        통과한다 — 안정적인 가짜는 안정적이라서 통과한다. 설계상 이걸 막기로 한
        것은 존재확률 P_D 인데 아직 미구현이다(설계 문서 P2). 그때까지 쓰는
        임시 안전밸브다.

        절대 임계라 부분 가림에 취약하다. datasets.md 실측으로 가림 중 점수가
        자기 baseline 의 50%까지 떨어지므로, 프롬프트가 약한 물체에 켜면 진짜
        물체의 발행이 끊긴다. 그래서 기본은 끄고 씬별로만 켠다.
        """
        return (self.obb is not None and self.filter is not None
                and not self.frozen and self.confirmed and self.fresh
                and self.score >= self.pub_score_min
                and float(self.filter.pos_std.max()) <= PUB_POS_STD_MAX)

    # ── association이 호출하는 상태 전이 ──────────────────
    def observe(self, det):
        """매칭 관측 커밋(2D box·score). pose 수락은 필터 게이트가 별도 판정."""
        self.box = np.asarray(det["box"], dtype=float)
        self.score = det["score"]
        self.missed = 0
        self.age += 1
        self.frozen = False

    def reactivate(self, det):
        """가림 후 재등장 복귀(rescue). 위치 불확실성을 rescue 탐색 반경
        수준으로 재팽창한다. 신선도는 리셋하지 않음 — 첫 수락 관측이
        들어올 때까지 발행되지 않는다."""
        self.observe(det)
        if self.filter is not None:
            self.filter.inflate_for_rescue(
                0.75 * float(self.filter.extent_sorted[0]))

    def update_obb(self, obb: ObbResult | None, rot_alpha=0.15,
                   rot_deadband_deg=2.0, allow_rot=True):
        """수락된 raw 관측으로 표시·발행용 OBB 재구성.

        center·extent는 필터 상태에서, 회전은 기존 검증 경로(match_axes +
        데드밴드 + slerp) 그대로 — flips=0은 회귀 지표라 무수정.
        allow_rot=False: extent 게이트가 기각한 관측(blob·절단)의 회전은
        slerp에 넣지 않고 이전 방향을 유지한다 — 오염 회전 차단."""
        if obb is None or self.filter is None:
            return
        es = self.filter.extent_sorted
        if es[0] / max(es[1], 1e-9) < 1.1:
            # 상위 두 축이 거의 같은 물체(정사각형에 가까움)는 축 정체성이
            # 관측 불가능 — 프레임마다 축이 뒤바뀌며 flip으로 집계된다
            # (C1 실측: e1/e2≈1.03인 검은 가방만 축 교환 17~22%). 회전 동결.
            allow_rot = False
        if self.obb is None:
            R_new, ext = obb.R, obb.extent
            order = np.argsort(-np.asarray(ext))
        elif not allow_rot:
            R_new = self.obb.R
            order = np.argsort(-self.obb.extent)
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
            order = np.argsort(-np.asarray(ext))
        # 필터의 정렬 extent를 축 크기 순위에 따라 R 열에 배치
        ext_out = np.empty(3)
        ext_out[order] = self.filter.extent_sorted
        self.obb = ObbResult(center=self.filter.center.copy(), extent=ext_out,
                             R=R_new, num_points=obb.num_points)


class IouTracker:
    def __init__(self, iou_threshold=0.3, max_missed=5, max_per_label=0,
                 occlusion_hold=12, rescue_buffer=2.5, pub_score_min=0.0):
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
        # 트랙 생성 시 각 Track 에 그대로 넘긴다 (Track.publishable 참고)
        self.pub_score_min = pub_score_min
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

    def update(self, detections, high_score=None):
        """detections: list of dicts with keys label, box (xyxy), score
        (+ optional "z": 마스크 depth 중앙값 — 매칭 비용에 반영됨).

        high_score: 지정 시 2-pass 매칭(ByteTrack) — 이 값 미만 저점수
        검출은 기존 트랙 유지에만 쓰고, 새 트랙·rescue는 불가. 저점수
        관측의 pose 오염은 필터 게이트가 막으므로 여기서 막지 않는다.
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

        # 1차: 고점수 검출 (depth 충돌 후보는 매칭에서 제외돼 가리개가
        # 진짜 검출의 매칭을 뺏지 못한다)
        matched_high = set()
        for ti, d in self._greedy_match(high, used_t):
            t = self.tracks[ti]
            matched_high.add(id(d))
            t.observe(d)
            pairs.append((t, d))

        # 2차: 저점수 검출 — 부분 가림 중인 트랙의 생존·위치 신호
        for ti, d in self._greedy_match(low, used_t,
                                        exclude_depth_conflict=False):
            t = self.tracks[ti]
            t.observe(d)
            pairs.append((t, d))

        # rescue·새 트랙은 고점수 검출만 가능
        unmatched = sorted((d for d in high if id(d) not in matched_high),
                           key=lambda d: -d["score"])
        for d in unmatched:
            # 가림 후 재등장 구조(rescue): buffered IoU(C-BIoU)로 범위를
            # 한정해 동결 트랙에 복귀시킨다 — 원거리 오매칭 방지.
            # depth 침입 검출은 가리개일 가능성이 높아 복귀 불가.
            frozen = [(ti, t) for ti, t in enumerate(self.tracks)
                      if t.label == d["label"] and t.missed > self.max_missed]
            if frozen:
                eb = _expand(d["box"], self.rescue_buffer)
                ti, t = max(frozen, key=lambda f: box_iou(
                    _expand(f[1].box, self.rescue_buffer), eb))
                if box_iou(_expand(t.box, self.rescue_buffer), eb) > 0 and \
                        not _depth_conflict(t, d):
                    t.reactivate(d)
                    used_t.add(ti)
                    pairs.append((t, d))
                continue  # 범위 밖·depth 충돌이면 동결 유지
            # 기존 트랙 안에 들어앉은 검출은 그 물체의 조각이다 (경계 절단
            # 물체에서 발생). 매칭은 IoU가 작아 실패했지만 새 트랙도 아니다.
            if any(t.label == d["label"]
                   and containment(d["box"], t.box) > FRAGMENT_CONTAIN
                   for t in self.tracks):
                continue
            alive = sum(1 for t in self.tracks if t.label == d["label"])
            if self.max_per_label > 0 and alive >= self.max_per_label:
                continue
            t = Track(self._next_id, d["label"], np.asarray(d["box"], dtype=float),
                      d["score"], pub_score_min=self.pub_score_min)
            self._next_id += 1
            self.tracks.append(t)
            pairs.append((t, d))

        for ti in range(n_old):
            if ti not in used_t:
                t = self.tracks[ti]
                t.missed += 1
                if t.missed > self.max_missed:
                    t.frozen = True  # 동결: 유지하되 발행·전파 중단
        self.tracks = [t for t in self.tracks
                       if t.missed <= self.max_missed + self.occlusion_hold]
        return pairs

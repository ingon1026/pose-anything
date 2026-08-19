"""발행 점수 하한(pub_score_min) 게이트.

이 영역은 그동안 테스트가 전혀 지키지 못했다 — conftest 의 픽스처가 모든
트랙을 score=0.9 로 만들어서, 발행 조건을 어떻게 바꿔도 55 개가 통과했다.
그래서 저점수 축을 명시적으로 흔드는 테스트를 따로 둔다.
"""
import numpy as np
import pytest

from conftest import make_det, make_filtered_track
from roboworld_perception.tracker import CONFIRM_N, IouTracker, Track


def _ready(t, score=None, now=10.0):
    """publishable 의 나머지 조건(obb/confirmed/fresh)을 충족시킨다."""
    from roboworld_perception.geometry import ObbResult
    t.obb = ObbResult(center=np.zeros(3), extent=np.array([0.2, 0.1, 0.05]),
                      R=np.eye(3), num_points=100)
    t.n_accepted = CONFIRM_N
    t.now = now
    t.last_accept_t = now
    if score is not None:
        t.score = score
    return t


def test_default_is_off_low_score_still_publishes():
    """기본값 0.0 은 아무것도 막지 않는다 — bag 재생 경로 동작 불변."""
    t = _ready(make_filtered_track(), score=0.12)
    assert t.pub_score_min == 0.0
    assert t.publishable


@pytest.mark.parametrize("score,expected", [
    (0.12, False),   # 허수 조각 수준
    (0.39, False),   # 문턱 바로 아래
    (0.40, True),    # 문턱 (>= 이므로 통과)
    (0.95, True),    # 정상 검출
])
def test_gate_filters_by_score(score, expected):
    t = _ready(make_filtered_track(pub_score_min=0.4), score=score)
    assert t.publishable is expected


def test_gate_does_not_affect_other_conditions():
    """점수가 높아도 나머지 조건이 안 되면 여전히 발행 금지."""
    t = _ready(make_filtered_track(pub_score_min=0.4), score=0.95)
    t.n_accepted = CONFIRM_N - 1          # 아직 미승격
    assert not t.publishable


def test_gate_reaches_track_via_tracker():
    """IouTracker 가 새 트랙에 값을 실제로 넘기는지 (배선 회귀)."""
    tr = IouTracker(pub_score_min=0.4)
    tr.update([make_det(score=0.9)], high_score=0.4)
    assert tr.tracks, "트랙이 생성되지 않았다"
    assert tr.tracks[0].pub_score_min == 0.4


def test_tracker_default_leaves_gate_off():
    tr = IouTracker()
    tr.update([make_det(score=0.9)], high_score=0.4)
    assert tr.tracks[0].pub_score_min == 0.0

import numpy as np
from conftest import make_det

from roboworld_perception.tracker import CONFIRM_N, IouTracker, box_iou


def det(box, label="cup", score=0.9):
    return make_det(box, label, score)


def test_id_persists_while_moving():
    tr = IouTracker(iou_threshold=0.3, max_missed=2)
    ids = []
    for dx in range(0, 100, 10):  # box moves right 10px/frame (conveyor)
        pairs = tr.update([det([100 + dx, 100, 200 + dx, 180])])
        ids.append(pairs[0][0].track_id)
    assert len(set(ids)) == 1


def test_two_objects_keep_distinct_ids():
    tr = IouTracker()
    for dx in range(0, 50, 10):
        pairs = tr.update([det([100 + dx, 100, 180 + dx, 160]),
                           det([400 + dx, 100, 480 + dx, 160])])
        got = sorted((p[0].box[0], p[0].track_id) for p in pairs)
        assert got[0][1] != got[1][1]


def test_track_dropped_after_max_missed():
    # occlusion_hold=0 이면 가림 유예 없이 기존처럼 즉시 삭제
    tr = IouTracker(max_missed=2, occlusion_hold=0)
    tr.update([det([100, 100, 200, 180])])
    for _ in range(3):
        tr.update([])
    assert len(tr.tracks) == 0


def test_label_mismatch_creates_new_track():
    tr = IouTracker()
    p1 = tr.update([det([100, 100, 200, 180], label="cup")])
    p2 = tr.update([det([100, 100, 200, 180], label="mouse")])
    assert p1[0][0].track_id != p2[0][0].track_id


def test_fragment_inside_track_makes_no_new_track():
    """경계 절단 물체의 부분 조각은 새 트랙을 만들지 않는다.

    조각↔본체 IoU는 작아(여기서 0.13) 매칭 문턱을 못 넘지만, 조각이
    본체 박스 안에 들어앉아 있으므로 포함률로 걸러낸다."""
    tr = IouTracker()
    t = tr.update([det(box=(500, 100, 640, 180))])[0][0]
    frag = det(box=(570, 110, 605, 170))          # 본체 길이의 1/4 짜리 조각
    assert box_iou(t.box, frag["box"]) < tr.iou_threshold   # 매칭은 실패한다
    assert tr.update([frag]) == []                # 그래도 새 트랙은 없다
    assert len(tr.tracks) == 1


def test_adjacent_object_still_creates_track():
    """겹치지 않는 이웃 물체는 정상적으로 새 트랙이 된다 (조각 규칙 오탐 방지)."""
    tr = IouTracker()
    tr.update([det(box=(100, 100, 200, 180))])
    pairs = tr.update([det(box=(100, 100, 200, 180)), det(box=(230, 100, 330, 180))])
    assert len(tr.tracks) == 2
    assert len({p[0].track_id for p in pairs}) == 2


# ── 중복 병합 ────────────────────────────────────────────

def _dup_track(tid, cx=0.0, extent=(0.2, 0.055, 0.03)):
    """원하는 위치·크기로 수렴시킨 트랙 하나 (수렴·신선 상태).

    시드 후 같은 값으로 반복 관측시킨다 — 수렴된 필터에 다른 값을 한 번
    넣으면 χ² 게이트가 기각해 상태가 안 움직인다."""
    from roboworld_perception.fusion import TrackFilter
    from roboworld_perception.tracker import Track
    c = np.array([cx, 0.0, 1.0])
    t = Track(tid, "obj", np.array([100, 100, 200, 180], float), 0.9)
    t.filter = TrackFilter(c, np.log(np.asarray(extent)))
    for _ in range(30):
        t.filter.predict(1 / 15)
        t.filter.fuse_pos(c)
        t.filter.fuse_extent(np.log(np.asarray(extent)))
    t.n_accepted, t.now, t.last_accept_t = 10, 1.0, 1.0
    t.obb = object()          # publishable 판정용 자리표시
    return t


def _dup_pair(tr, dx=0.002, extent_b=None):
    a = _dup_track(1, 0.0)
    b = _dup_track(2, dx, extent_b or (0.2, 0.055, 0.03))
    tr.tracks = [a, b]
    return a, b


def test_merge_removes_converged_duplicate():
    """같은 자리로 수렴한 중복은 CONFIRM_N 연속 성립 후 삭제된다."""
    tr = IouTracker(max_per_label=10, enable_merge=True)
    a, b = _dup_pair(tr)
    for _ in range(CONFIRM_N - 1):
        assert tr.merge_duplicates() == set()   # 아직 연속이 안 찼다
    assert tr.merge_duplicates() == {b.track_id}
    assert [t.track_id for t in tr.tracks] == [a.track_id]  # 오래된 id 생존


def test_merge_keeps_winner_state_untouched():
    """이긴 트랙의 필터 상태는 한 바이트도 안 바뀐다 (P 축소 = 교착 복귀)."""
    tr = IouTracker(max_per_label=10, enable_merge=True)
    a, _ = _dup_pair(tr)
    before = (a.filter.center.copy(), a.filter.extent_sorted.copy(),
              a.filter.pos_std.copy())
    for _ in range(CONFIRM_N):
        tr.merge_duplicates()
    assert np.allclose(a.filter.center, before[0])
    assert np.allclose(a.filter.extent_sorted, before[1])
    assert np.allclose(a.filter.pos_std, before[2])


def test_merge_spares_distinct_neighbours():
    """벨트 위 이웃 블록(간격 0.25m)은 아무리 오래 봐도 안 합쳐진다."""
    tr = IouTracker(max_per_label=10, enable_merge=True)
    _dup_pair(tr, dx=0.25)
    for _ in range(CONFIRM_N * 3):
        assert tr.merge_duplicates() == set()
    assert len(tr.tracks) == 2


def test_merge_holds_when_size_claims_disagree():
    """크기 주장이 어긋나면(절단 트랙) 보류 — 잘린 쪽이 이기면 파지 폭이 틀어진다."""
    tr = IouTracker(max_per_label=10, enable_merge=True)
    _dup_pair(tr, extent_b=(0.152, 0.05, 0.03))   # 한쪽만 절단돼 152mm
    for _ in range(CONFIRM_N * 3):
        assert tr.merge_duplicates() == set()


def test_merge_is_noop_when_one_track_per_label():
    """라벨당 트랙이 하나인 구성에서는 켜져 있어도 아무 일도 하지 않는다.

    이것이 기본 켜짐의 안전 근거다 — 실기 bag(test2~5)은 전부
    max_per_prompt=1 이라 이 경로로 조기 반환한다. 반대로 max_per_label 을
    열면 중복 위험도 함께 열리므로 두 설정은 짝이다(IouTracker 주석).
    조기 반환을 지워도 통과하지 않도록 CONFIRM_N 회 이상 반복해야 한다 —
    한 번만 부르면 연속 카운트가 안 차서 음성 대조가 성립하지 않는다."""
    tr = IouTracker(max_per_label=1)
    _dup_pair(tr)
    for _ in range(CONFIRM_N * 2):
        assert tr.merge_duplicates() == set()


def test_merge_on_by_default_when_label_allows_many():
    """max_per_label 을 열면 기본값만으로 중복이 병합된다."""
    tr = IouTracker(max_per_label=10)
    _dup_pair(tr)
    dead = set()
    for _ in range(CONFIRM_N * 2):
        dead |= tr.merge_duplicates()
    assert dead

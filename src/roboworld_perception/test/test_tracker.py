from conftest import make_det

from roboworld_perception.tracker import IouTracker, box_iou


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

    조각↔본체 IoU는 작아(여기서 0.19) 매칭 문턱을 못 넘지만, 조각이
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

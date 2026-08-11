import numpy as np

from roboworld_perception.tracker import IouTracker


def det(box, label="cup", score=0.9):
    return {"label": label, "box": np.array(box, dtype=float), "score": score,
            "mask": None}


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

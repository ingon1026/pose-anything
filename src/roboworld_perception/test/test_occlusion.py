import numpy as np

from roboworld_perception.tracker import IouTracker


def det(box=(100, 100, 200, 180), label="obj", score=0.9):
    return {"label": label, "box": np.array(box, dtype=float), "score": score,
            "mask": None}


def test_full_occlusion_keeps_id():
    tr = IouTracker(max_missed=3, occlusion_hold=10)
    tid = tr.update([det()])[0][0].track_id
    for _ in range(8):  # max_missed(3) < 8 < 3+10 — 가림 구간
        tr.update([])
    assert len(tr.tracks) == 1
    assert tr.tracks[0].occluded  # 동결 상태
    pairs = tr.update([det()])    # 같은 자리 재등장
    assert pairs[0][0].track_id == tid       # ID 복귀
    assert not pairs[0][0].occluded          # 동결 해제


def test_track_dropped_after_hold_exceeded():
    tr = IouTracker(max_missed=3, occlusion_hold=5)
    tr.update([det()])
    for _ in range(9):  # 3+5=8 초과
        tr.update([])
    assert len(tr.tracks) == 0


def test_partial_occlusion_flags_low_score():
    tr = IouTracker()
    for _ in range(5):  # 정상 score 기준선 형성 (ema ~0.9)
        t = tr.update([det(score=0.9)])[0][0]
    assert not t.occluded
    t = tr.update([det(score=0.3)])[0][0]  # 평균의 50% 미만 (test4 실측 패턴)
    assert t.occluded
    t = tr.update([det(score=0.88)])[0][0]  # 가림 해제
    assert not t.occluded


def test_normal_fluctuation_not_flagged():
    tr = IouTracker()
    for s in [0.75, 0.70, 0.78, 0.61, 0.73]:  # test3 실측 수준의 정상 요동
        t = tr.update([det(score=s)])[0][0]
        assert not t.occluded

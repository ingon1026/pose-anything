from conftest import make_det as det

from roboworld_perception.tracker import IouTracker


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


def test_validate_reject_does_not_commit():
    """검증(depth 침입 등) 실패 시 관측이 트랙에 커밋되지 않는다."""
    tr = IouTracker()
    t0 = tr.update([det(box=(100, 100, 200, 180), score=0.9)])[0][0]
    box_before = t0.box.copy()
    ema_before = t0.score_ema
    pairs = tr.update([det(box=(105, 100, 205, 180), score=0.9)],
                      validate=lambda t, d: False)
    t = pairs[0][0]
    assert t.occluded                       # 가림 처리
    assert (t.box == box_before).all()      # box 미오염
    assert t.score_ema == ema_before        # 기준선 미오염
    # 다음 프레임 검증 통과 → 정상 복귀 (depth 플래그 해제 경로)
    t = tr.update([det(box=(100, 100, 200, 180), score=0.9)])[0][0]
    assert not t.occluded


def test_rescue_rejected_when_validate_fails():
    """동결 트랙 rescue도 검증을 통과해야 한다 (가리개 오인 복귀 방지)."""
    tr = IouTracker(max_missed=2, occlusion_hold=10)
    tid = tr.update([det()])[0][0].track_id
    for _ in range(4):
        tr.update([])
    assert tr.tracks[0].occluded
    pairs = tr.update([det(box=(400, 100, 480, 180))],
                      validate=lambda t, d: False)
    assert pairs == []                      # 복귀 거부, 새 트랙도 없음
    assert tr.tracks[0].occluded
    pairs = tr.update([det(box=(400, 100, 480, 180))])  # 검증 통과 시 복귀
    assert pairs[0][0].track_id == tid


def test_reactivate_resets_depth_baseline():
    """rescue 복귀(다른 위치 재등장) 시 낡은 depth 기준선은 버린다 —
    가림 중 물체가 들리거나 교체됐을 수 있어서. 같은 자리 재등장(IoU
    매칭)은 observe 경로라 기준선을 유지한다."""
    tr = IouTracker(max_missed=2, occlusion_hold=10)
    t = tr.update([det()])[0][0]
    t.depth_ema = 0.9
    for _ in range(4):
        tr.update([])
    t = tr.update([det(box=(400, 100, 480, 180))])[0][0]  # 다른 위치 → rescue
    assert t.depth_ema == 0.0

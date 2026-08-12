from conftest import make_det as det
from conftest import make_filtered_track

from roboworld_perception.tracker import IouTracker


def give_filter(track, z=1.0):
    """트랙에 수렴된 융합 필터를 부여 (depth 판정·rescue 테스트용)."""
    track.filter = make_filtered_track(z=z).filter
    return track


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


def test_rescue_rejected_on_depth_conflict():
    """동결 트랙 rescue도 depth 침입 검출로는 안 된다 (가리개 오인 복귀 방지)."""
    tr = IouTracker(max_missed=2, occlusion_hold=10)
    tid = tr.update([det()])[0][0].track_id
    give_filter(tr.tracks[0], z=1.0)
    for _ in range(4):
        tr.update([])
    assert tr.tracks[0].occluded
    occ = det(box=(250, 100, 330, 180))  # rescue 버퍼 범위 내
    occ["z"] = 0.5                       # 추정 깊이의 50% — 침입
    assert tr.update([occ]) == []        # 복귀 거부, 새 트랙도 없음
    assert tr.tracks[0].occluded
    real = det(box=(250, 100, 330, 180))
    real["z"] = 0.98
    assert tr.update([real])[0][0].track_id == tid  # 정상 depth면 복귀


def test_reactivate_inflates_uncertainty():
    """rescue 복귀(다른 위치 재등장) 시 위치 불확실성을 키운다 — 가림 중
    물체가 이동·교체됐을 수 있어서. 첫 수락 관측 전까지는 발행 불가."""
    tr = IouTracker(max_missed=2, occlusion_hold=10)
    t = tr.update([det()])[0][0]
    give_filter(t)
    std_before = t.filter.pos_std.max()
    for _ in range(4):
        tr.update([])
    t = tr.update([det(box=(250, 100, 330, 180))])[0][0]  # 다른 위치 → rescue
    assert t.filter.pos_std.max() > 10 * std_before
    assert not t.publishable  # 신선도 미리셋 — 수락 관측이 다시 벌어와야 함


def test_low_score_second_pass_keeps_track_alive():
    """저점수 검출(ByteTrack 2차 매칭)은 트랙 생존·위치 신호 — pose 오염은
    필터 게이트 소관이라 여기서 box 갱신을 막지 않는다."""
    tr = IouTracker(max_missed=2, occlusion_hold=5)
    tr.update([det(score=0.9, box=(100, 100, 200, 180))], high_score=0.4)
    for _ in range(4):  # max_missed를 넘는 횟수 동안 저점수만 존재
        pairs = tr.update([det(score=0.2, box=(105, 100, 205, 180))],
                          high_score=0.4)
        t = pairs[0][0]
        assert t.missed == 0 and not t.frozen  # 동결로 안 넘어감
    assert t.box[0] == 105  # 저점수 매칭도 위치는 따라간다
    t = tr.update([det(score=0.9)], high_score=0.4)[0][0]
    assert len(tr.tracks) == 1


def test_low_score_never_creates_track():
    tr = IouTracker()
    assert tr.update([det(score=0.2)], high_score=0.4) == []
    assert tr.tracks == []


def test_depth_conflict_does_not_steal_match():
    """얕은 depth(가리개) 검출은 매칭 후보에서 제외 — 진짜 검출이 매칭됨."""
    tr = IouTracker()
    t = tr.update([det()], high_score=0.4)[0][0]
    give_filter(t, z=1.0)
    occluder = det(box=(100, 100, 200, 180), score=0.95)
    occluder["z"] = 0.5   # 추정 깊이의 50% — 침입
    real = det(box=(102, 100, 202, 180), score=0.85)
    real["z"] = 0.98
    pairs = tr.update([occluder, real], high_score=0.4)
    matched = [d for tt, d in pairs if tt is t]
    assert matched and matched[0]["z"] == 0.98


def test_buffered_rescue_bounded():
    """rescue는 buffered IoU 범위 안에서만 — 원거리 오매칭 방지."""
    tr = IouTracker(max_missed=2, occlusion_hold=20, rescue_buffer=2.0)
    tid = tr.update([det(box=(100, 100, 180, 160))])[0][0].track_id
    for _ in range(4):
        tr.update([])
    p = tr.update([det(box=(220, 100, 300, 160))])  # 버퍼 내 이동 재등장
    assert p and p[0][0].track_id == tid
    for _ in range(4):
        tr.update([])
    p = tr.update([det(box=(1000, 600, 1080, 660))])  # 화면 반대편
    assert p == []  # 복귀 거부 + alive 제한으로 새 트랙도 불가

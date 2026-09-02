"""트랙 상태 판정 — /perception/tracks 용 순수 함수. ROS import 없음."""

STATES = ("visible", "held", "pending", "occluded", "lost")


def track_state(track, now) -> tuple[str, str]:
    """(state, reason). state 는 '/perception/detections 에 지금 무엇이 실리는가' 를 말한다:
    visible/held 는 track.publishable 일 때만 — 그게 detections 의 실제 게이트다.
    visible  = publishable 이고 이번 프레임 관측이 수락됨 (track.last_accept_t == now)
    held     = publishable 이고 이번 프레임은 아니지만 마지막 pose 가 여전히 실림
    pending  = fresh 하고 frozen 아닌데 publishable 이 아님(승격 전·pos_std 초과·
               score·obb 없음) → detections 에 없음, 곧 나올 수 있음
    occluded = frozen 이거나 fresh 하지 않음 → 없음
    reason 은 '왜 이번 관측을 못 썼는가': track.last_reject 가 있으면 그것, pending 인데
    없으면 publishable 을 막은 게이트 이름으로 채운다.
    lost 는 여기서 안 나온다 — 노드가 tracker.dropped 로 낸다."""
    reason = track.last_reject or "none"
    if track.publishable:
        return ("visible" if track.last_accept_t == now else "held"), reason
    if track.frozen or not track.fresh:
        return "occluded", reason
    # fresh 인데 게이트에 막힘 — reason 이 비어 있으면 게이트 이름으로 채운다
    if reason == "none":
        if track.obb is None or track.filter is None:
            reason = "no_obb"
        elif not track.confirmed:
            reason = "unconfirmed"
        elif track.score < track.pub_score_min:
            reason = "score"
        else:
            reason = "pos_std"
    return "pending", reason


def level_of(state) -> int:
    """diagnostic_msgs: OK=0 visible, WARN=1 held·pending, ERROR=2 occluded, STALE=3 lost."""
    return {"visible": 0, "held": 1, "pending": 1, "occluded": 2, "lost": 3}[state]

"""track_status.track_state/level_of 단위 테스트 + _update_geometry 가
last_reject 를 채우는지 확인하는 파이프라인 통합 테스트 1개. ROS 없음."""
import types

import numpy as np

from conftest import K, rect_scene
from roboworld_perception.geometry import ObbResult
from roboworld_perception.pipeline import PerceptionPipeline
from roboworld_perception.tracker import T_STALE, Track
from roboworld_perception.track_status import level_of, track_state


def make_track(now=1.0, last_accept_t=None, last_reject=None, frozen=False,
               n_accepted=3, has_obb=True, pos_std=(0.0, 0.0, 0.0)):
    """publishable 을 기본으로 통과하는 Track — track_state 는 이 게이트로
    visible/held 와 pending 을 가른다(publishable 은 filter.pos_std.max() 만
    읽으므로 SimpleNamespace 로 충분하다)."""
    t = Track(1, "obj", np.zeros(4), 0.9)
    t.now = now
    t.last_accept_t = last_accept_t
    t.last_reject = last_reject
    t.frozen = frozen
    t.n_accepted = n_accepted
    if has_obb:
        t.obb = ObbResult(center=np.zeros(3), extent=np.ones(3) * 0.1,
                          R=np.eye(3), num_points=100)
        t.filter = types.SimpleNamespace(pos_std=np.array(pos_std, dtype=float))
    return t


def test_visible_when_accepted_this_frame():
    t = make_track(now=1.0, last_accept_t=1.0)
    assert track_state(t, 1.0) == ("visible", "none")


def test_held_when_stale_but_within_t_stale():
    t = make_track(now=1.0, last_accept_t=1.0 - T_STALE * 0.5, last_reject="chi2")
    assert track_state(t, 1.0) == ("held", "chi2")


def test_occluded_when_not_fresh_or_frozen():
    t = make_track(now=1.0, last_accept_t=1.0 - T_STALE * 2)
    assert track_state(t, 1.0) == ("occluded", "none")
    # frozen 이면 방금 수락됐어도(last_accept_t == now, publishable 이 이미 false) occluded.
    t2 = make_track(now=1.0, last_accept_t=1.0, frozen=True)
    assert track_state(t2, 1.0) == ("occluded", "none")


def test_occluded_when_never_accepted():
    t = make_track(now=1.0, last_accept_t=None, last_reject="border")
    assert track_state(t, 1.0) == ("occluded", "border")


def test_pending_when_unconfirmed():
    """방금 수락됐지만 승격 전(n_accepted<CONFIRM_N) → pending, reason unconfirmed."""
    t = make_track(now=1.0, last_accept_t=1.0, n_accepted=1)
    assert track_state(t, 1.0) == ("pending", "unconfirmed")


def test_pending_when_pos_std_exceeds_limit():
    """방금 수락됐지만 위치 불확실성이 발행 상한을 넘음 → pending, reason pos_std."""
    t = make_track(now=1.0, last_accept_t=1.0, pos_std=(1.0, 1.0, 1.0))
    assert track_state(t, 1.0) == ("pending", "pos_std")


def test_pending_reason_prefers_last_reject():
    """fresh·unconfirmed 라도 last_reject 가 있으면 그것이 우선한다."""
    t = make_track(now=1.0, last_accept_t=1.0, last_reject="footprint", n_accepted=1)
    assert track_state(t, 1.0) == ("pending", "footprint")


def test_frozen_is_occluded_even_though_publishable_conditions_met():
    """frozen 이면 나머지가 다 갖춰져도 publishable 이 이미 false 라 occluded."""
    t = make_track(now=1.0, last_accept_t=1.0, frozen=True)
    assert track_state(t, 1.0) == ("occluded", "none")


def test_level_of_mapping():
    assert level_of("visible") == 0
    assert level_of("held") == 1
    assert level_of("pending") == 1
    assert level_of("occluded") == 2
    assert level_of("lost") == 3


# ── 파이프라인 통합 테스트 (test_depth_occlusion.py 의 방식을 따른다) ──

def make(depth_mm):
    return rect_scene(depth_mm, 100, 180, 100, 160)


def make_partial(depth_mm, cover_px):
    """가리개가 마스크의 왼쪽 cover_px 만큼을 덮은 관측 — 풋프린트 게이트 기각."""
    return rect_scene(depth_mm, 100 + cover_px, 180, 100, 160)


class NoDetector:
    def detect(self, rgb, prompts):
        return []


def step(pipe, t, depth, dt=1 / 15):
    pipe.last_was_keyframe = True
    t.now += dt
    if t.filter is not None:
        t.filter.predict(dt)
    pipe._update_geometry(t, depth, K)


def seeded_track(pipe, depth_mm=1000, frames=5):
    t = Track(1, "obj", np.array([100, 100, 180, 160], float), 0.9)
    t.mask, depth = make(depth_mm)
    for _ in range(frames):
        step(pipe, t, depth)
    return t


def test_footprint_gate_rejection_sets_last_reject():
    """풋프린트 게이트가 기각한 프레임은 Track.last_reject == "footprint"."""
    pipe = PerceptionPipeline(NoDetector(), enable_footprint_gate=True)
    t = seeded_track(pipe)
    for _ in range(10):  # 절반쯤 덮인 관측만 계속 — 게이트가 계속 기각한다
        t.mask, depth = make_partial(1000, 40)
        step(pipe, t, depth)
    assert t.last_reject == "footprint"

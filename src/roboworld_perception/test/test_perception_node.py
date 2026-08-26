"""PerceptionNode 의 입력 계약·시계 판정 회귀 테스트.

노드 코드는 GPU 도 bag 도 필요 없다 — 필요한 것은 rclpy 뿐이다. 따라서
`Sam3Detector` 만 스텁으로 바꿔 **진짜 PerceptionPipeline** 을 쓴다.
파이프라인을 통째로 가짜로 바꾸면 `time_reset_required` /
`late_frame_drop_required` 의 진짜 의미가 사라져 "노드가 어떤 상태에서
무엇을 부르는가" 를 못 본다 — 그게 2026-08-25 에 실제로 났던 버그다.
"""
import os
import re
import time

import numpy as np
import pytest

# 테스트 노드가 살아 있는 스택에 붙어 /perception/detections 를 쏘지
# 않도록 격리한다. rclpy import/init 보다 먼저 세워야 한다.
os.environ.setdefault("ROS_DOMAIN_ID", "97")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "OFF")

# input_health.py 의 주석이 말하는 "경량 CI 환경" 에는 rclpy 가 없다.
rclpy = pytest.importorskip("rclpy")

from diagnostic_msgs.msg import DiagnosticStatus  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from roboworld_perception import perception_node as pn  # noqa: E402
from roboworld_perception.input_health import (DIAG_ERROR, DIAG_OK,  # noqa: E402
                                               DIAG_WARN)
from roboworld_perception.pipeline import LATE_DROP_STREAK_MAX  # noqa: E402

W, H = 64, 48
FRAME = "camera_color_optical_frame"


class _StubDetector:
    """SAM3 자리. 검출 0개 — 파이프라인의 시간·계약 경로만 태운다."""

    def __init__(self, *_, threshold=0.4, **__):
        self.threshold = threshold

    def detect(self, rgb, prompts):
        return []


def _info(fx=300.0, width=W, height=H, frame_id=FRAME):
    msg = CameraInfo()
    msg.width, msg.height = width, height
    msg.header.frame_id = frame_id
    msg.k = [fx, 0.0, width / 2, 0.0, fx, height / 2, 0.0, 0.0, 1.0]
    return msg


def _rgbd(stamp_s, width=W, height=H, frame_id=FRAME, depth_frame_id=None):
    """같은 (w, h, frame_id) 에서 color/depth 를 함께 만든다 — 어긋남 방지."""
    sec = int(stamp_s)
    nanosec = int(round((stamp_s - sec) * 1e9))
    color = Image()
    color.height, color.width = height, width
    color.encoding = "bgr8"
    color.step = width * 3
    color.data = np.zeros((height, width, 3), np.uint8).tobytes()
    color.header.frame_id = frame_id
    color.header.stamp.sec, color.header.stamp.nanosec = sec, nanosec
    depth = Image()
    depth.height, depth.width = height, width
    depth.encoding = "16UC1"
    depth.step = width * 2
    depth.data = np.full((height, width), 1000, np.uint16).tobytes()
    depth.header.frame_id = depth_frame_id or frame_id
    depth.header.stamp.sec, depth.header.stamp.nanosec = sec, nanosec
    return color, depth


class _Harness:
    """발행물과 error 로그를 가로채 보관한다.

    on_frames 는 예외를 통째로 삼키고 error 로그만 남긴다. 그것을 안 보면
    합성 메시지가 잘못돼 프레임이 사라져도 카운터 단언이 그대로 통과한다 —
    통과가 증거가 아닌 바로 그 형태다. 그래서 errors 를 항상 확인한다.
    """

    def __init__(self, node):
        self.node = node
        self.det, self.markers, self.status, self.errors = [], [], [], []
        node.pub_det.publish = self.det.append
        node.pub_markers.publish = self.markers.append
        node.pub_status.publish = self.status.append
        node.pub_debug.publish = lambda msg: None
        node.get_logger().error = self.errors.append
        self.process_calls = 0
        real_process = node.pipeline.process

        def spy(*a, **kw):
            self.process_calls += 1
            return real_process(*a, **kw)

        node.pipeline.process = spy

    @property
    def pipeline(self):
        return self.node.pipeline

    def frame(self, stamp_s, **kw):
        self.node.on_frames(*_rgbd(stamp_s, **kw))

    def deleteall_count(self):
        return sum(1 for m in self.markers
                   for mk in m.markers if mk.action == Marker.DELETEALL)

    def latest_status(self):
        """1 Hz 제한을 풀고 지금 상태를 한 장 받아온다."""
        self.node._last_status_time = None
        before = len(self.status)
        self.node._publish_status()
        assert len(self.status) == before + 1
        return self.status[-1].status[0]


def _level(status: DiagnosticStatus) -> int:
    # 이 WSL 오버레이의 생성 바인딩은 level 이 1바이트 bytes 다.
    return status.level[0] if isinstance(status.level, bytes) else status.level


def _values(status: DiagnosticStatus) -> dict:
    return {kv.key: kv.value for kv in status.values}


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(pn, "Sam3Detector", _StubDetector)
    rclpy.init()
    n = pn.PerceptionNode()
    try:
        yield _Harness(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


# --- 양성 대조: 정상 프레임이 실제로 파이프라인까지 간다 ------------------
# 이것이 없으면 아래 단언들은 "K 가 None 이라 첫 줄에서 return" 만으로도
# 전부 통과한다.

def test_normal_frame_reaches_pipeline(node):
    node.node.on_info(_info())
    node.frame(10.0)
    assert node.errors == []
    assert node.process_calls == 1
    assert node.pipeline._last_stamp == 10.0
    assert node.pipeline.late_frame_drops == 0
    assert node.deleteall_count() == 0
    assert len(node.det) == 1  # publish() 가 실제로 돌았다


# --- /clock 리셋과 지연 프레임 ------------------------------------------

def test_late_frame_is_dropped_inside_pipeline_not_by_the_node(node):
    """time_reset_required=False · late_frame_drop_required=True 상태.

    stamp 50.0 은 1.0s 를 넘으므로 시계 리셋 분기를 못 탄다. 노드가
    이 프레임에서 먼저 return 하면 process 가 안 돌아 late_frame_drops 가
    안 늘고 상계가 영원히 발화하지 않는다.
    """
    node.node.on_info(_info())
    node.frame(100.0)
    assert node.pipeline.time_reset_required(50.0) is False
    assert node.pipeline.late_frame_drop_required(50.0) is True

    node.frame(50.0)
    assert node.errors == []
    assert node.process_calls == 2          # 노드가 삼키지 않았다
    assert node.pipeline.late_frame_drops == 1
    assert node.pipeline._last_stamp == 100.0   # 지연 프레임은 시간을 되돌리지 않는다
    assert node.deleteall_count() == 0          # 시계 리셋이 아니므로 회수 없음


def test_late_drop_streak_reaches_its_bound_and_recovers(node):
    """상계에 닿으면 시계가 옮겨간 것으로 보고 빠져나온다 — 영구 락아웃 금지."""
    node.node.on_info(_info())
    node.frame(100.0)
    for _ in range(LATE_DROP_STREAK_MAX + 1):
        node.frame(50.0)
    assert node.errors == []
    # 정확히 30 — reset() 을 타고도 살아남는 누적 카운터여야 한다
    # (그래서 _reset_run_state 가 아니라 __init__ 에 있다).
    assert node.pipeline.late_frame_drops == LATE_DROP_STREAK_MAX
    # 락아웃이 풀렸다는 유일한 증거. 노드가 먼저 return 하면 100.0 에 영원히 갇힌다.
    assert node.pipeline._last_stamp == 50.0


def test_clock_reset_withdraws_detections_before_inference(node):
    node.node.on_info(_info())
    node.frame(100.0)
    node.frame(0.5)  # <= 1.0s + 뒤로 감김 = 새 런
    assert node.errors == []
    assert node.deleteall_count() == 1
    assert node.det[-2].detections == []      # 빈 Detection3DArray 로 회수
    assert node.pipeline._last_stamp == 0.5


# --- 프레임 계약 fail-closed --------------------------------------------

@pytest.mark.parametrize("kw", [
    {"width": 32},                       # 해상도 불일치
    {"height": 24},
    {"frame_id": "other_optical_frame"},  # color frame_id 불일치
    {"depth_frame_id": "other_optical_frame"},  # depth 만 불일치
])
def test_mismatched_frame_is_rejected_whole(node, kw):
    node.node.on_info(_info())
    node.frame(10.0, **kw)
    assert node.errors == []
    assert node.process_calls == 0
    assert node.pipeline._last_stamp is None
    assert node.node._last_input_contract_error is not None
    assert node.deleteall_count() == 1  # 회수는 즉시 보인다


def test_repeated_mismatch_resets_once_not_every_frame(node):
    node.node.on_info(_info())
    for _ in range(5):
        node.frame(10.0, width=32)
    assert node.errors == []
    assert node.process_calls == 0
    assert node.deleteall_count() == 1


def test_persistent_mismatch_keeps_warning(node):
    """위반이 지속되는 동안 warn 호출을 멈추지 않는다.

    회수(DELETEALL)는 첫 프레임에 한 번이면 되지만 경고까지 한 번이면
    "경고 1줄 뒤 영구 무발행" 이 된다 — 노드는 살아 있고 에러도 로그도
    없는데 아무것도 안 나가는, `8d45975` 가 무성 실패로 이름 붙인 그
    형태다. /perception/status 가 같은 사실을 싣지만 이 저장소에서
    그 토픽을 구독하는 곳이 없으므로, 사람이 실제로 보는 표면은
    콘솔이다. 억제는 rclpy 의 Throttle(1초)이 하고 노드는 호출을
    계속한다 — 그래서 노드에 상태도 파라미터도 안 늘어난다.
    여기서 warn 을 가로채므로 throttle 을 안 타고 호출 수가 그대로 보인다.
    """
    node.node.on_info(_info())
    warns = []
    node.node.get_logger().warn = lambda msg, **kw: warns.append(msg)
    for _ in range(5):
        node.frame(10.0, width=32)
    assert node.errors == []
    assert node.process_calls == 0
    assert node.deleteall_count() == 1  # 회수는 한 번
    assert len(warns) == 5              # 경고는 매 프레임
    # 진단도 계속 나간다. 단 이 루프는 _publish_status 를 직접 부르므로
    # "1Hz 타이머가 계속 뛴다" 가 아니라 "지금 물어보면 ERROR 다" 만 본다.
    for _ in range(3):
        s = node.latest_status()
        assert _level(s) == DIAG_ERROR
        assert _values(s)["input_contract_valid"] == "false"


def test_contract_error_clears_when_a_matching_frame_arrives(node):
    """fail-closed 가 다시 안 열리면 그건 락아웃과 같은 부류의 버그다."""
    node.node.on_info(_info())
    node.frame(10.0, width=32)
    node.frame(11.0)
    assert node.errors == []
    assert node.process_calls == 1
    assert node.pipeline._last_stamp == 11.0
    assert node.node._last_input_contract_error is None


# --- CameraInfo 서명 변경 ------------------------------------------------

def test_first_camera_info_does_not_reset(node):
    node.node.on_info(_info())
    assert node.deleteall_count() == 0
    assert node.det == []


def test_camera_info_signature_change_resets_and_is_visible(node):
    node.node.on_info(_info())
    node.frame(10.0)
    node.node.on_info(_info(fx=600.0))  # 재캘리브 = 다른 서명
    assert node.errors == []
    assert node.deleteall_count() == 1
    assert node.det[-1].detections == []
    assert node.pipeline._last_stamp is None  # 파이프라인 상태가 버려졌다
    assert node.node._last_frame_time is None


def test_identical_camera_info_does_not_reset(node):
    node.node.on_info(_info())
    node.frame(10.0)
    node.node.on_info(_info())
    assert node.errors == []
    assert node.deleteall_count() == 0
    assert node.pipeline._last_stamp == 10.0


def test_invalid_camera_info_drops_calibration_and_blocks_frames(node):
    node.node.on_info(_info())
    node.frame(10.0)
    bad = _info()
    bad.width = 0
    node.node.on_info(bad)
    assert node.node.K is None
    assert node.deleteall_count() == 1
    node.frame(11.0)
    assert node.process_calls == 1  # 10.0 한 번뿐 — 낡은 K 로 역투영하지 않는다


# --- /perception/status --------------------------------------------------

def test_status_reports_error_before_camera_info(node):
    s = node.latest_status()
    assert _level(s) == DIAG_ERROR
    assert s.message == "waiting for valid camera_info"
    assert s.name == "roboworld_perception/input"
    assert s.hardware_id == "rgbd_camera"
    v = _values(s)
    assert v["camera_info_valid"] == "false"
    assert v["last_frame_age_s"] == "never"
    assert v["last_processing_duration_ms"] == "never"
    assert v["camera_image_size"] == "unknown"


def test_status_warns_while_waiting_for_frames(node):
    node.node.on_info(_info())
    s = node.latest_status()
    assert _level(s) == DIAG_WARN
    assert s.message == "waiting for RGB-D frames"
    v = _values(s)
    assert v["camera_info_valid"] == "true"
    assert v["camera_image_size"] == f"{W}x{H}"
    assert v["camera_frame"] == FRAME


def test_status_ok_after_a_frame(node):
    node.node.on_info(_info())
    node.frame(10.0)
    assert node.errors == []
    s = node.latest_status()
    assert _level(s) == DIAG_OK
    assert s.message == "RGB-D input healthy"
    v = _values(s)
    assert float(v["last_frame_age_s"]) < 1.0
    assert float(v["last_processing_duration_ms"]) >= 0.0
    assert v["out_of_order_frame_drops"] == "0"
    assert v["input_error"] == ""


def test_status_warns_when_input_goes_stale(node):
    node.node.on_info(_info())
    node.frame(10.0)
    node.node._last_frame_time = time.monotonic() - 10.0
    assert node.errors == []
    s = node.latest_status()
    assert _level(s) == DIAG_WARN
    assert s.message == "RGB-D input stale"
    assert float(_values(s)["last_frame_age_s"]) > node.node._stale_timeout


def test_status_reports_the_contract_error_text(node):
    node.node.on_info(_info())
    node.frame(10.0, width=32)
    assert node.errors == []
    s = node.latest_status()
    assert _level(s) == DIAG_ERROR
    assert s.message == "input contract invalid"
    v = _values(s)
    assert v["input_contract_valid"] == "false"
    assert "do not match camera_info" in v["input_error"]


def test_status_counts_out_of_order_drops(node):
    """분기만 지우고 카운터가 없으면 이 진단은 항상 0 인 거짓말이 된다."""
    node.node.on_info(_info())
    node.frame(100.0)
    node.frame(50.0)
    assert node.errors == []
    assert _values(node.latest_status())["out_of_order_frame_drops"] == "1"


def test_status_is_rate_limited_to_1hz(node):
    node.node.on_info(_info())
    node.node._last_status_time = None
    node.node._publish_status()
    node.node._publish_status()
    assert len(node.status) == 1


def test_watchdog_publishes_the_heartbeat(node):
    node.node.on_info(_info())
    node.node._last_status_time = None
    node.node._watchdog()
    assert len(node.status) == 1


# --- 워치독: 마커만이 아니라 detections 도 회수한다 ----------------------

def test_watchdog_withdraws_detections_not_only_markers(node):
    """입력이 끊기면 로봇이 읽는 토픽도 비운다.

    마커만 지우면 RViz 는 깨끗해지는데 /perception/detections 에는
    stale_timeout 초 전의 pose 가 그대로 남는다 — 사람이 보는 쪽만
    정리되고 그리퍼는 옛 pose 로 계속 움직인다. tracker.T_STALE 의
    fresh/publishable 판정은 노드가 프레임을 처리하는 동안에만 돌아서
    입력이 끊기면 아무 도움이 안 된다.
    """
    node.node.on_info(_info())
    node.frame(10.0)
    assert node.errors == []
    assert len(node.det) == 1  # 정상 발행 1장

    node.node._last_frame_time = time.monotonic() - 10.0
    node.node._watchdog()
    assert node.deleteall_count() == 1
    assert len(node.det) == 2                  # 회수가 실제로 나갔다
    assert node.det[-1].detections == []
    assert node.det[-1].header.frame_id == FRAME
    # 스텁 검출기가 항상 [] 를 주므로 detections 만으로는 "마지막 배열을
    # 다시 쏘기" 와 구분되지 않는다. 스탬프가 진짜 판별자다 — 소비자의
    # staleness 필터가 회수를 최신으로 봐야 한다.
    assert node.det[-1].header.stamp.sec != node.det[0].header.stamp.sec
    assert node.node._prev_marker_ids == set()


def test_watchdog_does_not_withdraw_while_input_is_healthy(node):
    """반대쪽 사고: 회수를 if 밖으로 빼면 매 초 살아있는 검출을 지운다."""
    node.node.on_info(_info())
    node.frame(10.0)
    node.node._watchdog()
    assert node.errors == []
    assert len(node.det) == 1        # 정상 발행 1장 그대로
    assert node.deleteall_count() == 0


def test_watchdog_withdraws_once_not_every_tick(node):
    """_stale_cleared 가 반복 발행을 막는다 — 1 Hz 로 빈 배열을 쏘지 않는다."""
    node.node.on_info(_info())
    node.frame(10.0)
    node.node._last_frame_time = time.monotonic() - 10.0
    node.node._watchdog()
    node.node._watchdog()
    assert node.errors == []
    assert len(node.det) == 2
    assert node.deleteall_count() == 1


def test_watchdog_does_not_discard_the_pipeline_state(node):
    """회수만 한다. pipeline.reset() 을 타면 벨트 평면과 전 트랙이 날아가
    입력이 돌아왔을 때 재적합해야 한다 — 워치독이 살 이유가 없는 비용이다.
    """
    node.node.on_info(_info())
    node.frame(10.0)
    node.node._last_frame_time = time.monotonic() - 10.0
    node.node._watchdog()
    assert node.errors == []
    assert node.pipeline._last_stamp == 10.0
    assert node.node._last_frame_time is not None


# --- 프롬프트 교체도 회수 경로를 탄다 -----------------------------------

def test_prompt_change_withdraws_detections(node):
    """소비자가 붙잡은 pose 는 이제 **다른 물체 종류**의 것이다."""
    node.node.on_info(_info())
    node.frame(10.0)
    assert node.errors == []
    assert len(node.det) == 1

    node.node.on_prompt(String(data="thermos"))
    assert node.node.prompts == ["thermos"]     # 교체 자체는 그대로 일어났다
    assert len(node.det) == 2
    assert node.det[-1].detections == []
    assert node.det[-1].header.stamp.sec != node.det[0].header.stamp.sec
    assert node.deleteall_count() == 1
    assert node.pipeline._last_stamp is None    # pipeline.reset() 도 탔다


def test_prompt_change_resets_frame_age_and_recovers_next_frame(node):
    """_reset_input_state 의 부작용을 못 박는다 — 다음 프레임에 복구된다."""
    node.node.on_info(_info())
    node.frame(10.0)
    node.node.on_prompt(String(data="thermos"))
    assert node.node._last_frame_time is None
    assert _values(node.latest_status())["last_frame_age_s"] == "never"

    node.frame(11.0)
    assert node.errors == []
    assert node.process_calls == 2
    assert float(_values(node.latest_status())["last_frame_age_s"]) < 1.0


# --- 빈 프롬프트: 무성 실패 금지 ----------------------------------------

def _catch_warns(node):
    warns = []
    # throttle_duration_sec 이 kwarg 로 오므로 **kw 가 필요하다. 여기서
    # 가로채면 throttle 을 안 타고 호출 수가 그대로 보인다.
    node.node.get_logger().warn = lambda msg, **kw: warns.append(msg)
    return warns


@pytest.mark.parametrize("text", ["", "  ", ",", " , "])
def test_empty_prompt_drops_frames_loudly(node, text):
    """parse_prompts 가 [] 를 주는 모든 입력에서 로그 0 이면 안 된다.

    이 경로는 publish() 를 한 번도 안 타므로 옛 프롬프트의 마지막
    Detection3DArray 가 **무기한** 남는다. 게다가 _last_frame_time 이
    안 갱신돼 5초 뒤 status 가 멀쩡한 카메라를 stale 이라고 가리킨다.
    """
    node.node.on_info(_info())
    warns = _catch_warns(node)
    node.node.on_prompt(String(data=text))
    assert node.node.prompts == []

    for _ in range(3):
        node.frame(10.0)
    assert node.errors == []
    assert node.process_calls == 0
    assert len(warns) == 3          # 매 프레임 — 한 줄 뒤 침묵이 아니다
    assert all("prompt" in w for w in warns)


def test_waiting_for_camera_info_stays_silent(node):
    """기동 대기는 정상이다 — 여기까지 시끄러워지면 경고가 무의미해진다."""
    warns = _catch_warns(node)
    node.frame(10.0)
    assert node.errors == []
    assert node.process_calls == 0
    assert warns == []


def test_backpressure_drop_stays_silent(node):
    """추론 중 프레임 버리기는 설계된 동작이다 — 초당 여러 줄이 나오면 안 된다."""
    node.node.on_info(_info())
    warns = _catch_warns(node)
    node.node._busy = True
    node.frame(10.0)
    assert node.errors == []
    assert node.process_calls == 0
    assert warns == []


# --- 워치독: "한 장도 안 온다" 도 무성 실패다 --------------------------

def test_watchdog_warns_when_no_frame_ever_arrives(node):
    """냉시동에서 프레임이 아예 안 오면 콘솔이 침묵한다 — 그게 버그였다.

    stale 분기는 `_last_frame_time is not None` 을 요구하는데 초기값이
    None 이라 "처음부터 안 옴" 은 영원히 안 걸린다. 로그가 "SAM3 ready"
    에서 끝나고, mcap 플러그인 부재로 bag 이 못 뜬 것과 정상 대기가
    사람 눈에 똑같아 보인다.
    """
    warns = _catch_warns(node)
    node.node._input_wait_since = time.monotonic() - 10.0
    node.node._watchdog()
    assert node.errors == []
    assert len(warns) == 1
    assert "camera_info=없음" in warns[0]   # info_topic·bag 쪽을 가리킨다


def test_watchdog_keeps_warning_while_no_frame_arrives(node):
    """`test_persistent_mismatch_keeps_warning` 과 같은 판정이다 —
    "경고 1줄 뒤 침묵" 은 그 자체가 이 저장소가 금하는 형태다. 억제는
    rclpy Throttle 이 하고, 노드는 호출을 멈추지 않는다.
    """
    node.node.on_info(_info())
    warns = _catch_warns(node)
    node.node._input_wait_since = time.monotonic() - 10.0
    for _ in range(3):
        node.node._watchdog()
    assert node.errors == []
    assert len(warns) == 3
    assert all("camera_info=수신" in w for w in warns)  # 이번엔 토픽·동기화 쪽


@pytest.mark.parametrize("setup", ["contract", "empty_prompt"])
def test_watchdog_defers_to_the_warning_that_names_the_cause(node, setup):
    """프레임이 *오는데* 버려지는 중이면 on_frames 가 이미 이유까지 말한다.

    거기에 "bag 이 재생 중인지 확인하세요" 를 겹쳐 쏘면 재생은 멀쩡한데
    사람이 틀린 곳을 보게 된다 — ③ 에서 CPU 폴백을 900초 상계의 원인으로
    적으면 안 되는 것과 같은 종류의 오진이다.
    """
    node.node.on_info(_info())
    if setup == "contract":
        node.frame(10.0, width=32)          # 계약 위반 — on_frames 가 경고한다
    else:
        node.node.on_prompt(String(data=""))
        node.frame(10.0)                    # 빈 프롬프트 — on_frames 가 경고한다
    warns = _catch_warns(node)
    node.node._input_wait_since = time.monotonic() - 10.0
    node.node._watchdog()
    assert node.errors == []
    assert warns == []                      # 워치독은 한 줄도 겹치지 않는다


def test_watchdog_still_warns_while_camera_info_is_missing(node):
    """반대쪽 경계: camera_info 대기는 아무도 말하지 않는다.

    on_frames 는 K 가 None 이면 조용히 return 한다(설계된 동작,
    `test_waiting_for_camera_info_stays_silent`). 위 게이트를 넓게 잡아
    이 경우까지 막으면 냉시동 침묵이 그대로 돌아온다.
    """
    warns = _catch_warns(node)
    node.frame(10.0)                        # K is None — 조용히 버려진다
    assert warns == []
    node.node._input_wait_since = time.monotonic() - 10.0
    node.node._watchdog()
    assert node.errors == []
    assert len(warns) == 1
    assert "camera_info=없음" in warns[0]


def test_watchdog_stays_silent_during_normal_startup(node):
    """기동 직후 대기는 정상이다. 이 단언이 없으면 문턱을 0 으로 낮춰도
    위 두 테스트가 통과해, 정상 기동이 초당 한 줄씩 시끄러워진다.
    """
    warns = _catch_warns(node)
    node.node._watchdog()
    assert node.errors == []
    assert warns == []


def test_watchdog_stops_warning_once_frames_arrive(node):
    """첫 프레임이 대기를 끝낸다 — 정상 동작 중에 이 경고가 남으면 안 된다."""
    node.node.on_info(_info())
    node.frame(10.0)
    warns = _catch_warns(node)
    node.node._watchdog()
    assert node.errors == []
    assert warns == []


def test_watchdog_warns_again_after_a_reset_kills_the_stream(node):
    """리셋은 대기의 시작이다 — 여기가 이 커밋 이전에 조용히 뚫려 있었다.

    프레임이 오다가 프롬프트가 바뀌고 그 뒤 입력이 죽으면, _reset_input_state
    가 _last_frame_time 을 None 으로 되돌리므로 stale 분기가 죽는다. 새 분기의
    기준시각까지 같이 죽으면 두 경로가 동시에 막혀 콘솔이 완전히 침묵한다.

    ⚠ 상태를 손으로 꽂지 않는다. on_prompt 를 실제로 태워야 쓰기 지점 누락을
    잡을 수 있다 — 꽂아서 재면 그 누락이 가려진 채 초록으로 통과한다.
    """
    node.node.on_info(_info())
    node.frame(10.0)                                  # 정상 수신 -> 대기 종료
    node.node._input_wait_since -= 3600.0             # 기동한 지 한참 됐다고 치자
    node.node.on_prompt(String(data="물통"))           # 리셋 = 대기 재시작
    node.node._input_wait_since -= 10.0               # 그로부터 10초 경과
    warns = _catch_warns(node)
    node.node._watchdog()
    assert node.errors == []
    assert len(warns) == 1
    assert "오지 않습니다" in warns[0]
    # 경과는 리셋 기준이어야 한다. _reset_input_state 가 시계를 다시 감지
    # 않으면 노드 기동 시각부터 세어 "수백 초째" 같은 틀린 수를 보고한다.
    assert re.search(r"(\d+)초째", warns[0])
    assert int(re.search(r"(\d+)초째", warns[0]).group(1)) < 60


def test_status_reports_the_active_prompts(node):
    """콘솔 경고는 로컬이다. 원격 운영자가 보는 표면은 이 한 줄뿐이다."""
    node.node.on_info(_info())
    assert _values(node.latest_status())["prompts"] == "물통"

    node.node.on_prompt(String(data="thermos, cup"))
    assert _values(node.latest_status())["prompts"] == "thermos,cup"

    node.node.on_prompt(String(data=" , "))
    assert _values(node.latest_status())["prompts"] == ""

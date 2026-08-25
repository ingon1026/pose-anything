"""ROS 2 node: text-prompted detection + 3D OBB pose on RGB-D stream.

Subscribes color + aligned depth (+ camera_info), publishes Detection3DArray,
RViz markers and a debug overlay image. Runtime prompt switch via
/perception/prompt (std_msgs/String, comma-separated names).
"""
import csv
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray

import cv2

from .overlay import PALETTE, draw_objects, draw_status, show_window
from .input_health import classify_input_health
from .pipeline import (CSV_HEADER, PerceptionPipeline, csv_row, img_to_np,
                       parse_plane, status_text)
from .pose_covariance import published_pose_covariance
from .sam3_detector import PROMPT_ALIASES, Sam3Detector, parse_prompts

# ponytail: manual Image<->numpy instead of cv_bridge (its binary is built
# against numpy 1.x and may crash under the numpy 2.x in this env)


def np_to_imgmsg(bgr: np.ndarray) -> Image:
    msg = Image()
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr).tobytes()
    return msg


def camera_info_contract(msg: CameraInfo):
    """Return the calibration used by the pipeline and its input contract.

    The projection matrix is only valid for the exact image geometry and
    optical frame that produced it.  Keep those values together so callers
    cannot accidentally update ``K`` while retaining an old frame contract.
    """
    width, height = int(msg.width), int(msg.height)
    frame_id = msg.header.frame_id
    k = np.asarray(msg.k, dtype=float)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    if not frame_id:
        raise ValueError("camera_info frame_id is empty")
    if k.size != 9:
        raise ValueError(f"camera_info K has {k.size} values (expected 9)")
    K = k.reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("camera_info K must be finite with positive fx/fy")
    # A tuple makes equality exact and intentionally catches even a small
    # recalibration: mixing observations from before and after it is unsafe.
    signature = (width, height, frame_id, tuple(K.ravel()))
    return K, signature


def frame_contract_error(color_msg: Image, depth_msg: Image, signature):
    """Return why an RGB-D pair cannot be projected with ``signature``.

    Depth is subscribed from an *aligned-to-color* topic, therefore both
    images must have the CameraInfo geometry and optical frame exactly.
    """
    width, height, frame_id, _ = signature
    if (color_msg.width, color_msg.height) != (width, height):
        return ("color dimensions %dx%d do not match camera_info %dx%d" %
                (color_msg.width, color_msg.height, width, height))
    if (depth_msg.width, depth_msg.height) != (width, height):
        return ("aligned depth dimensions %dx%d do not match camera_info %dx%d" %
                (depth_msg.width, depth_msg.height, width, height))
    if color_msg.header.frame_id != frame_id:
        return ("color frame_id %r does not match camera_info %r" %
                (color_msg.header.frame_id, frame_id))
    if depth_msg.header.frame_id != frame_id:
        return ("aligned depth frame_id %r does not match camera_info %r" %
                (depth_msg.header.frame_id, frame_id))
    return None


def set_diagnostic_level(status, level):
    """Set ``DiagnosticStatus.level`` across ROS Python generator variants.

    Current upstream generators use an int.  This WSL overlay's older
    generated binding uses a one-byte ``bytes`` field and aborts in C on an
    int during publish.  Inspecting the freshly-created field keeps both
    ABIs working without weakening the DiagnosticStatus contract.
    """
    status.level = bytes((level,)) if isinstance(status.level, bytes) else level


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("roboworld_perception")
        self.declare_parameter("prompts", "물통")
        self.declare_parameter("score_threshold", 0.4)
        self.declare_parameter("max_per_prompt", 1)
        self.declare_parameter("detect_interval", 5)
        self.declare_parameter("display", False)  # 전체 크기 디버그 창 표시
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic",
                               "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("csv_path", "")
        # 입력이 이만큼 끊기면 잔상 마커를 지운다(_watchdog).
        # 2.0 초는 bag 재생 기준이라 실시간 Isaac Sim 에는 너무 짧다 —
        # WSL2 로 640x360 컬러(675 KB)를 보내면 3~4 초 공백이 예사라
        # 정상 동작 중에도 매번 DELETEALL 이 나가 마커가 통째로
        # 사라졌다 나타난다. 이것이 RViz 깜빡임의 주원인이다.
        self.declare_parameter("stale_timeout", 5.0)
        # color/depth 짝짓기 창. 브리지가 정상이면 두 스트림이 같은 렌더 틱에서
        # 나와 스탬프가 같으므로 0.05 로 100% 붙는다(2026-08-19 실측 3509/3509).
        # 안 붙으면 slop 을 올리기 전에 브리지부터 볼 것 — 양쪽에 LARGE_DATA
        # 전송 설정이 있는가. docs/bridge_contract.md 참고.
        self.declare_parameter("sync_slop", 0.05)
        # 동기화기 자체의 보관량도 작게 잡아야 input_qos_depth=1 의
        # "최신 프레임 우선" 계약이 지켜진다. 느린 추론 중 누적된 짝을
        # 순서대로 꺼내면 실제 벨트보다 과거 pose를 내기 때문이다.
        self.declare_parameter("sync_queue_size", 1)
        # 추론은 입력보다 느릴 수 있다. ROS 수신 큐가 여러 장을 보관하면
        # 단일 executor의 callback이 오래된 RGB-D 쌍을 순서대로 추론해
        # 실제 벨트 위치보다 한참 늦은 pose를 낸다. 각 스트림에는 최신 한
        # 장만 남긴다. depth/color는 동시 발행이므로 ATS의 짝짓기 창은
        # 별도로 유지한다.
        self.declare_parameter("input_qos_depth", 1)
        # 발행 점수 하한. 0.0 = 끔. 근거와 주의는 Track.publishable 의 주석.
        # 중복 병합 — 벨트 씬에서 보정된 상수라 기본 꺼짐 (tracker.KAPPA_PHYS 주석)
        self.declare_parameter("enable_merge", True)
        self.declare_parameter("publish_score_min", 0.0)
        # SAM3 입력 해상도. 0 = 기본 1008px.
        # Isaac Sim 과 GPU 를 나눠 쓰면 VRAM 이 부족해지고, 그러면 렌더프로덕트
        # 텍스처가 재할당되면서 ROS2 발행 노드가 들고 있던 CUDA 핸들이 무효가
        # 된다(OgnROS2PublishImage.cpp:466, cudaErrorInvalidResourceHandle).
        # 그 뒤로는 이미지 토픽이 통째로 죽는다 — 재시작 말고는 복구가 안 된다.
        # 카메라가 640x360 이므로 1008 은 업스케일이라 낭비다.
        self.declare_parameter("image_size", 0)
        # 벨트 평면 구속 OBB. 고정 카메라면 첫 프레임 추정으로 충분하지만,
        # 벨트가 화면의 20% 미만이거나 별도 캘리브 값이 있으면 직접 준다.
        self.declare_parameter("enable_footprint_gate", True)
        self.declare_parameter("use_belt_plane", True)
        self.declare_parameter("belt_plane", "")  # "a,b,c,d" (n·p+d=0), 빈 값=추정
        # 이 값은 캘리브레이션된 world TF가 아니라 RViz 전용의 공칭 카메라
        # 위치다. 로봇 base/world 좌표로 오인되면 위험하므로 기본 발행하지
        # 않는다. 실제 셀에서는 hand-eye/URDF TF를 외부에서 발행해야 한다.
        self.declare_parameter("publish_world_tf", False)
        # camera_link -> camera_color_optical_frame. RealSense 드라이버가 돌면
        # 드라이버가 채워주므로 건드릴 필요가 없지만, Isaac Sim 이 이미지를
        # 직접 발행하는 구성에는 드라이버가 없어 이 링크를 아무도 안 보낸다.
        # 그러면 world -> camera_link 가 있어도 검출이 실려오는 optical frame
        # 까지 사슬이 닿지 않아 RViz 가 마커를 놓지 못한다.
        # 기본값 False 인 이유: bag 재생처럼 RealSense 내부 트리가 이미 녹화된
        # 경우 같은 자식 프레임에 부모가 둘이 되어 멀쩡했던 TF 가 깨진다.
        # 켜서 생기는 고장은 증상이 엉뚱한 곳에 나타나 원인 찾기가 어렵고,
        # 꺼서 생기는 부족은 publish_optical_tf:=true 한 줄로 끝난다.
        self.declare_parameter("publish_optical_tf", False)
        # 프레임 이름. 정적 TF 는 __init__ 에서 한 번 발행하고 끝이라
        # camera_info 가 실어오는 실제 optical frame 이름을 기다릴 수 없다.
        # 하드코딩해 두면 다른 이름이 오는 순간 사슬이 조용히 끊기고 RViz 에
        # 마커만 안 보인다 — 그래서 밖에서 맞출 수 있게 뺀다.
        # optical_frame 은 camera_info 가 오기 전까지 self.frame_id 의
        # 기본값으로도 쓰이므로 두 곳이 항상 같은 값을 본다.
        # world_frame 을 바꾸면 rviz/perception.rviz 의 Fixed Frame 도 함께
        # 고쳐야 한다(현재 "world" 고정). 안 그러면 노드는 새 이름으로
        # 발행하는데 RViz 는 옛 이름을 기다려 "마커만 안 보이는" 증상이
        # 난다 — 이 파라미터가 없애려던 바로 그 증상이라 원인을 엉뚱한
        # 데서 찾게 된다.
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_link_frame", "camera_link")
        self.declare_parameter("optical_frame", "camera_color_optical_frame")
        world_frame = self.get_parameter("world_frame").value
        link_frame = self.get_parameter("camera_link_frame").value
        optical_frame = self.get_parameter("optical_frame").value
        static_tfs = []
        if self.get_parameter("publish_world_tf").value:
            self.get_logger().warn(
                "publish_world_tf=true: publishing an uncalibrated nominal "
                "RViz-only transform; do not use it for robot coordinates")
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = world_frame
            t.child_frame_id = link_frame
            t.transform.translation.z = 1.0  # 카메라 높이 (시각화용 공칭값)
            # Rz(90°)·Ry(90°): 수직 하방 시선 + 영상 우/하 = world +X/−Y —
            # RViz TopDownOrtho(위에서 보기)가 카메라 영상과 같은 방향이 된다
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = \
                (-0.5, 0.5, 0.5, 0.5)
            static_tfs.append(t)
        if self.get_parameter("publish_optical_tf").value:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = link_frame
            # camera_info 가 오면 self.frame_id 가 갱신되지만 정적 TF 는 여기서
            # 한 번 발행하고 끝이라 그때까지 기다릴 수 없다. 다른 이름이 오면
            # optical_frame:=<그 이름> 으로 맞춘다.
            t.child_frame_id = optical_frame
            # REP-103: camera_link(x앞 y왼 z위) -> optical(x오른 y아래 z앞).
            # 이 값으로 RViz TF 사슬이 이어지는 것을 실제로 확인했다.
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = \
                (-0.5, 0.5, -0.5, 0.5)
            static_tfs.append(t)
        if static_tfs:
            # 반드시 리스트로 한 번에 보낸다. StaticTransformBroadcaster 의
            # 발행자 QoS 는 depth=1 TRANSIENT_LOCAL 이라 늦게 켠 구독자는
            # "마지막 메시지" 하나만 받는다 — sendTransform 을 두 번 나눠
            # 부르면, 누적해서 재발행하지 않는 구현에서는 먼저 보낸 변환이
            # 통째로 사라진다.
            # latched 발행자는 살아 있어야 늦게 켠 RViz도 받는다 — 노드에 보관
            self._tf_bcast = StaticTransformBroadcaster(self)
            self._tf_bcast.sendTransform(static_tfs)

        self.prompts = parse_prompts(self.get_parameter("prompts").value)
        threshold = self.get_parameter("score_threshold").value
        self.get_logger().info(f"loading SAM3... prompts={self.prompts}")
        self.pipeline = PerceptionPipeline(
            Sam3Detector(threshold=threshold,
                         image_size=self.get_parameter("image_size").value),
            detect_interval=self.get_parameter("detect_interval").value,
            max_per_prompt=self.get_parameter("max_per_prompt").value,
            pub_score_min=self.get_parameter("publish_score_min").value,
            enable_merge=self.get_parameter("enable_merge").value,
            use_belt_plane=self.get_parameter("use_belt_plane").value,
            enable_footprint_gate=self.get_parameter(
                "enable_footprint_gate").value,
            belt_plane=parse_plane(self.get_parameter("belt_plane").value))
        # run.sh가 이 문자열을 grep으로 대기한다 — 문구 변경 시 run.sh도 수정
        self.get_logger().info("SAM3 ready")

        self.K = None
        self.frame_id = optical_frame
        self._camera_info_signature = None
        self._last_input_contract_error = None
        self._busy = False
        self._display = self.get_parameter("display").value
        self._fps_ema = 0.0
        self._last_frame_time = None
        self._last_process_ms = None
        self._last_status_time = None
        self._stale_cleared = True
        self._stale_timeout = float(self.get_parameter("stale_timeout").value)
        # 직전 사이클에 실제로 발행한 (ns, id). 사라진 것만 골라 지우기 위해
        # 들고 있는다 — 매번 DELETEALL 을 쏘면 RViz 가 지움과 다시 그림
        # 사이를 렌더링해 깜빡인다.
        self._prev_marker_ids = set()
        self.create_timer(1.0, self._watchdog)

        self.pub_det = self.create_publisher(Detection3DArray,
                                             "/perception/detections", 10)
        self.pub_markers = self.create_publisher(MarkerArray,
                                                 "/perception/markers", 10)
        self.pub_debug = self.create_publisher(Image, "/perception/debug_image", 10)
        self.pub_status = self.create_publisher(DiagnosticArray,
                                                "/perception/status", 10)

        self.create_subscription(CameraInfo,
                                 self.get_parameter("info_topic").value,
                                 self.on_info, 10)
        self.create_subscription(String, "/perception/prompt", self.on_prompt, 10)
        requested_qos_depth = self.get_parameter("input_qos_depth").value
        input_qos_depth = max(1, int(requested_qos_depth))
        if input_qos_depth != requested_qos_depth:
            self.get_logger().warn(
                "input_qos_depth must be >= 1; using latest-only depth=1")
        input_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=input_qos_depth,
            # 센서 데이터는 유실보다 최신성이 우선이다. RELIABLE publisher와도
            # 호환되며, BEST_EFFORT 카메라 publisher에도 붙을 수 있다.
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        requested_sync_queue_size = self.get_parameter("sync_queue_size").value
        sync_queue_size = max(1, int(requested_sync_queue_size))
        if sync_queue_size != requested_sync_queue_size:
            self.get_logger().warn(
                "sync_queue_size must be >= 1; using latest-only depth=1")
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, self.get_parameter("color_topic").value,
                        qos_profile=input_qos),
             Subscriber(self, Image, self.get_parameter("depth_topic").value,
                        qos_profile=input_qos)],
            queue_size=sync_queue_size,
            slop=self.get_parameter("sync_slop").value)
        sync.registerCallback(self.on_frames)

        self.csv_writer = None
        csv_path = self.get_parameter("csv_path").value
        if csv_path:
            # buffering=1: 노드가 SIGTERM으로 죽어도 기록이 남도록 라인 버퍼링
            self._csv_file = open(csv_path, "w", newline="", buffering=1)
            self.csv_writer = csv.writer(self._csv_file)
            self.csv_writer.writerow(CSV_HEADER)

    def on_info(self, msg: CameraInfo):
        try:
            K, signature = camera_info_contract(msg)
        except ValueError as e:
            # Without valid calibration, accepting the previous K for a new
            # stream would silently project pixels into the wrong 3D rays.
            if self.K is not None:
                self._reset_input_state("invalid camera_info")
            self.K = None
            self._camera_info_signature = None
            self._last_input_contract_error = f"invalid camera_info: {e}"
            self.get_logger().error(f"camera_info rejected: {e}")
            return

        was_configured = self._camera_info_signature is not None
        changed = was_configured and signature != self._camera_info_signature
        self.K = K
        self.frame_id = signature[2]
        self._camera_info_signature = signature
        self._last_input_contract_error = None
        if changed:
            self._reset_input_state("camera_info changed")
            self.get_logger().warn(
                "camera_info changed (size, intrinsics, or frame); "
                "tracker reset before accepting new RGB-D frames")
        elif not was_configured:
            self.get_logger().info(
                f"camera_info received, {signature[0]}x{signature[1]}, "
                f"frame={self.frame_id}")

    def _reset_input_state(self, reason):
        """Forget observations and explicitly withdraw output after a reset."""
        self.pipeline.reset()
        self._last_frame_time = None
        self._stale_cleared = True
        self._prev_marker_ids = set()

        # A reset must be visible to consumers immediately; otherwise a robot
        # can keep acting on the last detection while this node waits for a
        # compatible frame.  The next valid frame repopulates both topics.
        empty = Detection3DArray()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = self.frame_id
        self.pub_det.publish(empty)
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        self.pub_markers.publish(markers)
        self.get_logger().info(f"input contract reset: {reason}")

    def on_prompt(self, msg: String):
        self.prompts = parse_prompts(msg.data)
        self.pipeline.reset()
        self.get_logger().info(f"prompts -> {self.prompts}")

    def _watchdog(self):
        """Clear stale markers and publish the one-Hz input health heartbeat."""
        if (not self._stale_cleared and self._last_frame_time is not None and
                time.monotonic() - self._last_frame_time > self._stale_timeout):
            markers = MarkerArray()
            markers.markers.append(Marker(action=Marker.DELETEALL))
            self.pub_markers.publish(markers)
            self._stale_cleared = True
            self._prev_marker_ids = set()
            self.get_logger().info(
                "입력 없음 %.1f초 — 마커 정리" % self._stale_timeout)
        self._publish_status()

    def _publish_status(self):
        """Publish externally consumable input validity and timing state."""
        now = time.monotonic()
        if (self._last_status_time is not None
                and now - self._last_status_time < 1.0):
            return
        self._last_status_time = now
        level, message, frame_age_s = classify_input_health(
            now, self.K is not None, self._last_frame_time,
            self._stale_timeout, self._last_input_contract_error)
        status = DiagnosticStatus()
        set_diagnostic_level(status, level)
        status.name = "roboworld_perception/input"
        status.hardware_id = "rgbd_camera"
        status.message = message
        signature = self._camera_info_signature
        status.values = [
            KeyValue(key="camera_info_valid", value=str(self.K is not None).lower()),
            KeyValue(key="input_contract_valid",
                     value=str(self._last_input_contract_error is None).lower()),
            KeyValue(key="last_frame_age_s",
                     value="never" if frame_age_s is None else f"{frame_age_s:.3f}"),
            KeyValue(key="last_processing_duration_ms",
                     value=("never" if self._last_process_ms is None else
                            f"{self._last_process_ms:.1f}")),
            KeyValue(key="out_of_order_frame_drops",
                     value=str(self.pipeline.late_frame_drops)),
            KeyValue(key="stale_timeout_s", value=f"{self._stale_timeout:.1f}"),
            KeyValue(key="camera_frame", value=self.frame_id),
            KeyValue(key="camera_image_size",
                     value=("unknown" if signature is None else
                            f"{signature[0]}x{signature[1]}")),
            KeyValue(key="input_error", value=self._last_input_contract_error or ""),
        ]
        heartbeat = DiagnosticArray()
        heartbeat.header.stamp = self.get_clock().now().to_msg()
        heartbeat.status.append(status)
        self.pub_status.publish(heartbeat)

    def on_frames(self, color_msg: Image, depth_msg: Image):
        if self.K is None or self._busy or not self.prompts:
            return  # ponytail: drop frames while inference is running
        contract_error = frame_contract_error(
            color_msg, depth_msg, self._camera_info_signature)
        if contract_error:
            # Resolution/frame switches often deliver a few images before the
            # matching CameraInfo.  Drop those frames and reset once, instead
            # of mixing their depth with old intrinsics.
            if contract_error != self._last_input_contract_error:
                self._reset_input_state(contract_error)
                self._last_input_contract_error = contract_error
            # A mismatch that persists stops every publication for good.  One
            # warning followed by silence is the shape 8d45975 named as silent
            # failure - the node is alive with no error and no log.  The
            # /perception/status heartbeat carries the same fact, but nothing
            # in this repo subscribes to it, so the console is the only
            # surface a person actually watches.  rclpy's Throttle filter does
            # the suppressing, so no state and no parameter is added here.
            self.get_logger().warn(
                f"RGB-D frame rejected; waiting for matching input: "
                f"{contract_error}", throttle_duration_sec=1.0)
            return
        self._last_input_contract_error = None
        stamp_s = (color_msg.header.stamp.sec
                   + color_msg.header.stamp.nanosec * 1e-9)
        if self.pipeline.time_reset_required(stamp_s):
            # Withhold the previous run's detections before SAM inference;
            # otherwise a consumer can act on a latched old Detection3DArray
            # for the entire model latency after /clock resets.
            self._reset_input_state("camera timestamp reset")
        # 지연 프레임은 여기서 거르지 않는다 — pipeline.process() 안의
        # LATE_DROP_STREAK_MAX 상계를 타야 한다. 노드가 먼저 return 하면
        # pipeline._late_drops 가 안 늘어 상계가 영원히 발화하지 않고,
        # 그 경로는 _last_stamp 를 전진시키지 않으므로 한 번 빠지면 노드
        # 재시작 외에 회복이 없다 (docs/README.md 의 LATE_DROP_STREAK_MAX 행).
        self._busy = True
        self._last_frame_time = time.monotonic()
        self._stale_cleared = False
        t0 = time.perf_counter()
        try:
            img = img_to_np(color_msg)
            if color_msg.encoding == "rgb8":
                rgb, bgr = img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                rgb, bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img.copy()
            depth = img_to_np(depth_msg)
            objects = self.pipeline.process(rgb, depth, self.K, self.prompts,
                                            stamp_s)
            proc_ms = (time.perf_counter() - t0) * 1000
            self.publish(objects, color_msg.header.stamp, bgr, proc_ms)
        except Exception as e:  # keep the node alive on a bad frame
            self.get_logger().error(f"frame failed: {e}")
        finally:
            self._last_process_ms = (time.perf_counter() - t0) * 1000
            self._busy = False
            # The single-threaded executor can spend all its time in image
            # callbacks, starving the one-second timer. Publish from the
            # completed-work path too; _publish_status keeps the public rate
            # at at most 1 Hz.
            self._publish_status()

    def publish(self, objects, stamp, bgr, proc_ms):
        det_array = Detection3DArray()
        det_array.header.stamp = stamp
        det_array.header.frame_id = self.frame_id
        markers = MarkerArray()
        # DELETEALL 을 앞세우지 않는다. 같은 ns/id 로 다시 보내면 RViz 가
        # 알아서 덮어쓰므로, 지움 없이 갱신하면 깜빡이지 않는다.
        # 사라진 것만 아래에서 골라 DELETE 한다.
        cur_marker_ids = set()

        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        for obj in objects:
            if not obj.publishable:  # 가림 중이거나 obb 없음 — stale pose 발행 금지
                continue
            o = obj.obb
            qx, qy, qz, qw = o.quat_xyzw

            det = Detection3D()
            det.header = det_array.header
            det.id = f"{obj.label}#{obj.track_id}"
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = obj.label
            hyp.hypothesis.score = obj.score
            hyp.pose.pose.position.x, hyp.pose.pose.position.y, \
                hyp.pose.pose.position.z = o.center
            hyp.pose.pose.orientation.x = qx
            hyp.pose.pose.orientation.y = qy
            hyp.pose.pose.orientation.z = qz
            hyp.pose.pose.orientation.w = qw
            # 위치는 융합 필터의 분산(m²), OBB 회전은 미추정이므로 보수적인
            # 유한 분산(rad²)을 넣는다. 0은 "정확히 안다"라는 뜻이라 금지.
            hyp.pose.covariance = published_pose_covariance(obj.filter.pos_var)
            det.results.append(hyp)
            det.bbox.center = hyp.pose.pose
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = o.extent
            det_array.detections.append(det)

            color = PALETTE[obj.track_id % len(PALETTE)]
            cube = Marker()
            cube.header = det_array.header
            cube.ns, cube.id, cube.type = "obb", obj.track_id, Marker.CUBE
            cube.pose = hyp.pose.pose
            cube.scale.x, cube.scale.y, cube.scale.z = o.extent
            cube.color.b, cube.color.g, cube.color.r = [c / 255 for c in color]
            cube.color.a = 0.4
            markers.markers.append(cube)

            axes = Marker()
            axes.header = det_array.header
            axes.ns, axes.id, axes.type = "axes", obj.track_id, Marker.LINE_LIST
            axes.scale.x = 0.012
            axes.color.a = 1.0
            for k, rgb in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                tip = o.center + o.R[:, k] * (o.extent[k] / 2 + 0.03)
                for p in (o.center, tip):
                    axes.points.append(Point(x=p[0], y=p[1], z=p[2]))
                axes.colors.extend([ColorRGBA(r=float(rgb[0]), g=float(rgb[1]),
                                              b=float(rgb[2]), a=1.0)] * 2)
            markers.markers.append(axes)

            text = Marker()
            text.header = det_array.header
            text.ns, text.id, text.type = "label", obj.track_id, \
                Marker.TEXT_VIEW_FACING
            text.pose.position.x, text.pose.position.y, text.pose.position.z = \
                o.center + np.array([0, -o.extent[1] / 2 - 0.03, 0])
            text.scale.z = 0.03
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            # RViz는 한글 글리프를 렌더링하지 못하므로 마커 텍스트는 영어 별칭 사용
            en = PROMPT_ALIASES.get(obj.label, obj.label)
            text.text = f"{en}#{obj.track_id} {o.distance:.2f}m"
            markers.markers.append(text)

            cur_marker_ids.update(
                (("obb", obj.track_id), ("axes", obj.track_id),
                 ("label", obj.track_id)))

            if self.csv_writer:
                self.csv_writer.writerow(csv_row(obj, stamp_s, proc_ms))

        # 직전에 있었는데 이번에 빠진 것만 지운다.
        for ns, mid in self._prev_marker_ids - cur_marker_ids:
            gone = Marker()
            gone.header = det_array.header
            gone.ns, gone.id, gone.action = ns, mid, Marker.DELETE
            markers.markers.append(gone)
        self._prev_marker_ids = cur_marker_ids

        self.pub_det.publish(det_array)
        self.pub_markers.publish(markers)

        inst_fps = 1000.0 / max(proc_ms, 1e-3)
        self._fps_ema = inst_fps if self._fps_ema == 0 else \
            0.9 * self._fps_ema + 0.1 * inst_fps
        draw_objects(bgr, objects, self.K)
        draw_status(bgr, status_text(self._fps_ema, self.pipeline, objects))
        debug = np_to_imgmsg(bgr)
        debug.header = det_array.header
        self.pub_debug.publish(debug)
        if self._display:
            self._display = show_window(bgr)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()

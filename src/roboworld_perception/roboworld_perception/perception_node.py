"""ROS 2 node: text-prompted detection + 3D OBB pose on RGB-D stream.

Subscribes color + aligned depth (+ camera_info), publishes Detection3DArray,
RViz markers and a debug overlay image. Runtime prompt switch via
/perception/prompt (std_msgs/String, comma-separated names).
"""
import csv
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray

import cv2

from .overlay import PALETTE, draw_objects, draw_status, show_window
from .pipeline import (CSV_HEADER, PerceptionPipeline, csv_row, img_to_np,
                       parse_plane, status_text)
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
        self.declare_parameter("sync_queue_size", 5)
        # 발행 점수 하한. 0.0 = 끔. 근거와 주의는 Track.publishable 의 주석.
        # 중복 병합 — 벨트 씬에서 보정된 상수라 기본 꺼짐 (tracker.KAPPA_PHYS 주석)
        self.declare_parameter("enable_merge", False)
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
        # 카메라가 "위(1m)에서 아래를 본다"는 world TF. RealSense TF 트리의
        # 뿌리(camera_link) 위에 붙인다 — optical frame에 직접 붙이면 bag이
        # 함께 녹화한 내부 트리와 부모가 둘이 되어 TF가 깨진다. 실제 로봇
        # TF 트리와 통합할 때는 이 파라미터를 꺼서 충돌을 피한다.
        self.declare_parameter("publish_world_tf", True)
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
        self._busy = False
        self._display = self.get_parameter("display").value
        self._fps_ema = 0.0
        self._last_frame_time = None
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

        self.create_subscription(CameraInfo,
                                 self.get_parameter("info_topic").value,
                                 self.on_info, 10)
        self.create_subscription(String, "/perception/prompt", self.on_prompt, 10)
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, self.get_parameter("color_topic").value),
             Subscriber(self, Image, self.get_parameter("depth_topic").value)],
            queue_size=self.get_parameter("sync_queue_size").value,
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
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.frame_id = msg.header.frame_id
            self.get_logger().info(f"camera_info received, frame={self.frame_id}")

    def on_prompt(self, msg: String):
        self.prompts = parse_prompts(msg.data)
        self.pipeline.reset()
        self.get_logger().info(f"prompts -> {self.prompts}")

    def _watchdog(self):
        """입력이 끊기면(bag 종료 등) 잔상 마커를 정리한다."""
        if self._stale_cleared or self._last_frame_time is None:
            return
        if time.monotonic() - self._last_frame_time > self._stale_timeout:
            markers = MarkerArray()
            markers.markers.append(Marker(action=Marker.DELETEALL))
            self.pub_markers.publish(markers)
            self._stale_cleared = True
            self._prev_marker_ids = set()
            self.get_logger().info(
                "입력 없음 %.1f초 — 마커 정리" % self._stale_timeout)

    def on_frames(self, color_msg: Image, depth_msg: Image):
        if self.K is None or self._busy or not self.prompts:
            return  # ponytail: drop frames while inference is running
        self._busy = True
        self._last_frame_time = time.monotonic()
        self._stale_cleared = False
        try:
            t0 = time.perf_counter()
            img = img_to_np(color_msg)
            if color_msg.encoding == "rgb8":
                rgb, bgr = img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                rgb, bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img.copy()
            depth = img_to_np(depth_msg)
            stamp_s = color_msg.header.stamp.sec + \
                color_msg.header.stamp.nanosec * 1e-9
            objects = self.pipeline.process(rgb, depth, self.K, self.prompts,
                                            stamp_s)
            proc_ms = (time.perf_counter() - t0) * 1000
            self.publish(objects, color_msg.header.stamp, bgr, proc_ms)
        except Exception as e:  # keep the node alive on a bad frame
            self.get_logger().error(f"frame failed: {e}")
        finally:
            self._busy = False

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
            # 융합 필터의 위치 불확실성을 표준 covariance 필드로 전달 —
            # 로봇 측 파지 게이팅(방안2)이 이 값을 소비한다. 회전은 필터
            # 밖(미추정)이라 0 유지.
            var = obj.filter.pos_var.tolist()
            hyp.pose.covariance[0] = var[0]
            hyp.pose.covariance[7] = var[1]
            hyp.pose.covariance[14] = var[2]
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

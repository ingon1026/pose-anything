"""ROS 2 node: text-prompted detection + 3D OBB pose on RGB-D stream.

Subscribes color + aligned depth (+ camera_info), publishes Detection3DArray,
RViz markers and a debug overlay image. Runtime prompt switch via
/perception/prompt (std_msgs/String, comma-separated names).
"""
import csv
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray

import cv2

from .overlay import PALETTE, draw_objects, draw_status
from .pipeline import PerceptionPipeline
from .sam3_detector import PROMPT_ALIASES, Sam3Detector

# ponytail: manual Image<->numpy instead of cv_bridge (its binary is built
# against numpy 1.x and may crash under the numpy 2.x in this env)


def img_to_np(msg: Image) -> np.ndarray:
    if msg.encoding in ("rgb8", "bgr8"):
        ch, dtype = 3, np.uint8
    elif msg.encoding == "16UC1":
        ch, dtype = 1, np.uint16
    else:
        raise ValueError(f"unsupported encoding {msg.encoding}")
    itemsize = np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype).reshape(msg.height,
                                                 msg.step // itemsize)
    arr = arr[:, :msg.width * ch]
    return arr.reshape(msg.height, msg.width, ch).squeeze()


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

        self.prompts = [p.strip() for p in
                        self.get_parameter("prompts").value.split(",") if p.strip()]
        threshold = self.get_parameter("score_threshold").value
        self.get_logger().info(f"loading SAM3... prompts={self.prompts}")
        self.pipeline = PerceptionPipeline(
            Sam3Detector(threshold=threshold),
            detect_interval=self.get_parameter("detect_interval").value,
            max_per_prompt=self.get_parameter("max_per_prompt").value)
        self.get_logger().info("SAM3 ready")

        self.K = None
        self.frame_id = "camera_color_optical_frame"
        self._busy = False
        self._display = self.get_parameter("display").value
        self._fps_ema = 0.0
        self._last_frame_time = None
        self._stale_cleared = True
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
            queue_size=5, slop=0.05)
        sync.registerCallback(self.on_frames)

        self.csv_writer = None
        csv_path = self.get_parameter("csv_path").value
        if csv_path:
            # buffering=1: 노드가 SIGTERM으로 죽어도 기록이 남도록 라인 버퍼링
            self._csv_file = open(csv_path, "w", newline="", buffering=1)
            self.csv_writer = csv.writer(self._csv_file)
            self.csv_writer.writerow(
                ["stamp", "track_id", "label", "score", "x", "y", "z", "distance",
                 "w", "d", "h", "roll", "pitch", "yaw", "flips", "proc_ms"])

    def on_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.frame_id = msg.header.frame_id
            self.get_logger().info(f"camera_info received, frame={self.frame_id}")

    def on_prompt(self, msg: String):
        self.prompts = [p.strip() for p in msg.data.split(",") if p.strip()]
        self.pipeline.reset()
        self.get_logger().info(f"prompts -> {self.prompts}")

    def _watchdog(self):
        """입력이 끊기면(bag 종료 등) 잔상 마커를 정리한다."""
        if self._stale_cleared or self._last_frame_time is None:
            return
        if time.monotonic() - self._last_frame_time > 2.0:
            markers = MarkerArray()
            markers.markers.append(Marker(action=Marker.DELETEALL))
            self.pub_markers.publish(markers)
            self._stale_cleared = True
            self.get_logger().info("입력 없음 2초 — 마커 정리")

    def on_frames(self, color_msg: Image, depth_msg: Image):
        if self.K is None or self._busy or not self.prompts:
            return  # ponytail: drop frames while inference is running
        self._busy = True
        self._last_frame_time = time.monotonic()
        self._stale_cleared = False
        try:
            t0 = time.perf_counter()
            img = img_to_np(color_msg)
            rgb = img if color_msg.encoding == "rgb8" else img[:, :, ::-1]
            depth = img_to_np(depth_msg)
            objects = self.pipeline.process(np.ascontiguousarray(rgb), depth,
                                            self.K, self.prompts)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
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
        markers.markers.append(Marker(action=Marker.DELETEALL))

        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        for obj in objects:
            if obj.obb is None:
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

            if self.csv_writer:
                r, p, y = o.rpy
                self.csv_writer.writerow(
                    [f"{stamp_s:.6f}", obj.track_id, obj.label,
                     f"{obj.score:.3f}", *[f"{v:.4f}" for v in o.center],
                     f"{o.distance:.4f}", *[f"{v:.4f}" for v in o.extent],
                     f"{r:.2f}", f"{p:.2f}", f"{y:.2f}", obj.flip_count,
                     f"{proc_ms:.1f}"])

        self.pub_det.publish(det_array)
        self.pub_markers.publish(markers)

        inst_fps = 1000.0 / max(proc_ms, 1e-3)
        self._fps_ema = inst_fps if self._fps_ema == 0 else \
            0.9 * self._fps_ema + 0.1 * inst_fps
        draw_objects(bgr, objects, self.K)
        mode = "KEY" if self.pipeline.last_was_keyframe else "track"
        draw_status(bgr, f"{self._fps_ema:4.1f} FPS | {mode} | objects={len(objects)}")
        debug = np_to_imgmsg(bgr)
        debug.header = det_array.header
        self.pub_debug.publish(debug)
        if self._display:
            try:
                cv2.imshow("roboworld perception", bgr)
                cv2.waitKey(1)
            except cv2.error:
                self._display = False  # 디스플레이 없는 환경


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

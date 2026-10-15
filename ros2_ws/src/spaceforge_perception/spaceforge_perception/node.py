"""ROS 2 perception node; ROS imports remain optional for offline tests.

Independence boundary: this module never imports policy, mission-safety, or
planner code. It converts images to poses and publishes them.
"""
import json
import numpy as np
from .contracts import CameraCalibration, DetectionStatus, TargetKeypoints
from .detectors import DeterministicMockDetector, YOLOv8Detector
from .pipeline import PerceptionPipeline
from .pnp import OpenCVPnPSolver

#: ROS topic names under the node namespace.
TOPIC_IMAGE = "image"
TOPIC_CAMERA_INFO = "camera_info"
TOPIC_TARGET_POSE = "target_pose"
TOPIC_DETECTIONS = "detections"
TOPIC_DEBRIS = "debris"
TOPIC_STATUS = "status"


def detection_bbox_pixels(detection):
    """Bounding box (cx, cy, size_x, size_y) enclosing detection pixels."""
    pixels = np.asarray(detection.pixel_points, dtype=np.float64)
    lo = pixels.min(axis=0)
    hi = pixels.max(axis=0)
    center = (lo + hi) / 2.0
    size = np.maximum(hi - lo, 1.0)
    return float(center[0]), float(center[1]), float(size[0]), float(size[1])


def main():
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    except ImportError as exc:
        raise RuntimeError("ROS 2 sensor_msgs, geometry_msgs, std_msgs, and vision_msgs are required") from exc

    class PerceptionNode(Node):
        def __init__(self):
            super().__init__("perception")
            self.declare_parameter("camera_frame", "camera_optical_frame")
            self.declare_parameter("target_frame", "target_cad")
            self.declare_parameter("output_frame", "")
            self.declare_parameter("static_translation", [0.0, 0.0, 0.0])
            self.declare_parameter("static_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
            self.declare_parameter("max_image_age_ms", 200.0)
            self.declare_parameter("detector", "mock")
            self.declare_parameter("yolo_model", "")
            self.declare_parameter("keypoints_json", "")
            self.calibration = None
            self.keypoints = self._load_keypoints()
            detector_name = self.get_parameter("detector").value
            if detector_name == "yolov8":
                detector = YOLOv8Detector(self.get_parameter("yolo_model").value, len(self.keypoints.points_m))
            else:
                detector = DeterministicMockDetector()
            self.detector = detector
            self.pipeline = None
            self.pose_pub = self.create_publisher(PoseStamped, TOPIC_TARGET_POSE, 10)
            self.detection_pub = self.create_publisher(Detection2DArray, TOPIC_DETECTIONS, 10)
            self.debris_pub = self.create_publisher(Detection2DArray, TOPIC_DEBRIS, 10)
            self.status_pub = self.create_publisher(String, TOPIC_STATUS, 10)
            self.create_subscription(CameraInfo, TOPIC_CAMERA_INFO, self.on_camera_info, 10)
            self.create_subscription(Image, TOPIC_IMAGE, self.on_image, 10)

        def _load_keypoints(self):
            raw = self.get_parameter("keypoints_json").value
            if raw:
                data = json.loads(raw)
                return TargetKeypoints(tuple(data["names"]), tuple(tuple(p) for p in data["points_m"]))
            return TargetKeypoints(("p0", "p1", "p2", "p3"), ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)))

        def _static_transform(self):
            translation = tuple(float(v) for v in self.get_parameter("static_translation").value)
            quaternion = tuple(float(v) for v in self.get_parameter("static_quaternion_xyzw").value)
            if len(translation) != 3 or len(quaternion) != 4:
                raise ValueError("static_translation needs 3 values and static_quaternion_xyzw needs 4")
            return translation, quaternion

        def _build_pipeline(self):
            output_frame = self.get_parameter("output_frame").value or None
            translation, quaternion = self._static_transform()
            return PerceptionPipeline(
                self.keypoints, self.calibration, self.detector, OpenCVPnPSolver(),
                max_image_age_ns=int(self.get_parameter("max_image_age_ms").value * 1e6),
                output_frame=output_frame,
                static_translation_m=translation, static_quaternion_xyzw=quaternion,
            )

        def on_camera_info(self, msg):
            self.calibration = CameraCalibration(msg.width, msg.height, tuple(msg.k), tuple(msg.d), msg.header.frame_id or self.get_parameter("camera_frame").value)
            self.pipeline = self._build_pipeline()

        def on_image(self, msg):
            if self.pipeline is None:
                self.publish_status(DetectionStatus.BAD_CALIBRATION.value, 0, 0.0, False)
                return
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            channels = 1 if msg.encoding in ("mono8", "8UC1") else 3
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * channels]
            image = image.reshape(msg.height, msg.width, channels) if channels > 1 else image.reshape(msg.height, msg.width)
            pose = self.pipeline.process(image, stamp_ns, msg.header.frame_id, self.get_clock().now().nanoseconds)
            self.publish_status(pose.status.value, pose.latency_ns, pose.confidence, pose.clock_warning)
            if pose.status == DetectionStatus.VALID:
                out = PoseStamped()
                out.header.stamp = msg.header.stamp
                out.header.frame_id = pose.frame_id
                out.pose.position.x, out.pose.position.y, out.pose.position.z = pose.translation_m
                out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = pose.quaternion_xyzw
                self.pose_pub.publish(out)
                self.detection_pub.publish(self._detections_msg(msg, pose, "target"))
            else:
                failures = Detection2DArray()
                failures.header.stamp = msg.header.stamp
                failures.header.frame_id = msg.header.frame_id
                self.debris_pub.publish(failures)

        def _detections_msg(self, img_msg, pose, class_id):
            """Detection2DArray with one bbox hypothesis scored by pose confidence."""
            msg = Detection2DArray()
            msg.header.stamp = img_msg.header.stamp
            msg.header.frame_id = pose.frame_id
            detection = Detection2D()
            detection.header = msg.header
            pixels = getattr(self.pipeline.last_detection, "pixel_points", None)
            if pixels:
                cx, cy, sx, sy = detection_bbox_pixels(self.pipeline.last_detection)
            else:
                cx, cy, sx, sy = float(img_msg.width) / 2.0, float(img_msg.height) / 2.0, float(img_msg.width), float(img_msg.height)
            detection.bbox.center.position.x = cx
            detection.bbox.center.position.y = cy
            detection.bbox.size_x = sx
            detection.bbox.size_y = sy
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_id
            hypothesis.hypothesis.score = float(pose.confidence)
            detection.results.append(hypothesis)
            msg.detections.append(detection)
            return msg

        def publish_status(self, status, latency_ns, confidence, clock_warning=False):
            msg = String()
            msg.data = json.dumps({
                "status": status, "latency_ns": int(latency_ns),
                "confidence": float(confidence), "clock_warning": bool(clock_warning),
            })
            self.status_pub.publish(msg)

    rclpy.init()
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

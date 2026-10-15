# SpaceForge perception contracts

## Coordinate frames

The camera optical frame is right-handed ROS optical convention: x right, y down, z forward. CAD target coordinates are expressed in target_cad metres, with a fixed ordered keypoint list. The detector returns image pixels in the camera optical image plane. The pose output is target_cad expressed in camera_optical_frame unless a configured static transform is applied.

## Calibration contract

CameraCalibration contains width_px, height_px, K row-major 3x3 in pixels, distortion coefficients, and camera_frame. K must be finite, positive focal lengths, positive principal point within a 2x image margin, and match the image dimensions. Distortion is optional but defaults to zeros.

## Keypoint and detector contract

TargetKeypoints contains an ordered Nx3 CAD point array in metres and stable names. Detection contains the matching ordered Nx2 pixel array, per-point confidence, detector confidence, image stamp_ns, and image frame. A valid detection requires equal cardinality, finite values, confidence in [0,1], and at least four non-coplanar-capable points for PnP.

## Timestamp and latency contract

The image timestamp is the source of truth and is copied to PoseStamped and detection messages. Processing latency is measured as receive_wall_time_ns minus image_stamp_ns when clocks are comparable; negative latency is reported as zero with clock_warning set on the PoseEstimate and status message. Images older than max_image_age_ns are rejected as STALE_IMAGE. A detection answered in a different frame than requested is rejected as FRAME_ERROR.

## Pose and failure contract

PoseEstimate carries translation metres, quaternion xyzw, confidence, latency_ns, source image stamp, source camera frame, and status. Status is one of VALID, NO_DETECTION, AMBIGUOUS_PNP, BAD_CALIBRATION, STALE_IMAGE, INVALID_DETECTION, SOLVER_UNAVAILABLE, FRAME_ERROR. On VALID the detections message carries one Detection2D whose bbox encloses the keypoints and whose hypothesis score is the pose confidence; on failure an empty Detection2DArray is published on debris and the status topic carries the failure reason.

## ROS topics and parameters

Subscribe: ~/image (sensor_msgs/msg/Image), ~/camera_info (sensor_msgs/msg/CameraInfo). Publish: ~/target_pose (geometry_msgs/msg/PoseStamped), ~/detections (vision_msgs/msg/Detection2DArray), ~/debris (vision_msgs/msg/Detection2DArray), and ~/status (std_msgs/msg/String). Pose and detections preserve the image timestamp. Detection confidence is in [0,1]; debris uses the same schema with class_id debris.

Parameters: camera_frame, target_frame (contract-fixed CAD frame target_cad), output_frame (empty means camera frame), static_translation / static_quaternion_xyzw (fixed parent-frame transform applied only to VALID poses), max_image_age_ms, detector (mock|yolov8), yolo_model, keypoints_json.

Perception does not command vehicles, select actions, apply safety filters, call policy code, or contain LLM logic.

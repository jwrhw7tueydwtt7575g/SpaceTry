"""Optional OpenCV solvePnP backend and frame transform helpers."""
import numpy as np
from .contracts import CameraCalibration, Detection, DetectionStatus, PoseEstimate, TargetKeypoints

def _quat_multiply(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)

def _quat_rotate(q, v):
    return np.asarray(_quat_multiply(_quat_multiply(q, (float(v[0]), float(v[1]), float(v[2]), 0.0)), (-q[0], -q[1], -q[2], q[3]))[:3])

def transform_pose(pose: PoseEstimate, translation_m, quaternion_xyzw, output_frame: str) -> PoseEstimate:
    """Apply a fixed parent-frame transform to a pose without changing timestamp/confidence."""
    rotated = _quat_rotate(quaternion_xyzw, pose.translation_m)
    position = tuple((rotated + np.asarray(translation_m, dtype=np.float64)).tolist())
    orientation = _quat_multiply(quaternion_xyzw, pose.quaternion_xyzw)
    return PoseEstimate(position, orientation, pose.confidence, pose.latency_ns, pose.stamp_ns, output_frame, pose.status, pose.reprojection_error_px, pose.clock_warning)

class FixedPoseSolver:
    def __init__(self, pose: PoseEstimate): self.pose = pose
    def solve(self, keypoints, detection):
        del keypoints, detection
        return self.pose

class OpenCVPnPSolver:
    def solve(self, keypoints: TargetKeypoints, detection: Detection, calibration: CameraCalibration) -> PoseEstimate:
        try:
            import cv2
        except ImportError:
            return PoseEstimate.failure(DetectionStatus.SOLVER_UNAVAILABLE, detection.stamp_ns, detection.frame_id)
        if not calibration.validate() or not detection.validate(keypoints):
            return PoseEstimate.failure(DetectionStatus.BAD_CALIBRATION if not calibration.validate() else DetectionStatus.INVALID_DETECTION, detection.stamp_ns, detection.frame_id)
        object_points = np.asarray(keypoints.points_m, dtype=np.float64)
        image_points = np.asarray(detection.pixel_points, dtype=np.float64)
        camera_matrix = np.asarray(calibration.K, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(calibration.distortion or (0.0,) * 5, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return PoseEstimate.failure(DetectionStatus.AMBIGUOUS_PNP, detection.stamp_ns, calibration.camera_frame, confidence=detection.confidence)
        rotation, _ = cv2.Rodrigues(rvec)
        trace = float(np.trace(rotation))
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0); q = ((rotation[2,1]-rotation[1,2])/s, (rotation[0,2]-rotation[2,0])/s, (rotation[1,0]-rotation[0,1])/s, 0.25*s)
        else:
            q = (0.0, 0.0, 0.0, 1.0)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))
        confidence = float(np.clip(detection.confidence * np.exp(-error / 5.0), 0.0, 1.0))
        return PoseEstimate(tuple(float(v) for v in tvec.reshape(3)), q, confidence, 0, detection.stamp_ns, calibration.camera_frame, DetectionStatus.VALID, error)

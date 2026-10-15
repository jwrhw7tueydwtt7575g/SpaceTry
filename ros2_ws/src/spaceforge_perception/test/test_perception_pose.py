"""Ground-truth PnP accuracy and failure-path coverage (mock detector only).

Live YOLOv8 / ROS-camera paths are never exercised here; they require
ultralytics + ROS 2 + Gazebo and are dependency-gated in test_perception_live_ros.py.
"""
import sys
import numpy as np
import pytest
from spaceforge_perception.contracts import (
    CameraCalibration, Detection, DetectionStatus, PoseEstimate, TargetKeypoints,
)
from spaceforge_perception.detectors import DeterministicMockDetector, synthetic_target_image
from spaceforge_perception.node import detection_bbox_pixels
from spaceforge_perception.pipeline import PerceptionPipeline
from spaceforge_perception.pnp import FixedPoseSolver, OpenCVPnPSolver

cv2 = pytest.importorskip("cv2")

KEYPOINTS = TargetKeypoints(
    ("a", "b", "c", "d"),
    ((-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)),
)
CALIB = CameraCalibration(640, 480, (500, 0, 320, 0, 500, 240, 0, 0, 1), (0, 0, 0, 0, 0))
TRUTH_T = np.asarray((0.3, -0.2, 5.0), dtype=np.float64)
TRUTH_RVEC = np.zeros(3, dtype=np.float64)


def project_ground_truth_pixels():
    """Noise-free pixel projections of the CAD model at the known pose."""
    obj = np.asarray(KEYPOINTS.points_m, dtype=np.float64)
    K = np.asarray(CALIB.K, dtype=np.float64).reshape(3, 3)
    pixels, _ = cv2.projectPoints(obj, TRUTH_RVEC, TRUTH_T, K, np.zeros(5))
    return tuple(map(tuple, pixels.reshape(-1, 2).tolist()))


def test_opencv_solver_recovers_ground_truth_pose():
    pixels = project_ground_truth_pixels()
    detection = Detection(pixels, (1.0,) * 4, 1.0, 100, CALIB.camera_frame)
    pose = OpenCVPnPSolver().solve(KEYPOINTS, detection, CALIB)
    assert pose.status is DetectionStatus.VALID
    error_m = float(np.linalg.norm(np.asarray(pose.translation_m) - TRUTH_T))
    assert error_m < 0.01
    assert pose.reprojection_error_px < 1.0
    assert 0.0 <= pose.confidence <= 1.0


def test_end_to_end_pose_error_against_ground_truth():
    pixels = project_ground_truth_pixels()
    image = synthetic_target_image(640, 480, pixels)
    pipeline = PerceptionPipeline(
        KEYPOINTS, CALIB,
        detector=DeterministicMockDetector(pixels),
        solver=OpenCVPnPSolver(),
        max_image_age_ns=10 ** 12,
    )
    pose = pipeline.process(image, 100, now_ns=200)
    assert pose.status is DetectionStatus.VALID
    assert float(np.linalg.norm(np.asarray(pose.translation_m) - TRUTH_T)) < 0.02
    assert pipeline.last_detection is not None
    cx, cy, sx, sy = detection_bbox_pixels(pipeline.last_detection)
    assert 0 < cx < 640 and 0 < cy < 480 and sx > 0 and sy > 0


def test_invalid_detection_wrong_cardinality():
    pipeline = PerceptionPipeline(
        KEYPOINTS, CALIB,
        detector=DeterministicMockDetector(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))),
        solver=FixedPoseSolver(PoseEstimate.failure(DetectionStatus.VALID)),
        max_image_age_ns=10 ** 12,
    )
    result = pipeline.process(np.zeros((480, 640), dtype=np.uint8), 100, now_ns=200)
    assert result.status is DetectionStatus.INVALID_DETECTION


def test_solver_unavailable_without_opencv(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    pixels = ((300.0, 220.0), (340.0, 220.0), (340.0, 260.0), (300.0, 260.0))
    detection = Detection(pixels, (1.0,) * 4, 1.0, 100, CALIB.camera_frame)
    pose = OpenCVPnPSolver().solve(KEYPOINTS, detection, CALIB)
    assert pose.status is DetectionStatus.SOLVER_UNAVAILABLE


def test_frame_error_on_detector_frame_mismatch():
    class WrongFrame:
        def detect(self, image, stamp_ns, frame_id):
            del image, frame_id
            pixels = ((300.0, 220.0), (340.0, 220.0), (340.0, 260.0), (300.0, 260.0))
            return Detection(pixels, (1.0,) * 4, 0.9, int(stamp_ns), "some_other_frame")

    pipeline = PerceptionPipeline(
        KEYPOINTS, CALIB, detector=WrongFrame(),
        solver=FixedPoseSolver(PoseEstimate.failure(DetectionStatus.VALID)),
        max_image_age_ns=10 ** 12,
    )
    result = pipeline.process(np.zeros((480, 640), dtype=np.uint8), 100, now_ns=200)
    assert result.status is DetectionStatus.FRAME_ERROR


def test_negative_latency_clamped_with_clock_warning():
    pipeline = PerceptionPipeline(
        KEYPOINTS, CALIB,
        detector=DeterministicMockDetector(
            ((300.0, 220.0), (340.0, 220.0), (340.0, 260.0), (300.0, 260.0))),
        solver=FixedPoseSolver(
            PoseEstimate((0, 0, 5), (0, 0, 0, 1), 0.9, 0, 100, "camera_optical_frame", DetectionStatus.VALID)),
        max_image_age_ns=10 ** 12,
    )
    result = pipeline.process(np.zeros((480, 640), dtype=np.uint8), 500, now_ns=100)
    assert result.latency_ns == 0
    assert result.clock_warning is True

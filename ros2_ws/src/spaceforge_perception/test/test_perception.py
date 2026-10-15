import numpy as np
from spaceforge_perception.contracts import CameraCalibration, DetectionStatus, PoseEstimate, TargetKeypoints
from spaceforge_perception.detectors import DeterministicMockDetector, synthetic_target_image
from spaceforge_perception.pipeline import PerceptionPipeline
from spaceforge_perception.pnp import FixedPoseSolver, transform_pose

KEYPOINTS = TargetKeypoints(("a", "b", "c", "d"), ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)))
CALIB = CameraCalibration(640, 480, (500,0,320, 0,500,240, 0,0,1), (0,0,0,0,0))
PIXELS = ((300,220), (340,220), (340,260), (300,260))
IMAGE = synthetic_target_image(640, 480, PIXELS)

def pipeline(detector, solver=None, calibration=CALIB, max_age=1000):
    return PerceptionPipeline(KEYPOINTS, calibration, detector=detector, solver=solver or FixedPoseSolver(PoseEstimate((0,0,5), (0,0,0,1), 0.9, 0, 100, "camera_optical_frame", DetectionStatus.VALID)), max_image_age_ns=max_age)

def test_valid_detection_and_pose_error_against_ground_truth():
    expected = np.asarray((0, 0, 5), dtype=float)
    result = pipeline(DeterministicMockDetector(PIXELS),).process(IMAGE, 100, now_ns=200)
    assert result.status is DetectionStatus.VALID
    assert result.confidence == 0.9
    assert np.linalg.norm(np.asarray(result.translation_m) - expected) < 1e-9
    assert result.latency_ns == 100

def test_no_detection_failure():
    result = pipeline(DeterministicMockDetector(None)).process(IMAGE, 100, now_ns=200)
    assert result.status is DetectionStatus.NO_DETECTION

def test_ambiguous_pnp_failure():
    ambiguous = FixedPoseSolver(PoseEstimate.failure(DetectionStatus.AMBIGUOUS_PNP, 100, "camera_optical_frame"))
    result = pipeline(DeterministicMockDetector(PIXELS), ambiguous).process(IMAGE, 100, now_ns=200)
    assert result.status is DetectionStatus.AMBIGUOUS_PNP

def test_bad_calibration_failure():
    bad = CameraCalibration(640, 480, (0,)*9, (0,)*5)
    result = pipeline(DeterministicMockDetector(PIXELS), calibration=bad).process(IMAGE, 100, now_ns=200)
    assert result.status is DetectionStatus.BAD_CALIBRATION

def test_stale_image_failure():
    result = pipeline(DeterministicMockDetector(PIXELS), max_age=10).process(IMAGE, 100, now_ns=1000)
    assert result.status is DetectionStatus.STALE_IMAGE

def test_frame_transform_preserves_timestamp_and_changes_pose():
    pose = PoseEstimate((1, 0, 0), (0,0,0,1), 0.8, 4, 100, "camera_optical_frame", DetectionStatus.VALID)
    moved = transform_pose(pose, (2, 0, 0), (0,0,0,1), "target_frame")
    assert moved.frame_id == "target_frame"
    assert moved.translation_m == (3.0, 0.0, 0.0)
    assert moved.stamp_ns == pose.stamp_ns and moved.latency_ns == pose.latency_ns

"""Detector-to-pose perception data contracts."""
from .contracts import CameraCalibration, Detection, DetectionStatus, PoseEstimate, TargetKeypoints
from .detectors import DeterministicMockDetector, YOLOv8Detector
from .pipeline import PerceptionPipeline, synthetic_target_image
from .pnp import FixedPoseSolver, OpenCVPnPSolver, transform_pose

__all__ = [
    "CameraCalibration", "Detection", "DetectionStatus", "PoseEstimate", "TargetKeypoints",
    "DeterministicMockDetector", "YOLOv8Detector", "PerceptionPipeline",
    "FixedPoseSolver", "OpenCVPnPSolver", "transform_pose", "synthetic_target_image",
]

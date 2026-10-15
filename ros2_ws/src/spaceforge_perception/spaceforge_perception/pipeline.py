"""Detector-to-pose pipeline; no policy or safety dependencies."""
import time
from dataclasses import replace
import numpy as np
from .contracts import CameraCalibration, DetectionStatus, PoseEstimate, TargetKeypoints
from .detectors import DeterministicMockDetector, synthetic_target_image
from .pnp import OpenCVPnPSolver, transform_pose

__all__ = ["PerceptionPipeline", "synthetic_target_image"]

class PerceptionPipeline:
    def __init__(self, keypoints: TargetKeypoints, calibration: CameraCalibration, detector=None, solver=None, max_image_age_ns: int = 200_000_000, output_frame: str | None = None, static_translation_m=(0.0, 0.0, 0.0), static_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0)):
        self.keypoints = keypoints
        self.calibration = calibration
        self.detector = detector or DeterministicMockDetector()
        self.solver = solver or OpenCVPnPSolver()
        self.max_image_age_ns = int(max_image_age_ns)
        self.output_frame = output_frame
        self.static_translation_m = static_translation_m
        self.static_quaternion_xyzw = static_quaternion_xyzw
        # Most recent Detection (or None); lets publishers build 2D messages
        # without re-running the detector.
        self.last_detection = None

    def process(self, image: np.ndarray, stamp_ns: int, frame_id: str | None = None, now_ns: int | None = None) -> PoseEstimate:
        now_ns = int(time.time_ns() if now_ns is None else now_ns)
        frame_id = frame_id or self.calibration.camera_frame
        skew = now_ns - int(stamp_ns)
        clock_warning = bool(skew < 0)
        latency = max(0, skew)
        self.last_detection = None
        if not self.calibration.validate() or not self.keypoints.validate():
            return PoseEstimate.failure(DetectionStatus.BAD_CALIBRATION, stamp_ns, frame_id, latency,
                                        clock_warning=clock_warning)
        if skew > self.max_image_age_ns:
            return PoseEstimate.failure(DetectionStatus.STALE_IMAGE, stamp_ns, frame_id, latency,
                                        clock_warning=clock_warning)
        detection = self.detector.detect(image, int(stamp_ns), frame_id)
        self.last_detection = detection
        if detection is None:
            return PoseEstimate.failure(DetectionStatus.NO_DETECTION, stamp_ns, frame_id, latency,
                                        clock_warning=clock_warning)
        if detection.frame_id != frame_id:
            # Detector answered in a frame the pipeline never asked for: the
            # pixels cannot be interpreted, so fail loudly instead of solving.
            return PoseEstimate.failure(DetectionStatus.FRAME_ERROR, stamp_ns, frame_id, latency,
                                        confidence=detection.confidence, clock_warning=clock_warning)
        if not detection.validate(self.keypoints):
            return PoseEstimate.failure(DetectionStatus.INVALID_DETECTION, stamp_ns, frame_id, latency, detection.confidence,
                                        clock_warning=clock_warning)
        try:
            pose = self.solver.solve(self.keypoints, detection, self.calibration)
        except TypeError:
            pose = self.solver.solve(self.keypoints, detection)
        pose = replace(pose, latency_ns=latency, stamp_ns=int(stamp_ns), frame_id=frame_id,
                       clock_warning=clock_warning)
        if pose.status == DetectionStatus.VALID and self.output_frame and self.output_frame != frame_id:
            pose = transform_pose(pose, self.static_translation_m, self.static_quaternion_xyzw, self.output_frame)
        return pose

"""Validated, dependency-free perception data contracts."""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np

class DetectionStatus(str, Enum):
    VALID = "VALID"
    NO_DETECTION = "NO_DETECTION"
    AMBIGUOUS_PNP = "AMBIGUOUS_PNP"
    BAD_CALIBRATION = "BAD_CALIBRATION"
    STALE_IMAGE = "STALE_IMAGE"
    INVALID_DETECTION = "INVALID_DETECTION"
    SOLVER_UNAVAILABLE = "SOLVER_UNAVAILABLE"
    FRAME_ERROR = "FRAME_ERROR"

@dataclass(frozen=True)
class CameraCalibration:
    width_px: int
    height_px: int
    K: Tuple[float, ...]
    distortion: Tuple[float, ...] = ()
    camera_frame: str = "camera_optical_frame"

    def validate(self):
        matrix = np.asarray(self.K, dtype=np.float64)
        if self.width_px <= 0 or self.height_px <= 0 or matrix.size != 9 or not np.all(np.isfinite(matrix)): return False
        matrix = matrix.reshape(3, 3)
        return bool(matrix[0, 0] > 0 and matrix[1, 1] > 0 and -self.width_px <= matrix[0, 2] <= 2*self.width_px and -self.height_px <= matrix[1, 2] <= 2*self.height_px and self.camera_frame)

@dataclass(frozen=True)
class TargetKeypoints:
    names: Tuple[str, ...]
    points_m: Tuple[Tuple[float, float, float], ...]
    frame_id: str = "target_cad"

    def validate(self):
        points = np.asarray(self.points_m, dtype=np.float64)
        return bool(len(self.names) == len(points) >= 4 and points.ndim == 2 and points.shape[1] == 3 and np.all(np.isfinite(points)) and len(set(self.names)) == len(self.names) and self.frame_id == "target_cad")

@dataclass(frozen=True)
class Detection:
    pixel_points: Tuple[Tuple[float, float], ...]
    point_confidence: Tuple[float, ...]
    confidence: float
    stamp_ns: int
    frame_id: str
    class_id: str = "target"

    def validate(self, keypoints: TargetKeypoints):
        pixels = np.asarray(self.pixel_points, dtype=np.float64)
        confidence = np.asarray(self.point_confidence, dtype=np.float64)
        return bool(keypoints.validate() and len(pixels) == len(keypoints.points_m) and pixels.ndim == 2 and pixels.shape[1] == 2 and len(confidence) == len(pixels) and np.all(np.isfinite(pixels)) and np.all(np.isfinite(confidence)) and np.all((0 <= confidence) & (confidence <= 1)) and 0 <= self.confidence <= 1 and self.stamp_ns >= 0 and bool(self.frame_id))

@dataclass(frozen=True)
class PoseEstimate:
    translation_m: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    confidence: float
    latency_ns: int
    stamp_ns: int
    frame_id: str
    status: DetectionStatus
    reprojection_error_px: float = float("inf")
    # True when the image stamp was ahead of the receive clock (negative latency
    # clamped to zero). Trailing default: positional construction is unaffected.
    clock_warning: bool = False

    @classmethod
    def failure(cls, status, stamp_ns=0, frame_id="camera_optical_frame", latency_ns=0, confidence=0.0, clock_warning=False):
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), confidence, latency_ns, stamp_ns, frame_id, status,
                   float("inf"), bool(clock_warning))

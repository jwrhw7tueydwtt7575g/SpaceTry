"""Detector interfaces and deterministic/mock/optional YOLO implementations."""
from typing import Optional, Protocol
import numpy as np
from .contracts import Detection

class Detector(Protocol):
    def detect(self, image: np.ndarray, stamp_ns: int, frame_id: str) -> Optional[Detection]: ...

class DeterministicMockDetector:
    """Returns a configured detection; ideal for synthetic and unit tests."""
    def __init__(self, pixel_points=None, confidence: float = 1.0, point_confidence=None, class_id="target"):
        self.pixel_points = None if pixel_points is None else tuple(tuple(float(v) for v in p) for p in pixel_points)
        self.confidence = float(confidence)
        self.point_confidence = point_confidence
        self.class_id = class_id

    def detect(self, image, stamp_ns, frame_id):
        del image
        if self.pixel_points is None: return None
        points = self.point_confidence or tuple(self.confidence for _ in self.pixel_points)
        return Detection(self.pixel_points, tuple(float(v) for v in points), self.confidence, int(stamp_ns), frame_id, self.class_id)

class YOLOv8Detector:
    """Optional adapter; model loading is deferred until this class is used."""
    def __init__(self, model_path: str, keypoint_count: int):
        try:
            from ultralytics import YOLO
        except ImportError as exc: raise RuntimeError("ultralytics is required for YOLOv8Detector") from exc
        self.model = YOLO(model_path); self.keypoint_count = keypoint_count

    def detect(self, image, stamp_ns, frame_id):
        results = self.model.predict(image, verbose=False)
        if not results or getattr(results[0], "keypoints", None) is None: return None
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        if len(keypoints) != self.keypoint_count: return None
        confidence = float(results[0].keypoints.conf[0].mean().cpu().item()) if results[0].keypoints.conf is not None else 0.0
        return Detection(tuple(map(tuple, keypoints)), tuple([confidence] * len(keypoints)), confidence, int(stamp_ns), frame_id)

def synthetic_target_image(width: int, height: int, pixel_points, radius: int = 2):
    """Create a deterministic grayscale test image with bright keypoint dots."""
    image = np.zeros((height, width), dtype=np.uint8)
    for x, y in pixel_points:
        cx, cy = int(round(x)), int(round(y))
        image[max(0, cy-radius):min(height, cy+radius+1), max(0, cx-radius):min(width, cx+radius+1)] = 255
    return image

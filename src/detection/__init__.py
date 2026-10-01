"""Object detection components."""

from .yolo_detector import Detection, DetectionError, ModelLoadError, YOLODetector

__all__ = ["Detection", "DetectionError", "ModelLoadError", "YOLODetector"]

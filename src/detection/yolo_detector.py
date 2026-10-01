"""Pre-trained Ultralytics YOLO object detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, YOLO_MODEL_NAME


class ModelLoadError(RuntimeError):
    """Raised when pre-trained YOLO weights cannot be loaded."""


class DetectionError(RuntimeError):
    """Raised when a frame cannot be processed by YOLO."""


@dataclass(frozen=True)
class Detection:
    """A project-friendly object detection independent of YOLO result objects."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """Load a YOLO model once and run detection on individual OpenCV frames."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._model = YOLO(self.model_name)
        except Exception as error:
            raise ModelLoadError(
                f"Could not load YOLO model '{self.model_name}': {error}"
            ) from error

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one frame and return clean detection objects."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise DetectionError("Expected a non-empty OpenCV frame for inference.")

        try:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            detections: list[Detection] = []
            for result in results:
                detections.extend(self.parse_result(result))
            return detections
        except Exception as error:
            raise DetectionError(f"YOLO inference failed: {error}") from error

    @staticmethod
    def parse_result(result: Any) -> list[Detection]:
        """Convert one Ultralytics result object into project detection objects."""
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        coordinates = YOLODetector._to_list(boxes.xyxy)
        confidences = YOLODetector._to_list(boxes.conf)
        class_ids = YOLODetector._to_list(boxes.cls)
        names = getattr(result, "names", {})

        detections: list[Detection] = []
        for box, confidence, class_id in zip(coordinates, confidences, class_ids):
            numeric_class_id = int(class_id)
            detections.append(
                Detection(
                    x1=round(float(box[0])),
                    y1=round(float(box[1])),
                    x2=round(float(box[2])),
                    y2=round(float(box[3])),
                    confidence=float(confidence),
                    class_id=numeric_class_id,
                    class_name=YOLODetector._class_name(names, numeric_class_id),
                )
            )
        return detections

    @staticmethod
    def _to_list(values: Any) -> list[Any]:
        """Convert PyTorch tensors or compatible values into Python lists."""
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        """Look up a class name while preserving a useful fallback."""
        if isinstance(names, dict):
            return str(names.get(class_id, f"class_{class_id}"))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return f"class_{class_id}"

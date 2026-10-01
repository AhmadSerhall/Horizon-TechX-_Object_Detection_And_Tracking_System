"""OpenCV drawing helpers for detection output."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection import Detection


def draw_detections(
    frame: np.ndarray, detections: list[Detection], fps: float | None = None
) -> np.ndarray:
    """Return a copy of a frame annotated with classes, confidence, and FPS."""
    annotated_frame = frame.copy()

    for detection in detections:
        color = _class_color(detection.class_id)
        cv2.rectangle(
            annotated_frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        label_y = max(detection.y1 - 10, 20)
        cv2.putText(
            annotated_frame,
            label,
            (detection.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if fps is not None:
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Create a consistent, easy-to-see BGR color for one class."""
    return (
        (37 * class_id + 80) % 255,
        (17 * class_id + 150) % 255,
        (29 * class_id + 220) % 255,
    )

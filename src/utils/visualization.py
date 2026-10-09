"""OpenCV drawing helpers for detection output."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection import Detection
from src.tracking import TrackedObject


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


def draw_tracked_objects(
    frame: np.ndarray, tracked_objects: list[TrackedObject], fps: float | None = None
) -> np.ndarray:
    """Return a frame annotated with tracker IDs, confidence, and active-track count."""
    annotated_frame = frame.copy()

    for tracked_object in tracked_objects:
        color = _class_color(tracked_object.class_id)
        cv2.rectangle(
            annotated_frame,
            (tracked_object.x1, tracked_object.y1),
            (tracked_object.x2, tracked_object.y2),
            color,
            2,
        )
        label = (
            f"{tracked_object.class_name} #{tracked_object.track_id} "
            f"{tracked_object.confidence:.2f}"
        )
        label_y = max(tracked_object.y1 - 10, 20)
        cv2.putText(
            annotated_frame,
            label,
            (tracked_object.x1, label_y),
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
    cv2.putText(
        annotated_frame,
        f"Active Tracks: {len(tracked_objects)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated_frame

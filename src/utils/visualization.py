"""OpenCV drawing helpers for detection output."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection import Detection
from src.tracking import TrackedObject

from .statistics import FrameStatistics


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
    return annotated_frame


def draw_statistics_panel(
    frame: np.ndarray, statistics: FrameStatistics, max_classes: int = 5
) -> np.ndarray:
    """Draw a compact panel containing current-frame processing statistics."""
    annotated_frame = frame.copy()
    lines = [
        f"FPS: {statistics.processing_fps:.1f}",
        f"Tracking: {'ON' if statistics.tracking_enabled else 'OFF'}",
    ]
    if statistics.active_tracks is not None:
        lines.append(f"Active Tracks: {statistics.active_tracks}")
    lines.extend(
        [
            f"Confidence: {statistics.confidence_threshold:.2f}",
            f"Objects: {statistics.object_count}",
        ]
    )

    class_items = sorted(
        statistics.class_counts.items(), key=lambda item: (-item[1], item[0])
    )
    if class_items:
        lines.append("Objects:")
        lines.extend(
            f"{class_name.title()}: {count}"
            for class_name, count in class_items[:max_classes]
        )
        remaining = len(class_items) - max_classes
        if remaining > 0:
            lines.append(f"+{remaining} more classes")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 22
    padding = 10
    panel_width = max(
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    ) + padding * 2
    panel_height = line_height * len(lines) + padding * 2
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, annotated_frame, 0.35, 0, annotated_frame)

    for line_number, line in enumerate(lines, start=1):
        cv2.putText(
            annotated_frame,
            line,
            (padding, padding + line_height * line_number - 5),
            font,
            scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )
    return annotated_frame


def draw_help_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw the small in-window keyboard shortcut reference."""
    annotated_frame = frame.copy()
    lines = [
        "Controls",
        "SPACE  Pause / resume",
        "T      Toggle tracking",
        "+/-    Confidence",
        "H      Hide help",
        "Q/ESC  Quit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 24
    padding = 12
    panel_width = max(
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    ) + padding * 2
    panel_height = line_height * len(lines) + padding * 2
    frame_height, frame_width = annotated_frame.shape[:2]
    left = max(frame_width - panel_width - 10, 0)
    top = 10
    right = min(left + panel_width, frame_width - 1)
    bottom = min(top + panel_height, frame_height - 1)
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
    for line_number, line in enumerate(lines, start=1):
        cv2.putText(
            annotated_frame,
            line,
            (left + padding, top + padding + line_height * line_number - 6),
            font,
            scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )
    return annotated_frame

"""OpenCV drawing helpers for detections, tracking, and readable status panels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.detection import Detection
from src.tracking import TrackedObject

from .statistics import FrameStatistics


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_BACKGROUND = (18, 22, 28)
_PANEL_BORDER = (80, 110, 125)
_TEXT_PRIMARY = (245, 245, 245)
_TEXT_MUTED = (185, 185, 185)
_ACCENT = (65, 190, 255)
_STATUS_ON = (95, 225, 110)
_STATUS_OFF = (80, 95, 245)


@dataclass(frozen=True)
class PanelBounds:
    """Position and size of a statistics panel in the final display canvas."""

    x: int
    y: int
    width: int
    height: int
    placement: str

    @property
    def right(self) -> int:
        """Return the exclusive right boundary."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom boundary."""
        return self.y + self.height


@dataclass(frozen=True)
class _PanelRow:
    """One pre-formatted line in the statistics panel."""

    label: str
    value: str = ""
    value_color: tuple[int, int, int] = _TEXT_PRIMARY
    is_heading: bool = False


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    fps: float | None = None,
    show_boxes: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a copy of a frame annotated with classes and confidence values."""
    annotated_frame = frame.copy()

    for detection in detections:
        color = _class_color(detection.class_id)
        if show_boxes:
            cv2.rectangle(
                annotated_frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                color,
                2,
            )
        if show_labels:
            _draw_label(
                annotated_frame,
                f"{detection.class_name} {detection.confidence:.2f}",
                detection.x1,
                detection.y1,
                color,
            )

    if fps is not None:
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            _FONT,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated_frame


def draw_tracked_objects(
    frame: np.ndarray,
    tracked_objects: list[TrackedObject],
    fps: float | None = None,
    show_boxes: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Return a frame annotated with tracker IDs and confidence values."""
    annotated_frame = frame.copy()

    for tracked_object in tracked_objects:
        color = _class_color(tracked_object.class_id)
        if show_boxes:
            cv2.rectangle(
                annotated_frame,
                (tracked_object.x1, tracked_object.y1),
                (tracked_object.x2, tracked_object.y2),
                color,
                2,
            )
        if show_labels:
            _draw_label(
                annotated_frame,
                (
                    f"{tracked_object.class_name} #{tracked_object.track_id} "
                    f"{tracked_object.confidence:.2f}"
                ),
                tracked_object.x1,
                tracked_object.y1,
                color,
            )

    if fps is not None:
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            _FONT,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated_frame


def calculate_statistics_panel_layout(
    statistics: FrameStatistics,
    canvas_size: tuple[int, int],
    image_bounds: tuple[int, int, int, int],
    max_classes: int = 5,
    prefer_letterbox: bool = True,
) -> tuple[PanelBounds, tuple[_PanelRow, ...], float, int, int]:
    """Calculate a bounded panel, preferring unused left-letterbox space."""
    canvas_width, canvas_height = canvas_size
    image_x, image_y, image_width, image_height = image_bounds
    if min(canvas_width, canvas_height, image_width, image_height) <= 0:
        raise ValueError("Canvas and image bounds must have positive dimensions.")

    margin = max(6, round(min(canvas_width, canvas_height) * 0.018))
    side_space = image_x - margin * 2
    if prefer_letterbox and side_space >= 190:
        candidate = _build_panel(
            statistics,
            max_classes,
            side_space,
            canvas_height - margin * 2,
            canvas_size,
        )
        if candidate is not None:
            rows, font_scale, padding, line_height, width, height = candidate
            return (
                PanelBounds(
                    margin,
                    max(margin, (canvas_height - height) // 2),
                    width,
                    height,
                    "left_letterbox",
                ),
                rows,
                font_scale,
                padding,
                line_height,
            )

    available_width = max(1, image_width - margin * 2)
    available_height = max(1, image_height - margin * 2)
    candidate = _build_panel(
        statistics,
        max_classes,
        available_width,
        available_height,
        canvas_size,
    )
    if candidate is None:
        # A tiny display can still show the essential state without overflowing.
        candidate = _build_panel(
            statistics,
            0,
            available_width,
            available_height,
            canvas_size,
            minimum_scale=0.30,
        )
    if candidate is None:
        raise ValueError("Display area is too small for the statistics panel.")

    rows, font_scale, padding, line_height, width, height = candidate
    return (
        PanelBounds(
            image_x + margin,
            image_y + margin,
            width,
            height,
            "inside_video",
        ),
        rows,
        font_scale,
        padding,
        line_height,
    )


def draw_statistics_panel(
    frame: np.ndarray, statistics: FrameStatistics, max_classes: int = 5
) -> np.ndarray:
    """Draw a compact source-frame panel used only for annotated MP4 export."""
    frame_height, frame_width = frame.shape[:2]
    layout, rows, font_scale, padding, line_height = calculate_statistics_panel_layout(
        statistics,
        (frame_width, frame_height),
        (0, 0, frame_width, frame_height),
        max_classes,
        prefer_letterbox=False,
    )
    return _draw_panel(frame, layout, rows, font_scale, padding, line_height)


def draw_display_statistics_panel(
    canvas: np.ndarray,
    statistics: FrameStatistics,
    image_bounds: tuple[int, int, int, int],
    max_classes: int = 5,
) -> tuple[np.ndarray, PanelBounds]:
    """Draw stats on the final letterboxed display canvas, outside video when possible."""
    canvas_height, canvas_width = canvas.shape[:2]
    layout, rows, font_scale, padding, line_height = calculate_statistics_panel_layout(
        statistics,
        (canvas_width, canvas_height),
        image_bounds,
        max_classes,
    )
    return _draw_panel(canvas, layout, rows, font_scale, padding, line_height), layout


def draw_help_overlay(
    frame: np.ndarray, avoid_bounds: PanelBounds | None = None
) -> np.ndarray:
    """Draw a responsive keyboard-help panel away from the statistics panel."""
    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]
    margin = max(6, round(min(frame_width, frame_height) * 0.018))
    scale = min(0.70, max(0.38, min(frame_width, frame_height) / 1050))
    padding = max(7, round(12 * scale / 0.55))
    line_height = max(16, round(28 * scale / 0.55))
    lines = [
        "CONTROLS",
        "SPACE  Pause / resume",
        "T      Toggle tracking",
        "+/-    Confidence",
        "H      Hide help",
        "Q/ESC  Quit",
    ]
    max_text_width = max(1, frame_width - margin * 2 - padding * 2)
    lines = [_fit_text(line, max_text_width, scale) for line in lines]
    width = min(
        frame_width - margin * 2,
        max(_text_width(line, scale) for line in lines) + padding * 2,
    )
    height = min(frame_height - margin * 2, line_height * len(lines) + padding * 2)
    left = max(margin, frame_width - width - margin)
    top = margin
    bounds = PanelBounds(left, top, width, height, "help")
    if avoid_bounds is not None and _rectangles_overlap(bounds, avoid_bounds):
        below_top = avoid_bounds.bottom + margin
        if below_top + height <= frame_height - margin:
            bounds = PanelBounds(left, below_top, width, height, "help")
        else:
            bounds = PanelBounds(left, max(margin, frame_height - height - margin), width, height, "help")
    overlay = annotated_frame.copy()
    cv2.rectangle(
        overlay,
        (bounds.x, bounds.y),
        (bounds.right, bounds.bottom),
        _PANEL_BACKGROUND,
        -1,
    )
    cv2.addWeighted(overlay, 0.84, annotated_frame, 0.16, 0, annotated_frame)
    cv2.rectangle(
        annotated_frame,
        (bounds.x, bounds.y),
        (bounds.right, bounds.bottom),
        _PANEL_BORDER,
        1,
    )
    for index, line in enumerate(lines):
        color = _ACCENT if index == 0 else _TEXT_PRIMARY
        cv2.putText(
            annotated_frame,
            line,
            (bounds.x + padding, bounds.y + padding + line_height * (index + 1) - 6),
            _FONT,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated_frame


def _build_panel(
    statistics: FrameStatistics,
    max_classes: int,
    available_width: int,
    available_height: int,
    canvas_size: tuple[int, int],
    minimum_scale: float = 0.42,
) -> tuple[tuple[_PanelRow, ...], float, int, int, int, int] | None:
    """Fit the most informative set of rows into the available panel space."""
    canvas_width, canvas_height = canvas_size
    initial_scale = min(0.88, max(minimum_scale, min(canvas_width, canvas_height) / 980))
    for scale_step in range(10):
        font_scale = max(minimum_scale, initial_scale - scale_step * 0.05)
        padding = max(6, round(13 * font_scale / 0.60))
        line_height = max(15, round(27 * font_scale / 0.60))
        for visible_classes in range(max_classes, -1, -1):
            rows = _statistics_rows(statistics, visible_classes, available_width - padding * 2, font_scale)
            width = max(_row_width(row, font_scale) for row in rows) + padding * 2
            height = line_height * len(rows) + padding * 2
            if width <= available_width and height <= available_height:
                return rows, font_scale, padding, line_height, width, height
    for scale_step in range(10):
        font_scale = max(0.30, minimum_scale - scale_step * 0.03)
        padding = max(4, round(10 * font_scale / 0.50))
        line_height = max(13, round(22 * font_scale / 0.50))
        rows = _compact_statistics_rows(statistics)
        width = max(_row_width(row, font_scale) for row in rows) + padding * 2
        height = line_height * len(rows) + padding * 2
        if width <= available_width and height <= available_height:
            return rows, font_scale, padding, line_height, width, height
    return None


def _statistics_rows(
    statistics: FrameStatistics,
    visible_classes: int,
    text_width: int,
    font_scale: float,
) -> tuple[_PanelRow, ...]:
    """Format current-frame values with headings and concise class rows."""
    rows = [
        _PanelRow("LIVE STATISTICS", is_heading=True),
        _PanelRow("FPS", f"{statistics.processing_fps:.1f}", _ACCENT),
        _PanelRow(
            "Tracking",
            "ON" if statistics.tracking_enabled else "OFF",
            _STATUS_ON if statistics.tracking_enabled else _STATUS_OFF,
        ),
    ]
    if statistics.active_tracks is not None:
        rows.append(_PanelRow("Active tracks", str(statistics.active_tracks), _ACCENT))
    rows.extend(
        [
            _PanelRow("Confidence", f"{statistics.confidence_threshold:.2f}"),
            _PanelRow("Displayed", str(statistics.object_count), _ACCENT),
        ]
    )

    class_items = sorted(statistics.class_counts.items(), key=lambda item: (-item[1], item[0]))
    if class_items and visible_classes:
        rows.append(_PanelRow("CURRENT OBJECTS", is_heading=True))
        for class_name, count in class_items[:visible_classes]:
            value = f"x{count}"
            value_width = _text_width(value, font_scale)
            label_limit = max(1, text_width - value_width - _text_width("  ", font_scale))
            rows.append(_PanelRow(_fit_text(class_name.title(), label_limit, font_scale), value))
        remaining = len(class_items) - visible_classes
        if remaining > 0:
            rows.append(_PanelRow(f"+{remaining} more classes"))
    elif class_items:
        rows.append(_PanelRow(f"{len(class_items)} object classes"))
    else:
        rows.append(_PanelRow("No objects detected"))
    return tuple(rows)


def _compact_statistics_rows(statistics: FrameStatistics) -> tuple[_PanelRow, ...]:
    """Provide essential state when a very small viewport cannot fit full details."""
    return (
        _PanelRow("LIVE", is_heading=True),
        _PanelRow("FPS", f"{statistics.processing_fps:.1f}", _ACCENT),
        _PanelRow(
            "Tracking",
            "ON" if statistics.tracking_enabled else "OFF",
            _STATUS_ON if statistics.tracking_enabled else _STATUS_OFF,
        ),
        _PanelRow("Objects", str(statistics.object_count), _ACCENT),
    )


def _draw_panel(
    frame: np.ndarray,
    bounds: PanelBounds,
    rows: tuple[_PanelRow, ...],
    font_scale: float,
    padding: int,
    line_height: int,
) -> np.ndarray:
    """Render a dark, high-contrast panel without changing frame dimensions."""
    annotated_frame = frame.copy()
    overlay = annotated_frame.copy()
    cv2.rectangle(
        overlay,
        (bounds.x, bounds.y),
        (bounds.right, bounds.bottom),
        _PANEL_BACKGROUND,
        -1,
    )
    cv2.addWeighted(overlay, 0.87, annotated_frame, 0.13, 0, annotated_frame)
    cv2.rectangle(
        annotated_frame,
        (bounds.x, bounds.y),
        (bounds.right, bounds.bottom),
        _PANEL_BORDER,
        1,
    )
    for index, row in enumerate(rows):
        baseline = bounds.y + padding + line_height * (index + 1) - 6
        if row.is_heading:
            cv2.putText(
                annotated_frame,
                row.label,
                (bounds.x + padding, baseline),
                _FONT,
                font_scale,
                _ACCENT,
                1,
                cv2.LINE_AA,
            )
            continue
        cv2.putText(
            annotated_frame,
            row.label,
            (bounds.x + padding, baseline),
            _FONT,
            font_scale,
            _TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        if row.value:
            value_x = bounds.right - padding - _text_width(row.value, font_scale)
            cv2.putText(
                annotated_frame,
                row.value,
                (value_x, baseline),
                _FONT,
                font_scale,
                row.value_color,
                1,
                cv2.LINE_AA,
            )
    return annotated_frame


def _draw_label(frame: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw one detection label above its source-coordinate bounding box."""
    cv2.putText(
        frame,
        label,
        (x, max(y - 10, 20)),
        _FONT,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Create a consistent, easy-to-see BGR color for one class."""
    return (
        (37 * class_id + 80) % 255,
        (17 * class_id + 150) % 255,
        (29 * class_id + 220) % 255,
    )


def _fit_text(text: str, max_width: int, font_scale: float) -> str:
    """Truncate text with an ellipsis so it always stays inside its panel."""
    if _text_width(text, font_scale) <= max_width:
        return text
    ellipsis = "..."
    shortened = text
    while shortened and _text_width(shortened + ellipsis, font_scale) > max_width:
        shortened = shortened[:-1]
    return (shortened + ellipsis) if shortened else ellipsis


def _row_width(row: _PanelRow, font_scale: float) -> int:
    """Measure one panel row, including an optional right-aligned value."""
    if not row.value:
        return _text_width(row.label, font_scale)
    return _text_width(row.label, font_scale) + _text_width("  ", font_scale) + _text_width(
        row.value, font_scale
    )


def _text_width(text: str, font_scale: float) -> int:
    """Return pixel width for the common panel font."""
    return cv2.getTextSize(text, _FONT, font_scale, 1)[0][0]


def _rectangles_overlap(first: PanelBounds, second: PanelBounds) -> bool:
    """Return whether two panel rectangles share visible area."""
    return not (
        first.right <= second.x
        or second.right <= first.x
        or first.bottom <= second.y
        or second.bottom <= first.y
    )

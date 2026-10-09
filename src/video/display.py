"""Aspect-ratio-safe OpenCV display helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxLayout:
    """The scale and placement used to fit a source frame into a viewport."""

    viewport_width: int
    viewport_height: int
    image_width: int
    image_height: int
    offset_x: int
    offset_y: int
    scale: float

    def source_to_display(self, x: float, y: float) -> tuple[int, int]:
        """Map a source-frame coordinate to its displayed letterbox coordinate."""
        return (
            round(x * self.scale) + self.offset_x,
            round(y * self.scale) + self.offset_y,
        )


def calculate_letterbox_layout(
    source_size: tuple[int, int], viewport_size: tuple[int, int]
) -> LetterboxLayout:
    """Calculate an uncropped, centered, aspect-ratio-preserving display layout."""
    source_width, source_height = source_size
    viewport_width, viewport_height = viewport_size
    if min(source_width, source_height, viewport_width, viewport_height) <= 0:
        raise ValueError("Source and viewport dimensions must be positive.")

    scale = min(viewport_width / source_width, viewport_height / source_height)
    image_width = max(1, round(source_width * scale))
    image_height = max(1, round(source_height * scale))
    return LetterboxLayout(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        image_width=image_width,
        image_height=image_height,
        offset_x=(viewport_width - image_width) // 2,
        offset_y=(viewport_height - image_height) // 2,
        scale=scale,
    )


def letterbox_frame(
    frame: np.ndarray, viewport_size: tuple[int, int]
) -> tuple[np.ndarray, LetterboxLayout]:
    """Fit a frame into a black canvas without cropping or distorting it."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Expected a non-empty frame for display.")

    source_height, source_width = frame.shape[:2]
    layout = calculate_letterbox_layout((source_width, source_height), viewport_size)
    interpolation = cv2.INTER_AREA if layout.scale < 1 else cv2.INTER_LINEAR
    resized_frame = cv2.resize(
        frame,
        (layout.image_width, layout.image_height),
        interpolation=interpolation,
    )
    canvas_shape = (layout.viewport_height, layout.viewport_width, *frame.shape[2:])
    canvas = np.zeros(canvas_shape, dtype=frame.dtype)
    top = layout.offset_y
    left = layout.offset_x
    canvas[top : top + layout.image_height, left : left + layout.image_width] = resized_frame
    return canvas, layout

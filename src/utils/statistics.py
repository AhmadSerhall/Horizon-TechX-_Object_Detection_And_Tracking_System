"""Per-frame object statistics for the OpenCV overlay."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol


class DisplayObject(Protocol):
    """The shared fields needed to count displayed detections or tracks."""

    class_name: str


@dataclass(frozen=True)
class FrameStatistics:
    """Current-frame values shown in the application status panel."""

    processing_fps: float
    tracking_enabled: bool
    confidence_threshold: float
    object_count: int
    class_counts: dict[str, int]
    active_tracks: int | None


def calculate_frame_statistics(
    objects: Iterable[DisplayObject],
    processing_fps: float,
    tracking_enabled: bool,
    confidence_threshold: float,
) -> FrameStatistics:
    """Count only objects displayed on the current annotated frame."""
    displayed_objects = list(objects)
    class_counts = dict(Counter(item.class_name for item in displayed_objects))
    return FrameStatistics(
        processing_fps=processing_fps,
        tracking_enabled=tracking_enabled,
        confidence_threshold=confidence_threshold,
        object_count=len(displayed_objects),
        class_counts=class_counts,
        active_tracks=len(displayed_objects) if tracking_enabled else None,
    )

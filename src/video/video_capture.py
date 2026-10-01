"""Reusable OpenCV video source management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np


VideoSource: TypeAlias = int | str | Path


class VideoSourceError(RuntimeError):
    """Raised when a webcam or video file cannot be opened."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic properties reported by an OpenCV video source."""

    width: int
    height: int
    fps: float
    frame_count: int | None

    @property
    def duration_seconds(self) -> float | None:
        """Return the duration when frame count and FPS are available."""
        if self.frame_count is None or self.fps <= 0:
            return None
        return self.frame_count / self.fps


class VideoCapture:
    """Open, read, inspect, and release a webcam or video file source."""

    def __init__(self, source: VideoSource) -> None:
        self.source = source
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_webcam(self) -> bool:
        """Return whether the source identifies a webcam index."""
        return isinstance(self.source, int)

    @property
    def source_description(self) -> str:
        """Return a concise human-readable source label."""
        if self.is_webcam:
            return f"Webcam {self.source}"
        return str(Path(self.source))

    @property
    def is_opened(self) -> bool:
        """Return whether OpenCV has an active, opened source."""
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        """Open the configured webcam or existing video file."""
        if self.is_opened:
            return

        if not self.is_webcam:
            file_path = Path(self.source)
            if not file_path.is_file():
                raise VideoSourceError(f"Video file was not found: {file_path}")
            capture_source: int | str = str(file_path)
        else:
            capture_source = self.source

        self._capture = cv2.VideoCapture(capture_source)
        if self._capture.isOpened():
            return

        self.release()
        raise VideoSourceError(f"Could not open video source: {self.source_description}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame, or return ``(False, None)`` at end of source."""
        if not self.is_opened or self._capture is None:
            raise VideoSourceError("Video source is not open. Call open() first.")

        success, frame = self._capture.read()
        if not success:
            return False, None
        return True, frame

    def get_metadata(self) -> VideoMetadata:
        """Return source metadata after the source has been opened."""
        if not self.is_opened or self._capture is None:
            raise VideoSourceError("Video source is not open. Call open() first.")

        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMetadata(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._capture.get(cv2.CAP_PROP_FPS)),
            frame_count=frame_count if frame_count > 0 else None,
        )

    def release(self) -> None:
        """Release the underlying OpenCV capture resource."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

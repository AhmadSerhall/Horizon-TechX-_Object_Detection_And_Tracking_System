"""Safe MP4 export for annotated source-resolution frames."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class VideoWriterError(RuntimeError):
    """Raised when an annotated-video export cannot be created or written."""


def create_output_path(output_directory: Path, now: datetime | None = None) -> Path:
    """Create a non-conflicting timestamped MP4 path inside the output directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_path = output_directory / f"tracked_video_{timestamp}.mp4"
    if not base_path.exists():
        return base_path

    suffix = 1
    while True:
        candidate = output_directory / f"tracked_video_{timestamp}_{suffix}.mp4"
        if not candidate.exists():
            return candidate
        suffix += 1


class ProcessedVideoWriter:
    """Lazily create and reliably release an MP4 VideoWriter."""

    def __init__(self, output_directory: Path, codec: str = "mp4v") -> None:
        self.output_directory = output_directory
        self.codec = codec
        self.path: Path | None = None
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None

    @property
    def is_opened(self) -> bool:
        """Return whether the MP4 writer is ready to receive frames."""
        return self._writer is not None and self._writer.isOpened()

    def start(self, frame: np.ndarray, fps: float) -> Path:
        """Open an MP4 writer using the original annotated-frame dimensions."""
        if self.is_opened and self.path is not None:
            return self.path
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise VideoWriterError("Cannot start export without a valid annotated frame.")

        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        self.path = create_output_path(self.output_directory)
        safe_fps = fps if fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, safe_fps, self._frame_size)
        if self._writer.isOpened():
            return self.path

        self.release()
        raise VideoWriterError(f"Could not open video export: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """Write one source-resolution annotated frame to the active MP4 export."""
        if not self.is_opened or self._writer is None or self._frame_size is None:
            raise VideoWriterError("Video export has not been started.")
        height, width = frame.shape[:2]
        if (width, height) != self._frame_size:
            raise VideoWriterError(
                "Export frame dimensions changed; expected "
                f"{self._frame_size[0]}x{self._frame_size[1]}, got {width}x{height}."
            )
        self._writer.write(frame)

    def release(self) -> None:
        """Finalize the MP4 file if a writer was created."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

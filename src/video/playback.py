"""Simple source-FPS-aware pacing for processed video files."""

from __future__ import annotations


class PlaybackTiming:
    """Calculate OpenCV event-loop delays without dropping processed frames."""

    def __init__(self, source_fps: float) -> None:
        self.source_fps = source_fps

    @property
    def source_frame_interval(self) -> float | None:
        """Return the source-frame interval in seconds when it is known."""
        if self.source_fps <= 0:
            return None
        return 1 / self.source_fps

    def wait_delay_ms(self, processing_seconds: float) -> int:
        """Return a wait time that avoids playing a video faster than its source FPS."""
        frame_interval = self.source_frame_interval
        if frame_interval is None:
            return 1
        remaining_seconds = frame_interval - max(processing_seconds, 0.0)
        return max(1, round(max(remaining_seconds, 0.0) * 1000))

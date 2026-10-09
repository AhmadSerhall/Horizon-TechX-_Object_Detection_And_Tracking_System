"""Video input and output components."""

from .display import LetterboxLayout, calculate_letterbox_layout, letterbox_frame
from .playback import PlaybackTiming
from .video_capture import VideoCapture, VideoMetadata, VideoSourceError

__all__ = [
    "LetterboxLayout",
    "PlaybackTiming",
    "VideoCapture",
    "VideoMetadata",
    "VideoSourceError",
    "calculate_letterbox_layout",
    "letterbox_frame",
]

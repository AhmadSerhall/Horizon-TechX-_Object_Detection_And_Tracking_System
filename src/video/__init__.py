"""Video input and output components."""

from .display import LetterboxLayout, calculate_letterbox_layout, letterbox_frame
from .playback import PlaybackTiming
from .video_capture import VideoCapture, VideoMetadata, VideoSourceError
from .video_writer import ProcessedVideoWriter, VideoWriterError, create_output_path

__all__ = [
    "LetterboxLayout",
    "PlaybackTiming",
    "VideoCapture",
    "VideoMetadata",
    "VideoSourceError",
    "VideoWriterError",
    "ProcessedVideoWriter",
    "calculate_letterbox_layout",
    "create_output_path",
    "letterbox_frame",
]

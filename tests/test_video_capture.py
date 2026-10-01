"""Tests for source handling that do not require a physical camera."""

from pathlib import Path
import unittest

from src.video import VideoCapture, VideoMetadata, VideoSourceError


class VideoCaptureTests(unittest.TestCase):
    """Validate basic video-source logic without opening a webcam."""

    def test_webcam_source_description(self) -> None:
        video = VideoCapture(0)

        self.assertTrue(video.is_webcam)
        self.assertEqual(video.source_description, "Webcam 0")

    def test_video_file_source_description(self) -> None:
        video = VideoCapture(Path("input/example.mp4"))

        self.assertFalse(video.is_webcam)
        self.assertEqual(video.source_description, "input\\example.mp4")

    def test_missing_video_file_raises_useful_error(self) -> None:
        missing_path = Path("input/does-not-exist.mp4")
        video = VideoCapture(missing_path)

        with self.assertRaisesRegex(VideoSourceError, "Video file was not found"):
            video.open()

    def test_duration_is_calculated_when_metadata_is_available(self) -> None:
        metadata = VideoMetadata(width=1280, height=720, fps=25.0, frame_count=250)

        self.assertEqual(metadata.duration_seconds, 10.0)

    def test_duration_is_unknown_without_frame_count(self) -> None:
        metadata = VideoMetadata(width=640, height=480, fps=30.0, frame_count=None)

        self.assertIsNone(metadata.duration_seconds)


if __name__ == "__main__":
    unittest.main()

"""Tests for source-resolution MP4 export lifecycle without a real codec."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.video.video_writer import (
    ProcessedVideoWriter,
    VideoWriterError,
    create_output_path,
)


class ProcessedVideoWriterTests(unittest.TestCase):
    """Validate output naming, dimensions, and cleanup using a mocked writer."""

    def test_output_path_avoids_timestamp_collisions(self) -> None:
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            now = datetime(2026, 10, 9, 18, 0, 0)
            first_path = create_output_path(output_directory, now)
            first_path.touch()
            second_path = create_output_path(output_directory, now)

            self.assertEqual(first_path.name, "tracked_video_20261009_180000.mp4")
            self.assertEqual(second_path.name, "tracked_video_20261009_180000_1.mp4")

    @patch("src.video.video_writer.cv2.VideoWriter")
    def test_writer_uses_annotated_source_dimensions_and_releases(self, mock_writer: MagicMock) -> None:
        fake_writer = MagicMock()
        fake_writer.isOpened.return_value = True
        mock_writer.return_value = fake_writer
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with TemporaryDirectory() as directory:
            writer = ProcessedVideoWriter(Path(directory))
            output_path = writer.start(frame, 25.0)
            writer.write(frame)
            writer.release()

        self.assertEqual(mock_writer.call_args.args[3], (640, 480))
        self.assertEqual(mock_writer.call_args.args[2], 25.0)
        self.assertEqual(output_path.suffix, ".mp4")
        fake_writer.write.assert_called_once_with(frame)
        fake_writer.release.assert_called_once()
        self.assertFalse(writer.is_opened)

    @patch("src.video.video_writer.cv2.VideoWriter")
    def test_writer_rejects_a_changed_frame_size(self, mock_writer: MagicMock) -> None:
        fake_writer = MagicMock()
        fake_writer.isOpened.return_value = True
        mock_writer.return_value = fake_writer
        with TemporaryDirectory() as directory:
            writer = ProcessedVideoWriter(Path(directory))
            writer.start(np.zeros((20, 30, 3), dtype=np.uint8), 30.0)
            with self.assertRaisesRegex(VideoWriterError, "dimensions changed"):
                writer.write(np.zeros((30, 30, 3), dtype=np.uint8))
            writer.release()


if __name__ == "__main__":
    unittest.main()

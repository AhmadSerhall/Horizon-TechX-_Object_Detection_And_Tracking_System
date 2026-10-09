"""Tests for the tracking-enabled configuration branch in the app loop."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import app
from src.detection import Detection
from src.video import VideoMetadata


class FakeVideoCapture:
    """Single-frame video source used to avoid physical hardware in this test."""

    source_description = "Test source"

    def __init__(self, source: int) -> None:
        self.source = source

    def open(self) -> None:
        pass

    def get_metadata(self) -> VideoMetadata:
        return VideoMetadata(640, 480, 30.0, None)

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((20, 20, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class FakeDetector:
    """Detector replacement that avoids loading YOLO weights."""

    device = "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [Detection(1, 1, 10, 10, 0.9, 0, "person")]


class TrackingToggleTests(unittest.TestCase):
    """Verify disabled tracking uses detection-only visualization."""

    @patch.object(app, "print_video_info")
    @patch.object(app, "draw_tracked_objects")
    @patch.object(app, "draw_detections", return_value=np.zeros((20, 20, 3), dtype=np.uint8))
    @patch.object(app, "YOLODetector", FakeDetector)
    @patch.object(app, "VideoCapture", FakeVideoCapture)
    @patch.object(app, "TRACKING_ENABLED", False)
    @patch.object(app, "cv2")
    def test_tracking_disabled_uses_detection_visualization(
        self,
        mock_cv2: MagicMock,
        mock_draw_detections: MagicMock,
        mock_draw_tracked_objects: MagicMock,
        mock_print_video_info: MagicMock,
    ) -> None:
        mock_cv2.waitKey.return_value = ord("q")

        result = app.run_video_pipeline()

        self.assertEqual(result, 0)
        mock_draw_detections.assert_called_once()
        mock_draw_tracked_objects.assert_not_called()
        mock_print_video_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Headless tests for the PySide6 desktop shell and worker controls."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog

from src.gui.main_window import MainWindow
from src.gui.video_worker import VideoProcessingWorker
from src.utils import FrameStatistics


class DesktopGuiTests(unittest.TestCase):
    """Verify GUI wiring without a physical webcam, model download, or visible window."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow(start_worker=False)

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()

    def test_main_window_initializes_expected_controls(self) -> None:
        self.assertEqual(self.window.windowTitle(), "Object Detection & Tracking System")
        self.assertTrue(self.window.open_video_button.isEnabled())
        self.assertTrue(self.window.tracking_checkbox.isChecked())
        self.assertEqual(self.window.confidence_slider.value(), 50)

    @patch.object(QFileDialog, "getOpenFileName", return_value=("", ""))
    def test_cancelled_file_selection_leaves_source_unchanged(self, _: MagicMock) -> None:
        original_source = self.window.source_label.text()

        self.window.open_video_dialog()

        self.assertEqual(self.window.source_label.text(), original_source)

    def test_invalid_video_path_shows_friendly_error_without_starting_source(self) -> None:
        with patch.object(self.window, "_show_message") as show_message:
            started = self.window.start_source("input/not-a-real-video.mp4")

        self.assertFalse(started)
        show_message.assert_called_once()

    def test_source_requests_support_switching_between_file_and_webcam(self) -> None:
        requested_sources: list[object] = []
        self.window.request_source.connect(requested_sources.append)
        with TemporaryDirectory() as directory:
            video_path = Path(directory) / "sample.mp4"
            video_path.touch()
            self.assertTrue(self.window.start_source(str(video_path)))
        self.assertTrue(self.window.start_source(0))

        self.assertEqual(requested_sources[-1], 0)
        self.assertEqual(len(requested_sources), 2)

    def test_statistics_update_uses_current_frame_values(self) -> None:
        statistics = FrameStatistics(18.4, True, 0.45, 3, {"person": 2, "car": 1}, 3)

        self.window._update_statistics(statistics)

        self.assertEqual(self.window.stats_labels["fps"].text(), "18.4")
        self.assertEqual(self.window.stats_labels["objects"].text(), "3")
        self.assertEqual(self.window.stats_labels["confidence"].text(), "0.45")

    def test_worker_thread_stops_cleanly(self) -> None:
        threaded_window = MainWindow(start_worker=True)
        try:
            self.assertTrue(threaded_window._worker_thread.isRunning())
        finally:
            threaded_window.shutdown()
            threaded_window.deleteLater()
        self.assertFalse(threaded_window._worker_thread.isRunning())


class WorkerControlTests(unittest.TestCase):
    """Validate worker configuration and cleanup without starting inference."""

    def test_confidence_change_does_not_require_model_reload(self) -> None:
        worker = VideoProcessingWorker()
        detector = MagicMock()
        worker._detector = detector

        worker.set_confidence(0.35)

        self.assertEqual(worker._confidence, 0.35)
        detector.set_confidence_threshold.assert_called_once_with(0.35)

    @patch("src.gui.video_worker.ObjectTracker")
    def test_enabling_tracking_creates_a_fresh_tracker(self, mock_tracker: MagicMock) -> None:
        worker = VideoProcessingWorker()
        worker._video = MagicMock()

        worker.set_tracking_enabled(True)
        worker.set_tracking_enabled(False)

        mock_tracker.assert_called_once()
        self.assertIsNone(worker._tracker)

    def test_stop_releases_an_active_source_and_export_writer(self) -> None:
        worker = VideoProcessingWorker()
        video = MagicMock()
        writer = MagicMock()
        worker._video = video
        worker._writer = writer

        worker.stop()

        video.release.assert_called_once()
        writer.release.assert_called_once()
        self.assertIsNone(worker._video)
        self.assertIsNone(worker._writer)

    def test_pause_and_resume_preserve_the_open_source(self) -> None:
        worker = VideoProcessingWorker()
        video = MagicMock()
        video.is_opened = True
        worker._video = video

        worker.pause()
        self.assertTrue(worker._paused)
        self.assertIs(worker._video, video)
        worker.resume()

        self.assertFalse(worker._paused)
        self.assertIs(worker._video, video)


if __name__ == "__main__":
    unittest.main()

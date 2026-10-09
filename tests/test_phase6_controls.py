"""Tests for command-line controls and current-frame statistics."""

from contextlib import redirect_stderr
from io import StringIO
import unittest

import app
from src.detection import Detection
from src.tracking import TrackedObject
from src.utils.cli import confidence_value, parse_arguments, video_source_value
from src.utils.statistics import calculate_frame_statistics


class Phase6ControlTests(unittest.TestCase):
    """Validate Phase 6 logic without a model, webcam, or GUI."""

    def test_cli_interprets_webcam_indexes_and_video_paths(self) -> None:
        webcam = parse_arguments(["--source", "0", "--confidence", "0.35"])
        video_file = parse_arguments(["--source", "input/demo.mp4", "--no-tracking"])

        self.assertEqual(webcam.source, 0)
        self.assertEqual(webcam.confidence, 0.35)
        self.assertEqual(video_file.source, "input/demo.mp4")
        self.assertTrue(video_file.no_tracking)
        self.assertEqual(video_source_value("12"), 12)
        self.assertEqual(video_source_value("input/12.mp4"), "input/12.mp4")

    def test_invalid_confidence_is_rejected_and_keyboard_values_are_clamped(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_arguments(["--confidence", "1.01"])
        with self.assertRaises(Exception):
            confidence_value("not-a-number")

        self.assertEqual(app.clamp_confidence(1.04), 1.0)
        self.assertEqual(app.clamp_confidence(-0.04), 0.0)
        self.assertEqual(app.clamp_confidence(0.555), 0.56)

    def test_statistics_count_only_currently_displayed_objects(self) -> None:
        detections = [
            Detection(0, 0, 10, 10, 0.9, 0, "person"),
            Detection(20, 0, 30, 10, 0.8, 0, "person"),
            Detection(40, 0, 50, 10, 0.7, 2, "car"),
        ]
        statistics = calculate_frame_statistics(detections, 24.6, False, 0.5)

        self.assertEqual(statistics.object_count, 3)
        self.assertEqual(statistics.class_counts, {"person": 2, "car": 1})
        self.assertIsNone(statistics.active_tracks)

    def test_tracking_statistics_use_the_visualized_tracks(self) -> None:
        tracks = [
            TrackedObject(4, 0, 0, 10, 10, 0.9, 0, "person"),
            TrackedObject(8, 20, 0, 30, 10, 0.8, 2, "car"),
        ]
        statistics = calculate_frame_statistics(tracks, 15.0, True, 0.4)

        self.assertEqual(statistics.active_tracks, 2)
        self.assertEqual(statistics.object_count, 2)
        self.assertEqual(statistics.class_counts, {"person": 1, "car": 1})


if __name__ == "__main__":
    unittest.main()

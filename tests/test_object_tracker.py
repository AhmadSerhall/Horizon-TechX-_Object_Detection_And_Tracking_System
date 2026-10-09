"""Tracking tests that use synthetic detections instead of a camera or YOLO model."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.detection import Detection
from src.tracking import ObjectTracker, TrackedObject
from src.tracking.object_tracker import _TrackerDetections


def person(x1: int, y1: int, x2: int, y2: int) -> Detection:
    """Create a high-confidence person detection for test sequences."""
    return Detection(x1, y1, x2, y2, 0.95, 0, "person")


class ObjectTrackerTests(unittest.TestCase):
    """Verify ByteTrack integration without webcam, GUI, or model weights."""

    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_detection_conversion_uses_center_width_height(self) -> None:
        tracker_input = _TrackerDetections([person(10, 20, 110, 220)])

        np.testing.assert_allclose(tracker_input.xywh, [[60, 120, 100, 200]])
        np.testing.assert_allclose(tracker_input.conf, [0.95])
        np.testing.assert_allclose(tracker_input.cls, [0])

    def test_same_object_keeps_its_id_across_frames(self) -> None:
        tracker = ObjectTracker()
        first_frame = tracker.update(self.frame, [person(100, 100, 200, 300)])
        second_frame = tracker.update(self.frame, [person(106, 102, 206, 302)])

        self.assertEqual(len(first_frame), 1)
        self.assertEqual(len(second_frame), 1)
        self.assertEqual(first_frame[0].track_id, second_frame[0].track_id)
        self.assertIsInstance(first_frame[0], TrackedObject)
        self.assertEqual(second_frame[0].class_name, "person")

    def test_multiple_objects_receive_distinct_ids(self) -> None:
        tracker = ObjectTracker()
        tracked_objects = tracker.update(
            self.frame,
            [person(20, 20, 100, 180), person(400, 50, 500, 250)],
        )

        self.assertEqual(len(tracked_objects), 2)
        self.assertEqual(len({item.track_id for item in tracked_objects}), 2)

    def test_empty_detections_are_safe(self) -> None:
        tracker = ObjectTracker()

        self.assertEqual(tracker.update(self.frame, []), [])

    def test_reset_discards_the_underlying_tracker_state(self) -> None:
        tracker = ObjectTracker()
        tracker._tracker.reset = MagicMock()

        tracker.reset()

        tracker._tracker.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()

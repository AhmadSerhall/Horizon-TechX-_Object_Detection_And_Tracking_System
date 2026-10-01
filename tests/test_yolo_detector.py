"""Unit tests for detection parsing without loading real model weights."""

import unittest

from src.detection import Detection, YOLODetector


class FakeTensor:
    """Small tensor-like object used to test parser conversion."""

    def __init__(self, values: list[object]) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeBoxes:
    """Minimal stand-in for Ultralytics box results."""

    def __init__(self) -> None:
        self.xyxy = FakeTensor([[10.4, 20.6, 100.2, 200.8], [3.0, 4.0, 5.0, 6.0]])
        self.conf = FakeTensor([0.91, 0.75])
        self.cls = FakeTensor([0.0, 2.0])


class FakeResult:
    """Minimal stand-in for one Ultralytics result."""

    boxes = FakeBoxes()
    names = {0: "person", 2: "car"}


class YOLODetectorParsingTests(unittest.TestCase):
    """Verify conversion from raw model data to Detection objects."""

    def test_parse_result_creates_clean_detections(self) -> None:
        detections = YOLODetector.parse_result(FakeResult())

        self.assertEqual(
            detections,
            [
                Detection(10, 21, 100, 201, 0.91, 0, "person"),
                Detection(3, 4, 5, 6, 0.75, 2, "car"),
            ],
        )

    def test_parse_result_without_boxes_returns_empty_list(self) -> None:
        result = type("EmptyResult", (), {"boxes": None})()

        self.assertEqual(YOLODetector.parse_result(result), [])

    def test_unknown_class_uses_a_readable_fallback(self) -> None:
        self.assertEqual(YOLODetector._class_name({}, 42), "class_42")


if __name__ == "__main__":
    unittest.main()

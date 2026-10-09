"""Command-line parsing for the OpenCV application."""

from __future__ import annotations

import argparse

from config import CONFIDENCE_THRESHOLD, DEFAULT_VIDEO_SOURCE


def confidence_value(value: str) -> float:
    """Parse a confidence value in the inclusive range from 0 to 1."""
    try:
        confidence = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Confidence must be a number from 0 to 1.") from error
    if not 0.0 <= confidence <= 1.0:
        raise argparse.ArgumentTypeError("Confidence must be between 0 and 1.")
    return confidence


def video_source_value(value: str) -> int | str:
    """Interpret a non-negative integer as a webcam index, otherwise keep a path."""
    if value.isdigit():
        return int(value)
    return value


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Return validated command-line overrides while preserving config defaults."""
    parser = argparse.ArgumentParser(description="Object Detection & Tracking System")
    parser.add_argument(
        "--source",
        type=video_source_value,
        default=DEFAULT_VIDEO_SOURCE,
        help="Webcam index (for example 0) or path to a video file.",
    )
    parser.add_argument(
        "--confidence",
        type=confidence_value,
        default=CONFIDENCE_THRESHOLD,
        help="YOLO confidence threshold from 0 to 1.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Display detection boxes without ByteTrack IDs.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated source-resolution frames as an MP4 in output/.",
    )
    return parser.parse_args(arguments)

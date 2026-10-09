"""Persistent object tracking backed by Ultralytics ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from config import (
    NEW_TRACK_THRESH,
    TRACK_BUFFER,
    TRACK_FUSE_SCORE,
    TRACK_HIGH_THRESH,
    TRACK_LOW_THRESH,
    TRACK_MATCH_THRESH,
)
from src.detection import Detection


class TrackerError(RuntimeError):
    """Raised when ByteTrack cannot update its object tracks."""


@dataclass(frozen=True)
class TrackedObject:
    """A project-friendly object with a persistent tracker-generated ID."""

    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class _TrackerDetections:
    """Minimal NumPy-backed input adapter required by Ultralytics ByteTrack."""

    def __init__(self, detections: list[Detection]) -> None:
        self.conf = np.asarray([item.confidence for item in detections], dtype=np.float32)
        self.cls = np.asarray([item.class_id for item in detections], dtype=np.float32)
        self.xywh = np.asarray(
            [
                [
                    (item.x1 + item.x2) / 2,
                    (item.y1 + item.y2) / 2,
                    item.x2 - item.x1,
                    item.y2 - item.y1,
                ]
                for item in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, indices: Any) -> "_TrackerDetections":
        subset = self.__class__.__new__(self.__class__)
        subset.conf = self.conf[indices]
        subset.cls = self.cls[indices]
        subset.xywh = self.xywh[indices]
        if subset.xywh.ndim == 1:
            subset.xywh = subset.xywh.reshape(1, 4)
        return subset


class ObjectTracker:
    """Associate YOLO detections across frames using Ultralytics ByteTrack."""

    def __init__(self) -> None:
        tracker_arguments = SimpleNamespace(
            track_high_thresh=TRACK_HIGH_THRESH,
            track_low_thresh=TRACK_LOW_THRESH,
            new_track_thresh=NEW_TRACK_THRESH,
            track_buffer=TRACK_BUFFER,
            match_thresh=TRACK_MATCH_THRESH,
            fuse_score=TRACK_FUSE_SCORE,
        )
        self._tracker = BYTETracker(tracker_arguments)

    def update(self, frame: np.ndarray, detections: list[Detection]) -> list[TrackedObject]:
        """Update ByteTrack with one frame's detections and return active tracks."""
        tracker_input = _TrackerDetections(detections)
        class_names = {item.class_id: item.class_name for item in detections}

        try:
            tracker_output = self._tracker.update(tracker_input, img=frame)
        except Exception as error:
            raise TrackerError(f"ByteTrack update failed: {error}") from error

        tracked_objects: list[TrackedObject] = []
        for track in tracker_output:
            class_id = int(track[6])
            tracked_objects.append(
                TrackedObject(
                    track_id=int(track[4]),
                    x1=round(float(track[0])),
                    y1=round(float(track[1])),
                    x2=round(float(track[2])),
                    y2=round(float(track[3])),
                    confidence=float(track[5]),
                    class_id=class_id,
                    class_name=class_names.get(class_id, f"class_{class_id}"),
                )
            )
        return tracked_objects

    def reset(self) -> None:
        """Discard tracker state so future tracks begin a new ID sequence."""
        self._tracker.reset()

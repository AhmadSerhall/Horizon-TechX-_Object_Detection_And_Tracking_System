"""Thread-owned OpenCV, YOLO, and ByteTrack processing for the desktop UI."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QImage

from config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_WEBCAM_EXPORT_FPS,
    OUTPUT_DIR,
    STATISTICS_MAX_CLASSES,
    TRACKING_ENABLED,
)
from src.detection import DetectionError, ModelLoadError, YOLODetector
from src.tracking import ObjectTracker, TrackerError
from src.utils import (
    FrameStatistics,
    calculate_frame_statistics,
    draw_detections,
    draw_statistics_panel,
    draw_tracked_objects,
)
from src.video import (
    PlaybackTiming,
    ProcessedVideoWriter,
    VideoCapture,
    VideoMetadata,
    VideoSourceError,
    VideoWriterError,
)
from src.video.video_capture import VideoSource


class VideoProcessingWorker(QObject):
    """Process one video frame per timer tick in a dedicated Qt worker thread."""

    frame_ready = Signal(QImage, object, int, int)
    state_changed = Signal(str, str)
    source_changed = Signal(str, object)
    export_started = Signal(str)
    export_finished = Signal(str)
    error_occurred = Signal(str)
    shutdown_complete = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.process_next_frame)
        self._detector: YOLODetector | None = None
        self._tracker: ObjectTracker | None = None
        self._video: VideoCapture | None = None
        self._metadata: VideoMetadata | None = None
        self._playback_timing: PlaybackTiming | None = None
        self._writer: ProcessedVideoWriter | None = None
        self._frame_index = 0
        self._last_source_frame: np.ndarray | None = None
        self._paused = False
        self._tracking_enabled = TRACKING_ENABLED
        self._confidence = CONFIDENCE_THRESHOLD
        self._show_boxes = True
        self._show_labels = True
        self._save_session = False

    @Slot(object)
    def start_source(self, source: VideoSource) -> None:
        """Open a new source after safely stopping the previous session."""
        self.stop()
        self.state_changed.emit("loading", "Loading model and video source…")
        try:
            if self._detector is None:
                self._detector = YOLODetector(confidence_threshold=self._confidence)
            self._video = VideoCapture(source)
            self._video.open()
            self._metadata = self._video.get_metadata()
            self._playback_timing = (
                None if self._video.is_webcam else PlaybackTiming(self._metadata.fps)
            )
            self._tracker = ObjectTracker() if self._tracking_enabled else None
            self._frame_index = 0
            self._paused = False
            self.source_changed.emit(self._video.source_description, self._metadata)
            self.state_changed.emit("processing", "Processing video…")
            self._schedule_next_frame(1)
        except (VideoSourceError, ModelLoadError, TrackerError, ValueError) as error:
            self._handle_error(str(error))

    @Slot()
    def process_next_frame(self) -> None:
        """Read and process exactly one frame, then schedule the next timer tick."""
        self._process_one_frame(schedule_after=True)

    def _process_one_frame(self, schedule_after: bool, allow_paused: bool = False) -> None:
        """Process one source frame, optionally retaining paused state after review."""
        if (
            (self._paused and not allow_paused)
            or self._video is None
            or not self._video.is_opened
        ):
            return
        try:
            success, frame = self._video.read()
            if not success or frame is None:
                self._finish_source("Video completed.")
                return

            current_frame_index = self._frame_index
            self._last_source_frame = frame.copy()
            processing_seconds = self._analyze_frame(
                frame, current_frame_index, write_export=True
            )
            self._frame_index += 1
            delay = (
                self._playback_timing.wait_delay_ms(processing_seconds)
                if self._playback_timing is not None
                else 1
            )
            if schedule_after:
                self._schedule_next_frame(delay)
        except (DetectionError, TrackerError, VideoWriterError, VideoSourceError, cv2.error) as error:
            self._handle_error(str(error))

    def _analyze_frame(
        self, frame: np.ndarray, frame_index: int, write_export: bool
    ) -> float:
        """Run one YOLO/tracking pass and publish an annotated result to the GUI."""
        processing_start = perf_counter()
        detections = self._detector.detect(frame) if self._detector is not None else []
        if self._tracker is not None:
            displayed_objects = self._tracker.update(frame, detections)
            annotated_frame = draw_tracked_objects(
                frame,
                displayed_objects,
                show_boxes=self._show_boxes,
                show_labels=self._show_labels,
            )
        else:
            displayed_objects = detections
            annotated_frame = draw_detections(
                frame,
                displayed_objects,
                show_boxes=self._show_boxes,
                show_labels=self._show_labels,
            )
        processing_seconds = perf_counter() - processing_start
        statistics = calculate_frame_statistics(
            displayed_objects,
            processing_fps=1 / max(processing_seconds, 1e-9),
            tracking_enabled=self._tracker is not None,
            confidence_threshold=self._detector.confidence_threshold,
        )
        if write_export:
            self._write_export_frame(annotated_frame, statistics)
        self.frame_ready.emit(
            self._to_qimage(annotated_frame),
            statistics,
            frame_index,
            self._metadata.frame_count if self._metadata and self._metadata.frame_count else 0,
        )
        return processing_seconds

    @Slot()
    def pause(self) -> None:
        """Pause processing without discarding the displayed frame or tracker state."""
        if self._video is not None and self._video.is_opened:
            self._paused = True
            self._timer.stop()
            self.state_changed.emit("paused", "Playback paused.")

    @Slot()
    def resume(self) -> None:
        """Resume a paused source from its current frame."""
        if self._paused and self._video is not None and self._video.is_opened:
            self._paused = False
            self.state_changed.emit("processing", "Processing video…")
            self._schedule_next_frame(1)

    @Slot(int)
    def seek_to_frame(self, frame_index: int) -> None:
        """Pause a video file, reset tracking, and analyze the selected frame once."""
        if (
            self._video is None
            or self._metadata is None
            or self._video.is_webcam
            or self._metadata.frame_count is None
        ):
            return
        bounded_index = max(0, min(frame_index, self._metadata.frame_count - 1))
        self._timer.stop()
        if self._save_session:
            self.set_save_session(False)
        try:
            self._video.seek(bounded_index)
            self._frame_index = bounded_index
            self._paused = True
            self._tracker = ObjectTracker() if self._tracking_enabled else None
            self._process_one_frame(schedule_after=False, allow_paused=True)
            if self._video is not None:
                self.state_changed.emit("paused", f"Frame {bounded_index + 1} selected.")
        except (VideoSourceError, TrackerError) as error:
            self._handle_error(str(error))

    @Slot()
    def stop(self) -> None:
        """Release capture and export resources without terminating the worker thread."""
        self._timer.stop()
        self._paused = False
        output_path = self._release_writer()
        if output_path is not None:
            self.export_finished.emit(output_path)
        self._save_session = False
        if self._video is not None:
            self._video.release()
            self._video = None
        self._metadata = None
        self._playback_timing = None
        self._tracker = None
        self._frame_index = 0
        self._last_source_frame = None
        self.state_changed.emit("ready", "Ready. Select a video or webcam.")

    @Slot()
    def shutdown_worker(self) -> None:
        """Stop resources, then tell the owning QThread it may exit safely."""
        self.stop()
        self.shutdown_complete.emit()

    @Slot(float)
    def set_confidence(self, confidence: float) -> None:
        """Apply a later-frame confidence update without reloading YOLO."""
        self._confidence = confidence
        if self._detector is not None:
            self._detector.set_confidence_threshold(confidence)
        self._reanalyze_paused_frame()

    def _reanalyze_paused_frame(self) -> None:
        """Refresh the frozen frame so confidence changes immediately affect its objects."""
        if not self._paused or self._last_source_frame is None or self._detector is None:
            return
        try:
            # A fresh tracker prevents stale associations from the previous threshold.
            self._tracker = ObjectTracker() if self._tracking_enabled else None
            frame_index = max(0, self._frame_index - 1)
            self._analyze_frame(
                self._last_source_frame,
                frame_index,
                write_export=False,
            )
            self.state_changed.emit(
                "paused",
                f"Confidence {self._confidence:.2f} applied to the current frame.",
            )
        except (DetectionError, TrackerError, cv2.error) as error:
            self._handle_error(str(error))

    @Slot(bool)
    def set_tracking_enabled(self, enabled: bool) -> None:
        """Toggle ByteTrack and reset IDs whenever tracking is enabled again."""
        self._tracking_enabled = enabled
        self._tracker = ObjectTracker() if enabled and self._video is not None else None

    @Slot(bool)
    def set_show_boxes(self, enabled: bool) -> None:
        """Change only the visual bounding-box layer."""
        self._show_boxes = enabled

    @Slot(bool)
    def set_show_labels(self, enabled: bool) -> None:
        """Change only the visual label layer."""
        self._show_labels = enabled

    @Slot(bool)
    def set_save_session(self, enabled: bool) -> None:
        """Start or stop exporting the currently processed session from its next frame."""
        self._save_session = enabled
        if not enabled:
            output_path = self._release_writer()
            if output_path is not None:
                self.export_finished.emit(output_path)
            if self._video is not None and self._video.is_opened:
                self.state_changed.emit("processing", "Session export stopped.")

    def _schedule_next_frame(self, delay_ms: int) -> None:
        """Schedule another frame while keeping the worker event loop responsive."""
        if not self._paused and self._video is not None and self._video.is_opened:
            self._timer.start(max(1, delay_ms))

    def _write_export_frame(
        self, annotated_frame: np.ndarray, statistics: FrameStatistics
    ) -> None:
        """Write source-resolution annotations only when session export is enabled."""
        if not self._save_session or self._video is None or self._metadata is None:
            return
        export_frame = draw_statistics_panel(
            annotated_frame, statistics, STATISTICS_MAX_CLASSES
        )
        if self._writer is None:
            self._writer = ProcessedVideoWriter(OUTPUT_DIR)
        if not self._writer.is_opened:
            fps = self._metadata.fps if self._metadata.fps > 0 else DEFAULT_WEBCAM_EXPORT_FPS
            output_path = self._writer.start(export_frame, fps)
            self.export_started.emit(str(output_path))
        self._writer.write(export_frame)

    def _finish_source(self, message: str) -> None:
        """Mark end-of-file and release all source-specific resources."""
        self._timer.stop()
        output_path = self._release_writer()
        if output_path is not None:
            self.export_finished.emit(output_path)
        self._save_session = False
        if self._video is not None:
            self._video.release()
        self._video = None
        self._tracker = None
        self._paused = False
        self._last_source_frame = None
        self.state_changed.emit("completed", message)

    def _release_writer(self) -> str | None:
        """Finalize an active export file and return its path when it exists."""
        if self._writer is not None:
            output_path = str(self._writer.path) if self._writer.path is not None else None
            self._writer.release()
            self._writer = None
            return output_path
        return None

    def _handle_error(self, message: str) -> None:
        """Clean up processing resources and report one friendly worker error."""
        self._timer.stop()
        self._release_writer()
        if self._video is not None:
            self._video.release()
            self._video = None
        self._tracker = None
        self._paused = False
        self._last_source_frame = None
        self.state_changed.emit("error", "Processing stopped due to an error.")
        self.error_occurred.emit(message)

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        """Convert a BGR OpenCV frame to an owned Qt image for cross-thread display."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        return QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

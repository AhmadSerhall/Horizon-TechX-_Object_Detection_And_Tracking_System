"""OpenCV demonstration application for detection and ByteTrack tracking."""

from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from config import (
    CONFIDENCE_STEP,
    CONFIDENCE_THRESHOLD,
    DEFAULT_VIDEO_SOURCE,
    DEFAULT_WEBCAM_EXPORT_FPS,
    OUTPUT_DIR,
    STATISTICS_MAX_CLASSES,
    TRACKER_TYPE,
    TRACKING_ENABLED,
    YOLO_MODEL_NAME,
)
from src.detection import DetectionError, ModelLoadError, YOLODetector
from src.tracking import ObjectTracker, TrackerError
from src.utils import (
    calculate_frame_statistics,
    draw_detections,
    draw_help_overlay,
    draw_statistics_panel,
    draw_tracked_objects,
)
from src.utils.cli import parse_arguments
from src.video import (
    PlaybackTiming,
    ProcessedVideoWriter,
    VideoCapture,
    VideoMetadata,
    VideoSourceError,
    VideoWriterError,
    letterbox_frame,
)


WINDOW_TITLE = "Object Detection & Tracking System"


def print_environment_info() -> None:
    """Print the key runtime versions used by the project."""
    print("Object Detection & Tracking System is starting...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA/GPU acceleration available: {torch.cuda.is_available()}")
    print(f"Ultralytics import successful (YOLO: {YOLO.__name__}).")


def print_video_info(video: VideoCapture, metadata: VideoMetadata) -> None:
    """Print source information once after a successful open."""
    print("Video source opened successfully")
    print(f"Source: {video.source_description}")
    print(f"Resolution: {metadata.width}x{metadata.height}")
    print(f"FPS: {metadata.fps:.2f}")
    if metadata.frame_count is not None:
        print(f"Total frames: {metadata.frame_count}")
    if metadata.duration_seconds is not None:
        print(f"Duration: {metadata.duration_seconds:.2f} seconds")


def print_controls() -> None:
    """Print all keyboard shortcuts once at startup."""
    print("Controls:")
    print("  SPACE  Pause/resume")
    print("  T      Toggle tracking (IDs restart when turned back on)")
    print("  + or = Increase confidence by 0.05")
    print("  -      Decrease confidence by 0.05")
    print("  H      Show/hide help")
    print("  Q/ESC  Quit")


def is_window_open() -> bool:
    """Return whether the OpenCV display window is still visible."""
    try:
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def get_display_size(fallback_size: tuple[int, int]) -> tuple[int, int]:
    """Read the current OpenCV image area, with a safe source-size fallback."""
    try:
        _, _, width, height = cv2.getWindowImageRect(WINDOW_TITLE)
        if width > 0 and height > 0:
            return width, height
    except cv2.error:
        pass
    return fallback_size


def clamp_confidence(confidence: float) -> float:
    """Keep keyboard-adjusted YOLO confidence within its valid range."""
    return round(min(1.0, max(0.0, confidence)), 2)


def export_fps(video: VideoCapture, metadata: VideoMetadata) -> float:
    """Choose encoding FPS without tying output duration to CPU processing time."""
    if metadata.fps > 0:
        return metadata.fps
    return DEFAULT_WEBCAM_EXPORT_FPS


def run_video_pipeline(
    source: int | str | None = None,
    confidence_threshold: float | None = None,
    tracking_enabled: bool | None = None,
    save_output: bool = False,
) -> int:
    """Run detection, optional tracking, display, and optional source-size export."""
    source = DEFAULT_VIDEO_SOURCE if source is None else source
    confidence_threshold = (
        CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
    )
    tracking_enabled = TRACKING_ENABLED if tracking_enabled is None else tracking_enabled

    video = VideoCapture(source)
    writer = ProcessedVideoWriter(OUTPUT_DIR) if save_output else None
    base_display_frame: np.ndarray | None = None
    display_frame: np.ndarray | None = None
    paused = False
    show_help = False
    wait_delay_ms = 1

    try:
        print(f"Loading YOLO model: {YOLO_MODEL_NAME}")
        detector = YOLODetector(confidence_threshold=confidence_threshold)
        print("Model loaded successfully")
        print(f"Inference device: {detector.device.upper()}")
        tracker = ObjectTracker() if tracking_enabled else None
        print(f"Tracking enabled: {TRACKER_TYPE}" if tracker else "Tracking disabled")

        video.open()
        metadata = video.get_metadata()
        print_video_info(video, metadata)
        playback_timing = None if video.is_webcam else PlaybackTiming(metadata.fps)
        if playback_timing is not None:
            print("Video-file playback is capped by its source FPS when processing permits.")
        if writer is not None:
            print("Annotated MP4 export will start with the first processed frame.")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(WINDOW_TITLE, min(metadata.width, 960), min(metadata.height, 720))
        print_controls()

        while True:
            if not paused:
                success, current_frame = video.read()
                if not success:
                    print("Video source has no more frames.")
                    break

                processing_start = perf_counter()
                detections = detector.detect(current_frame)
                if tracker is not None:
                    displayed_objects = tracker.update(current_frame, detections)
                    annotated_frame = draw_tracked_objects(current_frame, displayed_objects)
                else:
                    displayed_objects = detections
                    annotated_frame = draw_detections(current_frame, displayed_objects)

                processing_seconds = perf_counter() - processing_start
                statistics = calculate_frame_statistics(
                    displayed_objects,
                    processing_fps=1 / max(processing_seconds, 1e-9),
                    tracking_enabled=tracker is not None,
                    confidence_threshold=detector.confidence_threshold,
                )
                base_display_frame = draw_statistics_panel(
                    annotated_frame, statistics, STATISTICS_MAX_CLASSES
                )
                display_frame = (
                    draw_help_overlay(base_display_frame)
                    if show_help
                    else base_display_frame
                )

                if writer is not None:
                    if not writer.is_opened:
                        output_path = writer.start(display_frame, export_fps(video, metadata))
                        print(f"Saving annotated video to: {output_path}")
                    writer.write(display_frame)

                wait_delay_ms = (
                    playback_timing.wait_delay_ms(processing_seconds)
                    if playback_timing is not None
                    else 1
                )

            if display_frame is not None:
                frame_size = (display_frame.shape[1], display_frame.shape[0])
                viewport_size = get_display_size(frame_size)
                output_frame, _ = letterbox_frame(display_frame, viewport_size)
                cv2.imshow(WINDOW_TITLE, output_frame)

            key = cv2.waitKey(30 if paused else wait_delay_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if not is_window_open():
                print("Display window closed.")
                break
            if key == ord(" "):
                paused = not paused
                print("Video paused." if paused else "Video resumed.")
            elif key in (ord("t"), ord("T")):
                tracking_enabled = not tracking_enabled
                if tracking_enabled:
                    tracker = ObjectTracker()
                    print("Tracking enabled; tracker state reset and IDs will restart.")
                else:
                    tracker = None
                    print("Tracking disabled: displaying detections only.")
            elif key in (ord("+"), ord("=")):
                detector.set_confidence_threshold(
                    clamp_confidence(detector.confidence_threshold + CONFIDENCE_STEP)
                )
                print(f"Confidence threshold: {detector.confidence_threshold:.2f}")
            elif key == ord("-"):
                detector.set_confidence_threshold(
                    clamp_confidence(detector.confidence_threshold - CONFIDENCE_STEP)
                )
                print(f"Confidence threshold: {detector.confidence_threshold:.2f}")
            elif key in (ord("h"), ord("H")):
                show_help = not show_help
                if base_display_frame is not None:
                    display_frame = (
                        draw_help_overlay(base_display_frame)
                        if show_help
                        else base_display_frame.copy()
                    )
                print("Keyboard help shown." if show_help else "Keyboard help hidden.")
    except (
        VideoSourceError,
        ModelLoadError,
        DetectionError,
        TrackerError,
        VideoWriterError,
    ) as error:
        print(f"Application error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Application interrupted.")
        return 0
    except cv2.error as error:
        print(f"OpenCV display error: {error}")
        return 1
    finally:
        if writer is not None:
            writer.release()
        video.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return 0


def main(arguments: list[str] | None = None) -> int:
    """Run environment checks and the configurable OpenCV pipeline."""
    arguments = parse_arguments(arguments)
    print_environment_info()
    return run_video_pipeline(
        source=arguments.source,
        confidence_threshold=arguments.confidence,
        tracking_enabled=not arguments.no_tracking,
        save_output=arguments.save,
    )


if __name__ == "__main__":
    raise SystemExit(main())

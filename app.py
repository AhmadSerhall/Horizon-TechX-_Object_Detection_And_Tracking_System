"""Phase 5 object detection, tracking, and playback demonstration."""

from time import perf_counter

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from config import DEFAULT_VIDEO_SOURCE, TRACKER_TYPE, TRACKING_ENABLED, YOLO_MODEL_NAME
from src.detection import DetectionError, ModelLoadError, YOLODetector
from src.tracking import ObjectTracker, TrackerError
from src.utils import draw_detections, draw_tracked_objects
from src.video import (
    PlaybackTiming,
    VideoCapture,
    VideoMetadata,
    VideoSourceError,
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


def run_video_pipeline() -> int:
    """Run frame-by-frame detection and display annotated output until exit."""
    video = VideoCapture(DEFAULT_VIDEO_SOURCE)
    current_frame: np.ndarray | None = None
    display_frame: np.ndarray | None = None
    paused = False
    wait_delay_ms = 1

    try:
        print(f"Loading YOLO model: {YOLO_MODEL_NAME}")
        detector = YOLODetector()
        print("Model loaded successfully")
        print(f"Inference device: {detector.device.upper()}")
        tracker = ObjectTracker() if TRACKING_ENABLED else None
        if tracker is not None:
            print(f"Tracking enabled: {TRACKER_TYPE}")
        else:
            print("Tracking disabled: displaying detections only")

        video.open()
        metadata = video.get_metadata()
        print_video_info(video, metadata)
        playback_timing = None if video.is_webcam else PlaybackTiming(metadata.fps)
        if playback_timing is not None:
            print(
                "Video-file playback uses the source FPS as an upper limit; "
                "slower processing plays at the processing rate."
            )

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow(
            WINDOW_TITLE,
            min(metadata.width, 960),
            min(metadata.height, 720),
        )
        print("Controls: SPACE = pause/resume, Q or ESC = quit")

        while True:
            if not paused:
                success, current_frame = video.read()
                if not success:
                    print("Video source has no more frames.")
                    break
                processing_start = perf_counter()
                detections = detector.detect(current_frame)
                if tracker is not None:
                    tracked_objects = tracker.update(current_frame, detections)
                    processing_seconds = perf_counter() - processing_start
                    processing_fps = 1 / max(processing_seconds, 1e-9)
                    display_frame = draw_tracked_objects(
                        current_frame, tracked_objects, processing_fps
                    )
                else:
                    processing_seconds = perf_counter() - processing_start
                    processing_fps = 1 / max(processing_seconds, 1e-9)
                    display_frame = draw_detections(current_frame, detections, processing_fps)
                if playback_timing is not None:
                    wait_delay_ms = playback_timing.wait_delay_ms(processing_seconds)
                else:
                    wait_delay_ms = 1

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
    except (VideoSourceError, ModelLoadError, DetectionError, TrackerError) as error:
        print(f"Application error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Application interrupted.")
        return 0
    except cv2.error as error:
        print(f"OpenCV display error: {error}")
        return 1
    finally:
        video.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    return 0


def main() -> int:
    """Run the environment checks and Phase 5 video pipeline."""
    print_environment_info()
    return run_video_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 3 detection demonstration for the project."""

from time import perf_counter

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from config import DEFAULT_VIDEO_SOURCE, YOLO_MODEL_NAME
from src.detection import DetectionError, ModelLoadError, YOLODetector
from src.utils import draw_detections
from src.video import VideoCapture, VideoMetadata, VideoSourceError


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


def run_video_pipeline() -> int:
    """Run frame-by-frame detection and display annotated output until exit."""
    video = VideoCapture(DEFAULT_VIDEO_SOURCE)
    current_frame: np.ndarray | None = None
    display_frame: np.ndarray | None = None
    paused = False

    try:
        print(f"Loading YOLO model: {YOLO_MODEL_NAME}")
        detector = YOLODetector()
        print("Model loaded successfully")
        print(f"Inference device: {detector.device.upper()}")

        video.open()
        print_video_info(video, video.get_metadata())
        print("Controls: SPACE = pause/resume, Q or ESC = quit")

        while True:
            if not paused:
                success, current_frame = video.read()
                if not success:
                    print("Video source has no more frames.")
                    break
                inference_start = perf_counter()
                detections = detector.detect(current_frame)
                inference_fps = 1 / (perf_counter() - inference_start)
                display_frame = draw_detections(current_frame, detections, inference_fps)

            if display_frame is not None:
                cv2.imshow(WINDOW_TITLE, display_frame)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                print("Video paused." if paused else "Video resumed.")
    except (VideoSourceError, ModelLoadError, DetectionError) as error:
        print(f"Application error: {error}")
        return 1
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
    """Run the environment checks and Phase 2 video pipeline."""
    print_environment_info()
    return run_video_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())

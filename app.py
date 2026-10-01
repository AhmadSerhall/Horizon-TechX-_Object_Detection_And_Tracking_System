"""Phase 2 video-input demonstration for the project."""

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from config import DEFAULT_VIDEO_SOURCE
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
    """Display unmodified frames from the configured source until the user exits."""
    video = VideoCapture(DEFAULT_VIDEO_SOURCE)
    current_frame: np.ndarray | None = None
    paused = False

    try:
        video.open()
        print_video_info(video, video.get_metadata())
        print("Controls: SPACE = pause/resume, Q or ESC = quit")

        while True:
            if not paused:
                success, current_frame = video.read()
                if not success:
                    print("Video source has no more frames.")
                    break

            if current_frame is not None:
                cv2.imshow(WINDOW_TITLE, current_frame)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                print("Video paused." if paused else "Video resumed.")
    except VideoSourceError as error:
        print(f"Video input error: {error}")
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

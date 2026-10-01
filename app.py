"""Environment smoke test for the Object Detection & Tracking System."""

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO


def main() -> None:
    """Print the key runtime versions and hardware availability."""
    print("Object Detection & Tracking System is starting...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA/GPU acceleration available: {torch.cuda.is_available()}")
    print(f"Ultralytics import successful (YOLO: {YOLO.__name__}).")
    print("Project environment is ready.")


if __name__ == "__main__":
    main()

"""Central, lightweight configuration for the application."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# The model will be downloaded by Ultralytics when inference is added later.
YOLO_MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

# 0 selects the default webcam. A path or URL can be supplied later.
DEFAULT_VIDEO_SOURCE: int | str = 0
TRACKING_ENABLED = True

# Deep OC-SORT is included in current Ultralytics releases and will be wired in later.
TRACKER_NAME = "deepocsort.yaml"

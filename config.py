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
CONFIDENCE_STEP = 0.05
STATISTICS_MAX_CLASSES = 5
DEFAULT_WEBCAM_EXPORT_FPS = 30.0

# 0 selects the default webcam. A path or URL can be supplied later.
DEFAULT_VIDEO_SOURCE: int | str = "input/final result.mp4"

# ByteTrack is provided by the installed Ultralytics package.
TRACKING_ENABLED = True
TRACKER_TYPE = "bytetrack"
TRACK_HIGH_THRESH = 0.50
TRACK_LOW_THRESH = 0.10
NEW_TRACK_THRESH = 0.50
TRACK_BUFFER = 30
TRACK_MATCH_THRESH = 0.80
TRACK_FUSE_SCORE = True

# Object Detection & Tracking System

A Python computer-vision project for real-time object detection and persistent multi-object tracking from a webcam or video file.

**HorizonTechX Internship — Task 4**

## Planned technologies

- Python
- OpenCV
- YOLO (Ultralytics)
- PyTorch
- ByteTrack (Ultralytics)

## Current status

YOLO11n detection and Ultralytics ByteTrack persistent tracking IDs are implemented.
Phase 6 adds command-line source and confidence controls, current-frame statistics,
and optional annotated MP4 export.

## Run examples

### Desktop application

```powershell
.\.venv\Scripts\python.exe desktop_app.py
```

Use **Open Video** or **Use Webcam** from the interface. The desktop application
includes playback controls, confidence and tracking settings, current-frame
statistics, and **Save Processed Session** MP4 export. Session export begins at
the next frame and therefore does not claim to create a complete source-video export.

### OpenCV command-line fallback

```powershell
.\.venv\Scripts\python.exe app.py --source 0
.\.venv\Scripts\python.exe app.py --source input\test_video.mp4 --no-tracking
.\.venv\Scripts\python.exe app.py --source input\test_video.mp4 --save
```

Use `--confidence 0.35` to override the default YOLO confidence threshold.

## Keyboard controls

- `Space`: pause/resume
- `T`: toggle ByteTrack on/off (IDs restart when tracking is re-enabled)
- `+` / `=` and `-`: raise/lower confidence by 0.05
- `H`: show/hide in-window help
- `Q` / `Esc`: quit cleanly

In the desktop application, `Space` pauses/resumes, `T` toggles tracking, `H`
opens shortcut help, `Q` exits, and `Esc` stops the active source.

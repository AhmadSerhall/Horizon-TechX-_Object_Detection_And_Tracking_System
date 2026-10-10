"""PySide6 main window for the object detection and tracking application."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMetaObject, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config import CONFIDENCE_STEP, CONFIDENCE_THRESHOLD, TRACKING_ENABLED
from src.utils import FrameStatistics
from src.video import VideoMetadata

from .video_worker import VideoProcessingWorker


class MainWindow(QMainWindow):
    """A dark, responsive desktop shell around the existing vision pipeline."""

    request_source = Signal(object)
    request_pause = Signal()
    request_resume = Signal()
    request_stop = Signal()
    request_seek = Signal(int)
    request_confidence = Signal(float)
    request_tracking = Signal(bool)
    request_show_boxes = Signal(bool)
    request_show_labels = Signal(bool)
    request_save_session = Signal(bool)

    def __init__(self, start_worker: bool = True) -> None:
        super().__init__()
        self._latest_pixmap: QPixmap | None = None
        self._source_name = "No source selected"
        self._source_metadata: VideoMetadata | None = None
        self._current_state = "ready"
        self._ignore_worker_frames = False
        self._worker_thread: QThread | None = None
        self._worker = VideoProcessingWorker()

        self.setWindowTitle("Object Detection & Tracking System")
        self.setMinimumSize(1120, 720)
        self.resize(1440, 900)
        self._build_ui()
        self._install_shortcuts()
        self._connect_worker(start_worker)
        self._set_state("ready", "Ready. Select a video or webcam.")

    def _build_ui(self) -> None:
        """Build the fixed header, responsive content area, and control strip."""
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(22, 18, 22, 18)
        root_layout.setSpacing(16)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 14)
        title_layout = QVBoxLayout()
        self.title_label = QLabel("Object Detection & Tracking System")
        self.title_label.setObjectName("titleLabel")
        subtitle = QLabel("AI Computer Vision")
        subtitle.setObjectName("subtitleLabel")
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        self.state_label = QLabel("READY")
        self.state_label.setObjectName("stateLabel")
        self.status_summary_label = QLabel()
        self.status_summary_label.setObjectName("statusSummary")
        status_layout = QVBoxLayout()
        status_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.state_label, alignment=Qt.AlignmentFlag.AlignRight)
        status_layout.addWidget(self.status_summary_label, alignment=Qt.AlignmentFlag.AlignRight)
        header_layout.addLayout(status_layout)
        root_layout.addWidget(header)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        self.video_label = QLabel("Select a video file or use the webcam to begin.")
        self.video_label.setObjectName("videoPreview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        content_layout.addWidget(self.video_label, stretch=4)
        content_layout.addWidget(self._build_statistics_sidebar(), stretch=1)
        root_layout.addLayout(content_layout, stretch=1)

        root_layout.addWidget(self._build_control_area())
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready. Select a video or webcam.")
        self.setStyleSheet(_APPLICATION_STYLESHEET)

    def _build_statistics_sidebar(self) -> QFrame:
        """Build a readable current-frame statistics sidebar."""
        sidebar = QFrame()
        sidebar.setObjectName("statisticsSidebar")
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(340)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        title = QLabel("LIVE STATISTICS")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.stats_labels: dict[str, QLabel] = {}
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        for key, text in (
            ("fps", "FPS"),
            ("tracking", "Tracking"),
            ("tracks", "Active tracks"),
            ("objects", "Displayed objects"),
            ("confidence", "Confidence"),
            ("source", "Source"),
        ):
            value = QLabel("—")
            value.setObjectName("statValue")
            value.setWordWrap(False)
            self.stats_labels[key] = value
            form.addRow(QLabel(text), value)
        layout.addLayout(form)
        classes_header = QHBoxLayout()
        classes_title = QLabel("CURRENT OBJECTS")
        classes_title.setObjectName("sectionTitle")
        self.class_total_label = QLabel("Total: 0")
        self.class_total_label.setObjectName("classTotal")
        classes_header.addWidget(classes_title)
        classes_header.addStretch()
        classes_header.addWidget(self.class_total_label)
        layout.addLayout(classes_header)
        self.class_counts_layout = QVBoxLayout()
        self.class_counts_layout.setContentsMargins(0, 0, 0, 0)
        self.class_counts_layout.setSpacing(2)
        class_container = QWidget()
        class_container.setObjectName("classContainer")
        class_container.setLayout(self.class_counts_layout)
        scroll = QScrollArea()
        scroll.setObjectName("classScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(class_container)
        layout.addWidget(scroll, stretch=1)
        return sidebar

    def _build_control_area(self) -> QFrame:
        """Build source, playback, settings, and session-export controls."""
        controls = QFrame()
        controls.setObjectName("controlArea")
        layout = QGridLayout(controls)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)

        source_group = QGroupBox("SOURCE")
        source_layout = QVBoxLayout(source_group)
        source_buttons = QHBoxLayout()
        self.open_video_button = QPushButton("Open Video")
        self.webcam_button = QPushButton("Use Webcam")
        source_buttons.addWidget(self.open_video_button)
        source_buttons.addWidget(self.webcam_button)
        source_layout.addLayout(source_buttons)
        self.source_label = QLabel(self._source_name)
        self.source_label.setObjectName("sourceLabel")
        self.source_label.setWordWrap(True)
        source_layout.addWidget(self.source_label)

        playback_group = QGroupBox("PLAYBACK")
        playback_layout = QVBoxLayout(playback_group)
        playback_buttons = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        for button in (self.play_pause_button, self.stop_button):
            playback_buttons.addWidget(button)
        playback_layout.addLayout(playback_buttons)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setEnabled(False)
        self.position_slider.setToolTip("Pause and drag to inspect a specific video frame.")
        self.position_label = QLabel("00:00 / —")
        playback_layout.addWidget(self.position_slider)
        playback_layout.addWidget(self.position_label)

        settings_group = QGroupBox("DETECTION & TRACKING")
        settings_layout = QVBoxLayout(settings_group)
        confidence_row = QHBoxLayout()
        confidence_row.addWidget(QLabel("Confidence"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(20, 90)
        self.confidence_slider.setValue(round(CONFIDENCE_THRESHOLD * 100))
        self.confidence_value_label = QLabel(f"{CONFIDENCE_THRESHOLD:.2f}")
        confidence_row.addWidget(self.confidence_slider, stretch=1)
        confidence_row.addWidget(self.confidence_value_label)
        settings_layout.addLayout(confidence_row)
        self.tracking_checkbox = QCheckBox("Enable tracking")
        self.tracking_checkbox.setChecked(TRACKING_ENABLED)
        self.boxes_checkbox = QCheckBox("Show bounding boxes")
        self.boxes_checkbox.setChecked(True)
        self.labels_checkbox = QCheckBox("Show labels and IDs")
        self.labels_checkbox.setChecked(True)
        settings_layout.addWidget(self.tracking_checkbox)
        settings_layout.addWidget(self.boxes_checkbox)
        settings_layout.addWidget(self.labels_checkbox)

        export_group = QGroupBox("EXPORT")
        export_layout = QVBoxLayout(export_group)
        self.export_button = QPushButton("Save Processed Session")
        self.export_button.setCheckable(True)
        self.export_hint = QLabel("Records from the next processed frame until stopped.")
        self.export_hint.setWordWrap(True)
        self.export_hint.setObjectName("exportHint")
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.export_hint)

        layout.addWidget(source_group, 0, 0)
        layout.addWidget(playback_group, 0, 1)
        layout.addWidget(settings_group, 0, 2)
        layout.addWidget(export_group, 0, 3)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 3)
        layout.setColumnStretch(3, 2)

        self.open_video_button.clicked.connect(self.open_video_dialog)
        self.webcam_button.clicked.connect(lambda: self.start_source(0))
        self.play_pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self.stop_current_source)
        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderMoved.connect(self._preview_seek_position)
        self.position_slider.sliderReleased.connect(self._seek_to_selected_frame)
        self.confidence_slider.valueChanged.connect(self._change_confidence)
        self.tracking_checkbox.toggled.connect(self.request_tracking.emit)
        self.boxes_checkbox.toggled.connect(self.request_show_boxes.emit)
        self.labels_checkbox.toggled.connect(self.request_show_labels.emit)
        self.export_button.toggled.connect(self._toggle_export)
        return controls

    def _connect_worker(self, start_worker: bool) -> None:
        """Move the processor to its own thread and connect queued UI signals."""
        if not start_worker:
            return
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self.request_source.connect(self._worker.start_source)
        self.request_pause.connect(self._worker.pause)
        self.request_resume.connect(self._worker.resume)
        self.request_stop.connect(self._worker.stop)
        self.request_seek.connect(self._worker.seek_to_frame)
        self.request_confidence.connect(self._worker.set_confidence)
        self.request_tracking.connect(self._worker.set_tracking_enabled)
        self.request_show_boxes.connect(self._worker.set_show_boxes)
        self.request_show_labels.connect(self._worker.set_show_labels)
        self.request_save_session.connect(self._worker.set_save_session)
        self._worker.frame_ready.connect(self._update_frame)
        self._worker.state_changed.connect(self._set_state)
        self._worker.source_changed.connect(self._source_opened)
        self._worker.export_started.connect(self._export_started)
        self._worker.export_finished.connect(self._export_finished)
        self._worker.error_occurred.connect(self._show_worker_error)
        self._worker.shutdown_complete.connect(
            self._worker_thread.quit, Qt.ConnectionType.DirectConnection
        )
        self._worker_thread.start()

    def open_video_dialog(self) -> None:
        """Choose a supported video without altering state when the dialog is cancelled."""
        selected_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if selected_file:
            self.start_source(selected_file)

    def start_source(self, source: int | str) -> bool:
        """Validate a requested source and forward it to the worker thread."""
        if isinstance(source, str) and not Path(source).is_file():
            self._show_message("Invalid video", f"Video file was not found:\n{source}", error=True)
            return False
        self.request_source.emit(source)
        return True

    def play_current_source(self) -> None:
        """Restart the selected source, rather than pretending a completed file resumes."""
        if self._source_name == "No source selected":
            self._show_message("No source", "Open a video or select the webcam first.")
            return
        if self._source_name.startswith("Webcam"):
            self.start_source(0)
        else:
            self.start_source(self._source_name)

    def _change_confidence(self, value: int) -> None:
        """Show and send a valid threshold for future or paused-frame inference."""
        confidence = value / 100
        self.confidence_value_label.setText(f"{confidence:.2f}")
        self.request_confidence.emit(confidence)

    def _adjust_confidence(self, adjustment: float) -> None:
        """Move the confidence slider by one configured keyboard shortcut step."""
        new_value = round(self.confidence_slider.value() + adjustment * 100)
        new_value = max(self.confidence_slider.minimum(), new_value)
        new_value = min(self.confidence_slider.maximum(), new_value)
        self.confidence_slider.setValue(new_value)

    def _toggle_export(self, enabled: bool) -> None:
        """Label the recording mode accurately as a current-session export."""
        self.export_button.setText("Stop Session Export" if enabled else "Save Processed Session")
        self.request_save_session.emit(enabled)
        if enabled:
            self.statusBar().showMessage("Session export starts with the next processed frame.")

    def _source_opened(self, source_name: str, metadata: VideoMetadata) -> None:
        """Update source labels and reset visible playback progress."""
        self._ignore_worker_frames = False
        self._source_name = source_name
        self._source_metadata = metadata
        self.source_label.setText(source_name)
        self.source_label.setToolTip(source_name)
        source_summary = _source_summary(source_name)
        self.stats_labels["source"].setText(source_summary)
        self.stats_labels["source"].setToolTip(source_name)
        if metadata.frame_count is not None:
            self.position_slider.setRange(0, max(0, metadata.frame_count - 1))
            self.position_slider.setValue(0)
            self.position_slider.setEnabled(True)
        else:
            self.position_slider.setRange(0, 0)
            self.position_slider.setEnabled(False)
        duration = metadata.duration_seconds
        self.position_label.setText(f"00:00 / {_format_duration(duration)}")

    def _update_frame(
        self, image, statistics: FrameStatistics, frame_index: int, total_frames: int
    ) -> None:
        """Receive an owned QImage from the worker and update GUI widgets only here."""
        if self._ignore_worker_frames:
            return
        self._latest_pixmap = QPixmap.fromImage(image)
        self._update_preview()
        self._update_statistics(statistics)
        if total_frames > 0:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(frame_index)
            self.position_slider.blockSignals(False)
        elapsed = (
            (frame_index + 1) / self._source_metadata.fps
            if self._source_metadata is not None and self._source_metadata.fps > 0
            else None
        )
        duration = self._source_metadata.duration_seconds if self._source_metadata else None
        self.position_label.setText(f"{_format_duration(elapsed)} / {_format_duration(duration)}")

    def _update_preview(self) -> None:
        """Scale the completed annotated frame only for the Qt preview widget."""
        if self._latest_pixmap is None:
            return
        self.video_label.setPixmap(
            self._latest_pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_statistics(self, statistics: FrameStatistics) -> None:
        """Present only current-frame detection or track counts."""
        self.stats_labels["fps"].setText(f"{statistics.processing_fps:.1f}")
        self.stats_labels["tracking"].setText("ON" if statistics.tracking_enabled else "OFF")
        self.stats_labels["tracks"].setText(
            str(statistics.active_tracks) if statistics.active_tracks is not None else "—"
        )
        self.stats_labels["objects"].setText(str(statistics.object_count))
        self.stats_labels["confidence"].setText(f"{statistics.confidence_threshold:.2f}")
        self.class_total_label.setText(f"Total: {statistics.object_count}")
        while self.class_counts_layout.count():
            item = self.class_counts_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if not statistics.class_counts:
            empty_label = QLabel("No objects in this frame")
            empty_label.setObjectName("emptyObjects")
            self.class_counts_layout.addWidget(empty_label)
            self.class_counts_layout.addStretch()
            return
        for class_name, count in sorted(
            statistics.class_counts.items(), key=lambda item: (-item[1], item[0])
        )[:8]:
            row = QLabel(f"{class_name.title()}  × {count}")
            row.setObjectName("classCount")
            row.setFixedHeight(24)
            row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.class_counts_layout.addWidget(row)
        self.class_counts_layout.addStretch()

    def _set_state(self, state: str, message: str) -> None:
        """Show a friendly high-level processing state in the header and status bar."""
        self._current_state = state
        self.state_label.setText(state.upper())
        self.state_label.setProperty("state", state)
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.status_summary_label.setText(message)
        self.statusBar().showMessage(message)
        if state == "processing":
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Pause")
        elif state == "paused":
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Resume")
        elif state == "loading":
            self.play_pause_button.setEnabled(False)
            self.play_pause_button.setText("Loading…")
        else:
            self.play_pause_button.setEnabled(True)
            self.play_pause_button.setText("Play")

    def _export_started(self, output_path: str) -> None:
        """Report the exact partial-session output path without claiming full export."""
        self.statusBar().showMessage(f"Saving processed session: {output_path}")

    def _export_finished(self, output_path: str) -> None:
        """Confirm that the current-session file was finalized and is safe to play."""
        self.export_button.blockSignals(True)
        self.export_button.setChecked(False)
        self.export_button.setText("Save Processed Session")
        self.export_button.blockSignals(False)
        self.statusBar().showMessage(f"Processed session saved: {output_path}")

    def _show_worker_error(self, message: str) -> None:
        """Display worker failures without leaving an unresponsive application."""
        self._show_message("Processing error", message, error=True)

    def _show_message(self, title: str, message: str, error: bool = False) -> None:
        """Use a consistent friendly dialog and status-bar message."""
        self.statusBar().showMessage(message)
        if error:
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def _install_shortcuts(self) -> None:
        """Keep useful controls accessible without conflicting with text entry widgets."""
        self._shortcut("Space", self._toggle_pause)
        self._shortcut("T", lambda: self.tracking_checkbox.toggle())
        self._shortcut("+", lambda: self._adjust_confidence(CONFIDENCE_STEP))
        self._shortcut("=", lambda: self._adjust_confidence(CONFIDENCE_STEP))
        self._shortcut("-", lambda: self._adjust_confidence(-CONFIDENCE_STEP))
        self._shortcut("H", self.show_help_dialog)
        self._shortcut("Q", self.close)
        self._shortcut("Escape", self.stop_current_source)

    def _shortcut(self, sequence: str, callback) -> None:
        """Register one window-level keyboard shortcut."""
        shortcut = QShortcut(QKeySequence(sequence), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(callback)

    def _toggle_pause(self) -> None:
        """Use one primary button for play, pause, and resume without extra controls."""
        if self._current_state == "paused":
            self.request_resume.emit()
        elif self._current_state == "processing":
            self.request_pause.emit()
        else:
            self.play_current_source()

    def stop_current_source(self) -> None:
        """Stop processing and restore the neutral first-launch interface state."""
        self._ignore_worker_frames = True
        self.request_stop.emit()
        self._set_state("ready", "Ready. Select a video or webcam.")
        self._reset_to_initial_state()

    def _reset_to_initial_state(self) -> None:
        """Clear source-specific preview, statistics, and playback UI state."""
        self._latest_pixmap = None
        self._source_name = "No source selected"
        self._source_metadata = None
        self.video_label.clear()
        self.video_label.setText("Select a video file or use the webcam to begin.")
        self.source_label.setText(self._source_name)
        self.source_label.setToolTip("")
        for label in self.stats_labels.values():
            label.setText("—")
            label.setToolTip("")
        self.class_total_label.setText("Total: 0")
        while self.class_counts_layout.count():
            item = self.class_counts_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, 0)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.position_slider.blockSignals(False)
        self.position_label.setText("00:00 / —")

    def _begin_seek(self) -> None:
        """Pause file playback before moving to a different frame."""
        if self.position_slider.isEnabled() and self._current_state == "processing":
            self.request_pause.emit()

    def _preview_seek_position(self, frame_index: int) -> None:
        """Show the requested file position while the user drags the frame slider."""
        if self._source_metadata is None or self._source_metadata.fps <= 0:
            return
        self.position_label.setText(
            f"{_format_duration((frame_index + 1) / self._source_metadata.fps)} / "
            f"{_format_duration(self._source_metadata.duration_seconds)}"
        )

    def _seek_to_selected_frame(self) -> None:
        """Analyze one selected video frame; ByteTrack resets to keep seek state honest."""
        if self.position_slider.isEnabled():
            self.request_seek.emit(self.position_slider.value())

    def show_help_dialog(self) -> None:
        """Show shortcuts in an accessible native Qt dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard shortcuts")
        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Space — Play, pause, or resume\n"
                "T — Toggle tracking\n"
                "+ / = — Increase confidence by 0.05\n"
                "- — Decrease confidence by 0.05\n"
                "H — Show this help\n"
                "Q — Exit application\n"
                "Esc — Stop current source\n"
                "Frame slider — Pause and inspect a selected video frame"
            )
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def resizeEvent(self, event) -> None:
        """Reapply aspect-ratio-preserving pixmap scaling after a window resize."""
        super().resizeEvent(event)
        self._update_preview()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop OpenCV resources and wait for the worker thread before exiting Qt."""
        self.shutdown()
        event.accept()

    def shutdown(self) -> None:
        """Perform idempotent worker cleanup for close events and automated tests."""
        if self._worker_thread is None or not self._worker_thread.isRunning():
            return
        QMetaObject.invokeMethod(
            self._worker,
            "shutdown_worker",
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker_thread.wait(5000)


def _format_duration(seconds: float | None) -> str:
    """Format a duration safely for file playback information."""
    if seconds is None:
        return "—"
    whole_seconds = max(0, round(seconds))
    return f"{whole_seconds // 60:02}:{whole_seconds % 60:02}"


def _source_summary(source_name: str) -> str:
    """Keep the narrow live-statistics source row readable without losing its tooltip."""
    if source_name.startswith("Webcam"):
        return source_name
    return Path(source_name).name or source_name


_APPLICATION_STYLESHEET = """
QWidget { color: #e6eef2; font-family: Segoe UI, Arial; font-size: 13px; }
QMainWindow, QWidget#centralWidget { background: #151b20; }
QLabel { background: transparent; }
QFrame#header, QFrame#statisticsSidebar, QFrame#controlArea { background: #202a31; border: 1px solid #334650; border-radius: 12px; }
QLabel#titleLabel { font-size: 25px; font-weight: 700; color: #f4fbfd; }
QLabel#subtitleLabel, QLabel#statusSummary, QLabel#sourceLabel, QLabel#exportHint { color: #9fb2bc; }
QLabel#stateLabel { color: #55d6d2; font-weight: 700; font-size: 13px; letter-spacing: 1px; }
QLabel#stateLabel[state="error"] { color: #ff8f7d; }
QLabel#stateLabel[state="paused"] { color: #ffc96b; }
QLabel#sectionTitle { color: #55d6d2; font-weight: 700; letter-spacing: 1px; }
QLabel#statValue { color: #f4fbfd; font-weight: 600; }
QLabel#classTotal { color: #9fb2bc; font-size: 12px; font-weight: 600; }
QLabel#classCount { background: #182229; border-radius: 5px; padding: 2px 7px; color: #d8e8ed; }
QLabel#emptyObjects { color: #9fb2bc; }
QLabel#videoPreview { background: #080b0d; border: 1px solid #334650; border-radius: 12px; color: #82949d; }
QGroupBox { background: #1a242a; border: 1px solid #3a4c55; border-radius: 9px; margin-top: 10px; padding: 12px 10px 8px 10px; color: #55d6d2; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #263841; border: 1px solid #43626c; border-radius: 7px; padding: 8px 11px; color: #eef9fb; font-weight: 600; }
QPushButton:hover { background: #31505a; border-color: #55d6d2; }
QPushButton:checked { background: #126d70; border-color: #55d6d2; }
QCheckBox { spacing: 7px; color: #d8e8ed; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #66818c; border-radius: 4px; background: #19242a; }
QCheckBox::indicator:checked { background: #35aeb0; border-color: #55d6d2; }
QSlider::groove:horizontal { height: 5px; background: #344850; border-radius: 2px; }
QSlider::handle:horizontal { background: #55d6d2; width: 16px; margin: -6px 0; border-radius: 8px; }
QProgressBar { border: 1px solid #405861; border-radius: 5px; text-align: center; background: #152027; }
QProgressBar::chunk { background: #2a9da0; border-radius: 4px; }
QScrollArea#classScroll, QScrollArea#classScroll QWidget, QWidget#classContainer { background: transparent; }
QStatusBar { background: #12181d; color: #9fb2bc; }
"""

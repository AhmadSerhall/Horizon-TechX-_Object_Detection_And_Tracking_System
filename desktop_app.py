"""Launch the PySide6 desktop interface for the detection and tracking system."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.gui import MainWindow


def main() -> int:
    """Create and run the native desktop application event loop."""
    application = QApplication(sys.argv)
    application.setApplicationName("Object Detection & Tracking System")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

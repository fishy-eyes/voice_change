"""Voice Changer GUI entry point.

Run standalone:
    python -m gui.app

Run with context (from main.py integration):
    from gui.app import create_app
    app, window = create_app(context)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional, Tuple

from PySide6.QtWidgets import QApplication

from config.settings import APP_NAME, APP_VERSION
from gui.main_window import MainWindow

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
    from core.context import AppContext


def create_app(
    context: Optional[AppContext] = None,
    argv: Optional[list] = None,
) -> Tuple[QApplication, QWidget]:
    """Create the Qt application and main window.

    Parameters
    ----------
    context : AppContext, optional
        Shared runtime references. Pass None for standalone mode.
    argv : list, optional
        Command-line arguments for QApplication. Defaults to sys.argv.

    Returns
    -------
    tuple[QApplication, MainWindow]
        The app instance and the main window.
    """
    app = QApplication(argv or sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    window = MainWindow(context=context)
    window.show()
    return app, window


def main() -> None:
    """Standalone entry point - no audio context."""
    app, _window = create_app()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

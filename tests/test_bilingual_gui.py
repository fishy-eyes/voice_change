"""Compact control-center language and visibility checks."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


class BilingualMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_adaptive_layout_and_language_toggle(self) -> None:
        window = MainWindow(None)
        try:
            self.assertIn("实时", window.windowTitle())
            self.assertFalse(window._details_widget.isVisible())
            self.assertTrue(window._scroll_area.widgetResizable())
            self.assertFalse(hasattr(window, "_rvc_pitch_slider"))
            self.assertTrue(hasattr(window._voice_panel, "settings_button"))
            self.assertEqual(window._gain_group.title(), "输出音量")
            window._toggle_language()
            self.assertEqual(window.windowTitle(), "Realtime AI Voice Control Center")
            self.assertEqual(window._start_button.text(), "Start Audio")
            self.assertEqual(window._gain_group.title(), "Output Gain")
            window._toggle_language()
            self.assertIn("实时", window.windowTitle())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

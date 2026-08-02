"""Offscreen checks for the two-column bilingual main window."""

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

    def test_horizontal_two_column_layout_and_language_toggle(self) -> None:
        window = MainWindow(None)
        try:
            window.show()
            QApplication.processEvents()

            self.assertGreater(window.width(), window.height())
            self.assertEqual(window._content_layout.count(), 2)
            self.assertEqual(window._language, "en")
            self.assertEqual(window.windowTitle(), "Voice Changer")
            self.assertEqual(window._device_group.title(), "Device Selection")
            self.assertEqual(window._language_button.text(), "中文")
            self.assertIn("lower voice", window._pitch_help_label.text())
            self.assertTrue(window._rvc_pitch_label.text().startswith("Pitch:"))
            self.assertEqual(window._rvc_pitch_slider.minimum(), -24)
            self.assertEqual(window._rvc_pitch_slider.maximum(), 24)
            window._rvc_pitch_slider.setValue(14)
            self.assertIn("+14", window._rvc_pitch_label.text())
            self.assertTrue(
                window._rvc_index_label.text().startswith("Index Rate:")
            )

            window._language_button.click()
            QApplication.processEvents()

            self.assertEqual(window._language, "zh")
            self.assertEqual(window.windowTitle(), "实时 AI 变声器")
            self.assertEqual(window._device_group.title(), "设备选择")
            self.assertEqual(window._effects_group.title(), "基础效果")
            self.assertEqual(window._language_button.text(), "English")
            self.assertIn("更低沉", window._pitch_help_label.text())
            self.assertIn("辅音", window._protect_help_label.text())
            self.assertEqual(window._ai_voice_panel.title(), "AI Voice")
            self.assertEqual(
                window._ai_voice_panel.import_button.text(),
                "导入 RVC 模型",
            )
            self.assertEqual(window._self_monitor_panel.title(), "自监听")
            self.assertTrue(window._rvc_pitch_label.text().startswith("Pitch:"))
            self.assertTrue(
                window._rvc_rms_label.text().startswith("RMS Mix Rate:")
            )

            window._language_button.click()
            QApplication.processEvents()
            self.assertEqual(window._language, "en")
            self.assertEqual(window.windowTitle(), "Voice Changer")
            self.assertEqual(
                window._ai_voice_panel.import_button.text(),
                "Import RVC Model",
            )
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

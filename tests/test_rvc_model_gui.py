"""Headless checks for RVC model selection controls."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.rvc_control_panel import RVCControlPanel


class FakeRuntime:
    def __init__(self) -> None:
        self.enabled = False
        self.selected_model = None
        self.state = SimpleNamespace(ready=False, error=None)
        self.model_manager = SimpleNamespace(
            discover_models=lambda: [SimpleNamespace(name="modelF")]
        )
        self.loads: list[tuple[str, object]] = []

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def load_model(self, name: str, *, audio_stream=None):
        self.loads.append((name, audio_stream))
        self.selected_model = name
        self.state = SimpleNamespace(ready=True, error=None)
        return self.state


class RVCModelGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_lists_loads_and_toggles_model(self) -> None:
        runtime = FakeRuntime()
        stream = object()
        context = SimpleNamespace(rvc_runtime=runtime, audio_stream=stream)
        panel = RVCControlPanel(context)

        self.assertEqual(panel.model_combo.currentData(), "modelF")
        panel.enable_checkbox.setChecked(True)
        panel.load_button.click()

        self.assertTrue(runtime.enabled)
        self.assertEqual(runtime.loads, [("modelF", stream)])
        self.assertEqual(panel.status_label.text(), "Loaded: modelF (Enabled)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

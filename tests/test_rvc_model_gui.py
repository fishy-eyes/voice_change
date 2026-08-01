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
        self.realtime_preset_key = "balanced"
        self.realtime_calls: list[str] = []

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def load_model(self, name: str, *, audio_stream=None):
        self.loads.append((name, audio_stream))
        self.selected_model = name
        self.state = SimpleNamespace(ready=True, error=None)
        return self.state

    def set_realtime_preset(self, key: str) -> None:
        self.realtime_calls.append(key)
        self.realtime_preset_key = key


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
        self.assertEqual(
            [panel.realtime_combo.itemText(index) for index in range(3)],
            ["Low Latency", "Balanced", "High Quality"],
        )
        self.assertEqual(panel.realtime_combo.currentData(), "balanced")
        self.assertIn(
            "500ms chunk / 50ms overlap", panel.realtime_detail_label.text()
        )
        panel.realtime_combo.setCurrentIndex(
            panel.realtime_combo.findData("low_latency")
        )
        self.assertEqual(runtime.realtime_calls, ["low_latency"])
        self.assertIn(
            "325ms chunk / 50ms overlap", panel.realtime_detail_label.text()
        )

        panel.realtime_combo.setCurrentIndex(
            panel.realtime_combo.findData("high_quality")
        )
        self.assertEqual(
            runtime.realtime_calls, ["low_latency", "high_quality"]
        )
        self.assertIn(
            "500ms chunk / 100ms overlap", panel.realtime_detail_label.text()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

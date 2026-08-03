"""Headless GUI checks for backend selection and async model loading."""

from __future__ import annotations

import os
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from ai.voice_conversion_manager import VoiceConversionStatus
from gui.voice_conversion_panel import VoiceConversionPanel


class FakeManager:
    def __init__(self) -> None:
        self.available_backends = ("rvc",)
        self.current_backend = "rvc"
        self.requested_enabled = True
        self.model = None
        self.loads = []

    def discover_models(self):
        return [SimpleNamespace(name="modelF")]

    def select_backend(self, backend):
        self.current_backend = backend

    def set_enabled(self, enabled):
        self.requested_enabled = bool(enabled)

    def switch_model_async(self, backend, model, *, audio_stream=None):
        self.loads.append((backend, model, audio_stream))
        self.model = model
        future = Future()
        future.set_result(SimpleNamespace(ready=True, error=None))
        return future

    def get_status(self):
        return VoiceConversionStatus(
            backend=self.current_backend,
            model=self.model,
            state="LOADED" if self.model else "IDLE",
            enabled=bool(self.model and self.requested_enabled),
            latency_ms=45.0,
        )


class VoiceConversionGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_panel_lists_backend_and_loads_without_duplicate_status(self) -> None:
        manager = FakeManager()
        stream = object()
        panel = VoiceConversionPanel(
            SimpleNamespace(voice_conversion_manager=manager, audio_stream=stream)
        )
        try:
            panel.load_button.click()
            self.app.processEvents()
            panel._poll_model_switch()
            self.assertEqual(panel.backend_combo.currentData(), "rvc")
            self.assertEqual(manager.loads, [("rvc", "modelF", stream)])
            self.assertFalse(hasattr(panel, "status_label"))
            visible_labels = [label.text() for label in panel.findChildren(QLabel)]
            self.assertNotIn("Backend: RVC", visible_labels)
            self.assertNotIn("Model: modelF", visible_labels)
            self.assertTrue(panel.settings_button.isEnabled())
        finally:
            panel.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Backend-neutral model-folder capability checks for the main panel."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.voice_conversion_panel import VoiceConversionPanel


class CapabilityManager:
    def __init__(self) -> None:
        self.available_backends = ("rvc", "beatrice")
        self.current_backend = "rvc"
        self.requested_enabled = False
        self.added = []

    def select_backend(self, backend):
        self.current_backend = backend

    def discover_models(self):
        return (
            [SimpleNamespace(name="modelF")]
            if self.current_backend == "rvc"
            else []
        )

    def get_preferred_model(self):
        return None

    def supports_model_folder_import(self, backend=None):
        return backend == "beatrice"

    def add_model_path(self, path, backend=None):
        self.added.append((Path(path).resolve(), backend))
        return SimpleNamespace(name="jvs"), True

    def validate_configuration(self, model, backend=None):
        if backend == "beatrice" and model is None:
            raise RuntimeError(
                "Please select a Beatrice Runtime folder in Model Settings first."
            )

    def set_enabled(self, enabled):
        self.requested_enabled = bool(enabled)


class VoiceConversionFolderCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_rvc_hides_generic_folder_button_and_beatrice_exposes_it(self) -> None:
        manager = CapabilityManager()
        panel = VoiceConversionPanel(
            SimpleNamespace(voice_conversion_manager=manager, audio_stream=None),
            language="en",
        )
        try:
            self.assertFalse(panel.add_model_button.isVisible())
            panel.show()
            beatrice_index = panel.backend_combo.findData("beatrice")
            panel.backend_combo.setCurrentIndex(beatrice_index)
            self.app.processEvents()
            self.assertTrue(panel.add_model_button.isVisible())
            with tempfile.TemporaryDirectory() as temp:
                with patch(
                    "gui.voice_conversion_panel.QFileDialog.getExistingDirectory",
                    return_value=temp,
                ):
                    panel._add_model_folder()
                self.assertEqual(manager.added, [(Path(temp).resolve(), "beatrice")])
        finally:
            panel.close()

    def test_configuration_error_is_shown_before_worker_switch(self) -> None:
        manager = CapabilityManager()
        manager.current_backend = "beatrice"
        panel = VoiceConversionPanel(
            SimpleNamespace(voice_conversion_manager=manager, audio_stream=None),
            language="en",
        )
        try:
            with patch("gui.voice_conversion_panel.QMessageBox.warning") as warning:
                panel._load_selected_model()
            warning.assert_called_once()
            self.assertIn("Runtime folder", panel.toolTip())
        finally:
            panel.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

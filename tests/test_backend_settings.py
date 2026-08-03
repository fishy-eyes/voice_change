"""Backend settings registry and RVC panel tests."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.backend_settings import create_default_registry


class FakeModelManager:
    def __init__(self) -> None:
        self.imports = []

    def inspect_import_directory(self, directory):
        return SimpleNamespace(
            directory=Path(directory),
            pth_candidates=(Path(directory) / "voice.pth",),
            index_candidates=(),
        )

    def import_model(self, directory, *, pth_path, index_path):
        self.imports.append((directory, pth_path, index_path))
        return SimpleNamespace(name="imported-voice")


class FakeManager:
    def __init__(self) -> None:
        self.parameters = {
            "pitch_shift": 2,
            "index_rate": 0.7,
            "protect": 0.3,
            "rms_mix_rate": 0.25,
        }
        self.preset = "balanced"
        self.descriptor = SimpleNamespace(name="modelF")
        self.updates = []

        self.model_manager = FakeModelManager()
    def get_current_parameters(self):
        return dict(self.parameters)

    def update_current_parameters(self, **changes):
        self.parameters.update(changes)
        self.updates.append(changes)
        return dict(self.parameters)

    def get_realtime_preset(self):
        return self.preset

    def set_realtime_preset(self, key):
        self.preset = key

    def get_current_model_descriptor(self):
        return self.descriptor

    @property
    def current_runtime(self):
        return SimpleNamespace(model_manager=self.model_manager)

    def get_status(self):
        return SimpleNamespace(state="LOADED")


class BackendSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_registry_only_exposes_real_rvc_panel(self) -> None:
        registry = create_default_registry()
        self.assertEqual(registry.backend_ids, ("rvc",))
        with self.assertRaises(LookupError):
            registry.create("beatrice", manager=FakeManager())

    def test_rvc_panel_reads_updates_and_refreshes_runtime_state(self) -> None:
        manager = FakeManager()
        panel = create_default_registry().create(
            "rvc", manager=manager, context=None, language="en"
        )
        try:
            self.assertEqual(panel.backend_id, "rvc")
            self.assertEqual(panel.pitch_spin.value(), 2)
            self.assertTrue(panel.customize_button.isEnabled())
            panel.pitch_spin.setValue(7)
            self.assertEqual(manager.parameters["pitch_shift"], 7)
            panel.realtime_combo.setCurrentIndex(
                panel.realtime_combo.findData("low_latency")
            )
            self.assertEqual(manager.preset, "low_latency")
            manager.parameters["pitch_shift"] = -3
            panel.refresh_from_runtime()
            self.assertEqual(panel.pitch_spin.value(), -3)
        finally:
            self.assertTrue(panel.import_button.isEnabled())
            panel.close_panel()

    def test_rvc_import_entry_preserves_existing_model_import_flow(self) -> None:
        manager = FakeManager()
        refreshed = []
        panel = create_default_registry().create(
            "rvc",
            manager=manager,
            context=None,
            language="en",
            on_models_changed=lambda: refreshed.append(True),
        )
        try:
            with patch(
                "gui.backend_settings.rvc.QFileDialog.getExistingDirectory",
                return_value="external-model",
            ):
                panel._import_model()
            self.assertEqual(len(manager.model_manager.imports), 1)
            self.assertEqual(refreshed, [True])
            self.assertIn("imported-voice", panel.status_label.text())
        finally:
            panel.close_panel()

if __name__ == "__main__":
    unittest.main(verbosity=2)

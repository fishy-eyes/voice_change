"""Beatrice settings registry, controls and bilingual copy."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.backend_settings import create_default_registry
from gui.beatrice_customization_dialog import BeatriceCustomizationDialog
from gui.i18n import tr


class FakeBeatriceManager:
    def __init__(self) -> None:
        self.parameters = {
            "target_speaker": 0,
            "pitch_shift_semitone": 0.0,
            "formant_shift": 0.0,
            "min_source_pitch": 30.0,
            "max_source_pitch": 1100.0,
            "vq_num_neighbors": 4,
        }
        self.descriptor = SimpleNamespace(
            model_name="JVS corpus",
            model_api_version="2.0.0-rc.0",
            runtime_requirement="2.0.0-rc.0",
            speaker_count=2,
            speaker_names=("jvs001", "jvs002"),
        )
        self.updates = []

    def get_current_parameters(self):
        return dict(self.parameters)

    def update_current_parameters(self, **changes):
        self.parameters.update(changes)
        self.updates.append(changes)
        return dict(self.parameters)

    def get_current_model_descriptor(self):
        return self.descriptor

    def get_status(self):
        return SimpleNamespace(state="LOADED")

    def get_info(self):
        return {
            "runtime": {
                "model_api_version": "2.0.0-rc.0",
                "runtime_implementation_version": "2.0.0-rc.2",
            }
        }


class BeatriceGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_registered_panel_updates_confirmed_parameters(self) -> None:
        manager = FakeBeatriceManager()
        panel = create_default_registry().create(
            "beatrice", manager=manager, language="en"
        )
        try:
            self.assertEqual(panel.backend_id, "beatrice")
            self.assertEqual(panel.target_combo.count(), 2)
            self.assertIn("2.0.0-rc.0", panel.info_label.text())
            self.assertIn("Model API", panel.info_label.text())
            self.assertNotIn("2.0.0-rc.2", panel.info_label.text())
            self.assertTrue(panel.customize_button.isEnabled())
            panel.target_combo.setCurrentIndex(1)
            panel.pitch_spin.setValue(3.5)
            panel.formant_spin.setValue(-1.0)
            panel.neighbors_spin.setValue(8)
            self.assertEqual(manager.parameters["target_speaker"], 1)
            self.assertEqual(manager.parameters["pitch_shift_semitone"], 3.5)
            self.assertEqual(manager.parameters["formant_shift"], -1.0)
            self.assertEqual(manager.parameters["vq_num_neighbors"], 8)
        finally:
            panel.close_panel()

    def test_beatrice_copy_is_bilingual(self) -> None:
        self.assertEqual(tr("en", "beatrice.target_speaker"), "Target Speaker")
        self.assertEqual(tr("zh", "beatrice.target_speaker"), "目标说话人")
        self.assertIn("Beatrice", tr("zh", "beatrice.load_first"))
        self.assertIn(
            "Detected Pitch Range",
            tr("en", "beatrice.detected_pitch_range", low="100", high="200"),
        )
        self.assertIn(
            "保持当前设置",
            tr(
                "zh",
                "beatrice.source_pitch_keep_current",
                low="30",
                high="1100",
            ),
        )
        self.assertNotIn("source_pitch", BeatriceCustomizationDialog.STAGE_NAMES)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Output Gain configuration and ordering tests without audio hardware."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from config.settings import APP_VERSION
from effects.base import BaseEffect
from effects.gain import GainEffect
from gui.main_window import MainWindow
from main import create_effect_manager


class AddEffect(BaseEffect):
    def process(self, audio_data, frames, time_info, status):
        return audio_data + 0.1


class MonitorTap:
    def __init__(self) -> None:
        self.blocks = []

    def submit(self, audio) -> None:
        self.blocks.append(np.array(audio, copy=True))


class OutputGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_default_chain_contains_only_unity_output_gain(self) -> None:
        manager = create_effect_manager()
        self.assertEqual([effect.name for effect in manager.effects], ["GainEffect"])
        gain = manager.effects[0]
        self.assertTrue(gain.enabled)
        self.assertEqual(gain.gain, 1.0)

    def test_gain_range_clipping_and_cannot_be_disabled(self) -> None:
        gain = GainEffect(2.0)
        audio = np.array([[-0.75], [0.25], [0.75]], dtype=np.float32)
        output = gain.process(audio, len(audio), None, None)
        np.testing.assert_allclose(output[:, 0], [-1.0, 0.5, 1.0])
        gain.enabled = False
        self.assertTrue(gain.enabled)
        with self.assertRaises(ValueError):
            gain.gain = 3.01
        gain.gain = 0.0
        np.testing.assert_array_equal(
            gain.process(audio, len(audio), None, None), np.zeros_like(audio)
        )

    def test_ai_precedes_gain_and_monitor_receives_final_output(self) -> None:
        monitor = MonitorTap()
        manager = create_effect_manager(AddEffect(), self_monitor=monitor)
        self.assertEqual(
            [effect.name for effect in manager.effects],
            ["AddEffect", "GainEffect"],
        )
        manager.get_by_name("GainEffect").gain = 2.0
        audio = np.full((4, 1), 0.2, dtype=np.float32)
        output = manager.process(audio, 4, None, None)
        np.testing.assert_allclose(output, 0.6)
        np.testing.assert_array_equal(monitor.blocks[-1], output)

    def test_ai_absent_still_applies_output_gain_and_gui_has_no_toggle(self) -> None:
        manager = create_effect_manager()
        context = SimpleNamespace(
            effect_manager=manager,
            audio_stream=None,
            device_manager=None,
            voice_conversion_manager=None,
            self_monitor=None,
        )
        window = MainWindow(context)
        try:
            self.assertEqual(window._gain_slider.minimum(), 0)
            self.assertEqual(window._gain_slider.maximum(), 30)
            window._gain_slider.setValue(25)
            self.assertEqual(manager.get_by_name("GainEffect").gain, 2.5)
            self.assertNotIn("Echo", window.windowTitle())
            self.assertNotIn("Robot", window.windowTitle())
            self.assertIn(f"v{APP_VERSION}", window.windowTitle())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

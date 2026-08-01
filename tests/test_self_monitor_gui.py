"""Offscreen tests for self-monitor GUI controls."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.self_monitor_panel import SelfMonitorPanel


class FakeMonitor:
    def __init__(self) -> None:
        self.volume = 0.5
        self.is_running = False
        self.output_device = None
        self.starts: list[object] = []
        self.stop_count = 0

    def start(self, *, output_device=None) -> None:
        self.starts.append(output_device)
        self.is_running = True
        self.output_device = output_device

    def stop(self) -> None:
        self.stop_count += 1
        self.is_running = False
        self.output_device = None


class FakeDeviceManager:
    @staticmethod
    def list_output_devices():
        return [
            SimpleNamespace(index=7, name="Headphones"),
            SimpleNamespace(index=9, name="CABLE Input (VB-Audio Virtual Cable)"),
        ]

    @staticmethod
    def find_virtual_input_device():
        return 4


class SelfMonitorGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_enable_volume_and_disable(self) -> None:
        monitor = FakeMonitor()
        context = SimpleNamespace(
            self_monitor=monitor,
            device_manager=FakeDeviceManager,
            output_device=9,
        )
        panel = SelfMonitorPanel(context)

        self.assertEqual(panel.device_combo.count(), 2)
        self.assertEqual(panel.device_combo.itemData(1), 7)
        panel.device_combo.setCurrentIndex(1)
        panel.volume_slider.setValue(35)
        self.assertEqual(monitor.volume, 0.35)

        panel.enable_checkbox.setChecked(True)
        self.assertEqual(monitor.starts, [7])
        self.assertFalse(panel.device_combo.isEnabled())
        self.assertIn("Headphones", panel.status_label.text())

        panel.enable_checkbox.setChecked(False)
        self.assertEqual(monitor.stop_count, 1)
        self.assertTrue(panel.device_combo.isEnabled())
        self.assertEqual(panel.status_label.text(), "Disabled")


if __name__ == "__main__":
    unittest.main(verbosity=2)

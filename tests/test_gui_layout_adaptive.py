"""Adaptive main-window layout and runtime-status regression tests."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)

from ai.voice_conversion_manager import VoiceConversionStatus
from gui.main_window import MainWindow


class FakeStream:
    def __init__(self) -> None:
        self.is_running = False
        self._callback_count = 4
        self._total_proc_ms = 20.0
        self._last_proc_ms = 4.5
        self._max_proc_ms = 7.25
        self._input_rms = 0.125
        self._output_rms = 0.25

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False


class FakeMonitor:
    def __init__(self) -> None:
        self.is_running = False
        self.output_device = None
        self.volume = 0.5

    def start(self, *, output_device=None) -> None:
        self.output_device = output_device
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False


class FakeManager:
    def __init__(self, model_name: str) -> None:
        self.available_backends = ("rvc",)
        self.current_backend = "rvc"
        self.requested_enabled = True
        self.model_name = model_name
        worker = SimpleNamespace(
            is_running=True,
            is_inferencing=False,
            infer_count=12,
            error_count=1,
        )
        effect = SimpleNamespace(worker=worker)
        self.current_runtime = SimpleNamespace(
            state=SimpleNamespace(effect=effect)
        )

    def discover_models(self):
        return [SimpleNamespace(name=self.model_name)]

    def select_backend(self, backend):
        self.current_backend = backend

    def set_enabled(self, enabled):
        self.requested_enabled = bool(enabled)

    def get_status(self):
        return VoiceConversionStatus(
            backend=self.current_backend,
            model=self.model_name,
            state="LOADED",
            enabled=self.requested_enabled,
            latency_ms=2477.7,
        )


class FakeEffectManager:
    def __init__(self) -> None:
        self.gain = SimpleNamespace(gain=1.0, enabled=True)

    def get_by_name(self, name):
        return self.gain if name == "GainEffect" else None


class FakeDeviceManager:
    def list_output_devices(self):
        return []


def make_context(model_name: str = "ruanmei_v2"):
    return SimpleNamespace(
        voice_conversion_manager=FakeManager(model_name),
        audio_stream=FakeStream(),
        self_monitor=FakeMonitor(),
        effect_manager=FakeEffectManager(),
        device_manager=FakeDeviceManager(),
        output_device=None,
    )


class AdaptiveMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow(make_context())
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def _visible_texts(self):
        texts = []
        for label in self.window.findChildren(QLabel):
            if label.isVisible():
                texts.append(label.text())
        for control_type in (QPushButton, QCheckBox):
            for control in self.window.findChildren(control_type):
                if control.isVisible():
                    texts.append(control.text())
        for group in self.window.findChildren(QGroupBox):
            if group.isVisible():
                texts.append(group.title())
        return texts

    def test_voice_panel_contains_operations_not_duplicate_runtime_status(self) -> None:
        panel = self.window._voice_panel
        self.assertFalse(hasattr(panel, "status_label"))
        self.assertEqual(panel.layout().rowCount(), 4)
        self.assertEqual(panel.backend_label.text(), "模型类型")
        self.assertEqual(panel.model_label.text(), "模型")

    def test_runtime_fields_and_details_layout_order(self) -> None:
        self.assertEqual(
            set(self.window._status_value_labels), set(MainWindow._STATUS_FIELDS)
        )
        layout = self.window._status_group.layout()
        self.assertIs(layout.itemAt(0).layout(), self.window._status_grid)
        self.assertIs(layout.itemAt(1).widget(), self.window._details_button)
        self.assertIs(layout.itemAt(2).widget(), self.window._details_widget)
        latency = self.window._status_value_labels["latency"]
        self.assertGreaterEqual(
            self.window._details_button.geometry().top(), latency.geometry().bottom()
        )

        self.window._details_button.setChecked(True)
        self.app.processEvents()
        self.assertTrue(self.window._details_widget.isVisible())
        self.assertTrue(
            all(label.isVisible() for label in self.window._detail_value_labels.values())
        )
        self.assertEqual(self.window._detail_value_labels["infer_count"].text(), "12")
        self.assertEqual(self.window._detail_value_labels["error_count"].text(), "1")

        self.window._details_button.setChecked(False)
        self.app.processEvents()
        self.assertFalse(self.window._details_widget.isVisible())
        self.assertTrue(
            all(not label.isVisible() for label in self.window._detail_value_labels.values())
        )

    def test_scroll_area_and_multiple_window_sizes(self) -> None:
        self.assertIsInstance(self.window._scroll_area, QScrollArea)
        self.assertTrue(self.window._scroll_area.widgetResizable())
        self.assertEqual(
            self.window._scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        for width, height in ((776, 792), (700, 650), (900, 700), (1100, 850)):
            self.window.resize(width, height)
            self.window._content_layout.activate()
            self.window._status_group.layout().activate()
            self.app.processEvents()
            self.assertGreater(self.window._details_button.width(), 0)
            self.assertGreater(self.window._status_group.minimumSizeHint().height(), 0)
        self.window.resize(700, 650)
        self.app.processEvents()
        self.assertGreater(
            self.window._scroll_area.verticalScrollBar().maximum(), 0
        )

    def test_language_switch_is_complete_immediate_and_does_not_duplicate_widgets(self) -> None:
        before = len(self.window.findChildren(QWidget))
        chinese = "\n".join(self._visible_texts())
        for forbidden in (
            "Output Gain",
            "Model Status",
            "Audio:",
            "Monitor:",
            "AI Latency",
            "Volume:",
        ):
            self.assertNotIn(forbidden, chinese)
        self.assertIn("输出音量", chinese)
        self.assertIn("监听音量：50%", chinese)
        self.assertEqual(
            self.window._status_value_labels["model_status"].text(), "已加载"
        )

        self.window._toggle_language()
        self.app.processEvents()
        english = "\n".join(self._visible_texts())
        for expected in (
            "Output Gain",
            "Model Status:",
            "Audio:",
            "Monitor:",
            "AI Latency:",
            "Volume: 50%",
        ):
            self.assertIn(expected, english)
        self.assertEqual(
            self.window._status_value_labels["model_status"].text(), "Loaded"
        )

        for _ in range(5):
            self.window._toggle_language()
            self.app.processEvents()
        self.assertEqual(len(self.window.findChildren(QWidget)), before)

    def test_status_updates_preserve_hierarchy_and_long_model_keeps_actions(self) -> None:
        old_window = self.window
        old_window.close()
        long_name = "ruanmei_v2_" + "very_long_voice_model_name_" * 4
        self.window = MainWindow(make_context(long_name))
        self.window.resize(700, 650)
        self.window.show()
        self.app.processEvents()

        status_layout = self.window._status_group.layout()
        item_count = status_layout.count()
        button_index = status_layout.indexOf(self.window._details_button)
        for _ in range(8):
            self.window._update_status_display()
        self.app.processEvents()
        self.assertEqual(status_layout.count(), item_count)
        self.assertEqual(status_layout.indexOf(self.window._details_button), button_index)
        self.assertEqual(
            self.window._status_value_labels["model"].text(), long_name
        )

        panel = self.window._voice_panel
        self.assertEqual(panel.model_combo.currentText(), long_name)
        self.assertTrue(panel.refresh_button.isVisible())
        self.assertTrue(panel.settings_button.isVisible())
        self.assertLess(
            panel.model_combo.geometry().right(), panel.refresh_button.geometry().left()
        )
        self.assertLessEqual(
            panel.settings_button.geometry().bottom(), panel.rect().bottom()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

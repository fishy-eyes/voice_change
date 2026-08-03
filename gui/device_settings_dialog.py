"""Audio device selection kept outside the compact main window."""

from __future__ import annotations

from loguru import logger
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from core.device_switching import switch_audio_devices


class DeviceSettingsDialog(QDialog):
    def __init__(self, context=None, *, language="zh", parent=None) -> None:
        super().__init__(parent)
        self._context = context
        self._language = language
        self.setWindowTitle("音频设备" if language == "zh" else "Audio Devices")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        form.addRow("输入设备" if language == "zh" else "Input", self.input_combo)
        form.addRow("输出设备" if language == "zh" else "Output", self.output_combo)
        layout.addLayout(form)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        refresh = QPushButton("刷新设备" if language == "zh" else "Refresh")
        apply = QPushButton("应用" if language == "zh" else "Apply")
        close = QPushButton("关闭" if language == "zh" else "Close")
        refresh.clicked.connect(self.refresh_devices)
        apply.clicked.connect(self.apply_devices)
        close.clicked.connect(self.close)
        buttons.addWidget(refresh)
        buttons.addWidget(apply)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.refresh_devices()

    def refresh_devices(self) -> None:
        manager = getattr(self._context, "device_manager", None)
        if manager is None:
            self.setEnabled(False)
            self.status_label.setText("Device manager unavailable")
            return
        input_selected = (
            self.input_combo.currentData()
            if self.input_combo.count()
            else getattr(self._context, "input_device", None)
        )
        output_selected = (
            self.output_combo.currentData()
            if self.output_combo.count()
            else getattr(self._context, "output_device", None)
        )
        self.input_combo.clear()
        self.output_combo.clear()
        try:
            for device in manager.list_input_devices():
                self.input_combo.addItem(f"[{device.index}] {device.name}", device.index)
            for device in manager.list_output_devices():
                self.output_combo.addItem(f"[{device.index}] {device.name}", device.index)
        except Exception as exc:
            logger.error("device refresh failed: {}", exc)
            self.status_label.setText(str(exc))
            return
        input_index = self.input_combo.findData(input_selected)
        output_index = self.output_combo.findData(output_selected)
        self.input_combo.setCurrentIndex(input_index if input_index >= 0 else 0)
        self.output_combo.setCurrentIndex(output_index if output_index >= 0 else 0)
        self.status_label.setText("设备列表已刷新。" if self._language == "zh" else "Device list refreshed.")

    def apply_devices(self) -> None:
        if self._context is None or self.input_combo.currentIndex() < 0 or self.output_combo.currentIndex() < 0:
            return
        result = switch_audio_devices(
            self._context,
            self.input_combo.currentData(),
            self.output_combo.currentData(),
        )
        if result.success:
            self.status_label.setText("设备已应用。" if self._language == "zh" else "Devices applied.")
        else:
            self.status_label.setText(str(result.error or "Device switch failed"))

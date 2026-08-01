"""Small developer-facing self-monitor control panel."""

from __future__ import annotations

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)


class SelfMonitorPanel(QGroupBox):
    """GUI adapter for the application-owned independent monitor stream."""

    def __init__(self, context=None) -> None:
        super().__init__("Self Monitor")
        self._context = context

        layout = QVBoxLayout(self)
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_checkbox)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Monitor Device:"))
        self.device_combo = QComboBox()
        device_row.addWidget(self.device_combo)
        layout.addLayout(device_row)

        volume_row = QHBoxLayout()
        self.volume_label = QLabel("Volume: 50%")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setSingleStep(1)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        volume_row.addWidget(self.volume_label)
        volume_row.addWidget(self.volume_slider)
        layout.addLayout(volume_row)

        self.status_label = QLabel("Disabled")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.refresh_devices()

    def _monitor(self):
        return (
            getattr(self._context, "self_monitor", None)
            if self._context is not None
            else None
        )

    def _device_manager(self):
        return (
            getattr(self._context, "device_manager", None)
            if self._context is not None
            else None
        )

    @staticmethod
    def _is_virtual_cable_output(name: str) -> bool:
        lower = name.lower()
        return "cable input" in lower or "vb-audio virtual cable" in lower

    def refresh_devices(self) -> None:
        monitor = self._monitor()
        manager = self._device_manager()
        selected_output = getattr(monitor, "output_device", None)
        self.device_combo.clear()
        if monitor is None or manager is None:
            self.setEnabled(False)
            self.status_label.setText("Self monitor unavailable")
            return

        self.setEnabled(True)
        self.device_combo.addItem("System Default", None)
        primary_output = getattr(self._context, "output_device", None)
        try:
            devices = manager.list_output_devices()
        except Exception as exc:
            logger.error("self-monitor device discovery failed: {}", exc)
            self.status_label.setText(f"Device discovery failed: {exc}")
            devices = []
        for device in devices:
            if device.index == primary_output:
                continue
            if self._is_virtual_cable_output(device.name):
                continue
            self.device_combo.addItem(f"[{device.index}] {device.name}", device.index)

        selected_index = self.device_combo.findData(selected_output)
        self.device_combo.setCurrentIndex(selected_index if selected_index >= 0 else 0)

        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(round(monitor.volume * 100))
        self.volume_slider.blockSignals(False)
        self._update_volume_label()
        self.enable_checkbox.blockSignals(True)
        self.enable_checkbox.setChecked(bool(monitor.is_running))
        self.enable_checkbox.blockSignals(False)
        self.device_combo.setEnabled(not monitor.is_running)
        self.status_label.setText("Monitoring" if monitor.is_running else "Disabled")

    def _on_enabled_toggled(self, enabled: bool) -> None:
        monitor = self._monitor()
        manager = self._device_manager()
        if monitor is None or manager is None:
            return
        if not enabled:
            monitor.stop()
            self.device_combo.setEnabled(True)
            self.status_label.setText("Disabled")
            return

        try:
            output_device = self.device_combo.currentData()
            monitor.start(output_device=output_device)
        except Exception as exc:
            logger.error("self-monitor start failed: {}", exc)
            self.enable_checkbox.blockSignals(True)
            self.enable_checkbox.setChecked(False)
            self.enable_checkbox.blockSignals(False)
            self.status_label.setText(f"Start failed: {exc}")
            return

        self.device_combo.setEnabled(False)
        device_name = self.device_combo.currentText()
        self.status_label.setText(f"Monitoring: {device_name}")

    def _on_volume_changed(self, value: int) -> None:
        monitor = self._monitor()
        if monitor is not None:
            monitor.volume = value / 100.0
        self._update_volume_label()

    def _update_volume_label(self) -> None:
        self.volume_label.setText(f"Volume: {self.volume_slider.value()}%")

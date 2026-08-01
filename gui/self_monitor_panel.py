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

from gui.i18n import tr


class SelfMonitorPanel(QGroupBox):
    """GUI adapter for the application-owned independent monitor stream."""

    def __init__(self, context=None, *, language: str = "en") -> None:
        super().__init__()
        self._context = context
        self._language = language

        layout = QVBoxLayout(self)
        self.enable_checkbox = QCheckBox()
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_checkbox)

        device_row = QHBoxLayout()
        self.device_label = QLabel()
        device_row.addWidget(self.device_label)
        self.device_combo = QComboBox()
        device_row.addWidget(self.device_combo, 1)
        layout.addLayout(device_row)

        volume_row = QHBoxLayout()
        self.volume_label = QLabel()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setSingleStep(1)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        volume_row.addWidget(self.volume_label)
        volume_row.addWidget(self.volume_slider, 1)
        layout.addLayout(volume_row)

        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        self.help_label.setObjectName("secondaryText")
        layout.addWidget(self.help_label)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._apply_static_language()
        self.refresh_devices()

    def _t(self, key: str, **values) -> str:
        return tr(self._language, key, **values)

    def set_language(self, language: str) -> None:
        """Switch visible text without starting or stopping monitoring."""
        self._language = "zh" if language == "zh" else "en"
        self._apply_static_language()
        self.refresh_devices()

    def _apply_static_language(self) -> None:
        self.setTitle(self._t("monitor.group"))
        self.enable_checkbox.setText(self._t("monitor.enable"))
        self.device_label.setText(self._t("monitor.device"))
        self.help_label.setText(self._t("monitor.help"))
        self._update_volume_label()

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
        selected_output = (
            getattr(monitor, "output_device", None)
            if getattr(monitor, "is_running", False)
            else self.device_combo.currentData()
            if self.device_combo.count()
            else None
        )
        self.device_combo.clear()
        if monitor is None or manager is None:
            self.setEnabled(False)
            self.status_label.setText(self._t("monitor.unavailable"))
            return

        self.setEnabled(True)
        self.device_combo.addItem(self._t("device.system_default"), None)
        primary_output = getattr(self._context, "output_device", None)
        try:
            devices = manager.list_output_devices()
        except Exception as exc:
            logger.error("self-monitor device discovery failed: {}", exc)
            self.status_label.setText(
                self._t("monitor.discovery_failed", error=exc)
            )
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
        self.status_label.setText(
            self._t("monitor.monitoring")
            if monitor.is_running
            else self._t("monitor.disabled")
        )

    def _on_enabled_toggled(self, enabled: bool) -> None:
        monitor = self._monitor()
        manager = self._device_manager()
        if monitor is None or manager is None:
            return
        if not enabled:
            monitor.stop()
            self.device_combo.setEnabled(True)
            self.status_label.setText(self._t("monitor.disabled"))
            return

        try:
            output_device = self.device_combo.currentData()
            monitor.start(output_device=output_device)
        except Exception as exc:
            logger.error("self-monitor start failed: {}", exc)
            self.enable_checkbox.blockSignals(True)
            self.enable_checkbox.setChecked(False)
            self.enable_checkbox.blockSignals(False)
            self.status_label.setText(
                self._t("monitor.start_failed", error=exc)
            )
            return

        self.device_combo.setEnabled(False)
        self.status_label.setText(
            self._t(
                "monitor.monitoring_device",
                device=self.device_combo.currentText(),
            )
        )

    def _on_volume_changed(self, value: int) -> None:
        monitor = self._monitor()
        if monitor is not None:
            monitor.volume = value / 100.0
        self._update_volume_label()

    def _update_volume_label(self) -> None:
        self.volume_label.setText(
            self._t("monitor.volume", value=self.volume_slider.value())
        )

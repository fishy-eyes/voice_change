"""Adaptive realtime voice-changer control center."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from gui.device_settings_dialog import DeviceSettingsDialog
from gui.i18n import tr
from gui.self_monitor_panel import SelfMonitorPanel
from gui.voice_conversion_panel import VoiceConversionPanel

if TYPE_CHECKING:
    from core.context import AppContext


class MainWindow(QMainWindow):
    """Realtime controls; backend-specific parameters live in separate panels."""

    _STATUS_FIELDS = ("backend", "model", "model_status", "audio", "monitor", "latency")
    _DETAIL_FIELDS = (
        "input_rms",
        "output_rms",
        "callback_last",
        "callback_avg",
        "callback_max",
        "worker",
        "infer_count",
        "error_count",
    )

    def __init__(self, context: Optional[AppContext] = None) -> None:
        super().__init__()
        self._context = context
        self._language = "zh"
        self._device_dialog: DeviceSettingsDialog | None = None
        self.resize(780, 760)
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6fa; }
            QWidget { font-family: "Segoe UI", "Microsoft YaHei UI", Arial; }
            QGroupBox { background: white; border: 1px solid #dfe4ec;
                        border-radius: 8px; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                               left: 12px; padding: 0 5px; background: white; }
            QPushButton { min-height: 28px; padding: 2px 10px; }
            QLabel#secondaryText { color: #687386; }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        outer.addWidget(self._scroll_area)

        self._content_widget = QWidget()
        self._content_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(18, 10, 18, 12)
        self._content_layout.setSpacing(8)
        self._content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._scroll_area.setWidget(self._content_widget)

        header = QHBoxLayout()
        header.setSpacing(10)
        self._app_title_label = QLabel()
        self._app_title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        self._app_title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        header.addWidget(self._app_title_label, 1)
        self._language_button = QPushButton()
        self._language_button.clicked.connect(self._toggle_language)
        header.addWidget(self._language_button)
        self._content_layout.addLayout(header)

        self._voice_panel = VoiceConversionPanel(
            context,
            on_changed=self._update_status_display,
            language=self._language,
        )
        self._content_layout.addWidget(self._voice_panel)

        self._audio_group = QGroupBox()
        audio_row = QHBoxLayout(self._audio_group)
        audio_row.setContentsMargins(14, 18, 14, 12)
        audio_row.setSpacing(8)
        self._start_button = QPushButton()
        self._stop_button = QPushButton()
        self._device_button = QPushButton()
        self._start_button.clicked.connect(self._start_audio)
        self._stop_button.clicked.connect(self._stop_audio)
        self._device_button.clicked.connect(self._open_device_settings)
        audio_row.addWidget(self._start_button)
        audio_row.addWidget(self._stop_button)
        audio_row.addStretch(1)
        audio_row.addWidget(self._device_button)
        self._content_layout.addWidget(self._audio_group)

        self._gain_group = QGroupBox()
        gain_row = QHBoxLayout(self._gain_group)
        gain_row.setContentsMargins(14, 18, 14, 12)
        gain_row.setSpacing(10)
        self._gain_label = QLabel()
        self._gain_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        gain_width = QFontMetrics(self._gain_label.font()).horizontalAdvance("3.0×")
        self._gain_label.setMinimumWidth(gain_width + 6)
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(0, 30)
        self._gain_slider.setSingleStep(1)
        self._gain_slider.valueChanged.connect(self._set_output_gain)
        gain_row.addWidget(self._gain_label)
        gain_row.addWidget(self._gain_slider, 1)
        self._content_layout.addWidget(self._gain_group)

        self._self_monitor_panel = SelfMonitorPanel(context, language=self._language)
        self._content_layout.addWidget(self._self_monitor_panel)

        self._status_group = QGroupBox()
        status_layout = QVBoxLayout(self._status_group)
        status_layout.setContentsMargins(14, 18, 14, 12)
        status_layout.setSpacing(8)

        self._status_grid = QGridLayout()
        self._status_grid.setHorizontalSpacing(14)
        self._status_grid.setVerticalSpacing(5)
        self._status_grid.setColumnStretch(1, 1)
        self._status_name_labels: dict[str, QLabel] = {}
        self._status_value_labels: dict[str, QLabel] = {}
        for row, field in enumerate(self._STATUS_FIELDS):
            name_label = QLabel()
            value_label = QLabel()
            value_label.setWordWrap(True)
            value_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            name_label.setObjectName(f"statusName_{field}")
            value_label.setObjectName(f"statusValue_{field}")
            self._status_name_labels[field] = name_label
            self._status_value_labels[field] = value_label
            self._status_grid.addWidget(name_label, row, 0, Qt.AlignmentFlag.AlignTop)
            self._status_grid.addWidget(value_label, row, 1)
        status_layout.addLayout(self._status_grid)

        self._details_button = QPushButton()
        self._details_button.setCheckable(True)
        self._details_button.toggled.connect(self._toggle_details)
        status_layout.addWidget(self._details_button)

        self._details_widget = QWidget()
        details_grid = QGridLayout(self._details_widget)
        details_grid.setContentsMargins(0, 4, 0, 0)
        details_grid.setHorizontalSpacing(14)
        details_grid.setVerticalSpacing(4)
        details_grid.setColumnStretch(1, 1)
        self._detail_name_labels: dict[str, QLabel] = {}
        self._detail_value_labels: dict[str, QLabel] = {}
        for row, field in enumerate(self._DETAIL_FIELDS):
            name_label = QLabel()
            value_label = QLabel()
            name_label.setObjectName(f"detailName_{field}")
            value_label.setObjectName(f"detailValue_{field}")
            value_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            self._detail_name_labels[field] = name_label
            self._detail_value_labels[field] = value_label
            details_grid.addWidget(name_label, row, 0, Qt.AlignmentFlag.AlignTop)
            details_grid.addWidget(value_label, row, 1)
        self._details_widget.setVisible(False)
        status_layout.addWidget(self._details_widget)
        self._content_layout.addWidget(self._status_group)
        self._content_layout.addStretch(1)

        self._sync_gain_control()
        self._apply_language()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_status_display)
        self._timer.start(250)
        self._update_status_display()

    def _t(self, key: str, **values) -> str:
        return tr(self._language, key, **values)

    def _effect_manager(self):
        return getattr(self._context, "effect_manager", None)

    def _gain_effect(self):
        manager = self._effect_manager()
        return manager.get_by_name("GainEffect") if manager is not None else None

    def _sync_gain_control(self) -> None:
        effect = self._gain_effect()
        gain = float(getattr(effect, "gain", 1.0))
        self._gain_slider.blockSignals(True)
        self._gain_slider.setValue(round(max(0.0, min(3.0, gain)) * 10))
        self._gain_slider.blockSignals(False)
        self._gain_slider.setEnabled(effect is not None)
        self._gain_label.setText(f"{gain:.1f}×")

    def _set_output_gain(self, value: int) -> None:
        effect = self._gain_effect()
        gain = value / 10.0
        if effect is not None:
            effect.gain = gain
            effect.enabled = True
        self._gain_label.setText(f"{gain:.1f}×")

    def _open_device_settings(self) -> None:
        if self._device_dialog is not None:
            self._device_dialog.raise_()
            self._device_dialog.activateWindow()
            return
        dialog = DeviceSettingsDialog(self._context, language=self._language, parent=self)
        dialog.finished.connect(self._device_settings_closed)
        self._device_dialog = dialog
        dialog.show()

    def _device_settings_closed(self, _result: int) -> None:
        self._device_dialog = None
        self._self_monitor_panel.refresh_devices()

    def _start_audio(self) -> None:
        stream = getattr(self._context, "audio_stream", None)
        if stream is None or stream.is_running:
            return
        try:
            stream.start()
        except Exception as exc:
            logger.error("GUI audio start failed: {}", exc)
            self.statusBar().showMessage(str(exc))
        self._update_status_display()

    def _stop_audio(self) -> None:
        stream = getattr(self._context, "audio_stream", None)
        if stream is None or not stream.is_running:
            return
        try:
            stream.stop()
        except Exception as exc:
            logger.error("GUI audio stop failed: {}", exc)
            self.statusBar().showMessage(str(exc))
        self._update_status_display()

    def _toggle_details(self, visible: bool) -> None:
        self._details_widget.setVisible(visible)
        self._details_button.setText(
            self._t("runtime.hide_details" if visible else "runtime.show_details")
        )
        self._details_widget.updateGeometry()
        self._status_group.updateGeometry()
        self._content_widget.updateGeometry()

    def _translated_state(self, state: str) -> str:
        normalized = str(state or "IDLE").upper()
        key = f"runtime.state.{normalized.lower()}"
        translated = self._t(key)
        return normalized if translated == key else translated

    def _worker_metrics(self, manager):
        runtime = getattr(manager, "current_runtime", None) if manager is not None else None
        state = getattr(runtime, "state", None)
        effect = getattr(state, "effect", None)
        worker = getattr(effect, "worker", None)
        if worker is None:
            return None, self._t("runtime.worker_stopped"), 0, 0
        if bool(getattr(worker, "is_inferencing", False)):
            worker_state = self._t("runtime.worker_inferencing")
        elif bool(getattr(worker, "is_running", False)):
            worker_state = self._t("runtime.worker_running")
        else:
            worker_state = self._t("runtime.worker_stopped")
        return (
            worker,
            worker_state,
            int(getattr(worker, "infer_count", 0)),
            (
                int(getattr(worker, "error_count", 0))
                + int(getattr(worker, "continuity_error_count", 0))
            ),
        )

    def _update_status_display(self) -> None:
        stream = getattr(self._context, "audio_stream", None)
        monitor = getattr(self._context, "self_monitor", None)
        manager = getattr(self._context, "voice_conversion_manager", None)
        running = bool(stream is not None and stream.is_running)
        monitor_running = bool(getattr(monitor, "is_running", False))
        if manager is not None:
            status = manager.get_status()
            backend = status.backend.upper() if status.backend else self._t("runtime.na")
            model = status.model or self._t("runtime.na")
            model_status = str(status.state or "IDLE").upper()
            latency = float(status.latency_ms)
        else:
            backend = model = self._t("runtime.na")
            model_status, latency = "IDLE", 0.0

        busy = model_status in {"LOADING", "SWITCHING", "UNLOADING"}
        self._start_button.setEnabled(stream is not None and not running and not busy)
        self._stop_button.setEnabled(stream is not None and running and not busy)
        self._device_button.setEnabled(not busy)
        values = {
            "backend": backend,
            "model": model,
            "model_status": self._translated_state(model_status),
            "audio": self._t("runtime.running" if running else "runtime.stopped"),
            "monitor": self._t("runtime.on" if monitor_running else "runtime.off"),
            "latency": f"{latency:.1f} ms",
        }
        for field, value in values.items():
            self._status_value_labels[field].setText(value)

        count = int(getattr(stream, "_callback_count", 0)) if stream is not None else 0
        total = float(getattr(stream, "_total_proc_ms", 0.0)) if stream is not None else 0.0
        average = total / count if count else 0.0
        _, worker_state, infer_count, error_count = self._worker_metrics(manager)
        details = {
            "input_rms": f"{float(getattr(stream, '_input_rms', 0.0)):.6f}",
            "output_rms": f"{float(getattr(stream, '_output_rms', 0.0)):.6f}",
            "callback_last": f"{float(getattr(stream, '_last_proc_ms', 0.0)):.2f} ms",
            "callback_avg": f"{average:.2f} ms",
            "callback_max": f"{float(getattr(stream, '_max_proc_ms', 0.0)):.2f} ms",
            "worker": worker_state,
            "infer_count": str(infer_count),
            "error_count": str(error_count),
        }
        for field, value in details.items():
            self._detail_value_labels[field].setText(value)
        self._voice_panel.update_status()

    def _apply_language(self) -> None:
        self.setWindowTitle(self._t("control.title"))
        self._app_title_label.setText(self.windowTitle())
        self._language_button.setText(self._t("control.language_switch"))
        self._audio_group.setTitle(self._t("control.audio_group"))
        self._start_button.setText(self._t("control.start_audio"))
        self._stop_button.setText(self._t("control.stop_audio"))
        self._device_button.setText(self._t("control.device_settings"))
        self._gain_group.setTitle(self._t("control.output_gain"))
        self._status_group.setTitle(self._t("runtime.group"))
        for field in self._STATUS_FIELDS:
            self._status_name_labels[field].setText(self._t(f"runtime.{field}"))
        for field in self._DETAIL_FIELDS:
            self._detail_name_labels[field].setText(self._t(f"runtime.{field}"))
        self._voice_panel.set_language(self._language)
        self._self_monitor_panel.set_language(self._language)
        self._toggle_details(self._details_button.isChecked())
        self._update_status_display()

    def _toggle_language(self) -> None:
        self._language = "en" if self._language == "zh" else "zh"
        self._apply_language()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._voice_panel.close_panel()
        if self._device_dialog is not None:
            self._device_dialog.close()
        super().closeEvent(event)

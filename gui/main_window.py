"""Voice Changer main window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.device_switching import switch_audio_devices
from gui.i18n import tr
from gui.rvc_control_panel import RVCControlPanel
from gui.self_monitor_panel import SelfMonitorPanel

if TYPE_CHECKING:
    from core.context import AppContext


class MainWindow(QMainWindow):
    """Application window bound directly to the shared runtime context."""

    def __init__(self, context: Optional[AppContext] = None) -> None:
        super().__init__()
        self._context = context
        self._language = "en"
        self.setMinimumSize(1020, 640)
        self.resize(1180, 760)
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6fa; }
            QWidget { font-family: "Segoe UI", "Microsoft YaHei UI", Arial; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #dfe4ec;
                border-radius: 9px;
                margin-top: 12px;
                padding: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            QPushButton {
                min-height: 28px;
                padding: 2px 10px;
                border: 1px solid #c9d1dc;
                border-radius: 6px;
                background: #ffffff;
            }
            QPushButton:hover { background: #eef4ff; border-color: #7ca7ee; }
            QPushButton:pressed { background: #dce9ff; }
            QComboBox {
                min-height: 27px;
                padding: 1px 7px;
                border: 1px solid #c9d1dc;
                border-radius: 5px;
                background: #ffffff;
            }
            QLabel#secondaryText { color: #657084; font-size: 11px; }
            QLabel#appTitle { color: #172033; font-size: 22px; font-weight: 700; }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 14, 18, 12)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        self._app_title_label = QLabel()
        self._app_title_label.setObjectName("appTitle")
        header.addWidget(self._app_title_label)
        header.addStretch()
        self._language_caption_label = QLabel()
        header.addWidget(self._language_caption_label)
        self._language_button = QPushButton()
        self._language_button.setMinimumWidth(88)
        self._language_button.clicked.connect(self._toggle_language)
        header.addWidget(self._language_button)
        root_layout.addLayout(header)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        root_layout.addWidget(scroll_area, 1)

        content = QWidget()
        scroll_area.setWidget(content)
        self._content_layout = QHBoxLayout(content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(16)

        self._left_column = QWidget()
        left_layout = QVBoxLayout(self._left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        self._right_column = QWidget()
        right_layout = QVBoxLayout(self._right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        self._content_layout.addWidget(self._left_column, 1)
        self._content_layout.addWidget(self._right_column, 1)

        # --- Left column: devices, basic effects, local monitoring ---
        self._device_group = QGroupBox()
        device_layout = QVBoxLayout(self._device_group)
        self._device_label = QLabel()
        self._device_label.setWordWrap(True)
        self._device_label.setStyleSheet("padding: 5px;")
        device_layout.addWidget(self._device_label)

        input_row = QHBoxLayout()
        self._input_title_label = QLabel()
        input_row.addWidget(self._input_title_label)
        self._input_combo = QComboBox()
        input_row.addWidget(self._input_combo, 1)
        device_layout.addLayout(input_row)

        output_row = QHBoxLayout()
        self._output_title_label = QLabel()
        output_row.addWidget(self._output_title_label)
        self._output_combo = QComboBox()
        output_row.addWidget(self._output_combo, 1)
        device_layout.addLayout(output_row)

        device_buttons = QHBoxLayout()
        self._refresh_device_btn = QPushButton()
        self._refresh_device_btn.clicked.connect(self._refresh_device_choices)
        self._apply_device_btn = QPushButton()
        self._apply_device_btn.clicked.connect(self._apply_devices)
        device_buttons.addWidget(self._refresh_device_btn)
        device_buttons.addWidget(self._apply_device_btn)
        device_layout.addLayout(device_buttons)
        left_layout.addWidget(self._device_group)

        self._effects_group = QGroupBox()
        effects_layout = QVBoxLayout(self._effects_group)
        self._effects_label = QLabel()
        self._effects_label.setStyleSheet("padding: 5px;")
        effects_layout.addWidget(self._effects_label)

        self._refresh_effects_btn = QPushButton()
        self._refresh_effects_btn.clicked.connect(self._update_effects_display)
        effects_layout.addWidget(self._refresh_effects_btn)

        effect_buttons = QHBoxLayout()
        self._effect_buttons: dict[str, QPushButton] = {}
        for name in ("RobotEffect", "EchoEffect", "GainEffect"):
            button = QPushButton()
            button.clicked.connect(
                lambda checked=False, effect_name=name: self._toggle_effect(
                    effect_name
                )
            )
            self._effect_buttons[name] = button
            effect_buttons.addWidget(button)
        effects_layout.addLayout(effect_buttons)

        gain_row = QHBoxLayout()
        self._gain_label = QLabel()
        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_slider.setRange(10, 50)
        self._gain_slider.setValue(20)
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self._gain_label)
        gain_row.addWidget(self._gain_slider, 1)
        effects_layout.addLayout(gain_row)
        self._gain_help_label = self._secondary_label()
        effects_layout.addWidget(self._gain_help_label)
        left_layout.addWidget(self._effects_group)

        self._self_monitor_panel = SelfMonitorPanel(
            context,
            language=self._language,
        )
        left_layout.addWidget(self._self_monitor_panel)
        left_layout.addStretch()

        # --- Right column: model controls, advanced RVC, runtime status ---
        self._ai_voice_panel = RVCControlPanel(
            context,
            self._update_effects_display,
            language=self._language,
        )
        right_layout.addWidget(self._ai_voice_panel)

        self._rvc_group = QGroupBox()
        rvc_layout = QVBoxLayout(self._rvc_group)
        self._rvc_status_label = QLabel()
        self._rvc_status_label.setWordWrap(True)
        rvc_layout.addWidget(self._rvc_status_label)

        pitch_row = QHBoxLayout()
        self._rvc_pitch_label = QLabel()
        self._rvc_pitch_slider = QSlider(Qt.Horizontal)
        self._rvc_pitch_slider.setRange(-24, 24)
        self._rvc_pitch_slider.setSingleStep(1)
        self._rvc_pitch_slider.setValue(0)
        self._rvc_pitch_slider.valueChanged.connect(self._on_rvc_config_changed)
        pitch_row.addWidget(self._rvc_pitch_label)
        pitch_row.addWidget(self._rvc_pitch_slider, 1)
        rvc_layout.addLayout(pitch_row)
        self._pitch_help_label = self._secondary_label()
        rvc_layout.addWidget(self._pitch_help_label)

        index_row = QHBoxLayout()
        self._rvc_index_label = QLabel()
        self._rvc_index_slider = QSlider(Qt.Horizontal)
        self._rvc_index_slider.setRange(0, 100)
        self._rvc_index_slider.setSingleStep(1)
        self._rvc_index_slider.setValue(75)
        self._rvc_index_slider.valueChanged.connect(self._on_rvc_config_changed)
        index_row.addWidget(self._rvc_index_label)
        index_row.addWidget(self._rvc_index_slider, 1)
        rvc_layout.addLayout(index_row)
        self._index_help_label = self._secondary_label()
        rvc_layout.addWidget(self._index_help_label)

        protect_row = QHBoxLayout()
        self._rvc_protect_label = QLabel()
        self._rvc_protect_slider = QSlider(Qt.Horizontal)
        self._rvc_protect_slider.setRange(0, 50)
        self._rvc_protect_slider.setSingleStep(1)
        self._rvc_protect_slider.setValue(33)
        self._rvc_protect_slider.valueChanged.connect(self._on_rvc_config_changed)
        protect_row.addWidget(self._rvc_protect_label)
        protect_row.addWidget(self._rvc_protect_slider, 1)
        rvc_layout.addLayout(protect_row)
        self._protect_help_label = self._secondary_label()
        rvc_layout.addWidget(self._protect_help_label)

        rms_row = QHBoxLayout()
        self._rvc_rms_label = QLabel()
        self._rvc_rms_slider = QSlider(Qt.Horizontal)
        self._rvc_rms_slider.setRange(0, 100)
        self._rvc_rms_slider.setSingleStep(1)
        self._rvc_rms_slider.setValue(25)
        self._rvc_rms_slider.valueChanged.connect(self._on_rvc_config_changed)
        rms_row.addWidget(self._rvc_rms_label)
        rms_row.addWidget(self._rvc_rms_slider, 1)
        rvc_layout.addLayout(rms_row)
        self._rms_help_label = self._secondary_label()
        rvc_layout.addWidget(self._rms_help_label)
        right_layout.addWidget(self._rvc_group)

        self._status_group = QGroupBox()
        status_layout = QVBoxLayout(self._status_group)
        self._status_label = QLabel()
        self._status_label.setStyleSheet("padding: 5px;")
        status_layout.addWidget(self._status_label)
        stream_buttons = QHBoxLayout()
        self._start_btn = QPushButton()
        self._start_btn.clicked.connect(self._start_audio)
        self._stop_btn = QPushButton()
        self._stop_btn.clicked.connect(self._stop_audio)
        stream_buttons.addWidget(self._start_btn)
        stream_buttons.addWidget(self._stop_btn)
        status_layout.addLayout(stream_buttons)
        right_layout.addWidget(self._status_group)
        right_layout.addStretch()

        self._apply_static_language()
        self.statusBar().showMessage(self._t("status.ready"))
        self._update_effects_display()
        self._refresh_device_choices()
        self._update_status_display()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_display)
        self._status_timer.start(500)

    @staticmethod
    def _secondary_label() -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        label.setObjectName("secondaryText")
        return label

    def _t(self, key: str, **values) -> str:
        return tr(self._language, key, **values)

    def _toggle_language(self) -> None:
        self._language = "zh" if self._language == "en" else "en"
        self._apply_static_language()
        self._ai_voice_panel.set_language(self._language)
        self._self_monitor_panel.set_language(self._language)
        self._update_effects_display()
        self._update_device_display()
        self._update_status_display()
        self.statusBar().showMessage(self._t("status.ready"))

    def _apply_static_language(self) -> None:
        self.setWindowTitle(self._t("window.title"))
        self._app_title_label.setText(self._t("window.title"))
        self._language_caption_label.setText(self._t("language.caption"))
        self._language_button.setText(self._t("language.switch"))
        self._device_group.setTitle(self._t("device.group"))
        self._input_title_label.setText(self._t("device.input"))
        self._output_title_label.setText(self._t("device.output"))
        self._refresh_device_btn.setText(self._t("device.refresh"))
        self._apply_device_btn.setText(self._t("device.apply"))
        self._effects_group.setTitle(self._t("effects.group"))
        self._refresh_effects_btn.setText(self._t("effects.refresh"))
        self._effect_buttons["RobotEffect"].setText(
            self._t("effects.toggle_robot")
        )
        self._effect_buttons["EchoEffect"].setText(
            self._t("effects.toggle_echo")
        )
        self._effect_buttons["GainEffect"].setText(
            self._t("effects.toggle_gain")
        )
        self._gain_help_label.setText(self._t("gain.help"))
        self._rvc_group.setTitle(self._t("advanced.group"))
        self._pitch_help_label.setText(self._t("pitch.help"))
        self._index_help_label.setText(self._t("index.help"))
        self._protect_help_label.setText(self._t("protect.help"))
        self._rms_help_label.setText(self._t("rms.help"))
        self._status_group.setTitle(self._t("status.group"))
        self._start_btn.setText(self._t("status.start"))
        self._stop_btn.setText(self._t("status.stop"))
        self._update_rvc_labels()

    def _effect_manager(self):
        return (
            getattr(self._context, "effect_manager", None)
            if self._context
            else None
        )

    def _rvc_engine(self):
        """Return the shared RVC engine without creating runtime resources."""
        if self._context is None:
            return None
        direct_engine = getattr(self._context, "rvc_engine", None)
        runtime = getattr(self._context, "rvc_runtime", None)
        if runtime is not None and runtime.state.engine is not None:
            return runtime.state.engine
        if direct_engine is not None:
            return direct_engine
        effect_manager = self._effect_manager()
        if effect_manager is None:
            return None
        ai_effect = effect_manager.get_by_name("AIVoiceEffect")
        return getattr(ai_effect, "engine", None) if ai_effect is not None else None

    def _sync_rvc_controls(self) -> None:
        """Refresh advanced controls from the current immutable config."""
        engine = self._rvc_engine()
        config = getattr(engine, "config", None) if engine is not None else None
        available = config is not None and callable(
            getattr(engine, "update_config", None)
        )
        self._rvc_group.setEnabled(available)
        if not available:
            self._rvc_status_label.setText(self._t("advanced.unavailable"))
            return

        controls = (
            (self._rvc_pitch_slider, int(config.pitch_shift)),
            (self._rvc_index_slider, round(config.index_rate * 100)),
            (self._rvc_protect_slider, round(config.protect * 100)),
            (self._rvc_rms_slider, round(config.rms_mix_rate * 100)),
        )
        for slider, value in controls:
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self._update_rvc_labels()
        self._rvc_status_label.setText(self._t("advanced.next_inference"))

    def _update_rvc_labels(self) -> None:
        pitch = self._rvc_pitch_slider.value()
        self._rvc_pitch_label.setText(f"Pitch: {pitch:+d}")
        self._rvc_index_label.setText(
            f"Index Rate: {self._rvc_index_slider.value() / 100.0:.2f}"
        )
        self._rvc_protect_label.setText(
            f"Protect: {self._rvc_protect_slider.value() / 100.0:.2f}"
        )
        self._rvc_rms_label.setText(
            f"RMS Mix Rate: {self._rvc_rms_slider.value() / 100.0:.2f}"
        )
        self._gain_label.setText(
            self._t("gain.value", value=self._gain_slider.value() / 10.0)
        )

    def _on_rvc_config_changed(self, _value: int) -> None:
        """Apply slider values to the existing engine without reloading it."""
        self._update_rvc_labels()
        engine = self._rvc_engine()
        if engine is None or not callable(getattr(engine, "update_config", None)):
            logger.warning("GUI RVC config change ignored: engine unavailable")
            self._rvc_group.setEnabled(False)
            return
        try:
            engine.update_config(
                pitch_shift=self._rvc_pitch_slider.value(),
                index_rate=self._rvc_index_slider.value() / 100.0,
                protect=self._rvc_protect_slider.value() / 100.0,
                rms_mix_rate=self._rvc_rms_slider.value() / 100.0,
            )
        except Exception as exc:
            logger.error("GUI RVC config update failed: {}", exc)
            self.statusBar().showMessage(
                self._t("status.rvc_update_failed", error=exc)
            )
            self._sync_rvc_controls()
            return
        self._rvc_status_label.setText(self._t("advanced.updated"))
        self.statusBar().showMessage(self._t("status.rvc_updated"))

    def _update_effects_display(self) -> None:
        """Read every displayed state from the shared EffectManager."""
        effect_manager = self._effect_manager()
        if effect_manager is None:
            self._effects_label.setText(self._t("effects.unavailable"))
            self._sync_rvc_controls()
            return

        lines = [
            f"{effect.name}: {'ON' if effect.enabled else 'OFF'}"
            for effect in effect_manager.effects
        ]
        self._effects_label.setText(
            "\n".join(lines) if lines else self._t("effects.none")
        )

        gain_effect = effect_manager.get_by_name("GainEffect")
        if gain_effect is not None:
            slider_value = max(
                self._gain_slider.minimum(),
                min(self._gain_slider.maximum(), round(gain_effect.gain * 10)),
            )
            self._gain_slider.blockSignals(True)
            self._gain_slider.setValue(slider_value)
            self._gain_slider.blockSignals(False)
            self._gain_label.setText(
                self._t("gain.value", value=gain_effect.gain)
            )
        self._sync_rvc_controls()

    def _toggle_effect(self, effect_name: str) -> None:
        """Toggle the effect instance owned by the shared EffectManager."""
        effect_manager = self._effect_manager()
        if effect_manager is None:
            logger.warning("GUI toggle ignored: EffectManager is unavailable")
            self.statusBar().showMessage(
                self._t("status.effect_manager_unavailable")
            )
            return
        effect = effect_manager.get_by_name(effect_name)
        if effect is None:
            logger.warning("GUI toggle ignored: effect not found: {}", effect_name)
            self.statusBar().showMessage(
                self._t("status.effect_not_found", name=effect_name)
            )
            return
        if effect.enabled:
            effect_manager.disable(effect_name)
        else:
            effect_manager.enable(effect_name)
        self._update_effects_display()

    def _on_gain_changed(self, value: int) -> None:
        """Update the shared GainEffect (integer 10-50 -> float 1.0-5.0)."""
        effect_manager = self._effect_manager()
        if effect_manager is None:
            logger.warning("GUI gain change ignored: EffectManager is unavailable")
            return
        gain_effect = effect_manager.get_by_name("GainEffect")
        if gain_effect is None:
            logger.warning("GUI gain change ignored: GainEffect not found")
            self.statusBar().showMessage(
                self._t("status.effect_not_found", name="GainEffect")
            )
            return
        gain_effect.gain = value / 10.0
        self._gain_label.setText(
            self._t("gain.value", value=gain_effect.gain)
        )

    def _refresh_device_choices(self) -> None:
        """Re-enumerate via DeviceManager without changing the audio stream."""
        device_manager = (
            getattr(self._context, "device_manager", None)
            if self._context
            else None
        )
        if device_manager is None:
            self._device_label.setText(
                f"{self._t('device.input')} (N/A)\n"
                f"{self._t('device.output')} (N/A)"
            )
            return

        input_preferred = (
            self._input_combo.currentData()
            if self._input_combo.count()
            else getattr(self._context, "input_device", None)
        )
        output_preferred = (
            self._output_combo.currentData()
            if self._output_combo.count()
            else getattr(self._context, "output_device", None)
        )
        try:
            input_devices = device_manager.list_input_devices()
            output_devices = device_manager.list_output_devices()
        except Exception as exc:
            logger.error("GUI device refresh failed: {}", exc)
            self.statusBar().showMessage(
                self._t("status.device_refresh_failed", error=exc)
            )
            return

        self._populate_device_combo(
            self._input_combo,
            input_devices,
            input_preferred,
        )
        self._populate_device_combo(
            self._output_combo,
            output_devices,
            output_preferred,
        )
        self._update_device_display()
        self.statusBar().showMessage(self._t("status.device_refreshed"))
        self._self_monitor_panel.refresh_devices()

    def _populate_device_combo(self, combo, devices, preferred_index) -> None:
        """Populate a combo with device indices as item data, never parsed text."""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._t("device.system_default"), None)
        for device in devices:
            combo.addItem(f"[{device.index}] {device.name}", device.index)
        selected = combo.findData(preferred_index)
        combo.setCurrentIndex(selected if selected >= 0 else 0)
        combo.blockSignals(False)

    def _apply_devices(self) -> None:
        """Apply selected indices through the application switching transaction."""
        if self._context is None:
            self.statusBar().showMessage(
                self._t("status.context_unavailable")
            )
            return
        if (
            self._input_combo.currentIndex() < 0
            or self._output_combo.currentIndex() < 0
        ):
            self.statusBar().showMessage(self._t("status.select_devices"))
            return

        input_device = self._input_combo.currentData()
        output_device = self._output_combo.currentData()
        result = switch_audio_devices(
            self._context,
            input_device,
            output_device,
        )
        if result.success:
            self._update_device_display()
            self._update_status_display()
            self.statusBar().showMessage(self._t("status.devices_applied"))
            self._self_monitor_panel.refresh_devices()
        else:
            self._update_device_display()
            self._update_status_display()
            suffix = (
                self._t("status.previous_restored")
                if result.restored_previous_stream
                else ""
            )
            self.statusBar().showMessage(
                self._t(
                    "status.device_switch_failed",
                    suffix=suffix,
                    error=result.error,
                )
            )

    def _update_device_display(self) -> None:
        """Display the devices used by the current context stream."""
        device_manager = (
            getattr(self._context, "device_manager", None)
            if self._context
            else None
        )
        if device_manager is None:
            self._device_label.setText(
                f"{self._t('device.input')} (N/A)\n"
                f"{self._t('device.output')} (N/A)"
            )
            return
        try:
            input_name = device_manager.get_device_name(self._context.input_device)
            output_name = device_manager.get_device_name(self._context.output_device)
        except Exception as exc:
            logger.error("GUI device status refresh failed: {}", exc)
            input_name = self._t("device.unknown")
            output_name = self._t("device.unknown")
        self._device_label.setText(
            f"{self._t('device.current_input', name=input_name)}\n"
            f"{self._t('device.current_output', name=output_name)}"
        )

    def _update_status_display(self) -> None:
        """Refresh stream metrics and AI readiness from current shared objects."""
        stream = (
            getattr(self._context, "audio_stream", None)
            if self._context
            else None
        )
        if stream is None:
            audio_lines = [self._t("status.audio_na")]
        elif stream.is_running:
            count = getattr(stream, "_callback_count", 0)
            total = getattr(stream, "_total_proc_ms", 0.0)
            average = total / count if count > 0 else 0.0
            audio_lines = [
                self._t("status.audio_running"),
                self._t("status.processing_avg", value=average),
                self._t(
                    "status.processing_max",
                    value=getattr(stream, "_max_proc_ms", 0.0),
                ),
            ]
        else:
            audio_lines = [self._t("status.audio_stopped")]

        ai_effect = None
        effect_manager = self._effect_manager()
        if effect_manager is not None:
            ai_effect = effect_manager.get_by_name("AIVoiceEffect")
        ai_ready = bool(
            ai_effect is not None
            and ai_effect.engine.is_loaded
            and ai_effect.is_running
        )
        audio_lines.append(
            self._t("status.ai_ready")
            if ai_ready
            else self._t("status.ai_not_loaded")
        )
        self._status_label.setText("\n".join(audio_lines))
        self._ai_voice_panel.update_status()

    def _start_audio(self) -> None:
        """Start the current audio stream via AppContext."""
        stream = (
            getattr(self._context, "audio_stream", None)
            if self._context
            else None
        )
        if stream is None or stream.is_running:
            return
        try:
            stream.start()
        except Exception as exc:
            logger.error("GUI audio start failed: {}", exc)
            self.statusBar().showMessage(
                self._t("status.audio_start_failed", error=exc)
            )
            return
        self._update_status_display()
        self.statusBar().showMessage(self._t("status.audio_started"))

    def _stop_audio(self) -> None:
        """Stop the current audio stream via AppContext."""
        stream = (
            getattr(self._context, "audio_stream", None)
            if self._context
            else None
        )
        if stream is None or not stream.is_running:
            return
        try:
            stream.stop()
        except Exception as exc:
            logger.error("GUI audio stop failed: {}", exc)
            self.statusBar().showMessage(
                self._t("status.audio_stop_failed", error=exc)
            )
            return
        self._update_status_display()
        self.statusBar().showMessage(self._t("status.audio_stopped"))

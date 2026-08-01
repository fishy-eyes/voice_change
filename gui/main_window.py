"""Voice Changer main window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.device_switching import switch_audio_devices
from gui.rvc_control_panel import RVCControlPanel
from gui.self_monitor_panel import SelfMonitorPanel

if TYPE_CHECKING:
    from core.context import AppContext


class MainWindow(QMainWindow):
    """Application window bound directly to the shared runtime context."""

    def __init__(self, context: Optional[AppContext] = None) -> None:
        super().__init__()
        self._context = context
        self.setWindowTitle("Voice Changer")
        self.resize(640, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Device selection ---
        device_group = QGroupBox("Device Selection")
        device_layout = QVBoxLayout(device_group)
        self._device_label = QLabel("Loading...")
        self._device_label.setStyleSheet("padding: 8px;")
        device_layout.addWidget(self._device_label)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input:"))
        self._input_combo = QComboBox()
        input_row.addWidget(self._input_combo)
        device_layout.addLayout(input_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self._output_combo = QComboBox()
        output_row.addWidget(self._output_combo)
        device_layout.addLayout(output_row)

        device_buttons = QHBoxLayout()
        refresh_device_btn = QPushButton("Refresh Devices")
        refresh_device_btn.clicked.connect(self._refresh_device_choices)
        apply_device_btn = QPushButton("Apply Devices")
        apply_device_btn.clicked.connect(self._apply_devices)
        device_buttons.addWidget(refresh_device_btn)
        device_buttons.addWidget(apply_device_btn)
        device_layout.addLayout(device_buttons)
        layout.addWidget(device_group)

        # --- Effects control ---
        effects_group = QGroupBox("Effects Control")
        effects_layout = QVBoxLayout(effects_group)
        self._effects_label = QLabel("No effects")
        self._effects_label.setStyleSheet("padding: 8px;")
        effects_layout.addWidget(self._effects_label)

        refresh_effects_btn = QPushButton("Refresh Effects")
        refresh_effects_btn.clicked.connect(self._update_effects_display)
        effects_layout.addWidget(refresh_effects_btn)
        for name in ("RobotEffect", "EchoEffect", "GainEffect"):
            button = QPushButton(f"Toggle {name.replace('Effect', '')}")
            button.clicked.connect(
                lambda checked=False, effect_name=name: self._toggle_effect(
                    effect_name
                )
            )
            effects_layout.addWidget(button)

        gain_row = QHBoxLayout()
        self._gain_label = QLabel("Gain: 2.0")
        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_slider.setRange(10, 50)
        self._gain_slider.setValue(20)
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self._gain_label)
        gain_row.addWidget(self._gain_slider)
        effects_layout.addLayout(gain_row)
        layout.addWidget(effects_group)

        self._ai_voice_panel = RVCControlPanel(context, self._update_effects_display)
        layout.addWidget(self._ai_voice_panel)

        self._self_monitor_panel = SelfMonitorPanel(context)
        layout.addWidget(self._self_monitor_panel)

        # --- Developer-only RVC tuning ---
        self._rvc_group = QGroupBox("AI Voice / RVC Advanced")
        rvc_layout = QVBoxLayout(self._rvc_group)
        self._rvc_status_label = QLabel("RVC engine unavailable")
        rvc_layout.addWidget(self._rvc_status_label)

        pitch_row = QHBoxLayout()
        self._rvc_pitch_label = QLabel("Pitch: 0")
        self._rvc_pitch_slider = QSlider(Qt.Horizontal)
        self._rvc_pitch_slider.setRange(-12, 12)
        self._rvc_pitch_slider.setSingleStep(1)
        self._rvc_pitch_slider.setValue(0)
        self._rvc_pitch_slider.valueChanged.connect(self._on_rvc_config_changed)
        pitch_row.addWidget(self._rvc_pitch_label)
        pitch_row.addWidget(self._rvc_pitch_slider)
        rvc_layout.addLayout(pitch_row)

        index_row = QHBoxLayout()
        self._rvc_index_label = QLabel("Index Rate: 0.75")
        self._rvc_index_slider = QSlider(Qt.Horizontal)
        self._rvc_index_slider.setRange(0, 100)
        self._rvc_index_slider.setSingleStep(1)
        self._rvc_index_slider.setValue(75)
        self._rvc_index_slider.valueChanged.connect(self._on_rvc_config_changed)
        index_row.addWidget(self._rvc_index_label)
        index_row.addWidget(self._rvc_index_slider)
        rvc_layout.addLayout(index_row)

        protect_row = QHBoxLayout()
        self._rvc_protect_label = QLabel("Protect: 0.33")
        self._rvc_protect_slider = QSlider(Qt.Horizontal)
        self._rvc_protect_slider.setRange(0, 50)
        self._rvc_protect_slider.setSingleStep(1)
        self._rvc_protect_slider.setValue(33)
        self._rvc_protect_slider.valueChanged.connect(self._on_rvc_config_changed)
        protect_row.addWidget(self._rvc_protect_label)
        protect_row.addWidget(self._rvc_protect_slider)
        rvc_layout.addLayout(protect_row)

        rms_row = QHBoxLayout()
        self._rvc_rms_label = QLabel("RMS Mix Rate: 0.25")
        self._rvc_rms_slider = QSlider(Qt.Horizontal)
        self._rvc_rms_slider.setRange(0, 100)
        self._rvc_rms_slider.setSingleStep(1)
        self._rvc_rms_slider.setValue(25)
        self._rvc_rms_slider.valueChanged.connect(self._on_rvc_config_changed)
        rms_row.addWidget(self._rvc_rms_label)
        rms_row.addWidget(self._rvc_rms_slider)
        rvc_layout.addLayout(rms_row)
        layout.addWidget(self._rvc_group)

        # --- Runtime status ---
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self._status_label = QLabel("Audio: not started")
        self._status_label.setStyleSheet("padding: 8px;")
        status_layout.addWidget(self._status_label)
        stream_buttons = QHBoxLayout()
        self._start_btn = QPushButton("Start Audio")
        self._start_btn.clicked.connect(self._start_audio)
        self._stop_btn = QPushButton("Stop Audio")
        self._stop_btn.clicked.connect(self._stop_audio)
        stream_buttons.addWidget(self._start_btn)
        stream_buttons.addWidget(self._stop_btn)
        status_layout.addLayout(stream_buttons)
        layout.addWidget(status_group)
        layout.addStretch()

        self.statusBar().showMessage("Ready")
        self._update_effects_display()
        self._refresh_device_choices()
        self._update_status_display()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_display)
        self._status_timer.start(500)

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
        available = config is not None and callable(getattr(engine, "update_config", None))
        self._rvc_group.setEnabled(available)
        if not available:
            self._rvc_status_label.setText("RVC engine unavailable")
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
        self._rvc_status_label.setText("Runtime updates apply to the next inference")

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
            self.statusBar().showMessage(f"RVC config update failed: {exc}")
            self._sync_rvc_controls()
            return
        self._rvc_status_label.setText("Runtime config updated")
        self.statusBar().showMessage("RVC runtime config updated")

    def _update_effects_display(self) -> None:
        """Read every displayed state from the shared EffectManager."""
        effect_manager = self._effect_manager()
        if effect_manager is None:
            self._effects_label.setText("Effects: (N/A)")
            self._sync_rvc_controls()
            return

        lines = [
            f"{effect.name}: {'ON' if effect.enabled else 'OFF'}"
            for effect in effect_manager.effects
        ]
        self._effects_label.setText("\n".join(lines) if lines else "No effects")

        gain_effect = effect_manager.get_by_name("GainEffect")
        if gain_effect is not None:
            slider_value = max(
                self._gain_slider.minimum(),
                min(self._gain_slider.maximum(), round(gain_effect.gain * 10)),
            )
            self._gain_slider.blockSignals(True)
            self._gain_slider.setValue(slider_value)
            self._gain_slider.blockSignals(False)
            self._gain_label.setText(f"Gain: {gain_effect.gain:.1f}")
        self._sync_rvc_controls()

    def _toggle_effect(self, effect_name: str) -> None:
        """Toggle the effect instance owned by the shared EffectManager."""
        effect_manager = self._effect_manager()
        if effect_manager is None:
            logger.warning("GUI toggle ignored: EffectManager is unavailable")
            self.statusBar().showMessage("Effect manager unavailable")
            return
        effect = effect_manager.get_by_name(effect_name)
        if effect is None:
            logger.warning("GUI toggle ignored: effect not found: {}", effect_name)
            self.statusBar().showMessage(f"Effect not found: {effect_name}")
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
            self.statusBar().showMessage("Effect not found: GainEffect")
            return
        gain_effect.gain = value / 10.0
        self._gain_label.setText(f"Gain: {gain_effect.gain:.1f}")

    def _refresh_device_choices(self) -> None:
        """Re-enumerate via DeviceManager without changing the audio stream."""
        device_manager = (
            getattr(self._context, "device_manager", None)
            if self._context
            else None
        )
        if device_manager is None:
            self._device_label.setText("Input: (N/A)\nOutput: (N/A)")
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
            self.statusBar().showMessage(f"Device refresh failed: {exc}")
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
        self.statusBar().showMessage("Device list refreshed")
        self._self_monitor_panel.refresh_devices()

    @staticmethod
    def _populate_device_combo(combo, devices, preferred_index) -> None:
        """Populate a combo with device indices as item data, never parsed text."""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("System Default", None)
        for device in devices:
            combo.addItem(f"[{device.index}] {device.name}", device.index)
        selected = combo.findData(preferred_index)
        combo.setCurrentIndex(selected if selected >= 0 else 0)
        combo.blockSignals(False)

    def _apply_devices(self) -> None:
        """Apply selected indices through the application switching transaction."""
        if self._context is None:
            self.statusBar().showMessage("Application context unavailable")
            return
        if self._input_combo.currentIndex() < 0 or self._output_combo.currentIndex() < 0:
            self.statusBar().showMessage("Select both input and output devices")
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
            self.statusBar().showMessage("Audio devices applied")
            self._self_monitor_panel.refresh_devices()
        else:
            self._update_device_display()
            self._update_status_display()
            suffix = " (previous stream restored)" if result.restored_previous_stream else ""
            self.statusBar().showMessage(f"Device switch failed{suffix}: {result.error}")

    def _update_device_display(self) -> None:
        """Display the devices used by the current context stream."""
        device_manager = (
            getattr(self._context, "device_manager", None)
            if self._context
            else None
        )
        if device_manager is None:
            self._device_label.setText("Input: (N/A)\nOutput: (N/A)")
            return
        try:
            input_name = device_manager.get_device_name(self._context.input_device)
            output_name = device_manager.get_device_name(self._context.output_device)
        except Exception as exc:
            logger.error("GUI device status refresh failed: {}", exc)
            input_name = "Unknown Device"
            output_name = "Unknown Device"
        self._device_label.setText(
            f"Current input: {input_name}\nCurrent output: {output_name}"
        )

    def _update_status_display(self) -> None:
        """Refresh stream metrics and AI readiness from current shared objects."""
        stream = (
            getattr(self._context, "audio_stream", None)
            if self._context
            else None
        )
        if stream is None:
            audio_lines = ["Audio: (N/A)"]
        elif stream.is_running:
            count = getattr(stream, "_callback_count", 0)
            total = getattr(stream, "_total_proc_ms", 0.0)
            average = total / count if count > 0 else 0.0
            audio_lines = [
                "Audio: Running",
                f"Processing avg: {average:.2f} ms",
                f"Processing max: {getattr(stream, '_max_proc_ms', 0.0):.2f} ms",
            ]
        else:
            audio_lines = ["Audio: Stopped"]

        ai_effect = None
        effect_manager = self._effect_manager()
        if effect_manager is not None:
            ai_effect = effect_manager.get_by_name("AIVoiceEffect")
        ai_ready = bool(
            ai_effect is not None
            and ai_effect.engine.is_loaded
            and ai_effect.is_running
        )
        audio_lines.append(f"AI model: {'Ready' if ai_ready else 'Not loaded'}")
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
            self.statusBar().showMessage(f"Audio start failed: {exc}")
            return
        self._update_status_display()
        self.statusBar().showMessage("Audio started")

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
            self.statusBar().showMessage(f"Audio stop failed: {exc}")
            return
        self._update_status_display()
        self.statusBar().showMessage("Audio stopped")

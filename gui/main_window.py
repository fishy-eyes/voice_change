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

if TYPE_CHECKING:
    from core.context import AppContext


class MainWindow(QMainWindow):
    """Application window bound directly to the shared runtime context."""

    def __init__(self, context: Optional[AppContext] = None) -> None:
        super().__init__()
        self._context = context
        self.setWindowTitle("Voice Changer")
        self.resize(640, 560)

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

    def _update_effects_display(self) -> None:
        """Read every displayed state from the shared EffectManager."""
        effect_manager = self._effect_manager()
        if effect_manager is None:
            self._effects_label.setText("Effects: (N/A)")
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

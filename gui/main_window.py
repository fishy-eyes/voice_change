"""Voice Changer main window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QHBoxLayout,
)
from PySide6.QtCore import Qt, QTimer

if TYPE_CHECKING:
    from core.context import AppContext


class MainWindow(QMainWindow):
    """Application main window with placeholder regions."""

    def __init__(self, context: Optional[AppContext] = None) -> None:
        super().__init__()
        self._context = context
        self.setWindowTitle("Voice Changer")
        self.resize(600, 450)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Device Info ---
        device_group = QGroupBox("Device Info")
        device_label = QLabel("Loading...")
        device_label.setStyleSheet("padding: 8px;")
        vg = QVBoxLayout(device_group)
        vg.addWidget(device_label)
        refresh_device_btn = QPushButton("刷新设备")
        refresh_device_btn.clicked.connect(self._update_device_display)
        vg.addWidget(refresh_device_btn)
        layout.addWidget(device_group)

        # --- Effects Control ---
        effects_group = QGroupBox("Effects Control")
        effects_label = QLabel(
            "Gain: 2.0  ·  Echo: OFF  ·  Robot: OFF"
        )
        effects_label.setStyleSheet("padding: 8px;")
        eg = QVBoxLayout(effects_group)
        eg.addWidget(effects_label)
        refresh_btn = QPushButton("刷新效果状态")
        refresh_btn.clicked.connect(self._update_effects_display)
        eg.addWidget(refresh_btn)
        for name in ("RobotEffect", "EchoEffect", "GainEffect"):
            btn = QPushButton(f"{name.replace('Effect', '')} 开关")
            btn.clicked.connect(lambda checked=False, n=name: self._toggle_effect(n))
            eg.addWidget(btn)
        gain_row = QHBoxLayout()
        self._gain_label = QLabel("Gain: 2.0")
        gain_slider = QSlider(Qt.Horizontal)
        gain_slider.setRange(10, 50)
        gain_slider.setValue(20)
        gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_row.addWidget(self._gain_label)
        gain_row.addWidget(gain_slider)
        eg.addLayout(gain_row)
        layout.addWidget(effects_group)

        # --- Status ---
        status_group = QGroupBox("Status")
        status_label = QLabel(
            "Audio: not started  ·  Latency: -- ms"
        )
        status_label.setStyleSheet("padding: 8px;")
        sg = QVBoxLayout(status_group)
        sg.addWidget(status_label)
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("启动音频")
        self._start_btn.clicked.connect(self._start_audio)
        self._stop_btn = QPushButton("停止音频")
        self._stop_btn.clicked.connect(self._stop_audio)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        sg.addLayout(btn_row)
        layout.addWidget(status_group)

        layout.addStretch()

        # store refs for future binding
        self._device_label = device_label
        self._effects_label = effects_label
        self._status_label = status_label

        self.statusBar().showMessage("Ready")

        # populate from context if available
        self._update_effects_display()
        self._update_device_display()

        # periodic status refresh
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_display)
        self._status_timer.start(500)

    def _update_effects_display(self) -> None:
        """Refresh the effects label from AppContext.effect_manager."""
        em = getattr(self._context, "effect_manager", None) if self._context else None
        if em is None:
            return
        lines = []
        for effect in em.effects:
            state = "ON" if effect.enabled else "OFF"
            lines.append(f"{effect.name}: {state}")
        self._effects_label.setText("  \n".join(lines))

    def _toggle_effect(self, effect_name: str) -> None:
        """Toggle an effect on/off and refresh the display."""
        em = getattr(self._context, "effect_manager", None) if self._context else None
        if em is None:
            return
        effect = em.get_by_name(effect_name)
        if effect is None:
            return
        if effect.enabled:
            em.disable(effect_name)
        else:
            em.enable(effect_name)
        self._update_effects_display()

    def _on_gain_changed(self, value: int) -> None:
        """Handle gain slider value change (integer 10-50 -> float 1.0-5.0)."""
        em = getattr(self._context, "effect_manager", None) if self._context else None
        if em is None:
            return
        effect = em.get_by_name("GainEffect")
        if effect is None:
            return
        gain = value / 10.0
        effect.gain = gain
        self._gain_label.setText(f"Gain: {gain:.1f}")

    def _update_status_display(self) -> None:
        """Periodically refresh the status label from AudioStream."""
        stream = getattr(self._context, "audio_stream", None) if self._context else None
        if stream is None:
            return
        if stream.is_running:
            count = stream._callback_count
            avg = (stream._total_proc_ms / count) if count > 0 else 0.0
            text = (
                f"Audio: Running\n"
                f"Processing avg: {avg:.2f} ms\n"
                f"Processing max: {stream._max_proc_ms:.2f} ms"
            )
        else:
            text = "Audio: Stopped"
        self._status_label.setText(text)

    def _update_device_display(self) -> None:
        """Refresh the device label from AppContext."""
        dm = getattr(self._context, "device_manager", None) if self._context else None
        if dm is None:
            self._device_label.setText("Input: (N/A)\nOutput: (N/A)")
            return
        try:
            input_name = dm.get_device_name(self._context.input_device)
            output_name = dm.get_device_name(self._context.output_device)
        except Exception:
            input_name = "Unknown Device"
            output_name = "Unknown Device"
        self._device_label.setText(f"Input: {input_name}\nOutput: {output_name}")
        self._update_status_display()

    def _start_audio(self) -> None:
        """Start the audio stream via AppContext."""
        stream = getattr(self._context, "audio_stream", None) if self._context else None
        if stream is None:
            return
        if stream.is_running:
            return
        stream.start()
        self._update_status_display()
        self.statusBar().showMessage("Audio started")

    def _stop_audio(self) -> None:
        """Stop the audio stream via AppContext."""
        stream = getattr(self._context, "audio_stream", None) if self._context else None
        if stream is None:
            return
        if not stream.is_running:
            return
        stream.stop()
        self._update_status_display()
        self.statusBar().showMessage("Audio stopped")

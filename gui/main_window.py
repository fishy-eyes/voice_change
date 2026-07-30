"""Voice Changer main window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QGroupBox, QLabel,
    QPushButton,
)

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
        device_label = QLabel(
            "[Input] System Default Mic  ·  [Output] VB-CABLE"
        )
        device_label.setStyleSheet("padding: 8px;")
        vg = QVBoxLayout(device_group)
        vg.addWidget(device_label)
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
        layout.addWidget(effects_group)

        # --- Status ---
        status_group = QGroupBox("Status")
        status_label = QLabel(
            "Audio: not started  ·  Latency: -- ms"
        )
        status_label.setStyleSheet("padding: 8px;")
        sg = QVBoxLayout(status_group)
        sg.addWidget(status_label)
        layout.addWidget(status_group)

        layout.addStretch()

        # store refs for future binding
        self._device_label = device_label
        self._effects_label = effects_label
        self._status_label = status_label

        self.statusBar().showMessage("Ready")

        # populate from context if available
        self._update_effects_display()

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

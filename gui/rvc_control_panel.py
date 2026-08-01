"""Developer-facing RVC model selection and enable controls."""

from __future__ import annotations

from typing import Callable

from loguru import logger
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from config.rvc_realtime import (
    RVC_DEFAULT_REALTIME_PRESET,
    RVC_REALTIME_PRESETS,
)


class RVCControlPanel(QGroupBox):
    """Thin GUI adapter around the application-owned ``RVCRuntime``."""

    def __init__(self, context=None, on_changed: Callable[[], None] | None = None):
        super().__init__("AI Voice")
        self._context = context
        self._on_changed = on_changed

        layout = QVBoxLayout(self)
        self.enable_checkbox = QCheckBox("Enable AI Voice")
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_checkbox)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        model_row.addWidget(self.model_combo)
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self._load_selected_model)
        model_row.addWidget(self.load_button)
        layout.addLayout(model_row)

        realtime_row = QHBoxLayout()
        realtime_row.addWidget(QLabel("Realtime Mode:"))
        self.realtime_combo = QComboBox()
        for key, preset in RVC_REALTIME_PRESETS.items():
            self.realtime_combo.addItem(preset.name, key)
        self.realtime_combo.currentIndexChanged.connect(
            self._on_realtime_mode_changed
        )
        realtime_row.addWidget(self.realtime_combo)
        layout.addLayout(realtime_row)

        self.realtime_detail_label = QLabel()
        layout.addWidget(self.realtime_detail_label)

        self.status_label = QLabel("Not loaded")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.refresh_models()

    def _runtime(self):
        return (
            getattr(self._context, "rvc_runtime", None)
            if self._context is not None
            else None
        )

    def _refresh_realtime_mode(self) -> None:
        runtime = self._runtime()
        available = runtime is not None and callable(
            getattr(runtime, "set_realtime_preset", None)
        )
        self.realtime_combo.setEnabled(available)
        if not available:
            self.realtime_detail_label.setText("Realtime mode unavailable")
            return

        key = getattr(runtime, "realtime_preset_key", None)
        if key is None:
            key = RVC_DEFAULT_REALTIME_PRESET
        index = self.realtime_combo.findData(key)
        self.realtime_combo.blockSignals(True)
        self.realtime_combo.setCurrentIndex(index if index >= 0 else 0)
        self.realtime_combo.blockSignals(False)
        self._update_realtime_detail()

    def _update_realtime_detail(self) -> None:
        key = self.realtime_combo.currentData()
        preset = RVC_REALTIME_PRESETS.get(key)
        if preset is None:
            self.realtime_detail_label.setText("Custom realtime settings")
            return
        self.realtime_detail_label.setText(
            f"{preset.name}\n"
            f"{preset.chunk_ms}ms chunk / {preset.overlap_ms}ms overlap"
        )

    def refresh_models(self) -> None:
        runtime = self._runtime()
        self._refresh_realtime_mode()
        selected = getattr(runtime, "selected_model", None)
        self.model_combo.clear()
        if runtime is None:
            self.model_combo.setEnabled(False)
            self.load_button.setEnabled(False)
            self.enable_checkbox.setEnabled(False)
            self.status_label.setText("RVC runtime unavailable")
            return

        self.enable_checkbox.blockSignals(True)
        self.enable_checkbox.setChecked(bool(runtime.enabled))
        self.enable_checkbox.blockSignals(False)
        try:
            models = runtime.model_manager.discover_models()
        except Exception as exc:
            logger.error("GUI RVC model discovery failed: {}", exc)
            models = []
            self.status_label.setText(f"Model discovery failed: {exc}")
        for descriptor in models:
            self.model_combo.addItem(descriptor.name, descriptor.name)
        model_index = self.model_combo.findData(selected)
        if model_index >= 0:
            self.model_combo.setCurrentIndex(model_index)
        has_models = bool(models)
        self.model_combo.setEnabled(has_models)
        self.load_button.setEnabled(has_models)
        self.enable_checkbox.setEnabled(True)
        self.update_status()

    def update_status(self) -> None:
        runtime = self._runtime()
        if runtime is None:
            self.status_label.setText("RVC runtime unavailable")
            return
        state = runtime.state
        if state.ready and runtime.selected_model:
            mode = "Enabled" if runtime.enabled else "Bypassed"
            self.status_label.setText(f"Loaded: {runtime.selected_model} ({mode})")
        elif state.error:
            self.status_label.setText(f"Not loaded: {state.error}")
        else:
            self.status_label.setText("Not loaded")

    def _on_enabled_toggled(self, enabled: bool) -> None:
        runtime = self._runtime()
        if runtime is None:
            return
        runtime.set_enabled(enabled)
        self.update_status()
        if self._on_changed is not None:
            self._on_changed()

    def _on_realtime_mode_changed(self, _index: int) -> None:
        runtime = self._runtime()
        key = self.realtime_combo.currentData()
        if runtime is None or not key:
            return
        try:
            runtime.set_realtime_preset(key)
        except Exception as exc:
            logger.error("GUI RVC realtime preset update failed: {}", exc)
            self.status_label.setText(f"Realtime mode update failed: {exc}")
            self._refresh_realtime_mode()
            return
        self._update_realtime_detail()
        if self._on_changed is not None:
            self._on_changed()

    def _load_selected_model(self) -> None:
        runtime = self._runtime()
        model_name = self.model_combo.currentData()
        if runtime is None or not model_name:
            return
        stream = getattr(self._context, "audio_stream", None)
        self.load_button.setEnabled(False)
        self.status_label.setText(f"Loading: {model_name} ...")
        QApplication.processEvents()
        try:
            state = runtime.load_model(model_name, audio_stream=stream)
            if not state.ready:
                logger.error("GUI RVC model load failed: {}", state.error)
        except Exception as exc:
            logger.error("GUI RVC model load failed: {}", exc)
        finally:
            self.load_button.setEnabled(self.model_combo.count() > 0)
            self.update_status()
            if self._on_changed is not None:
                self._on_changed()

"""RVC-only realtime settings panel."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Mapping

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.rvc_realtime import RVC_REALTIME_PRESETS
from gui.backend_settings.base import BackendSettingsPanel
from gui.customization_dialog import CustomizationDialog


class RVCSettingsPanel(QWidget, BackendSettingsPanel):
    backend_id = "rvc"

    def __init__(self, *, manager, context=None, language="zh", on_models_changed: Callable[[], None] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._context = context
        self._language = language
        self._refreshing = False

        self._on_models_changed = on_models_changed
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(-24, 24)
        self.index_spin = self._rate_spin(0.0, 1.0)
        self.protect_spin = self._rate_spin(0.0, 0.5)
        self.rms_spin = self._rate_spin(0.0, 1.0)
        self.realtime_combo = QComboBox()
        for key, preset in RVC_REALTIME_PRESETS.items():
            self.realtime_combo.addItem(
                f"{preset.name} ({preset.chunk_ms}/{preset.overlap_ms} ms)", key
            )
        form.addRow("Pitch", self.pitch_spin)
        form.addRow("Index Rate", self.index_spin)
        form.addRow("Protect", self.protect_spin)
        form.addRow("RMS Mix Rate", self.rms_spin)
        form.addRow("Realtime Preset", self.realtime_combo)
        layout.addLayout(form)

        self.customize_button = QPushButton(
            "智能定制…" if language == "zh" else "Smart Customization…"
        )
        self.customize_button.clicked.connect(self._open_customization)
        layout.addWidget(self.customize_button)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.import_button = QPushButton(
            "导入 RVC 模型…" if language == "zh" else "Import RVC Model…"
        )
        self.import_button.clicked.connect(self._import_model)
        layout.addWidget(self.import_button)

        self.pitch_spin.valueChanged.connect(self._apply_controls)
        self.index_spin.valueChanged.connect(self._apply_controls)
        self.protect_spin.valueChanged.connect(self._apply_controls)
        self.rms_spin.valueChanged.connect(self._apply_controls)
        self.realtime_combo.currentIndexChanged.connect(self._apply_realtime)
        self.refresh_from_runtime()

    @staticmethod
    def _rate_spin(minimum: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        return spin

    def refresh_from_runtime(self) -> None:
        self.apply_state(self._manager.get_current_parameters())
        preset = self._manager.get_realtime_preset()
        index = self.realtime_combo.findData(preset)
        self.realtime_combo.blockSignals(True)
        self.realtime_combo.setCurrentIndex(index if index >= 0 else 0)
        self.realtime_combo.blockSignals(False)
        descriptor = self._manager.get_current_model_descriptor()
        loaded = descriptor is not None and self._manager.get_status().state == "LOADED"
        for control in (
            self.pitch_spin, self.index_spin, self.protect_spin,
            self.rms_spin, self.realtime_combo,
        ):
            control.setEnabled(loaded)
        self.customize_button.setEnabled(loaded)
        self.import_button.setEnabled(True)
        self.status_label.setText(
            ("参数实时生效。" if self._language == "zh" else "Changes apply in realtime.")
            if loaded
            else ("请先加载 RVC 模型。" if self._language == "zh" else "Load an RVC model first.")
        )

    def apply_state(self, state: Mapping[str, Any]) -> None:
        self._refreshing = True
        controls = (
            (self.pitch_spin, state.get("pitch_shift", 0)),
            (self.index_spin, state.get("index_rate", 0.0)),
            (self.protect_spin, state.get("protect", 0.33)),
            (self.rms_spin, state.get("rms_mix_rate", 0.25)),
        )
        for control, value in controls:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._refreshing = False

    def close_panel(self) -> None:
        self.close()

    def _apply_controls(self, _value=None) -> None:
        if self._refreshing:
            return
        try:
            self._manager.update_current_parameters(
                pitch_shift=self.pitch_spin.value(),
                index_rate=self.index_spin.value(),
                protect=self.protect_spin.value(),
                rms_mix_rate=self.rms_spin.value(),
            )
            self.status_label.setText(
                "参数已应用。" if self._language == "zh" else "Parameters applied."
            )
        except Exception as exc:
            logger.error("RVC settings update failed: {}", exc)
            self.status_label.setText(str(exc))

    def _apply_realtime(self, _index: int) -> None:
        if self._refreshing:
            return
        key = self.realtime_combo.currentData()
        if not key:
            return
        try:
            self._manager.set_realtime_preset(key)
        except Exception as exc:
            logger.error("RVC realtime preset update failed: {}", exc)
            self.status_label.setText(str(exc))

    def _import_model(self) -> None:
        runtime = self._manager.current_runtime
        model_manager = getattr(runtime, "model_manager", None)
        if model_manager is None:
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择 RVC 模型目录" if self._language == "zh" else "Choose RVC Model Folder",
            "",
        )
        if not directory:
            return
        try:
            inspection = model_manager.inspect_import_directory(directory)
            if not inspection.pth_candidates:
                raise FileNotFoundError("No .pth model found in selected directory")
            pth_path = self._choose_candidate("RVC .pth", inspection.pth_candidates)
            if pth_path is None:
                return
            index_path = (
                self._choose_candidate("RVC .index", inspection.index_candidates)
                if inspection.index_candidates
                else None
            )
            if inspection.index_candidates and index_path is None:
                return
            descriptor = model_manager.import_model(
                inspection.directory,
                pth_path=pth_path,
                index_path=index_path,
            )
        except Exception as exc:
            logger.error("RVC model import failed: {}", exc)
            QMessageBox.warning(self, "RVC", str(exc))
            return
        if self._on_models_changed is not None:
            self._on_models_changed()
        self.status_label.setText(
            f"已导入：{descriptor.name}"
            if self._language == "zh"
            else f"Imported: {descriptor.name}"
        )

    def _choose_candidate(
        self,
        title: str,
        candidates: Iterable[Path],
    ) -> Path | None:
        options = tuple(candidates)
        if len(options) == 1:
            return options[0]
        names = [path.name for path in options]
        selected, accepted = QInputDialog.getItem(
            self, title, "File", names, 0, False
        )
        if not accepted:
            return None
        return options[names.index(selected)]

    def _open_customization(self) -> None:
        descriptor = self._manager.get_current_model_descriptor()
        if descriptor is None or self._context is None:
            QMessageBox.warning(self, "RVC", "Please load an RVC model first.")
            return
        dialog = CustomizationDialog(
            self._context,
            descriptor,
            language=self._language,
            parent=self,
        )
        dialog.exec()
        self.refresh_from_runtime()

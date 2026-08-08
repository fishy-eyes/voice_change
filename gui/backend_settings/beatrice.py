"""Registered settings panel for the Beatrice backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from loguru import logger
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai.voice_engine.beatrice import (
    DEFAULT_MAX_SOURCE_PITCH,
    DEFAULT_MIN_SOURCE_PITCH,
)
from gui.backend_settings.base import BackendSettingsPanel
from gui.i18n import tr


class BeatriceSettingsPanel(QWidget, BackendSettingsPanel):
    backend_id = "beatrice"

    def __init__(
        self,
        *,
        manager,
        context=None,
        language="zh",
        on_models_changed=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._context = context
        self._language = "zh" if language == "zh" else "en"
        self._on_models_changed = on_models_changed
        self._refreshing = False
        layout = QVBoxLayout(self)

        self.runtime_group = QGroupBox(
            tr(self._language, "beatrice.runtime_group")
        )
        runtime_layout = QVBoxLayout(self.runtime_group)
        self.runtime_path_label = QLabel()
        self.runtime_path_label.setWordWrap(True)
        runtime_layout.addWidget(self.runtime_path_label)
        self.runtime_select_button = QPushButton(
            tr(self._language, "beatrice.select_runtime")
        )
        self.runtime_select_button.clicked.connect(self._select_runtime_folder)
        runtime_layout.addWidget(self.runtime_select_button)
        self.runtime_status_label = QLabel()
        self.runtime_status_label.setWordWrap(True)
        runtime_layout.addWidget(self.runtime_status_label)
        layout.addWidget(self.runtime_group)

        self.registry_group = QGroupBox(
            tr(self._language, "beatrice.registered_model_folders")
        )
        registry_layout = QVBoxLayout(self.registry_group)
        self.registered_list = QListWidget()
        registry_layout.addWidget(self.registered_list)
        registry_buttons = QHBoxLayout()
        self.add_model_button = QPushButton(
            tr(self._language, "beatrice.add_model")
        )
        self.remove_model_button = QPushButton(
            tr(self._language, "beatrice.remove_from_list")
        )
        self.add_model_button.clicked.connect(self._add_model_folder)
        self.remove_model_button.clicked.connect(self._remove_model_folder)
        registry_buttons.addWidget(self.add_model_button)
        registry_buttons.addWidget(self.remove_model_button)
        registry_layout.addLayout(registry_buttons)
        layout.addWidget(self.registry_group)
        form = QFormLayout()
        self.target_combo = QComboBox()
        self.pitch_spin = self._float_spin(-24.0, 24.0, 0.25)
        self.formant_spin = self._float_spin(-16.0, 16.0, 0.25)
        self.min_pitch_spin = self._float_spin(20.0, 2000.0, 5.0)
        self.max_pitch_spin = self._float_spin(20.0, 4000.0, 10.0)
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(1, 256)
        form.addRow(tr(self._language, "beatrice.target_speaker"), self.target_combo)
        form.addRow(tr(self._language, "beatrice.pitch_shift"), self.pitch_spin)
        form.addRow(tr(self._language, "beatrice.formant_shift"), self.formant_spin)
        form.addRow(tr(self._language, "beatrice.min_pitch"), self.min_pitch_spin)
        form.addRow(tr(self._language, "beatrice.max_pitch"), self.max_pitch_spin)
        form.addRow(tr(self._language, "beatrice.vq_neighbors"), self.neighbors_spin)
        layout.addLayout(form)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.customize_button = QPushButton(
            "智能辅助调参…" if self._language == "zh" else "Assisted Tuning…"
        )
        self.customize_button.clicked.connect(self._open_customization)
        layout.addWidget(self.customize_button)
        layout.addStretch(1)

        self.target_combo.currentIndexChanged.connect(self._apply_controls)
        self.pitch_spin.valueChanged.connect(self._apply_controls)
        self.formant_spin.valueChanged.connect(self._apply_controls)
        self.min_pitch_spin.valueChanged.connect(self._apply_controls)
        self.max_pitch_spin.valueChanged.connect(self._apply_controls)
        self.neighbors_spin.valueChanged.connect(self._apply_controls)
        self.refresh_from_runtime()

    @staticmethod
    def _float_spin(minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        return spin

    def _runtime(self):
        return getattr(self._manager, "current_runtime", None)

    def _refresh_runtime_configuration(self) -> None:
        runtime = self._runtime()
        path = getattr(runtime, "runtime_path", None)
        if path is None:
            self.runtime_path_label.setText(
                tr(self._language, "beatrice.runtime_not_configured")
            )
            self.runtime_status_label.setText(
                tr(self._language, "beatrice.runtime_not_configured")
            )
        else:
            self.runtime_path_label.setText(str(path))
            status = getattr(runtime, "runtime_path_status", None)
            try:
                details = status() if callable(status) else {}
                if not details.get("available", False):
                    raise RuntimeError("Beatrice package was not found in this folder")
                version = details.get("version", "2.0.0-rc.0")
                self.runtime_status_label.setText(
                    tr(
                        self._language,
                        "beatrice.runtime_available_version",
                        version=version,
                    )
                )
            except Exception as exc:
                self.runtime_status_label.setText(
                    tr(
                        self._language,
                        "beatrice.runtime_invalid_detail",
                        error=str(exc),
                    )
                )
        self.registered_list.clear()
        paths = tuple(getattr(runtime, "registered_model_paths", ()))
        for registered in paths:
            self.registered_list.addItem(str(registered))
        self.remove_model_button.setEnabled(bool(paths))


    def refresh_from_runtime(self) -> None:
        self._refresh_runtime_configuration()
        descriptor = self._manager.get_current_model_descriptor()
        speaker_names = tuple(getattr(descriptor, "speaker_names", ()))
        current = self.target_combo.currentData()
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        for index, name in enumerate(speaker_names):
            self.target_combo.addItem(f"{index}: {name}", index)
        selected = self.target_combo.findData(current)
        self.target_combo.setCurrentIndex(selected if selected >= 0 else 0)
        self.target_combo.blockSignals(False)
        self.apply_state(self._manager.get_current_parameters())

        status = self._manager.get_status()
        loaded = descriptor is not None and status.state == "LOADED"
        for control in (
            self.target_combo,
            self.pitch_spin,
            self.formant_spin,
            self.min_pitch_spin,
            self.max_pitch_spin,
            self.neighbors_spin,
        ):
            control.setEnabled(loaded)
        self.customize_button.setEnabled(loaded)
        if loaded:
            capabilities = getattr(self._runtime(), "get_tuning_capabilities", lambda: None)()
            if capabilities is not None:
                self.pitch_spin.setRange(
                    capabilities.pitch_shift_min, capabilities.pitch_shift_max
                )
                maximum = capabilities.max_formant_shift
                self.formant_spin.setRange(-maximum, maximum)
                self.neighbors_spin.setRange(1, capabilities.codebook_size)
        info = self._manager.get_info()
        runtime = info.get("runtime", {}) if isinstance(info, Mapping) else {}
        model_name = getattr(descriptor, "model_name", None) or tr(
            self._language, "runtime.na"
        )
        version = runtime.get("version") or getattr(
            descriptor, "runtime_requirement", None
        ) or tr(self._language, "runtime.na")
        count = getattr(descriptor, "speaker_count", 0)
        self.info_label.setText(
            tr(
                self._language,
                "beatrice.info",
                model=model_name,
                version=version,
                speakers=count,
            )
        )
        self.status_label.setText(
            tr(
                self._language,
                "beatrice.ready" if loaded else "beatrice.load_first",
            )
        )

    def _select_runtime_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, tr(self._language, "beatrice.select_runtime"), ""
        )
        if not directory:
            return
        runtime = self._runtime()
        configure = getattr(runtime, "configure_runtime", None)
        try:
            if not callable(configure):
                raise RuntimeError("Beatrice runtime configuration is unavailable")
            details = configure(Path(directory))
        except Exception as exc:
            logger.error("Beatrice runtime selection failed: {}", exc)
            QMessageBox.warning(
                self,
                tr(self._language, "beatrice.runtime_invalid"),
                str(exc),
            )
            self._refresh_runtime_configuration()
            return
        self.runtime_path_label.setText(str(Path(directory).resolve()))
        self.runtime_status_label.setText(
            tr(
                self._language,
                "beatrice.runtime_available_version",
                version=details.get("version", "2.0.0-rc.0"),
            )
        )

    def _add_model_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, tr(self._language, "beatrice.select_model_folder"), ""
        )
        if not directory:
            return
        try:
            descriptor, added = self._manager.add_model_path(directory)
        except Exception as exc:
            logger.error("Beatrice model registration failed: {}", exc)
            QMessageBox.warning(
                self,
                tr(self._language, "beatrice.invalid_model"),
                str(exc),
            )
            return
        self._refresh_runtime_configuration()
        if self._on_models_changed is not None:
            self._on_models_changed()
        key = "beatrice.model_folder_added" if added else "beatrice.model_folder_exists"
        self.status_label.setText(
            tr(
                self._language,
                key,
                model=getattr(descriptor, "package", descriptor.name),
            )
        )

    def _remove_model_folder(self) -> None:
        item = self.registered_list.currentItem()
        if item is None:
            return
        if self._manager.remove_model_path(item.text()):
            self._refresh_runtime_configuration()
            if self._on_models_changed is not None:
                self._on_models_changed()

    def apply_state(self, state: Mapping[str, Any]) -> None:
        self._refreshing = True
        controls = (
            (self.pitch_spin, state.get("pitch_shift_semitone", 0.0)),
            (self.formant_spin, state.get("formant_shift", 0.0)),
            (
                self.min_pitch_spin,
                state.get("min_source_pitch", DEFAULT_MIN_SOURCE_PITCH),
            ),
            (
                self.max_pitch_spin,
                state.get("max_source_pitch", DEFAULT_MAX_SOURCE_PITCH),
            ),
            (self.neighbors_spin, state.get("vq_num_neighbors", 4)),
        )
        for control, value in controls:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        target = self.target_combo.findData(state.get("target_speaker", 0))
        self.target_combo.blockSignals(True)
        self.target_combo.setCurrentIndex(target if target >= 0 else 0)
        self.target_combo.blockSignals(False)
        self._refreshing = False

    def close_panel(self) -> None:
        self.close()

    def _apply_controls(self, _value=None) -> None:
        if self._refreshing or self.target_combo.currentData() is None:
            return
        try:
            state = self._manager.update_current_parameters(
                target_speaker=int(self.target_combo.currentData()),
                pitch_shift_semitone=self.pitch_spin.value(),
                formant_shift=self.formant_spin.value(),
                min_source_pitch=self.min_pitch_spin.value(),
                max_source_pitch=self.max_pitch_spin.value(),
                vq_num_neighbors=self.neighbors_spin.value(),
            )
            self.apply_state(state)
            self.status_label.setText(tr(self._language, "beatrice.applied"))
        except Exception as exc:
            logger.error("Beatrice settings update failed: {}", exc)
            self.status_label.setText(str(exc))

    def _open_customization(self) -> None:
        descriptor = self._manager.get_current_model_descriptor()
        if descriptor is None or self._context is None:
            QMessageBox.warning(self, "Beatrice", tr(self._language, "beatrice.load_first"))
            return
        from gui.beatrice_customization_dialog import BeatriceCustomizationDialog

        dialog = BeatriceCustomizationDialog(
            self._context,
            descriptor,
            language=self._language,
            parent=self,
        )
        dialog.exec()
        self.refresh_from_runtime()


__all__ = ["BeatriceSettingsPanel"]

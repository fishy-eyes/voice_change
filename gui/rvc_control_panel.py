"""Developer-facing RVC model selection, import and enable controls."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from loguru import logger
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
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
        self._descriptors: dict[str, object] = {}

        layout = QVBoxLayout(self)
        self.enable_checkbox = QCheckBox("Enable AI Voice")
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_checkbox)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(
            self._on_model_selection_changed
        )
        model_row.addWidget(self.model_combo)
        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self._load_selected_model)
        model_row.addWidget(self.load_button)
        self.import_button = QPushButton("导入RVC模型")
        self.import_button.clicked.connect(self._import_model)
        model_row.addWidget(self.import_button)
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
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self._descriptors.clear()
        if runtime is None:
            self.model_combo.blockSignals(False)
            self.model_combo.setEnabled(False)
            self.load_button.setEnabled(False)
            self.import_button.setEnabled(False)
            self.enable_checkbox.setEnabled(False)
            self.status_label.setText("RVC runtime unavailable")
            return

        manager = getattr(runtime, "model_manager", None)
        can_import = manager is not None and callable(
            getattr(manager, "import_model", None)
        )
        self.import_button.setEnabled(can_import)
        self.enable_checkbox.blockSignals(True)
        self.enable_checkbox.setChecked(bool(runtime.enabled))
        self.enable_checkbox.blockSignals(False)
        try:
            models = manager.discover_models() if manager is not None else []
        except Exception as exc:
            logger.error("GUI RVC model discovery failed: {}", exc)
            models = []
            self.status_label.setText(f"Model discovery failed: {exc}")
        for descriptor in models:
            self._descriptors[descriptor.name] = descriptor
            self.model_combo.addItem(descriptor.name, descriptor.name)
        model_index = self.model_combo.findData(selected)
        if model_index >= 0:
            self.model_combo.setCurrentIndex(model_index)
        self.model_combo.blockSignals(False)
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
        descriptor = self._current_descriptor()
        state = runtime.state
        if state.ready and runtime.selected_model:
            mode = "Enabled" if runtime.enabled else "Bypassed"
            heading = f"Loaded: {runtime.selected_model} ({mode})"
            try:
                descriptor = runtime.model_manager.get_model(runtime.selected_model)
            except Exception:
                pass
        elif state.error:
            heading = f"Not loaded: {state.error}"
        else:
            heading = "Not loaded"
        details = self._format_descriptor_status(descriptor)
        self.status_label.setText(
            heading if not details else f"{heading}\n{details}"
        )

    @staticmethod
    def _format_descriptor_status(descriptor) -> str:
        if descriptor is None or not hasattr(descriptor, "pth_path"):
            return ""
        pth_path = getattr(descriptor, "pth_path", None)
        index_path = getattr(descriptor, "index_path", None)
        profile_path = getattr(descriptor, "profile_path", None)
        profile_is_default = bool(
            getattr(descriptor, "profile_is_default", profile_path is None)
        )
        pth_status = f"✓ {Path(pth_path).name}" if pth_path else "missing"
        index_status = f"✓ {Path(index_path).name}" if index_path else "none"
        profile_status = (
            "default"
            if profile_is_default or profile_path is None
            else Path(profile_path).name
        )
        return (
            f"pth: {pth_status}\n"
            f"index: {index_status}\n"
            f"profile: {profile_status}"
        )

    def _current_descriptor(self):
        name = self.model_combo.currentData()
        return self._descriptors.get(name)

    def _on_model_selection_changed(self, _index: int) -> None:
        self.update_status()

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

    def _import_model(self) -> None:
        runtime = self._runtime()
        manager = getattr(runtime, "model_manager", None) if runtime else None
        if manager is None:
            return
        directory = self._choose_import_directory()
        if not directory:
            return
        try:
            inspection = manager.inspect_import_directory(directory)
            if not inspection.pth_candidates:
                self._show_warning("未发现RVC模型权重文件")
                return
            pth_path = self._choose_candidate(
                "选择RVC模型权重",
                "检测到多个 .pth 文件，请选择要导入的模型：",
                inspection.pth_candidates,
            )
            if pth_path is None:
                return

            if not inspection.index_candidates:
                index_path = None
                self._show_information("未找到index，将使用无index模式")
            else:
                index_path = self._choose_candidate(
                    "选择RVC索引",
                    "检测到多个 .index 文件，请选择要关联的索引：",
                    inspection.index_candidates,
                )
                if index_path is None:
                    return

            descriptor = manager.import_model(
                inspection.directory,
                pth_path=pth_path,
                index_path=index_path,
            )
        except Exception as exc:
            logger.error("GUI RVC model import failed: {}", exc)
            self._show_warning(f"导入RVC模型失败：{exc}")
            return

        self.refresh_models()
        selected_index = self.model_combo.findData(descriptor.name)
        if selected_index >= 0:
            self.model_combo.setCurrentIndex(selected_index)
        self._load_selected_model()

    def _choose_import_directory(self) -> str:
        return QFileDialog.getExistingDirectory(
            self,
            "选择RVC模型文件夹",
            "",
        )

    def _choose_candidate(
        self,
        title: str,
        label: str,
        candidates: Iterable[Path],
    ) -> Path | None:
        options = tuple(candidates)
        if len(options) == 1:
            return options[0]
        names = [path.name for path in options]
        selected, accepted = QInputDialog.getItem(
            self,
            title,
            label,
            names,
            0,
            False,
        )
        if not accepted:
            return None
        return options[names.index(selected)]

    def _show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "RVC模型导入", message)

    def _show_information(self, message: str) -> None:
        QMessageBox.information(self, "RVC模型导入", message)

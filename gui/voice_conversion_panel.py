"""Compact backend-neutral model controls for the main window."""

from __future__ import annotations

from concurrent.futures import Future

from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from gui.backend_settings import create_default_registry
from gui.i18n import tr


class VoiceConversionPanel(QGroupBox):
    """Select/load models without duplicating runtime status in this panel."""

    def __init__(self, context=None, on_changed=None, *, language="zh") -> None:
        super().__init__()
        self._context = context
        self._on_changed = on_changed
        self._language = "zh" if language == "zh" else "en"
        self._switch_future: Future | None = None
        self._descriptors: dict[str, object] = {}
        self._settings_registry = create_default_registry()
        self._settings_dialog: QDialog | None = None
        self._settings_panel = None

        layout = QGridLayout(self)
        layout.setContentsMargins(14, 18, 14, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        layout.setColumnStretch(1, 1)

        self.enable_checkbox = QCheckBox()
        self.enable_checkbox.toggled.connect(self._on_enabled_toggled)
        layout.addWidget(self.enable_checkbox, 0, 0, 1, 3)

        self.backend_label = QLabel()
        self.backend_combo = QComboBox()
        self.backend_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        layout.addWidget(self.backend_label, 1, 0)
        layout.addWidget(self.backend_combo, 1, 1, 1, 2)

        self.model_label = QLabel()
        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.model_combo.setMinimumContentsLength(16)
        self.model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.refresh_button = QPushButton()
        self.refresh_button.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred
        )
        self.refresh_button.clicked.connect(self.refresh_models)
        layout.addWidget(self.model_label, 2, 0)
        layout.addWidget(self.model_combo, 2, 1)
        layout.addWidget(self.refresh_button, 2, 2)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.load_button = QPushButton()
        self.load_button.clicked.connect(self._load_selected_model)
        self.settings_button = QPushButton()
        self.settings_button.clicked.connect(self._open_settings)
        buttons.addWidget(self.load_button, 1)
        buttons.addWidget(self.settings_button, 1)
        layout.addLayout(buttons, 3, 1, 1, 2)

        self._refresh_backends()
        self.refresh_models()

    def _t(self, key: str, **values) -> str:
        return tr(self._language, key, **values)

    def _manager(self):
        return getattr(self._context, "voice_conversion_manager", None)

    def set_language(self, language: str) -> None:
        self._language = "zh" if language == "zh" else "en"
        self.setTitle(self._t("vc.group"))
        self.enable_checkbox.setText(self._t("vc.enable"))
        self.backend_label.setText(self._t("vc.backend"))
        self.model_label.setText(self._t("vc.model"))
        self.refresh_button.setText(self._t("vc.refresh"))
        self.load_button.setText(self._t("vc.load"))
        self.settings_button.setText(self._t("vc.settings"))

    def _refresh_backends(self) -> None:
        manager = self._manager()
        backends = manager.available_backends if manager is not None else ()
        selected = manager.current_backend if manager is not None else None
        self.backend_combo.blockSignals(True)
        self.backend_combo.clear()
        for backend in backends:
            self.backend_combo.addItem(str(backend).upper(), backend)
        index = self.backend_combo.findData(selected)
        self.backend_combo.setCurrentIndex(index if index >= 0 else 0)
        self.backend_combo.setEnabled(len(backends) > 1)
        self.backend_combo.blockSignals(False)
        self.setEnabled(manager is not None)
        self.set_language(self._language)

    def refresh_models(self) -> None:
        manager = self._manager()
        current = self.model_combo.currentData()
        self._descriptors.clear()
        self.model_combo.clear()
        if manager is not None:
            try:
                for descriptor in manager.discover_models():
                    name = str(descriptor.name)
                    self._descriptors[name] = descriptor
                    self.model_combo.addItem(name, name)
            except Exception as exc:
                logger.error("model discovery failed: {}", exc)
        index = self.model_combo.findData(current)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.update_status()

    def _on_backend_changed(self, _index: int) -> None:
        manager = self._manager()
        backend = self.backend_combo.currentData()
        if manager is None or not backend:
            return
        try:
            manager.select_backend(backend)
        except Exception as exc:
            logger.error("backend selection failed: {}", exc)
            self._refresh_backends()
            return
        self.refresh_models()

    def _on_enabled_toggled(self, enabled: bool) -> None:
        manager = self._manager()
        if manager is not None:
            manager.set_enabled(enabled)
        self.update_status()
        if self._on_changed is not None:
            self._on_changed()

    def _load_selected_model(self) -> None:
        manager = self._manager()
        backend = self.backend_combo.currentData()
        model = self.model_combo.currentData()
        if manager is None or not backend or not model:
            return
        self.load_button.setEnabled(False)
        self.settings_button.setEnabled(False)
        try:
            self._switch_future = manager.switch_model_async(
                backend,
                model,
                audio_stream=getattr(self._context, "audio_stream", None),
            )
        except Exception as exc:
            logger.error("GUI model switch rejected: {}", exc)
            self._switch_future = None
            self.setToolTip(str(exc))
            self.update_status()
            if self._on_changed is not None:
                self._on_changed()
            return
        self.setToolTip("")
        self.update_status()
        QTimer.singleShot(50, self._poll_model_switch)

    def _poll_model_switch(self) -> None:
        future = self._switch_future
        if future is None:
            return
        if not future.done():
            QTimer.singleShot(50, self._poll_model_switch)
            return
        try:
            future.result()
        except Exception as exc:
            logger.error("GUI model switch failed: {}", exc)
            self.setToolTip(str(exc))
        self._switch_future = None
        if self._settings_panel is not None:
            self._settings_panel.refresh_from_runtime()
        self.update_status()
        if self._on_changed is not None:
            self._on_changed()

    def _open_settings(self) -> None:
        manager = self._manager()
        backend = self.backend_combo.currentData()
        if manager is None or not backend:
            return
        if self._settings_dialog is not None:
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        try:
            panel = self._settings_registry.create(
                backend,
                manager=manager,
                context=self._context,
                language=self._language,
                on_models_changed=self.refresh_models,
            )
        except Exception as exc:
            logger.error("backend settings unavailable: {}", exc)
            self.setToolTip(str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(
            self._t("vc.settings_title", backend=str(backend).upper())
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setMinimumWidth(440)
        QVBoxLayout(dialog).addWidget(panel)
        dialog.finished.connect(self._settings_closed)
        self._settings_dialog = dialog
        self._settings_panel = panel
        dialog.show()

    def _settings_closed(self, _result: int) -> None:
        self._settings_dialog = None
        self._settings_panel = None

    def update_status(self) -> None:
        """Refresh control state only; runtime fields belong to MainWindow."""
        manager = self._manager()
        if manager is None:
            self.load_button.setEnabled(False)
            self.settings_button.setEnabled(False)
            return
        self.enable_checkbox.blockSignals(True)
        self.enable_checkbox.setChecked(bool(manager.requested_enabled))
        self.enable_checkbox.blockSignals(False)
        switching = self._switch_future is not None
        self.load_button.setEnabled(self.model_combo.count() > 0 and not switching)
        self.settings_button.setEnabled(not switching)

    def close_panel(self) -> None:
        if self._settings_dialog is not None:
            self._settings_dialog.close()


__all__ = ["VoiceConversionPanel"]

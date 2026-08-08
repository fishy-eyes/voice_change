"""Headless GUI tests for Beatrice runtime and model folder pickers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.backend_settings.beatrice import BeatriceSettingsPanel
from gui.i18n import tr


class FakeRuntime:
    def __init__(self) -> None:
        self.runtime_path = None
        self.registered_model_paths: tuple[Path, ...] = ()
        self.configured: list[Path] = []

    def configure_runtime(self, path: Path):
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Runtime path does not exist: {path}")
        if path.name == "invalid-runtime":
            raise ValueError("SimpleBeatrice is missing")
        self.runtime_path = path
        self.configured.append(path)
        return {"version": "2.0.0-rc.0"}

    def validate_runtime(self):
        if self.runtime_path is None or not self.runtime_path.is_dir():
            raise FileNotFoundError("Runtime path does not exist")
        return {"version": "2.0.0-rc.0"}


class FakeManager:
    def __init__(self) -> None:
        self.current_runtime = FakeRuntime()
        self.added: list[Path] = []
        self.removed: list[Path] = []

    def get_current_model_descriptor(self):
        return None

    def get_current_parameters(self):
        return {}

    def get_status(self):
        return SimpleNamespace(state="IDLE")

    def get_info(self):
        return {}

    def add_model_path(self, directory):
        path = Path(directory).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        self.added.append(path)
        self.current_runtime.registered_model_paths = tuple(self.added)
        return SimpleNamespace(name=path.name, package=path.name), True

    def remove_model_path(self, directory):
        path = Path(directory).resolve()
        self.removed.append(path)
        self.added = [item for item in self.added if item != path]
        self.current_runtime.registered_model_paths = tuple(self.added)
        return True

    def update_current_parameters(self, **changes):
        return changes


class BeatricePathGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_runtime_picker_accepts_unicode_and_cancel_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime_dir = Path(temp) / "运行 库"
            runtime_dir.mkdir()
            manager = FakeManager()
            panel = BeatriceSettingsPanel(manager=manager, language="zh")
            try:
                with patch(
                    "gui.backend_settings.beatrice.QFileDialog.getExistingDirectory",
                    return_value="",
                ):
                    panel._select_runtime_folder()
                self.assertEqual(manager.current_runtime.configured, [])
                with patch(
                    "gui.backend_settings.beatrice.QFileDialog.getExistingDirectory",
                    return_value=str(runtime_dir),
                ):
                    panel._select_runtime_folder()
                self.assertEqual(manager.current_runtime.runtime_path, runtime_dir.resolve())
                self.assertIn("2.0.0-rc.0", panel.runtime_status_label.text())
            finally:
                panel.close()

    def test_runtime_picker_rejects_invalid_and_disappeared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            invalid = Path(temp) / "invalid-runtime"
            invalid.mkdir()
            missing = Path(temp) / "已移动-runtime"
            manager = FakeManager()
            panel = BeatriceSettingsPanel(manager=manager, language="en")
            try:
                with patch(
                    "gui.backend_settings.beatrice.QMessageBox.warning"
                ) as warning:
                    for path in (invalid, missing):
                        with patch(
                            "gui.backend_settings.beatrice.QFileDialog.getExistingDirectory",
                            return_value=str(path),
                        ):
                            panel._select_runtime_folder()
                self.assertEqual(warning.call_count, 2)
                self.assertIsNone(manager.current_runtime.runtime_path)
            finally:
                panel.close()

    def test_model_folder_add_remove_and_cancel_never_delete_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "jvs"
            package.mkdir()
            marker = package / "speaker_embeddings.bin"
            marker.write_bytes(b"keep")
            manager = FakeManager()
            refreshed = []
            panel = BeatriceSettingsPanel(
                manager=manager,
                language="en",
                on_models_changed=lambda: refreshed.append(True),
            )
            try:
                with patch(
                    "gui.backend_settings.beatrice.QFileDialog.getExistingDirectory",
                    return_value="",
                ):
                    panel._add_model_folder()
                self.assertEqual(manager.added, [])
                with patch(
                    "gui.backend_settings.beatrice.QFileDialog.getExistingDirectory",
                    return_value=str(package),
                ):
                    panel._add_model_folder()
                panel.registered_list.setCurrentRow(0)
                panel._remove_model_folder()
                self.assertTrue(marker.is_file())
                self.assertEqual(refreshed, [True, True])
            finally:
                panel.close()

    def test_path_copy_is_bilingual(self) -> None:
        self.assertEqual(tr("en", "beatrice.add_model"), "Add Model")
        self.assertEqual(tr("zh", "beatrice.add_model"), "添加模型")
        self.assertEqual(tr("en", "beatrice.remove_from_list"), "Remove From List")
        self.assertEqual(tr("zh", "beatrice.remove_from_list"), "从列表移除")


if __name__ == "__main__":
    unittest.main(verbosity=2)

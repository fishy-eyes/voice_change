"""Offscreen checks for external RVC model import controls."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.rvc_control_panel import RVCControlPanel


def descriptor(
    name: str,
    pth_path: Path,
    index_path: Path | None,
    *,
    profile_path: Path | None = None,
):
    return SimpleNamespace(
        name=name,
        pth_path=pth_path,
        index_path=index_path,
        profile_path=profile_path,
        profile_is_default=profile_path is None,
    )


class FakeImportManager:
    def __init__(self, inspection) -> None:
        self.inspection = inspection
        self.models = [
            descriptor(
                "modelF",
                Path("modelF.pth"),
                Path("modelF.index"),
                profile_path=Path("profile.json"),
            )
        ]
        self.imports = []

    def discover_models(self):
        return list(self.models)

    def inspect_import_directory(self, directory):
        self.inspected_directory = directory
        return self.inspection

    def import_model(self, directory, *, pth_path, index_path):
        self.imports.append((directory, pth_path, index_path))
        imported = descriptor("imported", pth_path, index_path)
        self.models.append(imported)
        return imported

    def get_model(self, name: str):
        return next(item for item in self.models if item.name == name)


class FakeRuntime:
    def __init__(self, manager) -> None:
        self.model_manager = manager
        self.enabled = True
        self.selected_model = None
        self.state = SimpleNamespace(ready=False, error=None)
        self.realtime_preset_key = "balanced"
        self.loads = []

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_realtime_preset(self, key: str) -> None:
        self.realtime_preset_key = key

    def load_model(self, name: str, *, audio_stream=None):
        self.loads.append((name, audio_stream))
        self.selected_model = name
        self.state = SimpleNamespace(ready=True, error=None)
        return self.state


class RVCModelImportGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _panel(inspection):
        manager = FakeImportManager(inspection)
        runtime = FakeRuntime(manager)
        stream = object()
        panel = RVCControlPanel(
            SimpleNamespace(rvc_runtime=runtime, audio_stream=stream)
        )
        return panel, manager, runtime, stream

    def test_multiple_files_use_explicit_choices_then_load(self) -> None:
        directory = Path("D:/external/modelA")
        pth_a = directory / "model.pth"
        pth_b = directory / "model_v2.pth"
        index_a = directory / "model.index"
        index_b = directory / "model_v2.index"
        inspection = SimpleNamespace(
            directory=directory,
            pth_candidates=(pth_a, pth_b),
            index_candidates=(index_a, index_b),
        )
        panel, manager, runtime, stream = self._panel(inspection)

        with (
            patch.object(
                panel,
                "_choose_import_directory",
                return_value=str(directory),
            ),
            patch.object(
                panel,
                "_choose_candidate",
                side_effect=[pth_b, index_a],
            ) as chooser,
        ):
            panel.import_button.click()

        self.assertEqual(chooser.call_count, 2)
        self.assertEqual(manager.imports, [(directory, pth_b, index_a)])
        self.assertEqual(runtime.loads, [("imported", stream)])
        self.assertEqual(panel.model_combo.currentData(), "imported")
        self.assertIn("Loaded: imported", panel.status_label.text())
        self.assertIn("pth: ✓ model_v2.pth", panel.status_label.text())
        self.assertIn("index: ✓ model.index", panel.status_label.text())
        self.assertIn("profile: default", panel.status_label.text())

    def test_no_index_shows_notice_and_loads_in_no_index_mode(self) -> None:
        directory = Path("D:/external/noIndex")
        pth = directory / "voice.pth"
        inspection = SimpleNamespace(
            directory=directory,
            pth_candidates=(pth,),
            index_candidates=(),
        )
        panel, manager, runtime, _stream = self._panel(inspection)

        with (
            patch.object(
                panel,
                "_choose_import_directory",
                return_value=str(directory),
            ),
            patch.object(panel, "_show_information") as notice,
        ):
            panel.import_button.click()

        notice.assert_called_once_with(
            "No index was found; no-index mode will be used."
        )
        self.assertEqual(manager.imports, [(directory, pth, None)])
        self.assertEqual(runtime.loads[0][0], "imported")
        self.assertIn("index: none", panel.status_label.text())

    def test_missing_pth_shows_required_message_without_import(self) -> None:
        directory = Path("D:/external/empty")
        inspection = SimpleNamespace(
            directory=directory,
            pth_candidates=(),
            index_candidates=(),
        )
        panel, manager, runtime, _stream = self._panel(inspection)

        with (
            patch.object(
                panel,
                "_choose_import_directory",
                return_value=str(directory),
            ),
            patch.object(panel, "_show_warning") as warning,
        ):
            panel.import_button.click()

        warning.assert_called_once_with("No RVC model weight file was found.")
        self.assertEqual(manager.imports, [])
        self.assertEqual(runtime.loads, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

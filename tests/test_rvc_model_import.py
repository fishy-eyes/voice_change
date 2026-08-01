"""Tests for external RVC model inspection, import and persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.rvc_model_manager import (
    RVCModelManager,
    RVCModelSelectionRequired,
)
from core.rvc_lifecycle import RVCApplicationState
from core.rvc_runtime import RVCRuntime


class RVCModelImportTests(unittest.TestCase):
    def _manager(self, root: Path) -> RVCModelManager:
        library = root / "library"
        library.mkdir()
        return RVCModelManager(
            library,
            user_models_path=root / "config" / "user_models.json",
        )

    def test_imports_pth_index_and_flat_profile_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self._manager(root)
            external = root / "external" / "modelA"
            external.mkdir(parents=True)
            pth = external / "voice_v2.pth"
            index = external / "voice_v2.index"
            pth.touch()
            index.touch()
            profile = external / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "name": "modelA",
                        "pitch": 12,
                        "f0_method": "rmvpe",
                        "index_rate": 0.3,
                        "protect": 0.33,
                        "rms_mix_rate": 0.25,
                    }
                ),
                encoding="utf-8",
            )

            descriptor = manager.import_model(external)

            self.assertEqual(descriptor.name, "modelA")
            self.assertEqual(descriptor.pth_path, pth.resolve())
            self.assertEqual(descriptor.index_path, index.resolve())
            self.assertEqual(descriptor.profile_path, profile.resolve())
            self.assertFalse(descriptor.profile_is_default)
            self.assertEqual(descriptor.profile.inference.pitch_shift, 12)
            self.assertEqual(descriptor.profile.inference.index_rate, 0.3)
            self.assertEqual(list(manager.models_root.iterdir()), [])

            registry = json.loads(manager.user_models_path.read_text(encoding="utf-8"))
            self.assertEqual(registry["version"], 1)
            self.assertEqual(registry["models"][0]["pth_path"], str(pth.resolve()))
            self.assertEqual(
                registry["models"][0]["index_path"],
                str(index.resolve()),
            )
            self.assertEqual(
                registry["models"][0]["profile_path"],
                str(profile.resolve()),
            )

    def test_imports_pth_only_with_required_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self._manager(root)
            external = root / "external" / "plainVoice"
            external.mkdir(parents=True)
            pth = external / "plain.pth"
            pth.touch()

            descriptor = manager.import_model(external)

            config = descriptor.profile.inference
            self.assertEqual(descriptor.pth_path, pth.resolve())
            self.assertIsNone(descriptor.index_path)
            self.assertIsNone(descriptor.profile_path)
            self.assertTrue(descriptor.profile_is_default)
            self.assertEqual(config.pitch_shift, 0)
            self.assertEqual(config.f0_method, "rmvpe")
            self.assertEqual(config.index_rate, 0.3)
            self.assertEqual(config.protect, 0.33)
            self.assertEqual(config.rms_mix_rate, 0.25)

    def test_multiple_pth_and_index_require_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self._manager(root)
            external = root / "multi"
            external.mkdir()
            pth_a = external / "model.pth"
            pth_b = external / "model_v2.pth"
            index_a = external / "model.index"
            index_b = external / "model_v2.index"
            for path in (pth_a, pth_b, index_a, index_b):
                path.touch()

            with self.assertRaises(RVCModelSelectionRequired) as pth_error:
                manager.import_model(external)
            self.assertEqual(pth_error.exception.file_type, "pth")

            with self.assertRaises(RVCModelSelectionRequired) as index_error:
                manager.import_model(external, pth_path=pth_b)
            self.assertEqual(index_error.exception.file_type, "index")

            descriptor = manager.import_model(
                external,
                pth_path=pth_b,
                index_path=index_b,
            )
            self.assertEqual(descriptor.pth_path, pth_b.resolve())
            self.assertEqual(descriptor.index_path, index_b.resolve())

    def test_bad_and_empty_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self._manager(root)
            with self.assertRaises(NotADirectoryError):
                manager.inspect_import_directory(root / "missing")

            empty = root / "empty"
            empty.mkdir()
            inspection = manager.inspect_import_directory(empty)
            self.assertEqual(inspection.pth_candidates, ())
            with self.assertRaises(FileNotFoundError):
                manager.import_model(empty)

    def test_saved_models_are_restored_by_a_new_manager(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self._manager(root)
            external = root / "restoredVoice"
            external.mkdir()
            pth = external / "restored.pth"
            pth.touch()
            imported = manager.import_model(external)

            restarted = RVCModelManager(
                manager.models_root,
                user_models_path=manager.user_models_path,
            )
            restored = restarted.get_model(imported.name)

            self.assertEqual(restored.pth_path, pth.resolve())
            self.assertTrue(restored.is_external)
            self.assertTrue(restored.profile_is_default)
            self.assertEqual(len(restarted.discover_models()), 1)

    def test_imported_model_uses_existing_runtime_profile_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self._manager(root)
            external = root / "runtimeVoice"
            external.mkdir()
            pth = external / "runtime.pth"
            pth.touch()
            imported = manager.import_model(external)
            fake_effect = SimpleNamespace(enabled=True)
            ready_state = RVCApplicationState(
                enabled=True,
                effect=fake_effect,
                ready=True,
            )
            runtime = RVCRuntime(manager, warmup_enabled=False)
            runtime.set_enabled(True)

            with patch(
                "core.rvc_runtime.initialize_rvc_application",
                return_value=ready_state,
            ) as initialize:
                state = runtime.load_model(imported.name)

            self.assertIs(state, ready_state)
            self.assertEqual(runtime.selected_model, imported.name)
            profile = initialize.call_args.kwargs["profile"]
            self.assertEqual(profile.voice_dir, external.resolve())
            self.assertEqual(profile.model_file, Path(pth.name))
            self.assertIsNone(profile.index_file)
            self.assertEqual(profile.inference.index_rate, 0.3)
            self.assertTrue(fake_effect.enabled)

if __name__ == "__main__":
    unittest.main(verbosity=2)

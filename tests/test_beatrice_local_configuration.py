"""Local path persistence and Beatrice folder registry tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai.beatrice.catalog import BeatriceModelCatalog
from ai.beatrice.model import EXPECTED_RUNTIME_VERSION, REQUIRED_MODEL_FILES
from config.local_settings import LocalSettingsStore


def create_package(root: Path, name: str = "voice") -> Path:
    package = root / name
    package.mkdir(parents=True)
    (package / f"beatrice_paraphernalia_{name}.toml").write_text(
        "\n".join(
            (
                "[model]",
                f'version = "{EXPECTED_RUNTIME_VERSION}"',
                f'name = "{name}"',
                "[voice.0]",
                f'name = "{name}001"',
            )
        ),
        encoding="utf-8",
    )
    for filename in REQUIRED_MODEL_FILES:
        (package / filename).write_bytes(b"test")
    return package


class LocalSettingsTests(unittest.TestCase):
    def test_missing_file_uses_defaults_and_round_trips_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "local_settings.json"
            store = LocalSettingsStore(path)
            self.assertEqual(store.beatrice["runtime_dir"], "")
            self.assertEqual(store.beatrice["model_roots"], [])
            store.update_beatrice(
                runtime_dir="D:\\Runtime 中文",
                model_roots=["E:\\Models\\jvs"],
                last_model="jvs",
            )
            reloaded = LocalSettingsStore(path)
            self.assertEqual(reloaded.beatrice["runtime_dir"], "D:\\Runtime 中文")
            self.assertEqual(reloaded.beatrice["model_roots"], ["E:\\Models\\jvs"])
            self.assertEqual(reloaded.beatrice["last_model"], "jvs")

    def test_malformed_json_recovers_without_rewriting_until_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "local_settings.json"
            path.write_text("{broken", encoding="utf-8")
            store = LocalSettingsStore(path)
            self.assertEqual(store.beatrice["runtime_dir"], "")
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")

    def test_updates_preserve_unrelated_user_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "local_settings.json"
            path.write_text(
                json.dumps({"theme": "dark", "beatrice": {"runtime_dir": ""}}),
                encoding="utf-8",
            )
            LocalSettingsStore(path).update_beatrice(runtime_dir="D:\\runtime")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["theme"], "dark")
            self.assertEqual(saved["beatrice"]["runtime_dir"], "D:\\runtime")


class BeatriceModelCatalogTests(unittest.TestCase):
    def test_register_duplicate_remove_and_restart_persistence_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = create_package(root / "external", "jvs")
            saved: list[tuple[str, ...]] = []
            catalog = BeatriceModelCatalog(
                root / "default",
                on_registered_paths_changed=saved.append,
            )
            descriptor, added = catalog.register_package(package)
            self.assertTrue(added)
            self.assertEqual(descriptor.package, "jvs")
            _, added_again = catalog.register_package(package)
            self.assertFalse(added_again)
            self.assertEqual(len(saved), 1)
            self.assertEqual(catalog.discover_models()[0].name, "jvs")
            self.assertTrue(catalog.remove_registered_package(package))
            self.assertTrue(package.is_dir(), "remove must not delete user files")
            self.assertEqual(saved[-1], ())

            restarted = BeatriceModelCatalog(
                root / "default", registered_packages=(package,)
            )
            self.assertEqual(restarted.get_model("jvs").directory, package.resolve())

    def test_default_registered_and_environment_roots_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = create_package(root / "models", "jvs")
            catalog = BeatriceModelCatalog(
                root / "models",
                registered_packages=(package,),
                additional_roots=(root / "models", package),
            )
            self.assertEqual(len(catalog.discover_models()), 1)

    def test_same_package_name_is_visible_and_never_silently_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = create_package(root / "default", "voice")
            second = create_package(root / "external", "voice")
            catalog = BeatriceModelCatalog(
                root / "default", registered_packages=(second,)
            )
            descriptors = catalog.discover_models()
            self.assertEqual(len(descriptors), 2)
            self.assertTrue(all(str(item.directory) in item.name for item in descriptors))
            with self.assertRaisesRegex(ValueError, "Multiple"):
                catalog.get_model("voice")
            self.assertEqual(catalog.get_model(descriptors[0].name).directory, first)

    def test_invalid_packages_report_missing_toml_bin_and_malformed_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = BeatriceModelCatalog(root / "default")
            missing_toml = root / "missing-toml"
            missing_toml.mkdir()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                catalog.register_package(missing_toml)

            missing_bin = create_package(root, "missing-bin")
            (missing_bin / REQUIRED_MODEL_FILES[-1]).unlink()
            with self.assertRaisesRegex(ValueError, REQUIRED_MODEL_FILES[-1]):
                catalog.register_package(missing_bin)

            malformed = create_package(root, "malformed")
            next(malformed.glob("*.toml")).write_text("[model", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "TOMLDecodeError"):
                catalog.register_package(malformed)


class ProductionIsolationTests(unittest.TestCase):
    def test_production_beatrice_modules_do_not_import_experiments(self) -> None:
        project = Path(__file__).resolve().parent.parent
        files = [
            *(project / "ai" / "beatrice").glob("*.py"),
            project / "ai" / "voice_engine" / "beatrice.py",
            project / "core" / "beatrice_runtime.py",
        ]
        for path in files:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("from experiments", source, path)
            self.assertNotIn("import experiments", source, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config.settings as settings
import main
import utils.logger as app_logger


class ReleasePortabilityTests(unittest.TestCase):
    def test_v2_version_metadata_is_consistent(self) -> None:
        self.assertEqual(settings.APP_VERSION, "2.0.0")
        project_root = Path(__file__).resolve().parents[1]
        expected = "2.0.0"
        for relative in (
            "README.md",
            "packaging/README-Windows.txt",
            "packaging/version_info.txt",
            "packaging/voice_change.spec",
            "packaging/build_windows.ps1",
        ):
            content = (project_root / relative).read_text(encoding="utf-8")
            self.assertIn(expected, content, relative)

    def test_frozen_application_root_is_executable_directory(self) -> None:
        executable = Path("C:/portable/VoiceChanger/VoiceChanger.exe")
        with (
            patch.object(settings.sys, "frozen", True, create=True),
            patch.object(settings.sys, "executable", str(executable)),
        ):
            self.assertEqual(
                settings._application_root(),
                executable.resolve().parent,
            )

    def test_source_defaults_are_project_local_without_machine_paths(self) -> None:
        self.assertEqual(
            Path(settings.RVC_SOURCE_DIR),
            settings.PROJECT_ROOT / "rvc_source",
        )
        self.assertEqual(
            Path(settings.RVC_MODELS_DIR),
            settings.PROJECT_ROOT
            / "local_assets"
            / "rvc"
            / "foundation_models",
        )
        self.assertEqual(
            Path(settings.RVC_MODEL_LIBRARY_DIR),
            settings.PROJECT_ROOT / "local_assets" / "rvc" / "voice_models",
        )

    def test_windows_package_keeps_public_rvc_asset_layout(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        build_script = (project_root / "packaging" / "build_windows.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('"rvc_models\\hubert"', build_script)
        self.assertIn('"rvc_models\\rmvpe"', build_script)
        self.assertIn('"models\\rvc"', build_script)

    @patch("main.DeviceManager.select_output_device")
    @patch("main.DeviceManager.find_virtual_output_device", return_value=None)
    @patch("main.INPUT_DEVICE", None)
    @patch("main.AUTO_SELECT_DEVICES", True)
    def test_auto_device_selection_never_prompts_without_vb_cable(
        self,
        _find_virtual,
        select_output,
    ) -> None:
        self.assertEqual(main._select_devices(), (None, None))
        select_output.assert_not_called()

    def test_windowed_logger_uses_file_when_stderr_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(app_logger, "PROJECT_ROOT", Path(directory)),
                patch.object(app_logger, "_configured", False),
                patch.object(app_logger.sys, "stderr", None),
                patch.object(app_logger.logger, "remove"),
                patch.object(app_logger.logger, "add") as add_sink,
                patch.object(app_logger.logger, "info"),
            ):
                app_logger.setup_logger()

            self.assertEqual(add_sink.call_count, 1)
            sink = Path(add_sink.call_args.args[0])
            self.assertEqual(sink.name, "voice_change.log")
            self.assertTrue(sink.parent.is_dir())

    def test_release_smoke_test_imports_packaged_runtime_modules(self) -> None:
        imported: list[str] = []
        with (
            patch.object(main, "RVC_SOURCE_DIR", "C:/portable/rvc_source"),
            patch.object(
                main.importlib,
                "import_module",
                side_effect=lambda name: imported.append(name) or SimpleNamespace(),
            ),
            patch.object(main.logger, "info"),
        ):
            main._run_release_smoke_test()

        self.assertIn("torch", imported)
        self.assertIn("infer.vc.pipeline", imported)


if __name__ == "__main__":
    unittest.main(verbosity=2)

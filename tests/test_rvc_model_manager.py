"""Tests for application RVC model discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.rvc_model_manager import RVCModelManager


class RVCModelManagerTests(unittest.TestCase):
    def test_discovers_colocated_relative_profile_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "modelF"
            model_dir.mkdir()
            (model_dir / "modelF.pth").touch()
            (model_dir / "modelF.index").touch()
            (model_dir / "profile.json").write_text(
                json.dumps(
                    {
                        "name": "modelF",
                        "voice_dir": ".",
                        "model_file": "modelF.pth",
                        "index_file": "modelF.index",
                        "inference": {
                            "pitch_shift": 12,
                            "f0_method": "rmvpe",
                            "index_rate": 0.3,
                            "rms_mix_rate": 0.25,
                            "protect": 0.33,
                        },
                    }
                ),
                encoding="utf-8",
            )

            descriptor = RVCModelManager(temp_dir).get_model("MODELF")

            self.assertEqual(descriptor.name, "modelF")
            self.assertEqual(descriptor.pth_path, (model_dir / "modelF.pth").resolve())
            self.assertEqual(
                descriptor.index_path,
                (model_dir / "modelF.index").resolve(),
            )
            self.assertTrue(descriptor.profile.voice_dir.is_absolute())
            self.assertEqual(descriptor.profile.inference.pitch_shift, 12)

    def test_invalid_model_does_not_hide_valid_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Path(temp_dir) / "broken"
            invalid.mkdir()
            (invalid / "profile.json").write_text("{}", encoding="utf-8")
            self.assertEqual(RVCModelManager(temp_dir).discover_models(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

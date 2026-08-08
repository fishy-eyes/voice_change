"""CI-safe Beatrice package and optional-runtime validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai.beatrice.model import (
    BeatriceModelManager,
    EXPECTED_RUNTIME_VERSION,
    REQUIRED_MODEL_FILES,
)
from ai.beatrice.runtime import (
    BeatriceRuntimeLoader,
    RuntimeUnavailableError,
    unpack_runtime_output,
)


def create_package(root: Path, name: str = "voice", *, version: str = EXPECTED_RUNTIME_VERSION) -> Path:
    package = root / name
    package.mkdir()
    (package / f"beatrice_paraphernalia_{name}.toml").write_text(
        "\n".join(
            (
                "[model]",
                f'version = "{version}"',
                'name = "Test Voice"',
                "[voice.0]",
                'name = "Alice"',
                "average_pitch = 57.5",
                "[voice.1]",
                'name = "Bob"',
            )
        ),
        encoding="utf-8",
    )
    for filename in REQUIRED_MODEL_FILES:
        (package / filename).write_bytes(b"test")
    return package


class BeatriceModelRuntimeTests(unittest.TestCase):
    def test_descriptor_validates_toml_bins_speakers_and_rates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = create_package(root)
            manager = BeatriceModelManager(root)
            descriptor = manager.inspect_package(package)
            self.assertTrue(descriptor.valid, descriptor.validation_error)
            self.assertEqual(descriptor.name, "voice")
            self.assertEqual(descriptor.version, EXPECTED_RUNTIME_VERSION)
            self.assertEqual(descriptor.speaker_names, ("Alice", "Bob"))
            self.assertEqual(descriptor.speaker_average_pitches, (57.5, None))
            self.assertEqual(len(descriptor.identity), 64)
            self.assertEqual(descriptor.input_sample_rate, 16_000)
            self.assertEqual(descriptor.output_sample_rate, 24_000)
            self.assertEqual(manager.discover_models(), [descriptor])
            self.assertEqual(manager.get_model("VOICE"), descriptor)

    def test_invalid_package_is_described_but_not_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = create_package(root)
            (package / REQUIRED_MODEL_FILES[-1]).unlink()
            manager = BeatriceModelManager(root)
            descriptor = manager.inspect_package(package)
            self.assertFalse(descriptor.valid)
            self.assertIn(REQUIRED_MODEL_FILES[-1], descriptor.validation_error or "")
            self.assertEqual(manager.discover_models(), [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                manager.get_model(package.name)

    def test_wrong_model_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = BeatriceModelManager(temp)
            descriptor = manager.inspect_package(
                create_package(Path(temp), version="2.0.0-beta.1")
            )
            self.assertFalse(descriptor.valid)
            self.assertIn(EXPECTED_RUNTIME_VERSION, descriptor.validation_error or "")

    def test_missing_runtime_is_clear_and_does_not_import_native_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            loader = BeatriceRuntimeLoader(Path(temp) / "missing")
            self.assertFalse(loader.available)
            with self.assertRaisesRegex(RuntimeUnavailableError, "未找到 Beatrice"):
                loader.load()

    def test_runtime_constants_and_convert_return_are_defensive(self) -> None:
        good = type(
            "Module",
            (),
            {
                "IN_SAMPLE_RATE": 16_000,
                "OUT_SAMPLE_RATE": 24_000,
                "IN_HOP_LENGTH": 160,
                "OUT_HOP_LENGTH": 240,
                "SimpleBeatrice": lambda: None,
            },
        )
        BeatriceRuntimeLoader._validate_module(good)
        good.OUT_HOP_LENGTH = 241
        with self.assertRaisesRegex(RuntimeUnavailableError, "constants"):
            BeatriceRuntimeLoader._validate_module(good)
        audio, auxiliary = unpack_runtime_output(
            (np.zeros(240, dtype=np.float32), {"ok": True})
        )
        self.assertEqual(audio.shape, (240,))
        self.assertEqual(auxiliary, {"ok": True})
        with self.assertRaisesRegex(RuntimeError, "3-item"):
            unpack_runtime_output((audio, None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)

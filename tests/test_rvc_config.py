"""Unit tests for RVC model profiles and runtime configuration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rvc_engine import RVCEngine
from config.rvc_profiles import (
    RVCInferenceConfig,
    RVCModelProfile,
    load_rvc_profile,
)


class CapturingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def pipeline(self, **kwargs):
        self.calls.append(kwargs)
        return np.asarray(kwargs["audio"], dtype=np.float32).copy()


class RVCConfigTests(unittest.TestCase):
    def test_model_f_example_loads_and_resolves_against_models_root(self) -> None:
        profile_path = PROJECT_ROOT / "config" / "rvc_profiles" / "modelF.example.json"
        profile = load_rvc_profile(profile_path)

        self.assertEqual(profile.name, "modelF")
        self.assertEqual(profile.inference.pitch_shift, 12)
        self.assertEqual(profile.inference.f0_method, "rmvpe")
        self.assertEqual(profile.inference.index_rate, 0.30)
        self.assertEqual(profile.inference.rms_mix_rate, 0.25)
        self.assertEqual(profile.inference.protect, 0.33)

        models_dir = Path("model-root")
        self.assertEqual(
            profile.resolve_voice_dir(models_dir),
            models_dir / "voices" / "modelF",
        )
        self.assertEqual(
            profile.resolve_model_file(models_dir),
            models_dir / "voices" / "modelF" / "zhoujie.pth",
        )
        self.assertEqual(
            profile.resolve_index_file(models_dir),
            models_dir
            / "voices"
            / "modelF"
            / "added_IVF5660_Flat_nprobe_1.index",
        )

    def test_json_and_toml_profiles_are_supported(self) -> None:
        values = {
            "name": "voice-a",
            "voice_dir": "voices/voice-a",
            "inference": {"pitch_shift": 3, "index_rate": 0.2},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "voice.json"
            json_path.write_text(json.dumps(values), encoding="utf-8")
            toml_path = root / "voice.toml"
            toml_path.write_text(
                'name = "voice-a"\n'
                'voice_dir = "voices/voice-a"\n'
                '[inference]\n'
                'pitch_shift = 3\n'
                'index_rate = 0.2\n',
                encoding="utf-8",
            )

            self.assertEqual(load_rvc_profile(json_path).inference.pitch_shift, 3)
            self.assertEqual(load_rvc_profile(toml_path).inference.index_rate, 0.2)

    def test_validation_rejects_invalid_or_unknown_settings(self) -> None:
        invalid_values = (
            {"f0_method": "harvest"},
            {"index_rate": 1.1},
            {"rms_mix_rate": -0.1},
            {"protect": 0.51},
            {"unknown": 1},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                RVCInferenceConfig.from_mapping(values)

    def test_engine_legacy_arguments_and_runtime_updates_remain_compatible(self) -> None:
        engine = RVCEngine(
            voice_dir="voices/legacy",
            source_dir="rvc-source",
            models_dir="models",
            pitch_shift=2,
            f0_method="pm",
            index_rate=0.4,
            rms_mix_rate=0.6,
            protect=0.2,
        )
        self.assertEqual(
            engine.config,
            RVCInferenceConfig(
                pitch_shift=2,
                f0_method="pm",
                index_rate=0.4,
                rms_mix_rate=0.6,
                protect=0.2,
            ),
        )

        updated = engine.update_config(pitch_shift=12, index_rate=0.3)
        self.assertEqual(updated.pitch_shift, 12)
        self.assertEqual(updated.index_rate, 0.3)
        self.assertEqual(updated.f0_method, "pm")
        self.assertEqual(engine.config, updated)

        before_invalid_update = engine.config
        with self.assertRaises(ValueError):
            engine.update_config(protect=0.8)
        self.assertEqual(engine.config, before_invalid_update)

    def test_engine_from_profile_selects_model_and_runtime_config(self) -> None:
        profile = RVCModelProfile(
            name="voice-a",
            voice_dir=Path("voices/voice-a"),
            model_file=Path("voice-a.pth"),
            index_file=Path("voice-a.index"),
            inference=RVCInferenceConfig(pitch_shift=7, index_rate=0.25),
        )
        engine = RVCEngine.from_profile(
            profile,
            source_dir="source",
            models_dir="models",
        )

        self.assertEqual(engine.voice_dir, Path("models/voices/voice-a"))
        self.assertEqual(engine.config, profile.inference)
        self.assertEqual(engine._voice_pth_path, Path("models/voices/voice-a/voice-a.pth"))
        self.assertEqual(
            engine._configured_index_path,
            Path("models/voices/voice-a/voice-a.index"),
        )

    def test_loaded_engine_uses_updated_snapshot_without_reloading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rmvpe_path = root / "rmvpe" / "rmvpe.pt"
            rmvpe_path.parent.mkdir()
            rmvpe_path.touch()
            engine = RVCEngine(
                voice_dir=root / "voice",
                source_dir=root / "source",
                models_dir=root,
                sample_rate=16000,
                config=RVCInferenceConfig(f0_method="rmvpe"),
            )
            pipeline = CapturingPipeline()
            engine._pipeline = pipeline
            engine._hubert_model = object()
            engine._net_g = object()
            engine._model_loaded = True
            engine._tgt_sr = 16000
            engine._version = "v1"
            engine._if_f0 = 1

            pipeline_identity = id(engine._pipeline)
            engine.update_config(
                pitch_shift=12,
                index_rate=0.30,
                rms_mix_rate=0.25,
                protect=0.33,
            )
            output = engine.infer(np.ones(320, dtype=np.float32) * 0.01)

            self.assertEqual(id(engine._pipeline), pipeline_identity)
            self.assertEqual(output.shape, (320,))
            call = pipeline.calls[-1]
            self.assertEqual(call["f0_up_key"], 12)
            self.assertEqual(call["f0_method"], "rmvpe")
            self.assertEqual(call["index_rate"], 0.30)
            self.assertEqual(call["rms_mix_rate"], 0.25)
            self.assertEqual(call["protect"], 0.33)


if __name__ == "__main__":
    unittest.main(verbosity=2)

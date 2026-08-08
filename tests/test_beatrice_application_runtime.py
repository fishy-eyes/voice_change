"""CI-safe Beatrice application lifecycle and effect ordering."""

from __future__ import annotations

from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

from ai.beatrice.model import BeatriceModelManager, REQUIRED_MODEL_FILES
from ai.voice_engine.beatrice import BeatriceConfig
from config.local_settings import LocalSettingsStore
from core.beatrice_runtime import BeatriceRuntime
from customization.beatrice import BeatriceParameterSet
from effects.gain import GainEffect
from effects.manager import EffectManager


class FakeModelManager:
    def __init__(self) -> None:
        self.descriptor = SimpleNamespace(name="voice")

    def get_model(self, name):
        if name != "voice":
            raise LookupError(name)
        return self.descriptor

    def discover_models(self):
        return [self.descriptor]


class FakeEngine:
    def __init__(self, descriptor, **kwargs) -> None:
        del kwargs
        self.descriptor = descriptor
        self.config = BeatriceConfig()
        self.is_loaded = False
        self.unloads = 0

    def load_model(self):
        self.is_loaded = True

    def unload_model(self):
        self.is_loaded = False
        self.unloads += 1

    def update_config(self, **changes):
        self.config = self.config.updated(**changes)

    def get_info(self):
        return {
            "runtime": {
                "pitch_shift_min": -24.0,
                "pitch_shift_max": 24.0,
                "max_formant_shift": 8,
                "codebook_size": 512,
            }
        }


class FakeEffect:
    name = "AIVoiceEffect"

    def __init__(self, engine, **kwargs) -> None:
        del engine, kwargs
        self.enabled = False
        self.worker = SimpleNamespace(thread_alive=False)
        self.stops = 0

    def start(self):
        return True

    def stop(self, timeout=0.0):
        del timeout
        self.stops += 1
        return True


class BeatriceApplicationRuntimeTests(unittest.TestCase):
    def test_load_enable_effect_order_and_repeated_shutdown(self) -> None:
        effects = EffectManager()
        gain = GainEffect(1.0)
        effects.add(gain)
        runtime = BeatriceRuntime(
            FakeModelManager(),
            engine_factory=FakeEngine,
            effect_factory=FakeEffect,
        )
        runtime.bind_effect_manager(effects)
        runtime.set_enabled(True)
        state = runtime.load_model("voice")
        self.assertTrue(state.ready, state.error)
        self.assertEqual([effect.name for effect in effects.effects], [
            "AIVoiceEffect", "GainEffect"
        ])
        self.assertTrue(state.effect.enabled)
        self.assertIs(runtime.load_model("voice"), state)
        self.assertTrue(runtime.shutdown())
        self.assertEqual(effects.effects, [gain])
        self.assertTrue(runtime.shutdown())

    def test_missing_model_fails_without_removing_base_gain(self) -> None:
        effects = EffectManager()
        gain = GainEffect(1.0)
        effects.add(gain)
        runtime = BeatriceRuntime(
            FakeModelManager(),
            engine_factory=FakeEngine,
            effect_factory=FakeEffect,
        )
        runtime.bind_effect_manager(effects)
        state = runtime.load_model("missing")
        self.assertFalse(state.ready)
        self.assertIn("LookupError", state.error or "")
        self.assertEqual(effects.effects, [gain])
        self.assertTrue(runtime.shutdown())

    def test_speaker_preset_is_keyed_and_restored_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "voice"
            package.mkdir()
            (package / "beatrice_paraphernalia_voice.toml").write_text(
                '\n'.join((
                    '[model]',
                    'version = "2.0.0-rc.0"',
                    'name = "Voice"',
                    '[voice.0]',
                    'name = "Alice"',
                    'average_pitch = 54.0',
                    '[voice.1]',
                    'name = "Bob"',
                    'average_pitch = 60.0',
                )),
                encoding="utf-8",
            )
            for filename in REQUIRED_MODEL_FILES:
                (package / filename).write_bytes(b"test")
            manager = BeatriceModelManager(root)
            settings_path = root / "local.json"
            settings = LocalSettingsStore(settings_path)

            first = BeatriceRuntime(
                manager,
                local_settings=settings,
                engine_factory=FakeEngine,
                effect_factory=FakeEffect,
            )
            self.assertTrue(first.load_model("voice").ready)
            first.update_parameters(pitch_shift_semitone=1.5)
            values = first.update_parameters(
                target_speaker=1,
                pitch_shift_semitone=5.5,
                formant_shift=2.0,
                min_source_pitch=95.0,
                max_source_pitch=410.0,
                vq_num_neighbors=8,
            )
            self.assertEqual(values["target_speaker"], 1)
            assisted = BeatriceParameterSet(
                target_speaker=1,
                pitch_shift_semitone=5.5,
                formant_shift=2.0,
                min_source_pitch=30.0,
                max_source_pitch=1100.0,
                vq_num_neighbors=8,
            )
            after_assisted = first.update_parameters(
                **assisted.to_assisted_changes()
            )
            self.assertEqual(after_assisted["min_source_pitch"], 95.0)
            self.assertEqual(after_assisted["max_source_pitch"], 410.0)
            self.assertTrue(first.shutdown())

            restarted_settings = LocalSettingsStore(settings_path)
            presets = restarted_settings.beatrice["speaker_presets"]
            self.assertEqual(len(presets), 2)
            second = BeatriceRuntime(
                manager,
                local_settings=restarted_settings,
                engine_factory=FakeEngine,
                effect_factory=FakeEffect,
            )
            state = second.load_model("voice")
            self.assertTrue(state.ready, state.error)
            restored = state.engine.config.to_dict()
            self.assertEqual(restored["target_speaker"], 1)
            self.assertEqual(restored["pitch_shift_semitone"], 5.5)
            self.assertEqual(restored["formant_shift"], 2.0)
            self.assertEqual(restored["vq_num_neighbors"], 8)
            self.assertEqual(restored["min_source_pitch"], 95.0)
            self.assertEqual(restored["max_source_pitch"], 410.0)
            alice = second.update_parameters(target_speaker=0)
            self.assertEqual(alice["pitch_shift_semitone"], 1.5)
            self.assertTrue(second.shutdown())


if __name__ == "__main__":
    unittest.main(verbosity=2)

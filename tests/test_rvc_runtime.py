"""Unit tests for application-owned RVC switching and cleanup."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config.rvc_profiles import RVCInferenceConfig, RVCModelProfile
from core.rvc_lifecycle import RVCApplicationState
from core.rvc_runtime import RVCRuntime
from effects.gain import GainEffect
from effects.manager import EffectManager


class FakeAI:
    name = "AIVoiceEffect"

    def __init__(self, engine) -> None:
        self.engine = engine
        self.enabled = True
        self.worker = object()
        self.realtime_updates: list[tuple[int, int]] = []

    def update_realtime_config(
        self,
        *,
        chunk_size: int,
        overlap_size: int,
    ) -> None:
        self.realtime_updates.append((chunk_size, overlap_size))


class FakeStream:
    def __init__(self) -> None:
        self.running = True
        self.stops = 0
        self.starts = 0

    @property
    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.stops += 1
        self.running = False

    def start(self) -> None:
        self.starts += 1
        self.running = True


class FakeModelManager:
    def get_model(self, name: str):
        profile = RVCModelProfile(
            name=name,
            voice_dir=".",
            model_file=f"{name}.pth",
            index_file=f"{name}.index",
            inference=RVCInferenceConfig(),
        )
        return SimpleNamespace(name=name, profile=profile)


def make_state(profile) -> RVCApplicationState:
    engine = SimpleNamespace(is_loaded=True, model=profile.name)
    return RVCApplicationState(
        enabled=True,
        engine=engine,
        effect=FakeAI(engine),
        ready=True,
    )


class RVCRuntimeTests(unittest.TestCase):
    @patch("core.rvc_runtime.cleanup_rvc_application", return_value=True)
    @patch("core.rvc_runtime.initialize_rvc_application")
    def test_switch_stops_stream_keeps_ai_first_and_releases_old_state(
        self,
        initialize_mock,
        cleanup_mock,
    ) -> None:
        initialize_mock.side_effect = lambda **kwargs: make_state(kwargs["profile"])
        manager = EffectManager()
        base_effect = GainEffect()
        manager.add(base_effect)
        runtime = RVCRuntime(FakeModelManager(), warmup_enabled=False)
        runtime.bind_effect_manager(manager)
        runtime.set_enabled(True)

        first = runtime.load_model("modelA")
        stream = FakeStream()
        second = runtime.load_model("modelB", audio_stream=stream)

        self.assertTrue(first.ready)
        self.assertTrue(second.ready)
        self.assertEqual(runtime.selected_model, "modelB")
        self.assertEqual([item.name for item in manager.effects], ["AIVoiceEffect", "GainEffect"])
        self.assertEqual(stream.stops, 1)
        self.assertEqual(stream.starts, 1)
        cleanup_mock.assert_called_once_with(first, timeout=runtime.stop_timeout)

        worker = second.effect.worker
        engine = second.effect.engine
        initialize_calls = initialize_mock.call_count
        preset = runtime.set_realtime_preset("low_latency")
        self.assertEqual(preset.name, "Low Latency")
        self.assertEqual(runtime.realtime_preset_key, "low_latency")
        self.assertEqual(
            second.effect.realtime_updates,
            [(
                preset.chunk_samples(runtime.sample_rate),
                preset.overlap_samples(runtime.sample_rate),
            )],
        )
        self.assertIs(second.effect.worker, worker)
        self.assertIs(second.effect.engine, engine)
        self.assertEqual(initialize_mock.call_count, initialize_calls)

        runtime.set_enabled(False)
        self.assertFalse(second.effect.enabled)
        self.assertTrue(runtime.shutdown())
        self.assertIsNone(manager.get_by_name("AIVoiceEffect"))
        self.assertEqual(cleanup_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

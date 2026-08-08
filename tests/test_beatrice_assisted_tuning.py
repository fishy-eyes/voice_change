from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from customization.beatrice import (
    BeatriceCandidateGenerator,
    BeatriceParameterSearch,
    BeatriceParameterSet,
    BeatriceTuningCapabilities,
    BeatriceVoiceAnalysis,
    metadata_pitch_to_hz,
    recommend_pitch_shift,
    recommend_source_pitch_range,
    analyze_beatrice_voice,
)


class FakeEngine:
    instances = []

    def __init__(self, descriptor, *, runtime_root, config) -> None:
        self.descriptor = descriptor
        self.runtime_root = runtime_root
        self.config = config
        self.blocks = 0
        self.unloaded = False
        type(self).instances.append(self)

    def load_model(self) -> None:
        pass

    def process_audio(self, block):
        self.blocks += 1
        return np.asarray(block, dtype=np.float32).copy()

    def unload_model(self) -> None:
        self.unloaded = True


class BeatriceAssistedTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeEngine.instances.clear()
        self.descriptor = SimpleNamespace(
            speaker_average_pitches=(69.0,),
            speaker_names=("speaker",),
        )
        self.analysis = BeatriceVoiceAnalysis(100.0, 220.0, 300.0, 0.1)
        self.capabilities = BeatriceTuningCapabilities(
            pitch_shift_min=-12.0,
            pitch_shift_max=12.0,
            source_pitch_min=90.0,
            source_pitch_max=350.0,
            max_formant_shift=8.0,
            codebook_size=10,
        )

    def test_f0_recommendations_use_percentiles_metadata_and_capabilities(self) -> None:
        self.assertAlmostEqual(metadata_pitch_to_hz(69.0), 440.0)
        self.assertEqual(
            recommend_source_pitch_range(self.analysis, self.capabilities),
            (90.0, 350.0),
        )
        self.assertAlmostEqual(
            recommend_pitch_shift(
                self.analysis, self.descriptor, 0, self.capabilities
            ),
            12.0,
        )
        missing = SimpleNamespace(speaker_average_pitches=(None,))
        self.assertEqual(
            recommend_pitch_shift(
                self.analysis,
                missing,
                0,
                self.capabilities,
                fallback=3.5,
            ),
            3.5,
        )
        self.assertIsNone(
            analyze_beatrice_voice(np.zeros(48_000, np.float32), 48_000).f0_p50
        )

    def test_search_changes_one_parameter_group_per_stage(self) -> None:
        base = BeatriceParameterSet(target_speaker=0)
        search = BeatriceParameterSearch(
            base, self.analysis, self.capabilities, self.descriptor
        )
        stages = []
        while search.final_parameters is None:
            stages.append(search.current.stage)
            before = search.current.candidates[0]
            differing = {
                key
                for key, value in before.to_engine_changes().items()
                if value != base.to_engine_changes()[key]
            }
            if search.current.stage == "source_pitch":
                self.assertLessEqual(differing, {"min_source_pitch", "max_source_pitch"})
            search.choose(len(search.current.candidates) // 2)
            base = search.history[-1].candidates[search.history[-1].selected_index]
        self.assertEqual(stages, list(BeatriceParameterSearch.STAGES))
        self.assertTrue(all(1 <= len(round_.candidates) <= 5 for round_ in search.history))

    def test_candidates_have_isolated_engines_and_cancel_after_block(self) -> None:
        rate = 48_000
        t = np.arange(rate * 2, dtype=np.float32) / rate
        audio = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        options = [
            BeatriceParameterSet(pitch_shift_semitone=-1.0),
            BeatriceParameterSet(pitch_shift_semitone=1.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            results = BeatriceCandidateGenerator(
                self.descriptor,
                directory,
                directory,
                engine_factory=FakeEngine,
            ).generate(audio, options)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(FakeEngine.instances), 2)
        self.assertTrue(all(instance.unloaded for instance in FakeEngine.instances))
        self.assertIsNot(FakeEngine.instances[0], FakeEngine.instances[1])

        cancelled = threading.Event()

        class CancellingEngine(FakeEngine):
            def process_audio(self, block):
                output = super().process_audio(block)
                cancelled.set()
                return output

        FakeEngine.instances.clear()
        with tempfile.TemporaryDirectory() as directory:
            results = BeatriceCandidateGenerator(
                self.descriptor,
                directory,
                directory,
                engine_factory=CancellingEngine,
            ).generate(audio, options, cancel_event=cancelled)
        self.assertEqual(results, [])
        self.assertEqual(FakeEngine.instances[0].blocks, 1)
        self.assertTrue(FakeEngine.instances[0].unloaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)

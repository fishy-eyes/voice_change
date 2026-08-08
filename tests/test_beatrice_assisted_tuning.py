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
    analyze_beatrice_voice,
)
from customization.candidate_evaluator import RawCandidateSafetyEvaluator
from customization.quality_checker import RecordingQualityChecker


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

    def test_f0_analysis_and_pitch_recommendation_keep_median_behavior(self) -> None:
        self.assertAlmostEqual(metadata_pitch_to_hz(69.0), 440.0)
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
        invalid = analyze_beatrice_voice(np.zeros(48_000, np.float32), 48_000)
        self.assertIsNone(invalid.f0_p5)
        self.assertIsNone(invalid.f0_p50)
        self.assertIsNone(invalid.f0_p95)
        self.assertEqual(invalid.f0_count, 0)
        self.assertFalse(
            RecordingQualityChecker().check(
                np.zeros(48_000, np.float32), 48_000
            ).is_acceptable
        )

    def test_search_changes_one_parameter_group_per_stage(self) -> None:
        base = BeatriceParameterSet(
            target_speaker=0,
            min_source_pitch=30.0,
            max_source_pitch=1100.0,
        )
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
            self.assertNotIn("min_source_pitch", differing)
            self.assertNotIn("max_source_pitch", differing)
            self.assertTrue(
                all(
                    candidate.min_source_pitch == 30.0
                    and candidate.max_source_pitch == 1100.0
                    for candidate in search.current.candidates
                )
            )
            search.choose(len(search.current.candidates) // 2)
            base = search.history[-1].candidates[search.history[-1].selected_index]
        self.assertEqual(stages, list(BeatriceParameterSearch.STAGES))
        self.assertNotIn("source_pitch", stages)
        self.assertEqual(search.final_parameters.min_source_pitch, 30.0)
        self.assertEqual(search.final_parameters.max_source_pitch, 1100.0)
        self.assertNotIn(
            "min_source_pitch", search.final_parameters.to_assisted_changes()
        )
        self.assertNotIn(
            "max_source_pitch", search.final_parameters.to_assisted_changes()
        )
        self.assertTrue(all(1 <= len(round_.candidates) <= 5 for round_ in search.history))

    def test_all_unsafe_rounds_keep_the_previous_parameters(self) -> None:
        base = BeatriceParameterSet(
            target_speaker=0,
            pitch_shift_semitone=2.5,
            min_source_pitch=95.0,
            max_source_pitch=410.0,
            formant_shift=-1.0,
            vq_num_neighbors=2,
        )
        search = BeatriceParameterSearch(
            base, self.analysis, self.capabilities, self.descriptor
        )
        stages = []
        while search.final_parameters is None:
            stages.append(search.current.stage)
            self.assertEqual(search.current.fallback, base)
            search.skip_unsafe_round()

        self.assertEqual(stages, list(BeatriceParameterSearch.STAGES))
        self.assertEqual(search.final_parameters, base)
        self.assertTrue(
            all(round_.selected_index is None for round_ in search.history)
        )

    def test_raw_safety_gate_rejects_damage_before_level_matching(self) -> None:
        evaluator = RawCandidateSafetyEvaluator()
        source = np.full(48_000, 0.1, dtype=np.float32)
        normal = evaluator.evaluate(source, source.copy(), 48_000)
        self.assertTrue(normal.is_safe)
        self.assertEqual(normal.clipping_ratio, 0.0)

        slight = source.copy()
        slight[:48] = 1.2
        slight_result = evaluator.evaluate(source, slight, 48_000)
        self.assertTrue(slight_result.is_safe)
        self.assertTrue(slight_result.would_clip_on_pcm_output)

        clipped = source.copy()
        clipped[:192] = 1.7
        clipped_result = evaluator.evaluate(source, clipped, 48_000)
        self.assertFalse(clipped_result.is_safe)
        self.assertAlmostEqual(clipped_result.peak, 1.7, places=5)
        self.assertAlmostEqual(clipped_result.clipping_ratio, 0.004, places=5)

        for damaged in (
            np.full_like(source, np.nan),
            np.full_like(source, np.inf),
            np.zeros_like(source),
        ):
            self.assertFalse(evaluator.evaluate(source, damaged, 48_000).is_safe)

        class ClippingEngine(FakeEngine):
            def process_audio(self, block):
                output = super().process_audio(block)
                output[0] = 1.7
                return output

        matcher_calls = []

        def level_matcher(reference, candidate):
            matcher_calls.append((reference, candidate))
            return candidate

        with tempfile.TemporaryDirectory() as directory:
            result = BeatriceCandidateGenerator(
                self.descriptor,
                directory,
                directory,
                engine_factory=ClippingEngine,
                level_matcher=level_matcher,
            ).generate(source, [BeatriceParameterSet()])[0]
        self.assertFalse(result.raw_safety.is_safe)
        self.assertIsNone(result.audio_path)
        self.assertEqual(matcher_calls, [])

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

from __future__ import annotations

import threading
import tempfile
import unittest

import numpy as np
import soundfile as sf

from customization.candidate_generator import CandidateGenerator
from customization.parameter_search import (
    PITCH_COARSE_VALUES,
    ParameterSearch,
    index_rate_round,
    pitch_coarse_round,
    pitch_fine_round,
)
from customization.schemas import RVCParameterSet


class FakeEngine:
    def __init__(self) -> None:
        self.config = "original-config"
        self.updates: list[object] = []

    def update_config(self, config=None, **changes):
        self.updates.append(config if config is not None else dict(changes))
        return config if config is not None else changes

    def infer(self, audio):
        return np.asarray(audio, dtype=np.float32) * 0.8


class ParameterSearchTests(unittest.TestCase):
    def test_pitch_coarse_candidates_are_correct(self) -> None:
        result = pitch_coarse_round(RVCParameterSet())
        self.assertEqual(
            tuple(candidate.pitch_shift for candidate in result.candidates),
            PITCH_COARSE_VALUES,
        )

    def test_pitch_fine_candidates_surround_center(self) -> None:
        result = pitch_fine_round(RVCParameterSet(pitch_shift=8))
        self.assertEqual(
            [candidate.pitch_shift for candidate in result.candidates],
            [6, 8, 10],
        )

    def test_no_index_skips_index_round(self) -> None:
        self.assertIsNone(index_rate_round(RVCParameterSet(), has_index=False))
        search = ParameterSearch(has_index=False)
        search.choose(3)
        next_round = search.choose(1)
        self.assertIsNotNone(next_round)
        self.assertEqual(next_round.stage, "protect")
        self.assertTrue(all(item.index_rate == 0 for item in next_round.candidates))

    def test_full_search_locks_previous_parameters_and_finishes_in_order(self) -> None:
        search = ParameterSearch(has_index=True)

        next_round = search.choose(4)
        self.assertEqual(next_round.stage, "pitch_fine")
        self.assertEqual([item.pitch_shift for item in next_round.candidates], [2, 4, 6])

        next_round = search.choose(1)
        self.assertEqual(next_round.stage, "index_rate")
        self.assertTrue(all(item.pitch_shift == 4 for item in next_round.candidates))

        next_round = search.choose(2)
        self.assertEqual(next_round.stage, "protect")
        self.assertTrue(all(item.index_rate == 0.80 for item in next_round.candidates))

        next_round = search.choose(0)
        self.assertEqual(next_round.stage, "rms_mix_rate")
        self.assertTrue(all(item.protect == 0.0 for item in next_round.candidates))

        self.assertIsNone(search.choose(1))
        self.assertEqual(
            [round_.stage for round_ in search.history],
            ["pitch_coarse", "pitch_fine", "index_rate", "protect", "rms_mix_rate"],
        )
        self.assertEqual(
            search.final_parameters, RVCParameterSet(4, "rmvpe", 0.80, 0.0, 0.50)
        )

    def test_search_advances_and_can_cancel(self) -> None:
        search = ParameterSearch(has_index=True)
        next_round = search.choose(4)
        self.assertEqual(next_round.stage, "pitch_fine")
        self.assertEqual(next_round.candidates[1].pitch_shift, 4)
        search.cancel()
        with self.assertRaises(RuntimeError):
            search.choose(0)

    def test_candidate_generation_restores_engine_and_can_cancel(self) -> None:
        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        engine = FakeEngine()
        with tempfile.TemporaryDirectory() as directory:
            generator = CandidateGenerator(engine, directory, sample_rate=sample_rate)
            results = generator.generate(
                audio,
                [RVCParameterSet(pitch_shift=-4), RVCParameterSet(pitch_shift=4)],
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(all(item.audio_path for item in results))
            self.assertEqual(engine.updates[-1], "original-config")
            audition, written_rate = sf.read(results[0].audio_path, dtype="float32")
            self.assertEqual(written_rate, sample_rate)
            self.assertEqual(len(audition), len(audio))
            source_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            audition_rms = float(np.sqrt(np.mean(np.square(audition, dtype=np.float64))))
            self.assertAlmostEqual(audition_rms, source_rms, delta=5e-4)

            cancelled = threading.Event()
            cancelled.set()
            self.assertEqual(generator.generate(audio, [RVCParameterSet()], cancel_event=cancelled), [])
            self.assertEqual(engine.updates[-1], "original-config")


if __name__ == "__main__":
    unittest.main(verbosity=2)

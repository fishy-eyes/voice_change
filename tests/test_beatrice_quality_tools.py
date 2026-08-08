from __future__ import annotations

import unittest

import numpy as np

from customization.beatrice import (
    BeatriceParameterSet,
    BeatriceTuningCapabilities,
    BeatriceVoiceAnalysis,
)
from experiments.beatrice_quality.compare_audio import compare_arrays, periodic_report
from experiments.beatrice_quality.parameter_sweep import build_candidates
from experiments.beatrice_quality.speaker_sweep import select_speakers


class _Descriptor:
    speaker_names = tuple(f"jvs{index:03d}" for index in range(1, 101))
    speaker_average_pitches = tuple([60.0] * 100)


class BeatriceQualityToolTests(unittest.TestCase):
    def test_lag_alignment_recovers_delayed_signal(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        reference = (0.2 * np.sin(2 * np.pi * 233.0 * time)).astype(np.float32)
        candidate = np.pad(reference, (321, 0))[: len(reference)]
        report = compare_arrays(reference, sample_rate, candidate, sample_rate)
        self.assertEqual(report["best_lag_samples"], 321)
        self.assertGreater(report["correlation"], 0.999)
        self.assertLess(report["rmse"], 1e-5)

    def test_periodic_report_has_requested_boundaries(self) -> None:
        report = periodic_report(np.zeros(48_000, dtype=np.float32), 48_000)
        self.assertEqual(report["callback_5_333ms"]["period_samples"], 256)
        self.assertEqual(report["native_10ms"]["period_samples"], 480)
        self.assertEqual(report["native_20ms"]["period_samples"], 960)

    def test_candidate_matrix_is_small_and_one_parameter_at_a_time(self) -> None:
        base = BeatriceParameterSet()
        analysis = BeatriceVoiceAnalysis(100.0, 140.0, 220.0, 0.1)
        capabilities = BeatriceTuningCapabilities(
            pitch_shift_min=-24,
            pitch_shift_max=24,
            max_formant_shift=8,
            codebook_size=512,
        )
        candidates = build_candidates(base, analysis, capabilities, _Descriptor())
        self.assertEqual(len(candidates["pitch_range"]), 3)
        self.assertEqual(len(candidates["pitch"]), 3)
        self.assertEqual(len(candidates["formant"]), 3)
        self.assertLessEqual(len(candidates["vq"]), 4)

    def test_speaker_sweep_uses_requested_jvs_names(self) -> None:
        selected = select_speakers(_Descriptor())
        self.assertEqual(
            [name for _, name in selected],
            ["jvs001", "jvs010", "jvs030", "jvs050", "jvs080"],
        )


if __name__ == "__main__":
    unittest.main()

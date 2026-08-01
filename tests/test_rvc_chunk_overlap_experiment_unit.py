"""Pure unit tests for the isolated chunk/overlap experiment helpers."""

from __future__ import annotations

import unittest

import numpy as np

from tests.test_rvc_chunk_overlap_experiment import (
    PRIORITY_CASES,
    build_cases,
    continuity_metrics,
    linear_overlap_add,
)
from tests.test_rvc_realtime_benchmark import build_overlapping_windows


class ChunkOverlapExperimentTests(unittest.TestCase):
    def test_default_priority_and_full_matrix(self) -> None:
        chunks = [200, 325, 500, 700]
        overlaps = [0, 25, 50, 100]
        self.assertEqual(build_cases(chunks, overlaps, False), list(PRIORITY_CASES))
        self.assertEqual(len(build_cases(chunks, overlaps, True)), 16)

    def test_overlap_add_reconstructs_unmodified_windows_and_exact_length(self) -> None:
        source = np.linspace(-0.8, 0.8, 1031, dtype=np.float32)
        windows, starts = build_overlapping_windows(source, 240, 50)
        output = linear_overlap_add(windows, starts, source.size, 50)
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_allclose(output, source, atol=1e-6)

    def test_continuity_metrics_detect_an_inserted_silent_gap(self) -> None:
        sample_rate = 1000
        timeline = np.arange(1000, dtype=np.float32) / sample_rate
        clean = np.sin(2 * np.pi * 10 * timeline).astype(np.float32)
        damaged = clean.copy()
        damaged[490:520] = 0.0
        metrics = continuity_metrics(damaged, [0, 500], 0, sample_rate)
        self.assertGreaterEqual(metrics["maximum_silent_run_ms"], 30.0)
        self.assertGreater(metrics["obvious_silence_boundary_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

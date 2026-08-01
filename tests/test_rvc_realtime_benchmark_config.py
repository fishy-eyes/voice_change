"""Fast tests for realtime benchmark configuration and windowing."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SAMPLE_RATE
from tests.test_rvc_realtime_benchmark import (
    build_overlapping_windows,
    milliseconds_to_samples,
    summarize_performance,
    validate_benchmark_settings,
)


class RealtimeBenchmarkConfigTests(unittest.TestCase):
    def test_required_chunk_shapes_at_configured_rate(self) -> None:
        self.assertEqual(SAMPLE_RATE, 48000)
        self.assertEqual(milliseconds_to_samples(100, SAMPLE_RATE), 4800)
        self.assertEqual(milliseconds_to_samples(200, SAMPLE_RATE), 9600)
        self.assertEqual(milliseconds_to_samples(325, SAMPLE_RATE), 15600)
        self.assertEqual(milliseconds_to_samples(500, SAMPLE_RATE), 24000)

    def test_settings_reject_overlap_at_or_above_smallest_chunk(self) -> None:
        validate_benchmark_settings([100, 200, 325, 500], 99.0, 10.0, 3, 2)
        with self.assertRaisesRegex(AssertionError, "overlap_ms"):
            validate_benchmark_settings([100, 200], 100.0, 10.0, 3, 2)

    def test_settings_reject_duplicates_and_invalid_counts(self) -> None:
        with self.assertRaisesRegex(AssertionError, "unique"):
            validate_benchmark_settings([100, 100], 0.0, 10.0, 3, 2)
        with self.assertRaisesRegex(AssertionError, "serial_count"):
            validate_benchmark_settings([100], 0.0, 10.0, 0, 2)
        with self.assertRaisesRegex(AssertionError, "queue_size"):
            validate_benchmark_settings([100], 0.0, 10.0, 3, 0)

    def test_windows_use_hop_and_zero_pad_only_the_tail(self) -> None:
        audio = np.arange(11, dtype=np.float32)
        windows, starts = build_overlapping_windows(audio, 4, 1)

        self.assertEqual(starts, [0, 3, 6, 9])
        self.assertTrue(all(window.shape == (4,) for window in windows))
        np.testing.assert_array_equal(windows[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(windows[-1], [9, 10, 0, 0])

    def test_short_input_produces_one_fixed_shape_window(self) -> None:
        windows, starts = build_overlapping_windows(
            np.array([0.25, -0.25], dtype=np.float32),
            4,
            0,
        )
        self.assertEqual(starts, [0])
        np.testing.assert_array_equal(windows[0], [0.25, -0.25, 0.0, 0.0])

    def test_performance_summary_reports_rtf_and_extrema(self) -> None:
        metrics = summarize_performance([0.1, 0.2, 0.3], 0.5)
        self.assertAlmostEqual(metrics["total_inference_seconds"], 0.6)
        self.assertAlmostEqual(metrics["average_inference_seconds"], 0.2)
        self.assertAlmostEqual(metrics["average_latency_ms"], 200.0)
        self.assertAlmostEqual(metrics["minimum_inference_seconds"], 0.1)
        self.assertAlmostEqual(metrics["maximum_inference_seconds"], 0.3)
        self.assertAlmostEqual(metrics["rtf"], 0.4)


if __name__ == "__main__":
    unittest.main()

"""Fast tests for RVCEngine timing snapshots and profiling helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rvc_engine import RVCEngine
from test_rvc_inference_profile_benchmark import (
    EXACT_STAGE_KEYS,
    build_cases,
    combine_timings,
    summarize_records,
)


class _FakePipeline:
    def pipeline(self, *, audio, times, **_kwargs):
        times[:] = [0.010, 0.020, 0.030]
        return np.asarray(audio * 0.5, dtype=np.float32)


class _FailingPipeline:
    def pipeline(self, **_kwargs):
        raise RuntimeError("expected failure")


def make_engine(pipeline) -> RVCEngine:
    engine = RVCEngine(
        voice_dir=Path("unused-voice"),
        source_dir=Path("unused-source"),
        models_dir=Path("unused-models"),
        sample_rate=16000,
    )
    engine._pipeline = pipeline
    engine._model_loaded = True
    engine._hubert_model = object()
    engine._net_g = object()
    engine._tgt_sr = 16000
    engine._if_f0 = 1
    engine._version = "v1"
    return engine


class RVCInferenceProfilingTests(unittest.TestCase):
    def test_engine_publishes_copy_of_native_timing_snapshot(self) -> None:
        engine = make_engine(_FakePipeline())
        output = engine.infer(np.linspace(-0.5, 0.5, 1600, dtype=np.float32))

        self.assertEqual(output.shape, (1600,))
        profile = engine.last_inference_profile
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["content_index_prepare_ms"], 10.0)
        self.assertEqual(profile["f0_ms"], 20.0)
        self.assertEqual(profile["index_synth_ms"], 30.0)
        self.assertGreaterEqual(profile["total_ms"], 0.0)
        profile["total_ms"] = -1.0
        self.assertNotEqual(engine.last_inference_profile["total_ms"], -1.0)

    def test_failed_inference_clears_previous_snapshot(self) -> None:
        engine = make_engine(_FakePipeline())
        source = np.ones(320, dtype=np.float32)
        engine.infer(source)
        self.assertIsNotNone(engine.last_inference_profile)
        engine._pipeline = _FailingPipeline()

        output = engine.infer(source)

        np.testing.assert_array_equal(output, source)
        self.assertIsNone(engine.last_inference_profile)

    def test_focused_matrix_contains_six_unique_cases(self) -> None:
        cases = build_cases(
            [325, 500], ["rmvpe", "pm", "fcpe"], [0.0, 0.3, 0.5],
            "rmvpe", 0.3,
        )
        keys = {
            (case["chunk_ms"], case["f0_method"], case["index_rate"])
            for case in cases
        }
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(keys), 6)
        self.assertIn((325, "rmvpe", 0.3), keys)
        self.assertIn((500, "pm", 0.3), keys)
        self.assertIn((500, "rmvpe", 0.0), keys)

    def test_combined_timing_preserves_native_and_exact_boundaries(self) -> None:
        native = {
            "total_ms": 100.0,
            "preprocess_ms": 5.0,
            "pipeline_ms": 90.0,
            "postprocess_ms": 4.0,
            "content_index_prepare_ms": 30.0,
            "f0_ms": 20.0,
            "index_synth_ms": 35.0,
            "pipeline_overhead_ms": 5.0,
        }
        exact = {key: 10.0 for key in EXACT_STAGE_KEYS}

        result = combine_timings(native, exact)

        self.assertEqual(result["hubert_ms"], 10.0)
        self.assertEqual(result["native_index_synth_ms"], 35.0)
        self.assertEqual(result["pipeline_unattributed_ms"], 30.0)
        self.assertEqual(result["total_unattributed_ms"], 1.0)

    def test_summary_reports_average_percentage_and_rtf(self) -> None:
        first = {"total_ms": 500.0, "hubert_ms": 100.0}
        second = {"total_ms": 600.0, "hubert_ms": 120.0}

        result = summarize_records([first, second], 500.0)

        self.assertEqual(result["average"]["total_ms"], 550.0)
        self.assertEqual(result["average"]["hubert_ms"], 110.0)
        self.assertEqual(result["percentage_of_total"]["hubert_ms"], 20.0)
        self.assertEqual(result["rtf"], 1.1)


if __name__ == "__main__":
    unittest.main()

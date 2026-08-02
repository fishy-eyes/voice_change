from __future__ import annotations

import unittest

import numpy as np

from customization.candidate_evaluator import CandidateEvaluator


class CandidateEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 16000
        t = np.arange(self.sample_rate * 2, dtype=np.float32) / self.sample_rate
        self.normal = (0.18 * np.sin(2 * np.pi * 170 * t)).astype(np.float32)
        self.evaluator = CandidateEvaluator()

    def test_empty_audio_is_rejected(self) -> None:
        result = self.evaluator.evaluate(self.normal, np.array([]), self.sample_rate)
        self.assertFalse(result.is_valid)
        self.assertIn("输出为空", result.rejection_reasons)

    def test_nan_audio_is_rejected(self) -> None:
        output = self.normal.copy()
        output[3] = np.nan
        result = self.evaluator.evaluate(self.normal, output, self.sample_rate)
        self.assertFalse(result.is_valid)
        self.assertIn("输出包含 NaN 或 Inf", result.rejection_reasons)

    def test_abnormal_duration_is_rejected(self) -> None:
        result = self.evaluator.evaluate(self.normal, self.normal[:1000], self.sample_rate)
        self.assertFalse(result.is_valid)
        self.assertIn("输出时长异常", result.rejection_reasons)

    def test_severe_clipping_is_rejected(self) -> None:
        result = self.evaluator.evaluate(self.normal, np.sign(self.normal), self.sample_rate)
        self.assertFalse(result.is_valid)
        self.assertIn("削波严重", result.rejection_reasons)

    def test_normal_audio_passes_with_stable_scores(self) -> None:
        first = self.evaluator.evaluate(self.normal, self.normal * 0.9, self.sample_rate)
        second = self.evaluator.evaluate(self.normal, self.normal * 0.9, self.sample_rate)
        self.assertTrue(first.is_valid, first.rejection_reasons)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.technical_quality, 0)
        self.assertLessEqual(first.technical_quality, 100)

    def test_technical_ranking_does_not_reward_louder_candidates(self) -> None:
        quiet = self.evaluator.evaluate(
            self.normal, self.normal * 0.35, self.sample_rate
        )
        loud = self.evaluator.evaluate(
            self.normal, self.normal * 0.90, self.sample_rate
        )
        self.assertTrue(quiet.is_valid, quiet.rejection_reasons)
        self.assertTrue(loud.is_valid, loud.rejection_reasons)
        self.assertNotEqual(quiet.volume_score, loud.volume_score)
        self.assertEqual(quiet.technical_quality, loud.technical_quality)


if __name__ == "__main__":
    unittest.main(verbosity=2)

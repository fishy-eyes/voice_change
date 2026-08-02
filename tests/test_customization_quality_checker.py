from __future__ import annotations

import unittest

import numpy as np

from customization.quality_checker import RecordingQualityChecker


SAMPLE_RATE = 16000


def synthetic_voice(seconds: float = 6.0, amplitude: float = 0.20) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    frequency = np.where(t < seconds / 2, 140.0, 190.0)
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t)
    gate = ((t % 1.0) < 0.78).astype(np.float32)
    return (signal * gate).astype(np.float32)


class RecordingQualityCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = RecordingQualityChecker(minimum_duration=5.0)

    def test_normal_voice_passes(self) -> None:
        result = self.checker.check(synthetic_voice(), SAMPLE_RATE)
        self.assertTrue(result.is_acceptable, result.rejection_reasons)
        self.assertTrue(result.has_valid_pitch)
        self.assertGreater(result.quality_score, 50)

    def test_silence_is_rejected(self) -> None:
        result = self.checker.check(np.zeros(SAMPLE_RATE * 6), SAMPLE_RATE)
        self.assertFalse(result.is_acceptable)
        self.assertIn("未检测到有效语音", result.rejection_reasons)

    def test_short_audio_is_rejected(self) -> None:
        result = self.checker.check(synthetic_voice(1.0), SAMPLE_RATE)
        self.assertIn("录音过短", result.rejection_reasons)

    def test_severe_clipping_is_rejected(self) -> None:
        clipped = np.sign(synthetic_voice())
        result = self.checker.check(clipped, SAMPLE_RATE)
        self.assertIn("严重削波", result.rejection_reasons)

    def test_very_low_volume_is_rejected(self) -> None:
        result = self.checker.check(synthetic_voice(amplitude=0.001), SAMPLE_RATE)
        self.assertIn("音量过小", result.rejection_reasons)

    def test_nan_is_rejected_without_crashing(self) -> None:
        audio = synthetic_voice()
        audio[10] = np.nan
        result = self.checker.check(audio, SAMPLE_RATE)
        self.assertFalse(result.is_acceptable)
        self.assertIn("audio contains NaN or Inf", result.rejection_reasons)

    def test_unpitched_noise_has_no_valid_pitch(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 0.05, SAMPLE_RATE * 6).astype(np.float32)
        result = self.checker.check(noise, SAMPLE_RATE)
        self.assertFalse(result.has_valid_pitch)
        self.assertIn("未检测到有效基频", result.rejection_reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)

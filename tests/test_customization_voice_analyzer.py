from __future__ import annotations

import unittest

import numpy as np

from customization.voice_analyzer import VoiceAnalyzer


class VoiceAnalyzerTests(unittest.TestCase):
    def test_reports_stable_pitch_statistics(self) -> None:
        sample_rate = 16000
        t = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
        audio = (0.15 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)

        result = VoiceAnalyzer().analyze(audio, sample_rate)

        self.assertAlmostEqual(result.duration_seconds, 3.0, places=2)
        self.assertIsNotNone(result.f0_median)
        self.assertGreater(result.f0_median or 0, 140)
        self.assertLess(result.f0_median or 999, 180)
        self.assertGreater(result.rms_mean, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)

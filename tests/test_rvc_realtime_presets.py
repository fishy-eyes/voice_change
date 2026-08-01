"""Fast tests for formal RVC realtime presets and buffer switching."""

from __future__ import annotations

import unittest

import numpy as np

from config.rvc_realtime import (
    RVC_DEFAULT_REALTIME_PRESET,
    RVC_REALTIME_PRESETS,
    RVCRealtimePreset,
)
from config.settings import (
    RVC_CHUNK_SIZE,
    RVC_OVERLAP_SIZE,
    SAMPLE_RATE,
)
from effects.ai_voice import AIVoiceEffect


class IdentityEngine:
    sample_rate = SAMPLE_RATE
    is_loaded = True

    @staticmethod
    def infer(audio: np.ndarray) -> np.ndarray:
        return np.asarray(audio, dtype=np.float32).copy()


class RVCRealtimePresetTests(unittest.TestCase):
    def test_named_presets_and_balanced_defaults(self) -> None:
        self.assertEqual(RVC_DEFAULT_REALTIME_PRESET, "balanced")
        self.assertEqual(
            [preset.name for preset in RVC_REALTIME_PRESETS.values()],
            ["Low Latency", "Balanced", "High Quality"],
        )
        self.assertEqual(
            [
                (preset.chunk_ms, preset.overlap_ms)
                for preset in RVC_REALTIME_PRESETS.values()
            ],
            [(325, 50), (500, 50), (500, 100)],
        )
        balanced = RVC_REALTIME_PRESETS["balanced"]
        self.assertEqual(RVC_CHUNK_SIZE, balanced.chunk_samples(SAMPLE_RATE))
        self.assertEqual(RVC_OVERLAP_SIZE, balanced.overlap_samples(SAMPLE_RATE))

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RVCRealtimePreset("Invalid", chunk_ms=100, overlap_ms=51)

    def test_linear_overlap_stitching(self) -> None:
        effect = AIVoiceEffect(IdentityEngine(), chunk_size=8, overlap_size=2)
        first = np.arange(8, dtype=np.float32)
        second = np.arange(100, 108, dtype=np.float32)

        effect._append_converted_window(first)
        effect._append_converted_window(second)

        output = effect._take_output(12)
        expected = np.concatenate(
            (
                first[:6],
                np.array([6.0, 101.0], dtype=np.float32),
                second[2:6],
            )
        )
        np.testing.assert_allclose(output, expected)

    def test_input_windows_advance_by_hop_size(self) -> None:
        effect = AIVoiceEffect(IdentityEngine(), chunk_size=8, overlap_size=2)
        submitted: list[np.ndarray] = []
        effect.worker.put = lambda audio, timeout=0.0: (
            submitted.append(audio.copy()) or True
        )

        effect._accumulate_input(np.arange(14, dtype=np.float32))

        self.assertEqual(len(submitted), 2)
        np.testing.assert_array_equal(submitted[0], np.arange(8, dtype=np.float32))
        np.testing.assert_array_equal(
            submitted[1], np.arange(6, 14, dtype=np.float32)
        )
        self.assertEqual(effect.input_buffered_samples, 2)

    def test_running_buffer_switch_keeps_engine_worker_and_thread(self) -> None:
        engine = IdentityEngine()
        effect = AIVoiceEffect(engine, chunk_size=8, overlap_size=2)
        self.assertTrue(effect.start())
        worker = effect.worker
        thread = worker._thread
        try:
            effect.update_realtime_config(chunk_size=10, overlap_size=3)
            self.assertIs(effect.engine, engine)
            self.assertIs(effect.worker, worker)
            self.assertIs(effect.worker._thread, thread)
            self.assertTrue(effect.worker.is_running)
            self.assertEqual(effect.chunk_size, 10)
            self.assertEqual(effect.overlap_size, 3)
            self.assertEqual(effect.hop_size, 7)
            self.assertEqual(effect.input_buffered_samples, 0)
            self.assertEqual(effect.output_buffered_samples, 0)
        finally:
            self.assertTrue(effect.stop())


if __name__ == "__main__":
    unittest.main(verbosity=2)

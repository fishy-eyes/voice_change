"""Fast deterministic tests for the isolated streaming adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from streaming_adapter import BeatriceStreamingAdapter, Float32FIFO


class FakeConverter:
    def convert(self, block: np.ndarray):
        positions = np.linspace(0.0, block.size - 1, 240)
        converted = np.interp(positions, np.arange(block.size), block).astype(
            np.float32
        )
        return converted, 0


class Factory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        module = SimpleNamespace(
            IN_SAMPLE_RATE=16_000,
            OUT_SAMPLE_RATE=24_000,
            IN_HOP_LENGTH=160,
            OUT_HOP_LENGTH=240,
        )
        return FakeConverter(), module, {"fake": True, "load_seconds": 0.0}


class StreamingAdapterTests(unittest.TestCase):
    def test_fifo_keeps_order_across_chunks(self) -> None:
        fifo = Float32FIFO()
        fifo.append(np.arange(3, dtype=np.float32))
        fifo.append(np.arange(3, 7, dtype=np.float32))
        np.testing.assert_array_equal(fifo.pop(5), np.arange(5, dtype=np.float32))
        np.testing.assert_array_equal(fifo.pop(2), np.arange(5, 7, dtype=np.float32))
        self.assertEqual(fifo.size, 0)

    def test_256_sample_contract_and_bounded_buffers(self) -> None:
        factory = Factory()
        adapter = BeatriceStreamingAdapter(converter_factory=factory)
        phase = np.arange(256 * 600, dtype=np.float64) / 48_000.0
        source = np.sin(2.0 * np.pi * 220.0 * phase).astype(np.float32)
        outputs = []
        for start in range(0, source.size, 256):
            output = adapter.process(source[start : start + 256])
            self.assertEqual(output.shape, (256,))
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            outputs.append(output)

        rendered = np.concatenate(outputs)
        self.assertFalse(np.all(rendered == 0.0))
        stats = adapter.stats()
        self.assertEqual(stats["buffer"]["underflow_count_after_start"], 0)
        self.assertEqual(stats["buffer"]["overflow_count"], 0)
        self.assertEqual(stats["buffer"]["dropped_samples"], 0)
        self.assertLessEqual(stats["buffer"]["input_fifo_max_samples_at_16khz"], 319)
        self.assertLess(stats["buffer"]["output_fifo_max_samples_at_48khz"], 1500)
        self.assertEqual(stats["timing"]["deadline_miss_count"], 0)
        adapter.close()

    def test_reset_recreates_converter_and_close_is_idempotent(self) -> None:
        factory = Factory()
        adapter = BeatriceStreamingAdapter(converter_factory=factory)
        for _ in range(4):
            adapter.process(np.ones(256, dtype=np.float32))
        self.assertGreater(adapter.stats()["work"]["beatrice_convert_count"], 0)

        adapter.reset()
        stats = adapter.stats()
        self.assertEqual(factory.calls, 2)
        self.assertEqual(stats["converter_generation"], 2)
        self.assertEqual(stats["work"]["input_callbacks"], 0)
        self.assertEqual(stats["buffer"]["input_fifo_current_samples_at_16khz"], 0)
        self.assertEqual(stats["buffer"]["output_fifo_current_samples_at_48khz"], 0)

        adapter.close()
        adapter.close()
        self.assertTrue(adapter.closed)
        with self.assertRaises(RuntimeError):
            adapter.process(np.zeros(256, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

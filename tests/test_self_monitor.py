"""Unit tests for the independent post-effect self-monitor output."""

from __future__ import annotations

import unittest

import numpy as np

from audio.monitor import SelfMonitor


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.active = False
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.closed = True


class SelfMonitorTests(unittest.TestCase):
    def test_processed_audio_is_copied_to_independent_output(self) -> None:
        output_streams: list[FakeStream] = []

        def output_factory(**kwargs):
            stream = FakeStream(**kwargs)
            output_streams.append(stream)
            return stream

        monitor = SelfMonitor(volume=0.5, output_stream_factory=output_factory)
        monitor.start(output_device=7)

        self.assertTrue(monitor.is_running)
        self.assertEqual(output_streams[0].kwargs["device"], 7)
        self.assertEqual(output_streams[0].kwargs["channels"], 1)
        source = np.array([[0.5], [-0.5]], dtype=np.float32)
        self.assertTrue(monitor.submit(source))
        output = np.empty_like(source)
        output_streams[0].kwargs["callback"](output, 2, None, None)
        np.testing.assert_allclose(output, source * 0.5)
        self.assertEqual(monitor.callback_count, 1)

        monitor.volume = 1.0
        self.assertTrue(monitor.submit(source))
        output_streams[0].kwargs["callback"](output, 2, None, None)
        np.testing.assert_allclose(output, source)
        monitor.stop()
        self.assertFalse(monitor.is_running)
        self.assertTrue(output_streams[0].closed)

    def test_output_callback_preserves_unconsumed_block_tail(self) -> None:
        output_streams: list[FakeStream] = []

        def output_factory(**kwargs):
            stream = FakeStream(**kwargs)
            output_streams.append(stream)
            return stream

        monitor = SelfMonitor(volume=1.0, output_stream_factory=output_factory)
        monitor.start(output_device=7)
        source = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        self.assertTrue(monitor.submit(source))
        callback = output_streams[0].kwargs["callback"]

        first = np.empty((2, 1), dtype=np.float32)
        second = np.empty((2, 1), dtype=np.float32)
        callback(first, 2, None, None)
        callback(second, 2, None, None)

        np.testing.assert_allclose(first, source[:2])
        np.testing.assert_allclose(second, source[2:])
        self.assertEqual(monitor.underflow_count, 0)
        monitor.stop()


    def test_queue_is_bounded_and_missing_audio_outputs_silence(self) -> None:
        output_streams: list[FakeStream] = []

        def output_factory(**kwargs):
            stream = FakeStream(**kwargs)
            output_streams.append(stream)
            return stream

        monitor = SelfMonitor(
            volume=0.5,
            queue_blocks=1,
            output_stream_factory=output_factory,
        )
        monitor.start(output_device=7)
        self.assertTrue(monitor.submit(np.ones((2, 1), dtype=np.float32)))
        self.assertTrue(
            monitor.submit(np.full((2, 1), 0.25, dtype=np.float32))
        )
        self.assertGreaterEqual(monitor.drop_count, 1)

        output = np.empty((2, 1), dtype=np.float32)
        callback = output_streams[0].kwargs["callback"]
        callback(output, 2, None, None)
        np.testing.assert_allclose(output, 0.125)
        callback(output, 2, None, None)
        np.testing.assert_array_equal(output, np.zeros_like(output))
        self.assertEqual(monitor.underflow_count, 1)
        monitor.stop()

    def test_stop_is_idempotent_and_volume_is_validated(self) -> None:
        monitor = SelfMonitor(output_stream_factory=FakeStream)
        monitor.stop()
        with self.assertRaises(ValueError):
            monitor.volume = 1.1


if __name__ == "__main__":
    unittest.main(verbosity=2)

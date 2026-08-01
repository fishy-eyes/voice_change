"""Tests for post-effect fan-out to the optional self monitor."""

from __future__ import annotations

import unittest

import numpy as np

from audio.output_router import OutputRoutingEffectManager
from effects.gain import GainEffect


class CapturingMonitor:
    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []

    def submit(self, audio: np.ndarray) -> bool:
        self.blocks.append(np.asarray(audio).copy())
        return True


class OutputRoutingEffectManagerTests(unittest.TestCase):
    def test_primary_result_is_unchanged_and_same_result_is_monitored(self) -> None:
        monitor = CapturingMonitor()
        manager = OutputRoutingEffectManager(monitor)
        gain = GainEffect(gain=2.0)
        gain.enabled = True
        manager.add(gain)
        source = np.array([[0.1], [-0.2]], dtype=np.float32)

        processed = manager.process(source, 2, None, None)

        np.testing.assert_allclose(processed, source * 2.0)
        self.assertEqual(len(monitor.blocks), 1)
        np.testing.assert_array_equal(monitor.blocks[0], processed)


if __name__ == "__main__":
    unittest.main(verbosity=2)

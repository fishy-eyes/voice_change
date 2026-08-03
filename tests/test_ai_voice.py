"""Current AIVoiceEffect passthrough and EffectManager tests."""

from __future__ import annotations

import unittest

import numpy as np

from effects.ai_voice import AIVoiceEffect
from effects.manager import EffectManager


class UnloadedEngine:
    is_loaded = False

    def infer(self, audio):
        raise AssertionError("unloaded engine must not be called")


class AIVoiceEffectTests(unittest.TestCase):
    def test_unloaded_engine_is_shape_preserving_float32_passthrough(self) -> None:
        effect = AIVoiceEffect(engine=UnloadedEngine())
        for shape in ((256,), (256, 1)):
            audio = np.random.default_rng(7).normal(size=shape).astype(np.float32)
            original = audio.copy()
            output = effect.process(audio, 256, None, None)
            self.assertEqual(output.shape, shape)
            self.assertEqual(output.dtype, np.float32)
            np.testing.assert_array_equal(output, original)
            np.testing.assert_array_equal(audio, original)

    def test_effect_manager_keeps_unloaded_ai_passthrough(self) -> None:
        manager = EffectManager()
        effect = AIVoiceEffect(engine=UnloadedEngine())
        manager.add(effect)
        audio = np.zeros((64, 1), dtype=np.float32)
        output = manager.process(audio, 64, None, None)
        self.assertIs(manager.get_by_name("AIVoiceEffect"), effect)
        np.testing.assert_array_equal(output, audio)


if __name__ == "__main__":
    unittest.main(verbosity=2)

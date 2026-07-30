"""Tests for RVCEngine and AIVoiceEffect."""

import numpy as np
import pytest

from ai.rvc_engine import RVCEngine
from effects.ai_voice import AIVoiceEffect
from effects.manager import EffectManager

# ── helpers ─────────────────────────────────────────────────────────

def _make_audio(frames: int = 256, channels: int = 1) -> np.ndarray:
    """Generate a simple float32 mono audio block."""
    return np.random.randn(frames, channels).astype(np.float32) * 0.5


# ── 1. RVCEngine basics ─────────────────────────────────────────────

class TestRVCEngine:
    def test_create_default(self):
        engine = RVCEngine(model_path="models/test.pth")
        assert engine.model_path.as_posix() == "models/test.pth"
        assert engine.pitch_shift == 0
        assert engine.sample_rate == 44100

    def test_not_loaded_by_default(self):
        engine = RVCEngine(model_path="models/test.pth")
        assert engine.is_loaded is False

    def test_pitch_shift_setter(self):
        engine = RVCEngine(model_path="models/test.pth")
        engine.pitch_shift = 12
        assert engine.pitch_shift == 12

    def test_load_nonexistent_raises(self):
        engine = RVCEngine(model_path="models/does_not_exist.pth")
        with pytest.raises(FileNotFoundError):
            engine.load_model()

    def test_unload_when_not_loaded(self):
        engine = RVCEngine(model_path="models/test.pth")
        # should warn but not raise
        engine.unload_model()
        assert engine.is_loaded is False


# ── 2. AIVoiceEffect ────────────────────────────────────────────────

class TestAIVoiceEffect:
    def test_passthrough_when_not_loaded(self):
        """Engine not loaded -> audio passes through unchanged."""
        engine = RVCEngine(model_path="models/test.pth")
        effect = AIVoiceEffect(engine=engine)

        audio = _make_audio()
        result = effect.process(audio, 256, None, None)

        assert result.shape == audio.shape, "shape must be preserved"
        assert np.array_equal(result, audio), "passthrough: output == input"

    def test_output_dtype_float32(self):
        engine = RVCEngine(model_path="models/test.pth")
        effect = AIVoiceEffect(engine=engine)

        audio = _make_audio()
        result = effect.process(audio, 256, None, None)

        assert result.dtype == np.float32

    def test_no_error_on_process(self):
        engine = RVCEngine(model_path="models/test.pth")
        effect = AIVoiceEffect(engine=engine)

        audio = _make_audio()
        # should not raise
        effect.process(audio, 256, None, None)

    def test_engine_property(self):
        engine = RVCEngine(model_path="models/test.pth")
        effect = AIVoiceEffect(engine=engine)
        assert effect.engine is engine


# ── 3. EffectManager integration ────────────────────────────────────

class TestEffectManagerIntegration:
    def test_chain_contains_ai_voice(self):
        em = EffectManager()
        engine = RVCEngine(model_path="models/test.pth")
        em.add(AIVoiceEffect(engine=engine))

        names = [e.name for e in em.effects]
        assert "AIVoiceEffect" in names

    def test_process_through_chain(self):
        em = EffectManager()
        engine = RVCEngine(model_path="models/test.pth")
        em.add(AIVoiceEffect(engine=engine))

        audio = _make_audio()
        result = em.process(audio, 256, None, None)

        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_passthrough_in_chain(self):
        """AIVoiceEffect (unloaded) in chain -> audio unchanged."""
        em = EffectManager()
        engine = RVCEngine(model_path="models/test.pth")
        em.add(AIVoiceEffect(engine=engine))

        audio = _make_audio()
        result = em.process(audio, 256, None, None)

        assert np.array_equal(result, audio)

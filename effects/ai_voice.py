"""AI voice effect adapter.

Wraps an RVCEngine into a BaseEffect so it can be plugged
into the existing EffectManager chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from effects.base import BaseEffect

if TYPE_CHECKING:
    from ai.rvc_engine import RVCEngine


class AIVoiceEffect(BaseEffect):
    """Adapter: RVCEngine -> BaseEffect.

    Delegates audio processing to an RVCEngine instance while
    conforming to the BaseEffect interface expected by EffectManager.

    Parameters
    ----------
    engine : RVCEngine
        An initialized (but not necessarily loaded) RVC engine.
    """

    def __init__(self, engine: RVCEngine) -> None:
        super().__init__()
        self._engine: RVCEngine = engine
        logger.debug("AIVoiceEffect created with engine: {}", engine)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def engine(self) -> RVCEngine:
        """The underlying RVC engine."""
        return self._engine

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------

    def process(
        self,
        audio_data: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> np.ndarray:
        """Run RVCEngine inference on the audio block.

        Handles shape conversion between EffectManager's (frames, channels)
        format and RVCEngine's flat (samples,) format.

        If the engine is not loaded, falls back to passthrough silently.
        """
        if not self._engine.is_loaded:
            return audio_data

        original_shape = audio_data.shape

        # Flatten to 1-D for engine (mono expected)
        flat = audio_data.ravel()

        try:
            result = self._engine.infer(flat)
        except Exception as e:
            logger.error("AIVoiceEffect: engine.infer() failed: {}", e)
            return audio_data  # fallback: passthrough

        # Validate shape
        if result.shape != flat.shape:
            logger.warning(
                "AIVoiceEffect: shape mismatch: expected {} got {}, passthrough",
                flat.shape, result.shape,
            )
            return audio_data

        np.clip(result, -1.0, 1.0, out=result)
        return result.reshape(original_shape)

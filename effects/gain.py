"""Gain (volume) effect."""

import numpy as np

from effects.base import BaseEffect


class GainEffect(BaseEffect):
    """Multiply audio signal by a gain factor.

    Parameters
    ----------
    gain : float
        Volume multiplier.  1.0 = unity, >1.0 louder, <1.0 softer.
    """

    def __init__(self, gain: float = 1.0) -> None:
        super().__init__()
        self.gain: float = gain

    def process(
        self,
        audio_data: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> np.ndarray:
        """Apply gain and clip to [-1.0, 1.0]."""
        result = audio_data * self.gain
        np.clip(result, -1.0, 1.0, out=result)
        return result

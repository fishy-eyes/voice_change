"""Final output gain effect."""

import math

import numpy as np

from effects.base import BaseEffect


class GainEffect(BaseEffect):
    """Apply the always-on final output gain with clipping protection."""

    MIN_GAIN = 0.0
    MAX_GAIN = 3.0

    def __init__(self, gain: float = 1.0) -> None:
        super().__init__()
        self._gain = 1.0
        self.gain = gain

    @property
    def enabled(self) -> bool:
        return True

    @enabled.setter
    def enabled(self, _value: bool) -> None:
        # The final output stage cannot be bypassed; unity gain is 1.0.
        self._enabled = True

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        selected = float(value)
        if not math.isfinite(selected) or not self.MIN_GAIN <= selected <= self.MAX_GAIN:
            raise ValueError("output gain must be between 0.0 and 3.0")
        self._gain = selected
        # Output Gain is part of the stable output path, not an optional effect.
        self.enabled = True

    def process(
        self,
        audio_data: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> np.ndarray:
        result = audio_data * self._gain
        np.clip(result, -1.0, 1.0, out=result)
        return result

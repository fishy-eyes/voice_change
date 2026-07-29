"""Robot voice effect using ring modulation."""

import numpy as np
from loguru import logger

from config.settings import SAMPLE_RATE
from effects.base import BaseEffect


class RobotEffect(BaseEffect):
    """Ring-modulation robot voice effect.

    Multiplies the input signal by a sine wave at *frequency* Hz.
    This produces the classic metallic / robotic timbre.

    Parameters
    ----------
    frequency : float
        Carrier frequency in Hz.  Higher values give a thinner,
        more nasal robot sound; lower values give a warble.
        Default is 80 Hz.
    """

    def __init__(self, frequency: float = 80) -> None:
        super().__init__()
        self._frequency: float = float(frequency)
        self._phase: float = 0.0
        logger.debug(
            "RobotEffect init: frequency={} Hz, sample_rate={}",
            self._frequency, SAMPLE_RATE,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = float(value)

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
        """Apply ring modulation to produce a robot voice."""
        # Generate carrier sine wave for this block
        t = np.arange(frames) / SAMPLE_RATE
        carrier = np.sin(2.0 * np.pi * self._frequency * t + self._phase)

        # Advance phase so the sine is continuous across blocks
        self._phase += 2.0 * np.pi * self._frequency * frames / SAMPLE_RATE
        self._phase %= 2.0 * np.pi

        # Ring modulate: multiply each channel by the carrier
        result = audio_data * carrier[:, np.newaxis]
        np.clip(result, -1.0, 1.0, out=result)
        return result

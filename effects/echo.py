"""Echo (delay) effect with feedback."""

import numpy as np
from loguru import logger

from config.settings import SAMPLE_RATE
from effects.base import BaseEffect


class EchoEffect(BaseEffect):
    """Simple delay-line echo effect.

    Parameters
    ----------
    delay_ms : float
        Echo delay in milliseconds.
    decay : float
        Feedback gain (0.0 = single echo, 0.5 = fading repeats, >=1.0 = infinite).
    """

    def __init__(self, delay_ms: float = 200.0, decay: float = 0.4) -> None:
        super().__init__()
        self._decay: float = float(decay)
        # pre-allocate a fixed-size ring buffer (never grows)
        self._delay_samples: int = max(1, int(SAMPLE_RATE * delay_ms / 1000.0))
        self._buffer: np.ndarray = np.zeros(self._delay_samples, dtype=np.float32)
        self._write_pos: int = 0
        logger.debug(
            "EchoEffect init: delay_ms={} samples={} decay={}",
            delay_ms, self._delay_samples, self._decay,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def delay_ms(self) -> float:
        return self._delay_samples * 1000.0 / SAMPLE_RATE

    @property
    def decay(self) -> float:
        return self._decay

    @decay.setter
    def decay(self, value: float) -> None:
        self._decay = float(value)

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
        """Apply echo: mix current signal with delayed feedback.

        Uses a fixed-size ring buffer -- memory usage is constant.
        """
        buf = self._buffer
        pos = self._write_pos
        delay = self._delay_samples
        decay = self._decay

        # flatten to 1-D for processing, restore shape afterwards
        flat = audio_data.ravel()

        for i in range(frames):
            # read from the delay line
            read_pos = (pos - delay + i) % delay
            delayed = buf[read_pos]

            # mix: current + decay * delayed
            sample = flat[i] + decay * delayed

            # soft-clip to [-1, 1]
            if sample > 1.0:
                sample = 1.0
            elif sample < -1.0:
                sample = -1.0

            # write back into delay line
            buf[(pos + i) % delay] = sample
            flat[i] = sample

        self._write_pos = (pos + frames) % delay

        return flat.reshape(audio_data.shape)

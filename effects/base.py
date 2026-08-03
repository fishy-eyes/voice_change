"""Unified audio effect interface.

Realtime effect modules inherit from BaseEffect and implement the
process() method.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEffect(ABC):
    """Abstract base class for all audio effects.

    Subclasses must implement:
        process(audio_data, frames, time_info, status) -> np.ndarray

    The returned array must have the same shape as audio_data.
    """

    @property
    def name(self) -> str:
        """Human-readable effect name, used for logging and UI display."""
        return self.__class__.__name__

    @property
    def enabled(self) -> bool:
        """Whether this effect is active. Can be toggled at runtime."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def __init__(self) -> None:
        self._enabled: bool = True

    @abstractmethod
    def process(
        self,
        audio_data: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> np.ndarray:
        """Process one block of audio.

        Parameters
        ----------
        audio_data : np.ndarray
            Input audio block, shape (frames, channels), dtype float32.
        frames : int
            Number of frames in this block.
        time_info : object
            PortAudio time info (sample_time, current_time, etc.).
        status : object
            PortAudio status flags (input_overflow, output_underflow, etc.).

        Returns
        -------
        np.ndarray
            Processed audio block with same shape as input.
        """
        ...

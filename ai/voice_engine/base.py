"""Small backend-neutral interface for real-time voice conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EngineCapabilities:
    """Immutable backend facts exposed to application and GUI code."""

    backend_id: str
    display_name: str
    backend_version: str | None
    model_name: str | None
    loaded: bool
    input_sample_rate: int
    output_sample_rate: int | None
    channels: int = 1
    dtype: str = "float32"
    stateful: bool = False
    supports_pitch: bool = False
    recommended_chunk_size: int | None = None
    recommended_hop_size: int | None = None
    recommended_overlap_size: int | None = None
    parameter_names: tuple[str, ...] = ()
    latency_ms: float = 0.0


class VoiceConversionEngine(ABC):
    """Contract implemented by one loaded voice-conversion backend."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model resources prepared when the engine was constructed."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources."""

    @abstractmethod
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert one mono float32 block outside the audio callback."""

    @abstractmethod
    def get_latency(self) -> float:
        """Return the latest backend processing time in milliseconds."""

    @abstractmethod
    def get_info(self) -> Mapping[str, Any]:
        """Return a read-only snapshot suitable for status displays."""

"""Unified voice-conversion engine for external Beatrice v2 packages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ai.beatrice.model import BeatriceModelDescriptor
from ai.beatrice.runtime import BeatriceRuntimeLoader
from ai.beatrice.streaming_adapter import BeatriceStreamingAdapter
from ai.voice_engine.base import EngineCapabilities, VoiceConversionEngine


@dataclass(frozen=True)
class BeatriceConfig:
    target_speaker: int = 0
    formant_shift: float = 0.0
    pitch_shift_semitone: float = 0.0
    min_source_pitch: float = 30.0
    max_source_pitch: float = 1100.0
    vq_num_neighbors: int = 4

    def to_dict(self) -> dict[str, int | float]:
        return {
            "target_speaker": self.target_speaker,
            "formant_shift": self.formant_shift,
            "pitch_shift_semitone": self.pitch_shift_semitone,
            "min_source_pitch": self.min_source_pitch,
            "max_source_pitch": self.max_source_pitch,
            "vq_num_neighbors": self.vq_num_neighbors,
        }

    def updated(self, **changes: Any) -> "BeatriceConfig":
        unknown = set(changes) - set(self.to_dict())
        if unknown:
            raise TypeError(f"Unknown Beatrice parameters: {sorted(unknown)}")
        updated = replace(self, **changes)
        if updated.target_speaker < 0:
            raise ValueError("target_speaker must not be negative")
        if updated.min_source_pitch <= 0:
            raise ValueError("min_source_pitch must be positive")
        if updated.max_source_pitch <= updated.min_source_pitch:
            raise ValueError("max_source_pitch must exceed min_source_pitch")
        if updated.vq_num_neighbors <= 0:
            raise ValueError("vq_num_neighbors must be positive")
        return updated


class BeatriceVoiceEngine(VoiceConversionEngine):
    """Long-lived Beatrice stream. All inference must run in the Worker."""

    backend = "beatrice"
    requires_contiguous_input = True
    sample_rate = 48_000

    def __init__(
        self,
        descriptor: BeatriceModelDescriptor,
        *,
        runtime_root: str | Path | None = None,
        config: BeatriceConfig | None = None,
        callback_samples: int = 256,
        startup_buffer_samples: int = 512,
        loader: BeatriceRuntimeLoader | None = None,
        adapter_factory=BeatriceStreamingAdapter,
    ) -> None:
        if not descriptor.valid:
            raise ValueError(descriptor.validation_error or "Invalid Beatrice model")
        if callback_samples != 256:
            raise ValueError("Beatrice production callback size must remain 256")
        self.descriptor = descriptor
        self.config = config or BeatriceConfig()
        if self.config.target_speaker >= descriptor.speaker_count:
            raise ValueError(
                f"target_speaker must be below {descriptor.speaker_count}"
            )
        self._loader = loader or BeatriceRuntimeLoader(runtime_root)
        self._adapter_factory = adapter_factory
        self._callback_samples = int(callback_samples)
        self._startup_buffer_samples = int(startup_buffer_samples)
        self._adapter: BeatriceStreamingAdapter | None = None
        self._last_process_ms = 0.0
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._adapter is not None and self._adapter.ready

    @property
    def adapter(self) -> BeatriceStreamingAdapter | None:
        return self._adapter

    def _create_converter(self):
        return self._loader.create_converter(self.descriptor, self.config)

    def load_model(self) -> None:
        with self._lock:
            if self.is_loaded:
                return
            self._adapter = self._adapter_factory(
                self._create_converter,
                callback_samples=self._callback_samples,
                startup_buffer_samples=self._startup_buffer_samples,
            )

    def unload_model(self) -> None:
        with self._lock:
            adapter, self._adapter = self._adapter, None
            if adapter is not None:
                adapter.close()

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self.is_loaded or self._adapter is None:
                raise RuntimeError("Beatrice model is not loaded")
            started = perf_counter()
            output = self._adapter.process(audio)
            self._last_process_ms = (perf_counter() - started) * 1000.0
            return output

    def reset_stream(self) -> None:
        """Recreate state after a detected discontinuity, never in callback."""
        with self._lock:
            if not self.is_loaded or self._adapter is None:
                raise RuntimeError("Beatrice model is not loaded")
            self._adapter.reset()

    def update_config(self, **changes: Any) -> None:
        with self._lock:
            updated = self.config.updated(**changes)
            if updated.target_speaker >= self.descriptor.speaker_count:
                raise ValueError(
                    f"target_speaker must be below {self.descriptor.speaker_count}"
                )
            if self._adapter is not None:
                self._adapter.update_config(**updated.to_dict())
            self.config = updated

    def get_latency(self) -> float:
        return float(self._last_process_ms)

    def get_info(self) -> Mapping[str, Any]:
        adapter = self._adapter
        details = dict(adapter.runtime_details) if adapter is not None else {}
        stats = adapter.stats() if adapter is not None else {}
        capabilities = EngineCapabilities(
            backend_id=self.backend,
            display_name="Beatrice v2",
            backend_version=self.descriptor.runtime_requirement,
            model_name=self.descriptor.model_name,
            loaded=self.is_loaded,
            input_sample_rate=self.sample_rate,
            output_sample_rate=self.sample_rate,
            stateful=True,
            supports_pitch=True,
            recommended_chunk_size=self._callback_samples,
            recommended_hop_size=self._callback_samples,
            parameter_names=tuple(self.config.to_dict()),
            latency_ms=self.get_latency(),
        )
        return {
            "backend": self.backend,
            "capabilities": capabilities,
            "loaded": self.is_loaded,
            "latency_ms": self.get_latency(),
            "parameters": self.config.to_dict(),
            "model": self.descriptor,
            "runtime": details,
            "streaming": stats,
        }


__all__ = ["BeatriceConfig", "BeatriceVoiceEngine"]

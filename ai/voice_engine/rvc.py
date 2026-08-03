"""RVC adapter for the unified voice-conversion engine interface."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ai.rvc_engine import RVCEngine
from ai.voice_engine.base import EngineCapabilities, VoiceConversionEngine
from config.settings import RVC_CHUNK_SIZE, RVC_OVERLAP_SIZE
from config.rvc_profiles import RVCModelProfile


class RVCVoiceEngine(VoiceConversionEngine):
    """Delegate unified engine calls to the established :class:`RVCEngine`.

    Attribute fallback intentionally preserves the existing RVC configuration,
    profiling and cache APIs for current GUI and diagnostic callers.
    """

    backend = "rvc"

    def __init__(
        self,
        *args,
        engine: RVCEngine | None = None,
        **kwargs,
    ) -> None:
        self._rvc_engine = engine or RVCEngine(*args, **kwargs)
        self._last_process_ms = 0.0

    @classmethod
    def from_profile(
        cls,
        profile: RVCModelProfile | str | Path,
        **kwargs,
    ) -> "RVCVoiceEngine":
        return cls(engine=RVCEngine.from_profile(profile, **kwargs))

    @property
    def is_loaded(self) -> bool:
        return self._rvc_engine.is_loaded

    @property
    def core_engine(self) -> RVCEngine:
        """The unchanged RVC implementation wrapped by this adapter."""
        return self._rvc_engine

    def load_model(self) -> None:
        self._rvc_engine.load_model()

    def unload_model(self) -> None:
        self._rvc_engine.unload_model()

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        started = perf_counter()
        result = self._rvc_engine.infer(audio)
        self._last_process_ms = (perf_counter() - started) * 1000.0
        return result

    def infer(self, audio: np.ndarray) -> np.ndarray:
        """Compatibility alias used by existing RVC tests and utilities."""
        return self.process_audio(audio)

    def get_latency(self) -> float:
        return float(self._last_process_ms)

    def get_info(self) -> Mapping[str, Any]:
        core = self._rvc_engine
        capabilities = EngineCapabilities(
            backend_id=self.backend,
            display_name="RVC",
            backend_version=getattr(core, "_version", None),
            model_name=getattr(getattr(core, "_voice_dir", None), "name", None),
            loaded=self.is_loaded,
            input_sample_rate=int(getattr(core, "_sample_rate", 0)),
            output_sample_rate=int(getattr(core, "_tgt_sr", 0)) or None,
            supports_pitch=True,
            stateful=True,
            recommended_chunk_size=RVC_CHUNK_SIZE,
            recommended_hop_size=RVC_CHUNK_SIZE - RVC_OVERLAP_SIZE,
            recommended_overlap_size=RVC_OVERLAP_SIZE,
            parameter_names=tuple(core.config.to_dict()),
            latency_ms=self.get_latency(),
        )

        return {
            "backend": self.backend,
            "capabilities": capabilities,
            "loaded": self.is_loaded,
            "device": self._rvc_engine.device,
            "half_precision": self._rvc_engine.is_half,
            "latency_ms": self.get_latency(),
            "parameters": self._rvc_engine.config.to_dict(),
            "index_cache": self._rvc_engine.index_cache_info,
        }

    def __getattr__(self, name: str):
        return getattr(self._rvc_engine, name)

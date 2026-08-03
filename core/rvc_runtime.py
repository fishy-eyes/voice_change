"""Application-level RVC model selection without audio-chain redesign."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from config.rvc_realtime import (
    RVCRealtimePreset,
    get_rvc_realtime_preset,
)
from config.settings import (
    RVC_INPUT_QUEUE_SIZE,
    RVC_MODELS_DIR,
    RVC_REALTIME_PRESET,
    RVC_SOURCE_DIR,
    RVC_WARMUP_ENABLED,
    RVC_WARMUP_TIMEOUT,
    RVC_WORKER_STOP_TIMEOUT,
    SAMPLE_RATE,
)
from core.rvc_lifecycle import (
    RVCApplicationState,
    cleanup_rvc_application,
    initialize_rvc_application,
)
from core.rvc_model_manager import RVCModelManager

if TYPE_CHECKING:
    from audio.stream import AudioStream
    from effects.ai_voice import AIVoiceEffect
    from effects.manager import EffectManager


class RVCRuntime:
    """Own the selected model, Worker, engine and effect lifecycle."""

    def __init__(
        self,
        model_manager: RVCModelManager,
        *,
        source_dir: str | Path = RVC_SOURCE_DIR,
        backend_models_dir: str | Path = RVC_MODELS_DIR,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int | None = None,
        overlap_size: int | None = None,
        realtime_preset: str = RVC_REALTIME_PRESET,
        input_queue_size: int = RVC_INPUT_QUEUE_SIZE,
        warmup_enabled: bool = RVC_WARMUP_ENABLED,
        warmup_timeout: float = RVC_WARMUP_TIMEOUT,
        stop_timeout: float = RVC_WORKER_STOP_TIMEOUT,
    ) -> None:
        self.model_manager = model_manager
        self.source_dir = Path(source_dir)
        self.backend_models_dir = Path(backend_models_dir)
        self.sample_rate = int(sample_rate)
        preset = get_rvc_realtime_preset(realtime_preset)
        preset_chunk_size = preset.chunk_samples(self.sample_rate)
        preset_overlap_size = preset.overlap_samples(self.sample_rate)
        self.chunk_size = int(
            preset_chunk_size if chunk_size is None else chunk_size
        )
        self.overlap_size = int(
            preset_overlap_size if overlap_size is None else overlap_size
        )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap_size < 0 or self.overlap_size * 2 > self.chunk_size:
            raise ValueError(
                "overlap_size must be between 0 and half the chunk_size"
            )
        self.realtime_preset_key: str | None = (
            realtime_preset
            if self.chunk_size == preset_chunk_size
            and self.overlap_size == preset_overlap_size
            else None
        )
        self.input_queue_size = int(input_queue_size)
        self.warmup_enabled = bool(warmup_enabled)
        self.warmup_timeout = float(warmup_timeout)
        self.stop_timeout = float(stop_timeout)
        self.state = RVCApplicationState(enabled=False)
        self.selected_model: str | None = None
        self.enabled = False
        self._effect_manager: EffectManager | None = None
        self._lock = threading.RLock()

    def bind_effect_manager(self, effect_manager: EffectManager) -> None:
        self._effect_manager = effect_manager
        if self.state.ready and self.state.effect is not None:
            self._attach_ai_first(self.state.effect)

    def load_model(
        self,
        name: str,
        *,
        audio_stream: AudioStream | None = None,
    ) -> RVCApplicationState:
        """Replace the selected model while the stream is safely stopped."""
        with self._lock:
            try:
                descriptor = self.model_manager.get_model(name)
                # A valid retry must not inherit an earlier selection error.
                self.state.error = None
            except Exception as exc:
                if not self.state.ready:
                    self.state = RVCApplicationState(
                        enabled=self.enabled,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    self.state.error = f"{type(exc).__name__}: {exc}"
                logger.error("RVC model selection failed: {}", self.state.error)
                return self.state
            was_running = bool(audio_stream is not None and audio_stream.is_running)
            if was_running:
                audio_stream.stop()
            try:
                if self.state.ready and self.selected_model == descriptor.name:
                    if self.state.effect is not None:
                        self.state.effect.enabled = self.enabled
                    return self.state

                self._detach_ai_effect()
                if self.state.engine is not None or self.state.effect is not None:
                    if not cleanup_rvc_application(self.state, timeout=self.stop_timeout):
                        self.state.error = "Existing RVC resources could not be stopped"
                        logger.error(self.state.error)
                        return self.state

                self.state = initialize_rvc_application(
                    enabled=True,
                    profile=descriptor.profile,
                    source_dir=self.source_dir,
                    models_dir=self.backend_models_dir,
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    input_queue_size=self.input_queue_size,
                    warmup_enabled=self.warmup_enabled,
                    overlap_size=self.overlap_size,
                    warmup_timeout=self.warmup_timeout,
                    stop_timeout=self.stop_timeout,
                )
                if self.state.ready and self.state.effect is not None:
                    self.selected_model = descriptor.name
                    self.state.effect.enabled = self.enabled
                    self._attach_ai_first(self.state.effect)
                else:
                    self.selected_model = None
                return self.state
            finally:
                if was_running:
                    audio_stream.start()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/bypass the loaded AI effect without model reload."""
        with self._lock:
            self.enabled = bool(enabled)
            self.state.enabled = self.enabled
            if self.state.effect is not None:
                self.state.effect.enabled = self.enabled

    @property
    def realtime_preset(self) -> RVCRealtimePreset | None:
        """The selected named preset, or ``None`` for custom sample sizes."""
        if self.realtime_preset_key is None:
            return None
        return get_rvc_realtime_preset(self.realtime_preset_key)

    def set_realtime_preset(self, key: str) -> RVCRealtimePreset:
        """Apply chunk/overlap settings without replacing model or Worker."""
        preset = get_rvc_realtime_preset(key)
        chunk_size = preset.chunk_samples(self.sample_rate)
        overlap_size = preset.overlap_samples(self.sample_rate)
        with self._lock:
            effect = self.state.effect
            if effect is not None:
                effect.update_realtime_config(
                    chunk_size=chunk_size,
                    overlap_size=overlap_size,
                )
            self.chunk_size = chunk_size
            self.overlap_size = overlap_size
            self.realtime_preset_key = key
            logger.info(
                "RVC realtime preset: {} (chunk={}ms, overlap={}ms)",
                preset.name,
                preset.chunk_ms,
                preset.overlap_ms,
            )
            return preset

    def shutdown(self) -> bool:
        """Detach the effect, stop its Worker, then unload model and caches."""
        with self._lock:
            self._detach_ai_effect()
            cleaned = cleanup_rvc_application(self.state, timeout=self.stop_timeout)
            if cleaned:
                self.selected_model = None
            return cleaned

    def _detach_ai_effect(self) -> None:
        manager = self._effect_manager
        if manager is not None and manager.get_by_name("AIVoiceEffect") is not None:
            manager.remove_by_name("AIVoiceEffect")

    def _attach_ai_first(self, effect: AIVoiceEffect) -> None:
        manager = self._effect_manager
        if manager is None:
            return
        self._detach_ai_effect()
        existing = manager.effects
        for current in existing:
            manager.remove(current)
        manager.add(effect)
        for current in existing:
            manager.add(current)

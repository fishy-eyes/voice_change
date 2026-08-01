"""Application-level RVC model selection without audio-chain redesign."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from config.settings import (
    RVC_CHUNK_SIZE,
    RVC_INPUT_QUEUE_SIZE,
    RVC_MODELS_DIR,
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
        chunk_size: int = RVC_CHUNK_SIZE,
        input_queue_size: int = RVC_INPUT_QUEUE_SIZE,
        warmup_enabled: bool = RVC_WARMUP_ENABLED,
        warmup_timeout: float = RVC_WARMUP_TIMEOUT,
        stop_timeout: float = RVC_WORKER_STOP_TIMEOUT,
    ) -> None:
        self.model_manager = model_manager
        self.source_dir = Path(source_dir)
        self.backend_models_dir = Path(backend_models_dir)
        self.sample_rate = int(sample_rate)
        self.chunk_size = int(chunk_size)
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

"""Application lifecycle for the optional Beatrice backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Optional

from loguru import logger

from ai.beatrice.model import BeatriceModelManager
from ai.voice_engine.beatrice import BeatriceVoiceEngine
from ai.beatrice.runtime import BeatriceRuntimeLoader, RuntimeUnavailableError
from customization.beatrice import (
    BeatriceParameterSet,
    BeatriceTuningCapabilities,
    speaker_preset_key,
)
from effects.ai_voice import AIVoiceEffect


@dataclass
class BeatriceApplicationState:
    enabled: bool
    engine: Optional[BeatriceVoiceEngine] = None
    effect: Optional[AIVoiceEffect] = None
    ready: bool = False
    error: str | None = None


class BeatriceRuntime:
    """Own one Beatrice engine/effect and keep failures isolated from RVC."""

    supports_model_folder_import = True

    def __init__(
        self,
        model_manager: BeatriceModelManager,
        *,
        runtime_root: str | Path | None = None,
        sample_rate: int = 48_000,
        local_settings=None,
        callback_samples: int = 256,
        input_queue_size: int = 8,
        startup_buffer_samples: int = 512,
        stop_timeout: float = 5.0,
        engine_factory=BeatriceVoiceEngine,
        effect_factory=AIVoiceEffect,
    ) -> None:
        if sample_rate != 48_000:
            raise ValueError("Beatrice application boundary must remain 48 kHz")
        self.model_manager = model_manager
        self.runtime_root = Path(runtime_root).expanduser().resolve() if runtime_root else None
        self.sample_rate = int(sample_rate)
        self.local_settings = local_settings
        self.callback_samples = int(callback_samples)
        self.input_queue_size = int(input_queue_size)
        self.startup_buffer_samples = int(startup_buffer_samples)
        self.stop_timeout = float(stop_timeout)
        self._engine_factory = engine_factory
        self._effect_factory = effect_factory
        self.state = BeatriceApplicationState(enabled=False)
        self.selected_model: str | None = None
        self.enabled = False
        self._effect_manager = None
        self._lock = threading.RLock()

    def bind_effect_manager(self, effect_manager) -> None:
        self._effect_manager = effect_manager
        if self.state.ready and self.state.effect is not None:
            self._attach_ai_first(self.state.effect)


    @property
    def runtime_path(self) -> Path | None:
        return self.runtime_root

    @property
    def registered_model_paths(self) -> tuple[Path, ...]:
        return tuple(getattr(self.model_manager, "registered_paths", ()))

    @property
    def preferred_model(self) -> str | None:
        if self.local_settings is None:
            return None
        value = self.local_settings.beatrice.get("last_model", "")
        return str(value).strip() or None

    def validate_runtime(self, path: str | Path | None = None) -> dict:
        selected = self.runtime_root if path is None else Path(path).expanduser().resolve()
        if selected is None:
            raise RuntimeUnavailableError("Beatrice runtime is not configured")
        return BeatriceRuntimeLoader(selected).validate()

    def runtime_path_status(self) -> dict:
        """Lightweight startup/settings check that never imports native code."""
        loader = BeatriceRuntimeLoader(self.runtime_root)
        return {
            "configured": self.runtime_root is not None,
            "available": loader.available,
            "runtime_root": str(loader.runtime_root) if loader.runtime_root else None,
            "version": "2.0.0-rc.0",
        }

    def configure_runtime(self, path: str | Path) -> dict:
        selected = Path(path).expanduser().resolve()
        loader = BeatriceRuntimeLoader(selected)
        if not loader.available:
            raise RuntimeUnavailableError(
                f"Beatrice package was not found in runtime folder: {selected}"
            )
        details = {
            "version": "2.0.0-rc.0",
            "runtime_root": str(loader.runtime_root),
            "available": True,
        }
        with self._lock:
            if self.state.ready and not self.shutdown():
                raise RuntimeError("Existing Beatrice Worker could not be stopped")
            self.runtime_root = selected
            if self.local_settings is not None:
                self.local_settings.update_beatrice(runtime_dir=str(selected))
        return details

    def validate_configuration(self, model: str | None = None) -> None:
        if self.runtime_root is None:
            raise RuntimeUnavailableError(
                "Please select a Beatrice Runtime folder in Model Settings first."
            )
        loader = BeatriceRuntimeLoader(self.runtime_root)
        if not loader.available:
            raise RuntimeUnavailableError(
                f"Beatrice Runtime path does not exist or is invalid: {self.runtime_root}"
            )
        if model is None or not str(model).strip():
            raise LookupError("Please add a Beatrice model folder first.")
        self.model_manager.get_model(model)

    def add_model_path(self, path: str | Path):
        register = getattr(self.model_manager, "register_package", None)
        if not callable(register):
            raise RuntimeError("Beatrice model registry is unavailable")
        return register(path)

    def remove_model_path(self, path: str | Path) -> bool:
        remove = getattr(self.model_manager, "remove_registered_package", None)
        if not callable(remove):
            return False
        return bool(remove(path))

    def update_parameters(self, **changes):
        engine = self.state.engine
        update = getattr(engine, "update_config", None)
        if not callable(update):
            raise RuntimeError("current backend has no loaded parameter interface")
        descriptor = self.model_manager.get_model(self.selected_model)
        current = BeatriceParameterSet.from_mapping(engine.config.to_dict())
        target = int(changes.get("target_speaker", current.target_speaker))
        if target != current.target_speaker:
            restored = self._load_speaker_preset(descriptor, target)
            if restored is not None:
                changes = restored.to_engine_changes()
                changes["target_speaker"] = target
        update(**changes)
        values = dict(engine.config.to_dict())
        if self.local_settings is not None:
            index = int(values["target_speaker"])
            speakers = tuple(getattr(descriptor, "speaker_names", ()))
            speaker = speakers[index] if 0 <= index < len(speakers) else str(index)
            presets = dict(self.local_settings.beatrice.get("speaker_presets", {}))
            presets[speaker_preset_key(descriptor, index)] = values
            self.local_settings.update_beatrice(
                last_speaker=speaker,
                speaker_presets=presets,
            )
        return values

    def get_tuning_capabilities(self) -> BeatriceTuningCapabilities:
        engine = self.state.engine
        info = engine.get_info() if engine is not None else {}
        runtime = info.get("runtime", {}) if isinstance(info, dict) else {}
        return BeatriceTuningCapabilities.from_runtime(runtime)

    def _load_speaker_preset(self, descriptor, index: int):
        if self.local_settings is None:
            return None
        presets = self.local_settings.beatrice.get("speaker_presets", {})
        if not isinstance(presets, dict):
            return None
        values = presets.get(speaker_preset_key(descriptor, index))
        if not isinstance(values, dict):
            return None
        try:
            return BeatriceParameterSet.from_mapping(values)
        except (TypeError, ValueError):
            return None

    def load_model(self, name: str, *, audio_stream=None) -> BeatriceApplicationState:
        with self._lock:
            was_running = bool(audio_stream is not None and audio_stream.is_running)
            if was_running:
                audio_stream.stop()
            try:
                descriptor = self.model_manager.get_model(name)
                if self.state.ready and self.selected_model == descriptor.name:
                    if self.state.effect is not None:
                        self.state.effect.enabled = self.enabled
                    return self.state
                self._detach_ai_effect()
                if not self._cleanup_locked():
                    self.state.error = "Existing Beatrice Worker could not be stopped"
                    logger.error(self.state.error)
                    return self.state
                state = BeatriceApplicationState(enabled=self.enabled)
                self.state = state
                state.engine = self._engine_factory(
                    descriptor,
                    runtime_root=self.runtime_root,
                    callback_samples=self.callback_samples,
                    startup_buffer_samples=self.startup_buffer_samples,
                )
                state.engine.load_model()
                if self.local_settings is not None:
                    last_speaker = str(
                        self.local_settings.beatrice.get("last_speaker", "")
                    ).strip()
                    if last_speaker in descriptor.speaker_names:
                        target = descriptor.speaker_names.index(last_speaker)
                        preset = self._load_speaker_preset(descriptor, target)
                        changes = (
                            preset.to_engine_changes()
                            if preset is not None
                            else {"target_speaker": target}
                        )
                        state.engine.update_config(**changes)
                state.effect = self._effect_factory(
                    state.engine,
                    chunk_size=self.callback_samples,
                    overlap_size=0,
                    max_queue_size=self.input_queue_size,
                )
                if not state.effect.start():
                    raise RuntimeError("Beatrice VoiceConversionWorker failed to start")
                state.effect.enabled = self.enabled
                state.ready = True
                self.state = state
                self.selected_model = descriptor.name
                if self.local_settings is not None:
                    self.local_settings.update_beatrice(last_model=descriptor.name)
                self._attach_ai_first(state.effect)
                return state
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                cleaned = self._cleanup_locked()
                self.selected_model = None
                if not cleaned:
                    self.state.error = error
                    logger.error(
                        "Beatrice cleanup after failure was incomplete: {}", error
                    )
                    return self.state
                self.state = BeatriceApplicationState(
                    enabled=self.enabled,
                    error=error,
                )
                logger.error(
                    "Beatrice initialization failed; RVC/base audio remain available: {}",
                    self.state.error,
                )
                return self.state
            finally:
                if was_running:
                    audio_stream.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            self.state.enabled = self.enabled
            if self.state.effect is not None:
                self.state.effect.enabled = self.enabled

    def shutdown(self) -> bool:
        with self._lock:
            self._detach_ai_effect()
            cleaned = self._cleanup_locked()
            if cleaned:
                self.selected_model = None
            return cleaned

    def _cleanup_locked(self) -> bool:
        state = self.state
        state.ready = False
        effect = state.effect
        if effect is not None:
            try:
                if not effect.stop(timeout=self.stop_timeout) or effect.worker.thread_alive:
                    return False
            except Exception as exc:
                logger.error("Beatrice Worker stop failed: {}", exc)
                return False
        engine = state.engine
        if engine is not None:
            try:
                engine.unload_model()
            except Exception as exc:
                logger.error("Beatrice unload failed: {}", exc)
                return False
        self.state = BeatriceApplicationState(enabled=self.enabled)
        return True

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


__all__ = ["BeatriceApplicationState", "BeatriceRuntime"]

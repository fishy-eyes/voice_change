"""Application-level selection and status for voice-conversion backends."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from loguru import logger


class VoiceConversionState(str, Enum):
    """Backend-neutral model lifecycle states."""

    IDLE = "IDLE"
    LOADING = "LOADING"
    LOADED = "LOADED"
    SWITCHING = "SWITCHING"
    FAILED = "FAILED"
    UNLOADING = "UNLOADING"


@dataclass(frozen=True)
class VoiceConversionStatus:
    backend: str | None
    model: str | None
    state: str
    enabled: bool
    latency_ms: float
    error: str | None = None


class VoiceConversionManager:
    """Serialize backend model switches outside the audio callback."""

    def __init__(
        self,
        backends: Mapping[str, object] | None = None,
        *,
        default_backend: str | None = None,
    ) -> None:
        self._backends: dict[str, object] = {}
        self._current_backend: str | None = None
        self._requested_enabled = False
        self._loading = False
        self._last_error: str | None = None
        self._lifecycle_state = VoiceConversionState.IDLE
        self._active_switch: Future | None = None
        self._lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="voice-conversion-loader",
        )
        self._closed = False
        for name, runtime in (backends or {}).items():
            self.register_backend(name, runtime)
        if default_backend is not None:
            self.select_backend(default_backend)
        elif self._backends:
            self._current_backend = next(iter(self._backends))

    @staticmethod
    def _normalize_backend(name: str) -> str:
        selected = str(name).strip().lower()
        if not selected:
            raise ValueError("backend name must not be empty")
        return selected

    def register_backend(self, name: str, runtime: object) -> None:
        key = self._normalize_backend(name)
        if runtime is None:
            raise TypeError("backend runtime must not be None")
        with self._lock:
            if self._closed:
                raise RuntimeError("voice conversion manager is closed")
            if key in self._backends:
                raise ValueError(f"backend already registered: {key}")
            self._backends[key] = runtime
            if self._current_backend is None:
                self._current_backend = key

    @property
    def available_backends(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._backends)

    @property
    def requested_enabled(self) -> bool:
        with self._lock:
            return self._requested_enabled

    @property
    def current_backend(self) -> str | None:
        with self._lock:
            return self._current_backend

    @property
    def current_runtime(self):
        with self._lock:
            if self._current_backend is None:
                return None
            return self._backends.get(self._current_backend)

    @property
    def current_engine(self):
        runtime = self.current_runtime
        state = getattr(runtime, "state", None)
        return getattr(state, "engine", None)

    def select_backend(self, name: str) -> None:
        key = self._normalize_backend(name)
        with self._lock:
            if key not in self._backends:
                raise LookupError(f"voice conversion backend not found: {name}")
            if self._loading:
                raise RuntimeError("voice conversion model switch is in progress")
            self._current_backend = key
            self._last_error = None

    def discover_models(self, backend: str | None = None) -> list[object]:
        runtime = self._runtime_for(backend)
        model_manager = getattr(runtime, "model_manager", None)
        discover = getattr(model_manager, "discover_models", None)
        return list(discover()) if callable(discover) else []

    def set_enabled(self, enabled: bool) -> None:
        requested = bool(enabled)
        with self._lock:
            self._requested_enabled = requested
            runtime = self.current_runtime
        if runtime is not None:
            runtime.set_enabled(requested)

    def load_model(self, backend: str, model: str, *, audio_stream=None):
        """Switch synchronously; GUI callers normally use the async wrapper."""
        key = self._normalize_backend(backend)
        with self._switch_lock:
            with self._lock:
                # A switch accepted before shutdown may finish before cleanup.
                if self._closed and not self._loading:
                    raise RuntimeError("voice conversion manager is closed")
                if key not in self._backends:
                    raise LookupError(f"voice conversion backend not found: {backend}")
                old_key = self._current_backend
                old_runtime = self._backends.get(old_key) if old_key else None
                runtime = self._backends[key]
                old_ready = bool(
                    getattr(getattr(old_runtime, "state", None), "ready", False)
                )
                self._loading = True
                self._lifecycle_state = (
                    VoiceConversionState.SWITCHING
                    if old_ready
                    else VoiceConversionState.LOADING
                )
                self._last_error = None
                requested_enabled = self._requested_enabled
            try:
                if old_runtime is not None and old_runtime is not runtime:
                    old_runtime.set_enabled(False)
                    if not old_runtime.shutdown():
                        raise RuntimeError(
                            f"backend {old_key} could not release its resources"
                        )
                with self._lock:
                    self._current_backend = key
                runtime.set_enabled(requested_enabled)
                state = (
                    runtime.load_model(model)
                    if audio_stream is None
                    else runtime.load_model(model, audio_stream=audio_stream)
                )
                state_error = getattr(state, "error", None)
                if not bool(getattr(state, "ready", False)) or state_error:
                    raise RuntimeError(str(state_error or "model load failed"))
                runtime.set_enabled(requested_enabled)
                with self._lock:
                    self._lifecycle_state = VoiceConversionState.LOADED
                return state
            except Exception as exc:
                try:
                    runtime.set_enabled(False)
                except Exception as bypass_exc:
                    logger.error(
                        "Voice conversion bypass after failure also failed: {}",
                        bypass_exc,
                    )
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._lifecycle_state = VoiceConversionState.FAILED
                logger.error("Voice conversion model switch failed: {}", exc)
                state = getattr(runtime, "state", None)
                if state is not None:
                    state.error = self._last_error
                return state
            finally:
                with self._lock:
                    self._loading = False

    def switch_model_async(
        self,
        backend: str,
        model: str,
        *,
        audio_stream=None,
    ) -> Future:
        """Submit at most one model switch; repeated clicks never queue work."""
        with self._lock:
            if self._closed:
                raise RuntimeError("voice conversion manager is closed")
            if self._loading or (
                self._active_switch is not None and not self._active_switch.done()
            ):
                raise RuntimeError("voice conversion model switch is in progress")
            self._loading = True
            self._lifecycle_state = (
                VoiceConversionState.SWITCHING
                if bool(
                    getattr(getattr(self.current_runtime, "state", None), "ready", False)
                )
                else VoiceConversionState.LOADING
            )
            try:
                future = self._executor.submit(
                    self.load_model,
                    backend,
                    model,
                    audio_stream=audio_stream,
                )
            except Exception:
                self._loading = False
                self._lifecycle_state = VoiceConversionState.FAILED
                raise
            self._active_switch = future
            return future

    def get_status(self) -> VoiceConversionStatus:
        with self._lock:
            backend = self._current_backend
            runtime = self.current_runtime
            error = self._last_error
            requested_enabled = self._requested_enabled
            lifecycle_state = self._lifecycle_state
        state = getattr(runtime, "state", None)
        ready = bool(getattr(state, "ready", False))
        runtime_error = getattr(state, "error", None)
        if error or runtime_error:
            lifecycle_state = VoiceConversionState.FAILED
        return VoiceConversionStatus(
            backend=backend,
            model=getattr(runtime, "selected_model", None),
            state=lifecycle_state.value,
            enabled=bool(ready and requested_enabled and not (error or runtime_error)),
            latency_ms=self._runtime_latency(runtime),
            error=error or runtime_error,
        )

    def get_info(self) -> Mapping[str, Any]:
        engine = self.current_engine
        getter = getattr(engine, "get_info", None)
        return dict(getter()) if callable(getter) else {}

    def get_current_parameters(self) -> Mapping[str, Any]:
        engine = self.current_engine
        config = getattr(engine, "config", None)
        getter = getattr(config, "to_dict", None)
        return dict(getter()) if callable(getter) else {}

    def update_current_parameters(self, **changes: Any) -> Mapping[str, Any]:
        engine = self.current_engine
        update = getattr(engine, "update_config", None)
        if not callable(update):
            raise RuntimeError("current backend has no loaded parameter interface")
        update(**changes)
        return self.get_current_parameters()

    def get_realtime_preset(self) -> str | None:
        return getattr(self.current_runtime, "realtime_preset_key", None)

    def set_realtime_preset(self, key: str):
        setter = getattr(self.current_runtime, "set_realtime_preset", None)
        if not callable(setter):
            raise RuntimeError("current backend has no realtime preset interface")
        return setter(key)

    def get_current_model_descriptor(self):
        runtime = self.current_runtime
        name = getattr(runtime, "selected_model", None)
        model_manager = getattr(runtime, "model_manager", None)
        getter = getattr(model_manager, "get_model", None)
        return getter(name) if name and callable(getter) else None

    def shutdown(self) -> bool:
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            self._lifecycle_state = VoiceConversionState.UNLOADING
        self._executor.shutdown(wait=True, cancel_futures=True)
        cleaned = True
        for runtime in tuple(self._backends.values()):
            try:
                cleaned = bool(runtime.shutdown()) and cleaned
            except Exception as exc:
                cleaned = False
                logger.error("Voice conversion backend shutdown failed: {}", exc)
        with self._lock:
            self._lifecycle_state = (
                VoiceConversionState.IDLE if cleaned else VoiceConversionState.FAILED
            )
        return cleaned

    def _runtime_for(self, backend: str | None):
        key = self.current_backend if backend is None else self._normalize_backend(backend)
        with self._lock:
            if key is None or key not in self._backends:
                raise LookupError(f"voice conversion backend not found: {backend}")
            return self._backends[key]

    @staticmethod
    def _runtime_latency(runtime) -> float:
        state = getattr(runtime, "state", None)
        effect = getattr(state, "effect", None)
        worker = getattr(effect, "worker", None)
        value = getattr(worker, "last_infer_ms", None)
        if value is not None:
            return float(value)
        engine = getattr(state, "engine", None)
        getter = getattr(engine, "get_latency", None)
        return float(getter()) if callable(getter) else 0.0

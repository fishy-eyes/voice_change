"""Application-owned RVC initialization and cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ai.rvc_engine import RVCEngine
from config.rvc_profiles import RVCModelProfile, load_rvc_profile
from effects.ai_voice import AIVoiceEffect
from config.settings import (
    ENABLE_AI_VOICE,
    RVC_CHUNK_SIZE,
    RVC_F0_METHOD,
    RVC_INDEX_RATE,
    RVC_INPUT_QUEUE_SIZE,
    RVC_MODELS_DIR,
    RVC_OVERLAP_SIZE,
    RVC_PITCH_SHIFT,
    RVC_PROTECT,
    RVC_RMS_MIX_RATE,
    RVC_SOURCE_DIR,
    RVC_VOICE_DIR,
    RVC_WARMUP_ENABLED,
    RVC_WARMUP_TIMEOUT,
    RVC_WORKER_STOP_TIMEOUT,
    SAMPLE_RATE,
)


@dataclass
class RVCApplicationState:
    """RVC resources owned by the application layer."""

    enabled: bool
    engine: Optional[RVCEngine] = None
    effect: Optional[AIVoiceEffect] = None
    ready: bool = False
    error: Optional[str] = None
    warmup_seconds: float = 0.0


def _validate_paths(
    voice_dir: str | Path,
    source_dir: str | Path,
    models_dir: str | Path,
) -> None:
    voice_path = Path(voice_dir)
    source_path = Path(source_dir)
    models_path = Path(models_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"RVC source directory not found: {source_path}")
    if not models_path.is_dir():
        raise FileNotFoundError(f"RVC models directory not found: {models_path}")
    if not voice_path.is_dir():
        raise FileNotFoundError(f"RVC voice directory not found: {voice_path}")
    if not any(voice_path.glob("*.pth")):
        raise FileNotFoundError(f"No RVC .pth model in: {voice_path}")
    if not any(voice_path.glob("*.index")):
        logger.warning("No RVC .index file in {}; continuing without index", voice_path)


def initialize_rvc_application(
    *,
    enabled: bool = ENABLE_AI_VOICE,
    voice_dir: str | Path = RVC_VOICE_DIR,
    source_dir: str | Path = RVC_SOURCE_DIR,
    models_dir: str | Path = RVC_MODELS_DIR,
    sample_rate: int = SAMPLE_RATE,
    chunk_size: int = RVC_CHUNK_SIZE,
    overlap_size: int = RVC_OVERLAP_SIZE,
    input_queue_size: int = RVC_INPUT_QUEUE_SIZE,
    warmup_enabled: bool = RVC_WARMUP_ENABLED,
    warmup_timeout: float = RVC_WARMUP_TIMEOUT,
    stop_timeout: float = RVC_WORKER_STOP_TIMEOUT,
    pitch_shift: int = RVC_PITCH_SHIFT,
    f0_method: str = RVC_F0_METHOD,
    index_rate: float = RVC_INDEX_RATE,
    rms_mix_rate: float = RVC_RMS_MIX_RATE,
    protect: float = RVC_PROTECT,
    profile: RVCModelProfile | str | Path | None = None,
    engine_factory: Callable[..., RVCEngine] = RVCEngine,
    effect_factory: Callable[..., AIVoiceEffect] = AIVoiceEffect,
    validate_paths: bool = True,
) -> RVCApplicationState:
    """Load, start, and optionally warm up RVC before audio starts.

    Failure is returned as state instead of aborting the base voice changer.
    Resources that cannot be stopped immediately remain attached to the state
    so the application's final cleanup can retry without unloading a live model.
    """
    state = RVCApplicationState(enabled=bool(enabled))
    if not enabled:
        logger.info("AI voice disabled; starting base effect chain only")
        return state

    try:
        selected_profile = (
            profile
            if isinstance(profile, RVCModelProfile) or profile is None
            else load_rvc_profile(profile)
        )
        if selected_profile is not None:
            voice_dir = selected_profile.resolve_voice_dir(models_dir)
            logger.info("Using RVC model profile: {}", selected_profile.name)

        if validate_paths:
            _validate_paths(voice_dir, source_dir, models_dir)

        engine_kwargs = {
            "voice_dir": voice_dir,
            "source_dir": source_dir,
            "models_dir": models_dir,
            "pitch_shift": pitch_shift,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "rms_mix_rate": rms_mix_rate,
            "protect": protect,
            "sample_rate": sample_rate,
        }
        if selected_profile is not None:
            engine_kwargs.update(
                config=selected_profile.inference,
                voice_pth_path=selected_profile.model_file,
                index_path=selected_profile.index_file,
            )
        if selected_profile is not None and engine_factory is RVCEngine:
            state.engine = RVCEngine.from_profile(
                selected_profile,
                source_dir=source_dir,
                models_dir=models_dir,
                sample_rate=sample_rate,
            )
        else:
            state.engine = engine_factory(**engine_kwargs)
        state.engine.load_model()

        state.effect = effect_factory(
            state.engine,
            chunk_size=chunk_size,
            max_queue_size=input_queue_size,
            overlap_size=overlap_size,
        )
        if not state.effect.start():
            raise RuntimeError("AIVoiceEffect worker failed to start")

        if warmup_enabled:
            if not state.effect.warmup(timeout=warmup_timeout):
                raise TimeoutError(
                    f"RVC warmup failed or timed out after {warmup_timeout:.3f}s"
                )
            state.warmup_seconds = state.effect.last_warmup_ms / 1000.0
        else:
            logger.info("RVC warmup disabled by configuration")

        state.ready = True
        logger.info(
            "RVC application resources ready (warmup={:.3f}s)",
            state.warmup_seconds,
        )
        return state
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
        logger.error("RVC initialization failed; using base effects: {}", state.error)
        cleanup_rvc_application(state, timeout=stop_timeout)
        return state


def cleanup_rvc_application(
    state: RVCApplicationState,
    *,
    timeout: float = RVC_WORKER_STOP_TIMEOUT,
) -> bool:
    """Stop the Worker before unloading the application-owned engine.

    Returns False when a Worker remains alive or unloading fails. In that case
    the engine is intentionally retained so inference never races model release.
    Repeated calls are safe and can finish a previous timed-out cleanup.
    """
    state.ready = False
    effect = state.effect
    if effect is not None:
        try:
            stopped = effect.stop(timeout=timeout)
        except Exception as exc:
            logger.error("AIVoiceEffect stop failed: {}", exc)
            stopped = False
        if effect.worker.thread_alive:
            logger.warning(
                "RVC worker is still alive after {:.3f}s; engine will not unload",
                timeout,
            )
            return False
        if not stopped:
            logger.warning("RVC worker reported a stop timeout; engine retained")
            return False

    engine = state.engine
    if engine is not None and engine.is_loaded:
        try:
            engine.unload_model()
        except Exception as exc:
            logger.error("RVCEngine unload failed: {}", exc)
            return False

    logger.info("RVC application resources cleaned up")
    return True

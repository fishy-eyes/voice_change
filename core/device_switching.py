"""Application-level audio device switching without touching RVC resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from loguru import logger

from audio.player import AudioPlayer
from audio.recorder import AudioRecorder
from audio.stream import AudioStream

if TYPE_CHECKING:
    from core.context import AppContext


@dataclass(frozen=True)
class DeviceSwitchResult:
    """Outcome of one stop/recreate/start device switch transaction."""

    success: bool
    error: Optional[str] = None
    restored_previous_stream: bool = False


def switch_audio_devices(
    context: AppContext,
    input_device: Optional[int],
    output_device: Optional[int],
    *,
    recorder_factory: Callable[..., AudioRecorder] = AudioRecorder,
    player_factory: Callable[..., AudioPlayer] = AudioPlayer,
    stream_factory: Callable[..., AudioStream] = AudioStream,
) -> DeviceSwitchResult:
    """Replace the context's stream while preserving its EffectManager.

    The old stream is restarted on failure when it had been running before the
    switch. Context references are committed only after the new stream is ready.
    """
    old_stream = getattr(context, "audio_stream", None)
    effect_manager = getattr(context, "effect_manager", None)
    if old_stream is None:
        message = "cannot switch devices without an existing AudioStream"
        logger.error(message)
        return DeviceSwitchResult(False, message)
    if effect_manager is None:
        message = "cannot switch devices without an EffectManager"
        logger.error(message)
        return DeviceSwitchResult(False, message)

    was_running = bool(old_stream.is_running)
    old_stopped = False
    new_stream = None
    try:
        # stop() is idempotent and also closes an inactive underlying handle.
        old_stream.stop()
        old_stopped = True

        recorder = recorder_factory(device=input_device)
        player = player_factory(device=output_device)
        new_stream = stream_factory(
            recorder,
            player,
            effect_manager=effect_manager,
        )
        if was_running:
            new_stream.start()

        context.audio_stream = new_stream
        context.input_device = input_device
        context.output_device = output_device
        logger.info(
            "audio devices switched: input={} output={} restarted={}",
            input_device,
            output_device,
            was_running,
        )
        return DeviceSwitchResult(True)
    except Exception as exc:
        message = f"audio device switch failed: {type(exc).__name__}: {exc}"
        logger.exception(message)
        if new_stream is not None:
            try:
                new_stream.stop()
            except Exception as stop_exc:
                logger.error("new AudioStream cleanup failed: {}", stop_exc)

        restored = False
        if was_running and old_stopped:
            try:
                old_stream.start()
                restored = bool(old_stream.is_running)
                if restored:
                    logger.info("previous AudioStream restored after switch failure")
                else:
                    logger.error("previous AudioStream restart did not become active")
            except Exception as restore_exc:
                logger.error("previous AudioStream restore failed: {}", restore_exc)
        return DeviceSwitchResult(False, message, restored)


def stop_current_audio_stream(
    context: Optional[AppContext],
    *,
    fallback: Optional[AudioStream] = None,
) -> None:
    """Stop the latest context stream, falling back to the startup stream."""
    stream = getattr(context, "audio_stream", None) if context is not None else None
    stream = stream or fallback
    if stream is None:
        return
    try:
        stream.stop()
    except Exception as exc:
        logger.error("AudioStream stop failed: {}", exc)
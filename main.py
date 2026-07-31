"""Voice Changer application entry point."""

from __future__ import annotations

import signal
import threading
from typing import Callable, Optional

from loguru import logger

from audio.device_manager import DeviceManager
from audio.player import AudioPlayer
from audio.recorder import AudioRecorder
from audio.stream import AudioStream
from config.settings import (
    AUTO_SELECT_DEVICES,
    ECHO_DECAY,
    ECHO_DELAY,
    ENABLE_AI_VOICE,
    ENABLE_ECHO,
    ENABLE_GAIN,
    ENABLE_ROBOT,
    GAIN_VALUE,
    INPUT_DEVICE,
    ROBOT_FREQUENCY,
    RVC_WORKER_STOP_TIMEOUT,
    SHOW_DEVICE_LIST,
)
from core.context import AppContext
from core.rvc_lifecycle import (
    RVCApplicationState,
    cleanup_rvc_application,
    initialize_rvc_application,
)
from effects.ai_voice import AIVoiceEffect
from effects.echo import EchoEffect
from effects.gain import GainEffect
from effects.manager import EffectManager
from effects.robot import RobotEffect
from gui.app import create_app
from utils.logger import setup_logger

_stop_event = threading.Event()
_quit_callback: Optional[Callable[[], None]] = None


def _on_signal(sig, frame) -> None:
    """Route Ctrl+C/termination through the same Qt/finally cleanup path."""
    del frame
    logger.info("received signal {}; shutting down", sig)
    _stop_event.set()
    callback = _quit_callback
    if callback is not None:
        callback()


def _cli_loop(effect_manager: EffectManager, quit_fn=None) -> None:
    """Background thread: read CLI commands without blocking audio."""
    while not _stop_event.is_set():
        try:
            cmd = input("> ").strip().lower()
        except EOFError:
            break
        if cmd == "exit":
            logger.info("CLI: exit")
            _stop_event.set()
            if quit_fn:
                quit_fn()
            break
        if cmd == "status":
            for effect in effect_manager.effects:
                print(f"{effect.name}: enabled={effect.enabled}")
        elif cmd == "robot on":
            effect_manager.enable("RobotEffect")
        elif cmd == "robot off":
            effect_manager.disable("RobotEffect")
        elif cmd == "echo on":
            effect_manager.enable("EchoEffect")
        elif cmd == "echo off":
            effect_manager.disable("EchoEffect")
        elif cmd.startswith("gain "):
            effect = effect_manager.get_by_name("GainEffect")
            if effect is None:
                print("effect not found: GainEffect")
            else:
                try:
                    value = float(cmd.split(" ", 1)[1])
                    effect.gain = value
                    print(f"gain set to {value}")
                except ValueError:
                    print("invalid gain value")
        elif cmd:
            print(f"unknown command: {cmd!r}  (type 'exit' to quit)")


def create_effect_manager(
    ai_voice_effect: Optional[AIVoiceEffect] = None,
) -> EffectManager:
    """Build the established effect chain, with ready AI first when present."""
    effect_manager = EffectManager()
    if ai_voice_effect is not None:
        effect_manager.add(ai_voice_effect)
    if ENABLE_GAIN:
        effect_manager.add(GainEffect(gain=GAIN_VALUE))
    if ENABLE_ECHO:
        effect_manager.add(EchoEffect(delay_ms=ECHO_DELAY, decay=ECHO_DECAY))
    if ENABLE_ROBOT:
        effect_manager.add(RobotEffect(frequency=ROBOT_FREQUENCY))
    return effect_manager


def _select_devices() -> tuple[Optional[int], Optional[int]]:
    if SHOW_DEVICE_LIST:
        DeviceManager.print_devices()

    if AUTO_SELECT_DEVICES:
        if INPUT_DEVICE is not None:
            input_idx = INPUT_DEVICE
            logger.info("using configured input device {}", input_idx)
        else:
            input_idx = None
            logger.info("using system default input device")
        output_idx = DeviceManager.find_virtual_output_device()
        if output_idx is not None:
            logger.info("using detected VB-CABLE output")
        else:
            logger.warning("VB-CABLE not found; falling back to output selection")
            output_idx = DeviceManager.select_output_device()
    else:
        input_idx = DeviceManager.select_input_device()
        output_idx = DeviceManager.select_output_device()
    return input_idx, output_idx


def main() -> None:
    """Initialize RVC before exposing AudioStream, then clean up in order."""
    global _quit_callback

    setup_logger()
    _stop_event.clear()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    rvc_state = RVCApplicationState(enabled=ENABLE_AI_VOICE)
    stream: Optional[AudioStream] = None
    cli_thread: Optional[threading.Thread] = None

    try:
        input_idx, output_idx = _select_devices()
        recorder = AudioRecorder(device=input_idx)
        player = AudioPlayer(device=output_idx)

        # Required order: model load -> Worker start -> warmup -> effect chain.
        rvc_state = initialize_rvc_application(enabled=ENABLE_AI_VOICE)
        if _stop_event.is_set():
            logger.info("shutdown requested during initialization")
            return

        ai_effect = rvc_state.effect if rvc_state.ready else None
        effect_manager = create_effect_manager(ai_effect)

        # AudioStream is only created after AI is ready or has safely fallen
        # back to the base effect chain. The GUI may start it later.
        stream = AudioStream(recorder, player, effect_manager=effect_manager)
        context = AppContext(
            effect_manager=effect_manager,
            device_manager=DeviceManager,
            audio_stream=stream,
            input_device=input_idx,
            output_device=output_idx,
        )
        app, _window = create_app(context)
        _quit_callback = app.quit

        cli_thread = threading.Thread(
            target=_cli_loop,
            args=(effect_manager, app.quit),
            daemon=True,
            name="cli",
        )
        cli_thread.start()
        app.exec()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt; shutting down")
    except Exception:
        logger.exception("application startup/runtime failed")
    finally:
        _stop_event.set()
        _quit_callback = None

        # Ownership order: AudioStream -> Effect/Worker -> Engine.
        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                logger.error("AudioStream stop failed: {}", exc)

        cleaned = cleanup_rvc_application(
            rvc_state,
            timeout=RVC_WORKER_STOP_TIMEOUT,
        )
        if not cleaned:
            logger.warning(
                "RVC cleanup incomplete; live Worker retained its loaded engine"
            )

        if cli_thread is not None:
            cli_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

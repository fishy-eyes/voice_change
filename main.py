"""Voice Changer application entry point."""

from __future__ import annotations

import importlib
import signal
import sys
import threading
from typing import Callable, Optional

from loguru import logger

from audio.device_manager import DeviceManager
from audio.monitor import SelfMonitor
from audio.output_router import OutputRoutingEffectManager
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
    RVC_USER_MODELS_FILE,
    RVC_DEFAULT_MODEL,
    RVC_MODEL_LIBRARY_DIR,
    RVC_SOURCE_DIR,
    SHOW_DEVICE_LIST,
)
from core.context import AppContext
from core.device_switching import stop_current_audio_stream
from core.rvc_model_manager import RVCModelManager
from core.rvc_runtime import RVCRuntime
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
        except (EOFError, RuntimeError):
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
    *,
    self_monitor: Optional[SelfMonitor] = None,
    gain_enabled: bool = ENABLE_GAIN,
    echo_enabled: bool = ENABLE_ECHO,
    robot_enabled: bool = ENABLE_ROBOT,
) -> EffectManager:
    """Build the established effect chain, with ready AI first when present.

    Base effects are always registered so runtime controls can enable them.
    Configuration only determines their initial enabled state.
    """
    effect_manager = (
        OutputRoutingEffectManager(self_monitor)
        if self_monitor is not None
        else EffectManager()
    )
    if ai_voice_effect is not None:
        effect_manager.add(ai_voice_effect)

    gain = GainEffect(gain=GAIN_VALUE)
    gain.enabled = gain_enabled
    effect_manager.add(gain)

    echo = EchoEffect(delay_ms=ECHO_DELAY, decay=ECHO_DECAY)
    echo.enabled = echo_enabled
    effect_manager.add(echo)

    robot = RobotEffect(frequency=ROBOT_FREQUENCY)
    robot.enabled = robot_enabled
    effect_manager.add(robot)
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
            logger.warning("VB-CABLE not found; using the system default output")
            output_idx = None
    else:
        input_idx = DeviceManager.select_input_device()
        output_idx = DeviceManager.select_output_device()
    return input_idx, output_idx


def _run_release_smoke_test() -> None:
    """Import bundled native/ML dependencies without loading model weights."""
    source_dir = str(RVC_SOURCE_DIR)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    modules = (
        "torch",
        "torchaudio",
        "transformers",
        "faiss",
        "librosa",
        "parselmouth",
        "infer.hubert",
        "infer.module.models",
        "infer.rmvpe",
        "infer.vc.pipeline",
    )
    for module_name in modules:
        logger.info("release smoke test importing {}", module_name)
        importlib.import_module(module_name)
        logger.info("release smoke test imported {}", module_name)
    logger.info(
        "release smoke test passed; imported {} modules from RVC source {}",
        len(modules),
        source_dir,
    )


def main() -> None:
    """Initialize RVC before exposing AudioStream, then clean up in order."""
    global _quit_callback

    setup_logger()
    _stop_event.clear()
    if "--release-smoke-test" in sys.argv:
        try:
            _run_release_smoke_test()
        except Exception:
            logger.exception("release smoke test failed")
            raise SystemExit(1)
        return

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    rvc_runtime: Optional[RVCRuntime] = None
    stream: Optional[AudioStream] = None
    context: Optional[AppContext] = None
    self_monitor: Optional[SelfMonitor] = None
    cli_thread: Optional[threading.Thread] = None

    try:
        input_idx, output_idx = _select_devices()
        recorder = AudioRecorder(device=input_idx)
        player = AudioPlayer(device=output_idx)

        self_monitor = SelfMonitor()
        effect_manager = create_effect_manager(self_monitor=self_monitor)
        model_manager = RVCModelManager(
            RVC_MODEL_LIBRARY_DIR,
            user_models_path=RVC_USER_MODELS_FILE,
        )
        rvc_runtime = RVCRuntime(model_manager)
        rvc_runtime.bind_effect_manager(effect_manager)
        rvc_runtime.set_enabled(ENABLE_AI_VOICE)
        if ENABLE_AI_VOICE:
            rvc_state = rvc_runtime.load_model(RVC_DEFAULT_MODEL)
            if not rvc_state.ready:
                logger.warning(
                    "Default RVC model failed to load; base effects remain available: {}",
                    rvc_state.error,
                )
        if _stop_event.is_set():
            logger.info("shutdown requested during initialization")
            return

        # AudioStream is only created after AI is ready or has safely fallen
        # back to the base effect chain. The GUI may start it later.
        stream = AudioStream(recorder, player, effect_manager=effect_manager)
        context = AppContext(
            effect_manager=effect_manager,
            device_manager=DeviceManager,
            audio_stream=stream,
            input_device=input_idx,
            output_device=output_idx,
            rvc_runtime=rvc_runtime,
            self_monitor=self_monitor,
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

        # Ownership order: monitor -> AudioStream -> Effect/Worker -> Engine.
        if self_monitor is not None:
            try:
                self_monitor.stop()
            except Exception as exc:
                logger.error("SelfMonitor stop failed: {}", exc)

        stop_current_audio_stream(context, fallback=stream)

        cleaned = (
            rvc_runtime.shutdown()
            if rvc_runtime is not None
            else True
        )
        if not cleaned:
            logger.warning(
                "RVC cleanup incomplete; live Worker retained its loaded engine"
            )

        if cli_thread is not None:
            cli_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

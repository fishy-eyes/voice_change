"""Voice Changer application entry point."""

from __future__ import annotations

import importlib
from pathlib import Path
import signal
import sys
import threading
from time import perf_counter
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
    BEATRICE_CALLBACK_SIZE,
    BEATRICE_DEFAULT_RUNTIME_DIR,
    BEATRICE_ENV_MODELS_DIR,
    BEATRICE_INPUT_QUEUE_SIZE,
    BEATRICE_MODEL_LIBRARY_DIR,
    BEATRICE_RUNTIME_DIR,
    BEATRICE_STARTUP_BUFFER_SIZE,
    BEATRICE_WORKER_STOP_TIMEOUT,
    ENABLE_AI_VOICE,
    LOCAL_SETTINGS_FILE,
    GAIN_VALUE,
    INPUT_DEVICE,
    RVC_USER_MODELS_FILE,
    RVC_DEFAULT_MODEL,
    RVC_MODEL_LIBRARY_DIR,
    RVC_SOURCE_DIR,
    SHOW_DEVICE_LIST,
)
from core.context import AppContext
from ai.voice_conversion_manager import VoiceConversionManager
from ai.beatrice.catalog import BeatriceModelCatalog
from config.local_settings import LocalSettingsStore
from core.beatrice_runtime import BeatriceRuntime
from core.device_switching import stop_current_audio_stream
from core.rvc_model_manager import RVCModelManager
from core.rvc_runtime import RVCRuntime
from effects.ai_voice import AIVoiceEffect
from effects.gain import GainEffect
from effects.manager import EffectManager
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
) -> EffectManager:
    """Build AI followed by the always-on final Output Gain.

    Output routing fans this result to both primary output and Self Monitor.
    """
    effect_manager = (
        OutputRoutingEffectManager(self_monitor)
        if self_monitor is not None
        else EffectManager()
    )
    if ai_voice_effect is not None:
        effect_manager.add(ai_voice_effect)

    gain = GainEffect(gain=GAIN_VALUE)
    effect_manager.add(gain)
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
    """Show the base application first; heavyweight backends load on demand."""
    global _quit_callback

    startup_started = perf_counter()
    setup_logger()
    logger.info(
        "Startup timing: logger ready {:.1f} ms",
        (perf_counter() - startup_started) * 1000.0,
    )
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
    beatrice_runtime: Optional[BeatriceRuntime] = None
    voice_conversion_manager: Optional[VoiceConversionManager] = None
    stream: Optional[AudioStream] = None
    context: Optional[AppContext] = None
    self_monitor: Optional[SelfMonitor] = None
    cli_thread: Optional[threading.Thread] = None

    try:
        phase_started = perf_counter()
        input_idx, output_idx = _select_devices()
        logger.info(
            "Startup timing: device discovery {:.1f} ms",
            (perf_counter() - phase_started) * 1000.0,
        )
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
        local_settings = LocalSettingsStore(LOCAL_SETTINGS_FILE)
        beatrice_settings = local_settings.beatrice
        local_runtime = str(beatrice_settings.get("runtime_dir", "")).strip()
        default_runtime = Path(BEATRICE_DEFAULT_RUNTIME_DIR)
        beatrice_runtime_root = (
            local_runtime
            or BEATRICE_RUNTIME_DIR
            or (str(default_runtime) if default_runtime.is_dir() else None)
        )
        registered_packages = beatrice_settings.get("model_roots", [])
        environment_roots = (
            (BEATRICE_ENV_MODELS_DIR,) if BEATRICE_ENV_MODELS_DIR else ()
        )
        beatrice_model_manager = BeatriceModelCatalog(
            BEATRICE_MODEL_LIBRARY_DIR,
            registered_packages=registered_packages,
            additional_roots=environment_roots,
            on_registered_paths_changed=lambda roots: local_settings.update_beatrice(
                model_roots=list(roots)
            ),
        )
        beatrice_runtime = BeatriceRuntime(
            beatrice_model_manager,
            runtime_root=beatrice_runtime_root,
            local_settings=local_settings,
            callback_samples=BEATRICE_CALLBACK_SIZE,
            input_queue_size=BEATRICE_INPUT_QUEUE_SIZE,
            startup_buffer_samples=BEATRICE_STARTUP_BUFFER_SIZE,
            stop_timeout=BEATRICE_WORKER_STOP_TIMEOUT,
        )
        beatrice_runtime.bind_effect_manager(effect_manager)
        voice_conversion_manager = VoiceConversionManager(
            {"rvc": rvc_runtime, "beatrice": beatrice_runtime},
            default_backend="rvc",
        )
        # The initial chain is Gain-only.  Model metadata is lightweight and
        # scanned exactly once here; the GUI consumes the manager cache.
        voice_conversion_manager.set_enabled(False)
        phase_started = perf_counter()
        for backend in voice_conversion_manager.available_backends:
            voice_conversion_manager.discover_models(backend)
        logger.info(
            "Startup timing: model metadata discovery {:.1f} ms",
            (perf_counter() - phase_started) * 1000.0,
        )
        if _stop_event.is_set():
            logger.info("shutdown requested during initialization")
            return

        # AudioStream starts from the safe base chain; the GUI may start it
        # independently of any optional model backend.
        stream = AudioStream(recorder, player, effect_manager=effect_manager)
        context = AppContext(
            effect_manager=effect_manager,
            device_manager=DeviceManager,
            audio_stream=stream,
            input_device=input_idx,
            output_device=output_idx,
            rvc_runtime=rvc_runtime,
            beatrice_runtime=beatrice_runtime,
            voice_conversion_manager=voice_conversion_manager,
            self_monitor=self_monitor,
            local_settings=local_settings,
        )
        phase_started = perf_counter()
        app, _window = create_app(context)
        logger.info(
            "Startup timing: GUI creation {:.1f} ms",
            (perf_counter() - phase_started) * 1000.0,
        )
        logger.info(
            "Startup timing: GUI visible {:.1f} ms (visible={})",
            (perf_counter() - startup_started) * 1000.0,
            bool(getattr(_window, "isVisible", lambda: True)()),
        )
        _quit_callback = app.quit

        startup = local_settings.startup
        if bool(startup.get("autoload_last_model", False)):
            backend = str(startup.get("last_backend", "")).strip()
            model = str(startup.get("last_model", "")).strip()
            if backend in voice_conversion_manager.available_backends and model:
                logger.info(
                    "Scheduling optional post-GUI model autoload: {}/{}",
                    backend,
                    model,
                )
                voice_conversion_manager.set_enabled(bool(ENABLE_AI_VOICE))
                voice_conversion_manager.switch_model_async(
                    backend,
                    model,
                    audio_stream=stream,
                )

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
            voice_conversion_manager.shutdown()
            if voice_conversion_manager is not None
            else (
                rvc_runtime.shutdown()
                if rvc_runtime is not None
                else True
            )
        )
        if not cleaned:
            logger.warning(
                "Voice conversion cleanup incomplete; live Worker retained its engine"
            )

        if cli_thread is not None:
            cli_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

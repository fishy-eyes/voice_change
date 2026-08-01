"""RVC application lifecycle integration test without GUI or audio devices."""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import SAMPLE_RATE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def log(message: str = "") -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    log(f"  OK: {message}")


class SlowFakeEngine:
    """Loaded engine whose warmup stays active until the test releases it."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, **kwargs) -> None:
        del kwargs
        self.sample_rate = sample_rate
        self.is_loaded = False
        self.infer_started = threading.Event()
        self.release_infer = threading.Event()
        self.unload_calls = 0

    def load_model(self) -> None:
        self.is_loaded = True

    def infer(self, audio: np.ndarray) -> np.ndarray:
        self.infer_started.set()
        self.release_infer.wait(timeout=5.0)
        return np.asarray(audio, dtype=np.float32).copy()

    def unload_model(self) -> None:
        self.unload_calls += 1
        self.is_loaded = False


def main() -> int:
    from config.settings import RVC_WORKER_STOP_TIMEOUT
    from core.rvc_lifecycle import (
        cleanup_rvc_application,
        initialize_rvc_application,
    )
    import main as app_main

    create_effect_manager = app_main.create_effect_manager
    real_state = None
    slow_state = None
    started_at = time.perf_counter()

    try:
        section("AI disabled: base chain remains available")
        disabled_state = initialize_rvc_application(enabled=False)
        base_manager = create_effect_manager()
        require(not disabled_state.ready, "disabled AI is not marked ready")
        require(disabled_state.engine is None, "disabled AI creates no engine")
        require(disabled_state.effect is None, "disabled AI creates no Worker/effect")
        require(not base_manager.is_empty, "base EffectManager is usable")
        require(
            base_manager.get_by_name("AIVoiceEffect") is None,
            "disabled AI is absent from effect chain",
        )

        quit_called = threading.Event()
        app_main._stop_event.clear()
        app_main._quit_callback = quit_called.set
        app_main._on_signal(signal.SIGINT, None)
        require(app_main._stop_event.is_set(), "Ctrl+C sets shared stop event")
        require(quit_called.is_set(), "Ctrl+C requests Qt event-loop exit")
        app_main._quit_callback = None
        app_main._stop_event.clear()

        section("Invalid model path: graceful fallback")
        invalid_state = initialize_rvc_application(
            enabled=True,
            voice_dir=PROJECT_ROOT / "tests" / "missing-rvc-voice",
        )
        invalid_manager = create_effect_manager(
            invalid_state.effect if invalid_state.ready else None
        )
        require(not invalid_state.ready, "invalid model returns failure state")
        require(bool(invalid_state.error), "invalid model exposes a clear error")
        require(
            invalid_manager.get_by_name("AIVoiceEffect") is None,
            "invalid AI effect is not added",
        )
        require(not invalid_manager.is_empty, "base chain survives AI failure")
        require(cleanup_rvc_application(invalid_state), "invalid state cleanup is safe")

        section("Warmup timeout: never unload under a live Worker")
        slow_state = initialize_rvc_application(
            enabled=True,
            warmup_enabled=True,
            warmup_timeout=0.01,
            stop_timeout=0.01,
            engine_factory=SlowFakeEngine,
            validate_paths=False,
        )
        require(not slow_state.ready, "warmup timeout returns failure state")
        require(bool(slow_state.error), "warmup timeout exposes an error")
        require(slow_state.effect is not None, "timed-out effect remains owned")
        require(slow_state.engine is not None, "timed-out engine remains owned")
        require(
            slow_state.effect.worker.thread_alive,
            "timed-out Worker is still observable for later cleanup",
        )
        require(slow_state.engine.is_loaded, "live Worker keeps model loaded")
        require(slow_state.engine.unload_calls == 0, "live model was not unloaded")
        slow_state.engine.release_infer.set()
        require(
            cleanup_rvc_application(slow_state, timeout=2.0),
            "cleanup succeeds after timed-out inference exits",
        )
        require(not slow_state.effect.worker.thread_alive, "slow Worker exited")
        require(not slow_state.engine.is_loaded, "slow engine unloaded after exit")
        require(slow_state.engine.unload_calls == 1, "slow engine unloaded exactly once")
        require(
            cleanup_rvc_application(slow_state, timeout=0.1),
            "repeated slow-state cleanup is safe",
        )

        section("Real model: load -> Worker -> warmup -> effect chain")
        real_state = initialize_rvc_application(enabled=True)
        require(real_state.ready, f"real RVC initialized: {real_state.error}")
        require(real_state.engine is not None, "real engine created")
        require(real_state.engine.is_loaded, "real model loaded")
        require(real_state.effect is not None, "AIVoiceEffect created")
        require(real_state.effect.is_running, "Worker started")
        require(real_state.warmup_seconds > 0.0, "warmup duration recorded")
        log(f"  warmup: {real_state.warmup_seconds:.3f}s")
        require(
            real_state.effect.worker.input_queue_size == 0,
            "warmup input queue cleared",
        )
        require(
            real_state.effect.worker.output_queue_size == 0,
            "warmup output queue cleared",
        )
        require(
            real_state.effect.input_buffered_samples == 0,
            "warmup local input buffer cleared",
        )
        require(
            real_state.effect.output_buffered_samples == 0,
            "warmup local output buffer cleared",
        )

        require(
            real_state.effect.warmup(timeout=30.0),
            "repeated warmup is safe",
        )
        require(
            real_state.effect.worker.input_queue_size == 0
            and real_state.effect.worker.output_queue_size == 0,
            "repeated warmup leaves queues empty",
        )

        manager = create_effect_manager(real_state.effect)
        require(
            manager.effects[0] is real_state.effect,
            "ready AI effect is first in EffectManager",
        )
        require(
            manager.get_by_name("AIVoiceEffect") is real_state.effect,
            "ready AI effect is discoverable",
        )
        block = np.ones((256, 1), dtype=np.float32) * 0.01
        output = real_state.effect.process(block, 256, None, None)
        require(output.shape == block.shape, "AI effect preserves callback shape")
        require(output.dtype == np.float32, "AI effect returns float32")

        section("Normal and repeated cleanup")
        effect = real_state.effect
        engine = real_state.engine
        require(
            cleanup_rvc_application(
                real_state,
                timeout=RVC_WORKER_STOP_TIMEOUT,
            ),
            "normal cleanup succeeds",
        )
        require(not effect.worker.thread_alive, "Worker exits before unload")
        require(not engine.is_loaded, "Engine is unloaded")
        require(
            cleanup_rvc_application(real_state, timeout=0.1),
            "repeated normal cleanup is safe",
        )
        require(not effect.worker.thread_alive, "no Worker thread remains")

        section("RESULT")
        log(f"  total elapsed: {time.perf_counter() - started_at:.2f}s")
        log("  ALL TESTS PASSED")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if slow_state is not None and slow_state.engine is not None:
            slow_state.engine.release_infer.set()
            cleanup_rvc_application(slow_state, timeout=2.0)
        if real_state is not None:
            cleanup_rvc_application(real_state, timeout=120.0)


if __name__ == "__main__":
    sys.exit(main())

"""Real-model smoke test for profile loading and runtime updates."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  OK: {message}", flush=True)


def main() -> int:
    from config.settings import RVC_MODELS_DIR, RVC_SOURCE_DIR, SAMPLE_RATE
    from core.rvc_lifecycle import cleanup_rvc_application, initialize_rvc_application

    profile_path = PROJECT_ROOT / "config" / "rvc_profiles" / "modelF.example.json"
    state = None
    try:
        print("RVC profile integration smoke test", flush=True)
        state = initialize_rvc_application(
            enabled=True,
            profile=profile_path,
            source_dir=RVC_SOURCE_DIR,
            models_dir=RVC_MODELS_DIR,
            sample_rate=SAMPLE_RATE,
            warmup_enabled=True,
            warmup_timeout=120.0,
        )
        require(state.ready, f"profile application initialized: {state.error}")
        require(state.engine is not None and state.engine.is_loaded, "engine loaded")
        require(state.effect is not None and state.effect.is_running, "worker started")
        require(state.engine.config.pitch_shift == 12, "profile pitch loaded")
        require(state.engine.config.index_rate == 0.30, "profile index rate loaded")

        pipeline_identity = id(state.engine._pipeline)
        state.engine.update_config(index_rate=0.15)
        require(id(state.engine._pipeline) == pipeline_identity, "model was not reloaded")
        require(state.engine.config.index_rate == 0.15, "runtime update applied")

        timeline = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        samples = (0.05 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
        require(state.effect.worker.put(samples, timeout=1.0), "updated work submitted")
        result = state.effect.worker.get(timeout=120.0)
        require(result is not None, "updated inference completed")
        require(result.shape == samples.shape, "updated inference shape preserved")
        require(result.dtype == np.float32, "updated inference dtype preserved")
        require(bool(np.all(np.isfinite(result))), "updated inference is finite")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if state is not None:
            cleanup_rvc_application(state, timeout=120.0)


if __name__ == "__main__":
    sys.exit(main())

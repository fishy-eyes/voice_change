"""Real formal-model smoke test for ModelManager -> RVCRuntime -> cleanup."""

from __future__ import annotations

import traceback

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import RVC_DEFAULT_MODEL, RVC_MODEL_LIBRARY_DIR
from core.rvc_model_manager import RVCModelManager
from core.rvc_runtime import RVCRuntime
from effects.manager import EffectManager


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  OK: {message}", flush=True)


def main() -> int:
    runtime = RVCRuntime(RVCModelManager(RVC_MODEL_LIBRARY_DIR))
    manager = EffectManager()
    runtime.bind_effect_manager(manager)
    runtime.set_enabled(True)
    state = None
    try:
        state = runtime.load_model(RVC_DEFAULT_MODEL)
        require(state.ready, f"formal model loaded: {state.error}")
        require(runtime.selected_model == "modelF", "modelF selected")
        require(state.engine.config.pitch_shift == 12, "profile pitch +12 applied")
        require(state.engine.config.index_rate == 0.30, "profile index 0.30 applied")
        require(state.engine.config.protect == 0.33, "profile protect 0.33 applied")
        require(state.engine.config.rms_mix_rate == 0.25, "profile RMS 0.25 applied")
        require(manager.effects[0] is state.effect, "AI effect is first in chain")
        require(state.effect.is_running, "RVC Worker is running")
        require(state.engine.index_cache_info["enabled"], "index cache is active")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        cleaned = runtime.shutdown()
        print(f"  cleanup={cleaned}", flush=True)
        if state is not None and state.engine is not None:
            print(f"  engine_loaded={state.engine.is_loaded}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

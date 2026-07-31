"""RVC Worker smoke test.

Verifies:
    1. RVCWorker can be created from RVCEngine
    2. start() launches worker thread
    3. put() submits audio chunk
    4. get() returns processed result
    5. stop() cleanly shuts down

Usage:
    python -u tests\test_rvc_worker.py
"""

import sys
import os
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def log(msg):
    print(msg, flush=True)

def section(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")


def main():
    section("RVC Worker Smoke Test")

    # ------------------------------------------------------------------
    # 1. Load engine
    # ------------------------------------------------------------------
    log("\n[1/5] Loading RVCEngine ...")
    t0 = time.perf_counter()
    try:
        from config.settings import (
            RVC_SOURCE_DIR, RVC_MODELS_DIR, RVC_VOICE_DIR,
            RVC_PITCH_SHIFT, RVC_F0_METHOD,
            RVC_INDEX_RATE, RVC_RMS_MIX_RATE, RVC_PROTECT,
            SAMPLE_RATE,
        )
        from ai.rvc_engine import RVCEngine
        from ai.rvc_worker import RVCWorker
        import numpy as np

        engine = RVCEngine(
            voice_dir=RVC_VOICE_DIR,
            source_dir=RVC_SOURCE_DIR,
            models_dir=RVC_MODELS_DIR,
            pitch_shift=RVC_PITCH_SHIFT,
            f0_method=RVC_F0_METHOD,
            index_rate=RVC_INDEX_RATE,
            rms_mix_rate=RVC_RMS_MIX_RATE,
            protect=RVC_PROTECT,
            sample_rate=SAMPLE_RATE,
        )
        engine.load_model()
        log(f"  OK: loaded in {time.perf_counter() - t0:.2f}s")
    except Exception:
        log("  FAILED")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Create and start worker
    # ------------------------------------------------------------------
    log("\n[2/5] Creating RVCWorker ...")
    try:
        worker = RVCWorker(engine, chunk_size=SAMPLE_RATE)
        log(f"  OK: {worker!r}")
    except Exception:
        log("  FAILED")
        traceback.print_exc()
        return 1

    log("\n[3/5] Starting worker ...")
    try:
        worker.start()
        log(f"  OK: is_running={worker.is_running}")
    except Exception:
        log("  FAILED")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 3. Submit audio and retrieve result
    # ------------------------------------------------------------------
    log("\n[4/5] Submitting 1-second audio chunk ...")
    duration = 1.0
    n_samples = int(SAMPLE_RATE * duration)
    t_arr = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)
    audio_in = (0.3 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
    log(f"  input: shape={audio_in.shape} dtype={audio_in.dtype}")

    try:
        enqueued = worker.put(audio_in, timeout=1.0)
        log(f"  put() returned: {enqueued}")
        if not enqueued:
            log("  FAILED: could not enqueue")
            return 1

        log("  waiting for result (timeout=120s) ...")
        t_wait = time.perf_counter()
        result = worker.get(timeout=120)
        wait_ms = (time.perf_counter() - t_wait) * 1000
        log(f"  get() returned in {wait_ms:.0f}ms")

        if result is None:
            log("  FAILED: result is None (inference error)")
            return 1

        log(f"  output: shape={result.shape} dtype={result.dtype}")
        if result.shape == audio_in.shape:
            log("  OK: shape matches input")
        else:
            log(f"  WARNING: shape mismatch")
    except Exception:
        log("  FAILED")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 4. Stop worker
    # ------------------------------------------------------------------
    log("\n[5/5] Stopping worker ...")
    try:
        worker.stop(timeout=5.0)
        log(f"  OK: {worker!r}")
    except Exception:
        log("  FAILED")
        traceback.print_exc()
        return 1

    # Cleanup
    log("\n  Unloading model ...")
    engine.unload_model()

    section("RESULT")
    log("  ALL PASSED")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
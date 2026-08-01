"""RVC Inference Profiling Test

Runs 10 consecutive inferences on a fixed 1-second audio input
to measure first-run overhead, steady-state latency, and GPU memory.

Usage:
    python -u tests\test_rvc_profile.py
"""

import sys
import os
import time
import gc
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def log(msg: str) -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")


def gpu_mem_mb() -> float:
    """Return current GPU memory usage in MB, or -1 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except Exception:
        pass
    return -1.0


def gpu_max_mem_mb() -> float:
    """Return peak GPU memory since last reset in MB, or -1."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return -1.0


def main() -> int:
    section("RVC Inference Profiling")
    log("  10 consecutive calls, 1-second float32 input")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    log("\n[1/4] Loading model ...")
    t0 = time.perf_counter()
    try:
        from config.settings import (
            RVC_SOURCE_DIR, RVC_MODELS_DIR, RVC_VOICE_DIR,
            RVC_PITCH_SHIFT, RVC_F0_METHOD,
            RVC_INDEX_RATE, RVC_RMS_MIX_RATE, RVC_PROTECT,
            SAMPLE_RATE,
        )
        from ai.rvc_engine import RVCEngine
        import numpy as np
        import torch

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
        log(f"  Device: {engine._device}  half={engine._is_half}")
    except Exception:
        log("  FAILED: load error")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Prepare fixed 1-second input
    # ------------------------------------------------------------------
    log("\n[2/4] Preparing 1-second test audio ...")
    duration = 1.0
    sr = SAMPLE_RATE
    n_samples = int(sr * duration)
    t_arr = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)
    audio_in = (0.3 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
    log(f"  shape={audio_in.shape}  dtype={audio_in.dtype}  duration={duration:.1f}s")

    # ------------------------------------------------------------------
    # 3. Run 10 inferences
    # ------------------------------------------------------------------
    section("[3/4] Running 10 inferences")
    N = 10
    times = []
    mem_before = []
    mem_after = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    mem_baseline = gpu_mem_mb()
    log(f"  GPU baseline: {mem_baseline:.1f} MB" if mem_baseline >= 0 else "  GPU: N/A (CPU mode)")
    log("")

    for i in range(N):
        # Record memory before
        mem_before.append(gpu_mem_mb())

        # Time the inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = engine.infer(audio_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Record memory after
        mem_after.append(gpu_mem_mb())

        label = "FIRST" if i == 0 else f"#{i+1:>2d}"
        mem_delta = mem_after[-1] - mem_before[-1] if mem_after[-1] >= 0 else 0
        log(
            f"  {label:>5s}  |  {elapsed*1000:>8.1f} ms  |  "
            f"mem={mem_after[-1]:>7.1f} MB  |  delta={mem_delta:>+7.1f} MB"
        )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    section("[4/4] Summary")

    first_ms = times[0] * 1000
    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    steady_avg_ms = sum(times[1:]) / len(times[1:]) * 1000 if len(times) > 1 else avg_ms
    rtf_first = times[0] / duration
    rtf_avg = (sum(times) / len(times)) / duration
    rtf_min = min(times) / duration
    gpu_peak = gpu_max_mem_mb()

    log("")
    log(f"  First run:       {first_ms:>8.1f} ms  (RTF={rtf_first:.2f})")
    log(f"  Average (all):   {avg_ms:>8.1f} ms  (RTF={rtf_avg:.2f})")
    log(f"  Average (2-10):  {steady_avg_ms:>8.1f} ms")
    log(f"  Fastest:         {min_ms:>8.1f} ms  (RTF={rtf_min:.2f})")
    log(f"  Slowest:         {max_ms:>8.1f} ms")
    log("")

    overhead_ms = first_ms - steady_avg_ms
    if overhead_ms > 10:
        log(f"  First-run overhead: ~{overhead_ms:.0f}ms (warmup / caching effect)")
    else:
        log(f"  First-run overhead: negligible ({overhead_ms:.1f}ms)")

    if gpu_peak >= 0:
        log(f"  GPU peak memory: {gpu_peak:.1f} MB")
        log(f"  GPU mem at end:  {gpu_mem_mb():.1f} MB")
    else:
        log(f"  GPU: N/A (CPU mode)")

    log("")
    if rtf_min < 1.0:
        log(f"  >>> Best RTF {rtf_min:.2f}: can process 1s audio in {min_ms:.0f}ms")
        log(f"  >>> Real-time capable for 1-second chunks")
    else:
        log(f"  >>> Best RTF {rtf_min:.2f}: NOT real-time for 1-second chunks")
        log(f"  >>> Need RTF < 1.0 for real-time")

    # Cleanup
    log("\n  Unloading model ...")
    engine.unload_model()
    log("  Done.")

    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
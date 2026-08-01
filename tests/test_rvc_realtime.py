"""RVC Real-time Inference Benchmark

Tests different chunk sizes and measures inference latency to determine
the real-time capability of the current RVCEngine implementation.

Usage:
    python -u tests\test_rvc_realtime.py

Output:
    Table of chunk sizes, latency, and RTF (Real-Time Factor).
    RTF < 1.0 means real-time capable for that chunk size.
"""

import sys
import os
import time
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


def main() -> int:
    section("RVC Real-time Inference Benchmark")
    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Import & load model
    # ------------------------------------------------------------------
    log("\n[1/3] Loading RVCEngine ...")
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
        log(f"  OK: model loaded in {time.perf_counter() - t0:.2f}s")
    except Exception:
        log("  FAILED: could not load model")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Benchmark different chunk sizes
    # ------------------------------------------------------------------
    section("[2/3] Benchmarking chunk sizes")
    sr = SAMPLE_RATE
    # chunk sizes in samples: 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    # plus one and two configured seconds.
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, sr, sr * 2]

    results = []
    warmup_done = False

    for chunk_len in chunk_sizes:
        duration_ms = chunk_len / sr * 1000
        # Generate test audio: 440 Hz sine wave
        t_arr = np.linspace(0, chunk_len / sr, chunk_len, endpoint=False, dtype=np.float32)
        audio_chunk = (0.3 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)

        # Warmup run (first chunk, not counted)
        if not warmup_done:
            log(f"  warmup run ({chunk_len} samples / {duration_ms:.1f}ms) ...")
            try:
                _ = engine.infer(audio_chunk)
            except Exception:
                pass
            warmup_done = True

        # Timed run
        N_RUNS = 3
        times = []
        ok = True
        for i in range(N_RUNS):
            t1 = time.perf_counter()
            try:
                result = engine.infer(audio_chunk)
                elapsed = time.perf_counter() - t1
                times.append(elapsed)
            except Exception as e:
                log(f"  FAIL at {chunk_len} samples: {e}")
                ok = False
                break

        if ok and times:
            avg_ms = sum(times) / len(times) * 1000
            min_ms = min(times) * 1000
            max_ms = max(times) * 1000
            audio_dur_ms = chunk_len / sr * 1000
            rtf = (sum(times) / len(times)) / (chunk_len / sr)
            results.append({
                "chunk": chunk_len,
                "dur_ms": audio_dur_ms,
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "rtf": rtf,
            })
            status = "REALTIME" if rtf < 1.0 else "TOO SLOW"
            log(
                f"  {chunk_len:>6d} samples | "
                f"dur={audio_dur_ms:>8.1f}ms | "
                f"avg={avg_ms:>8.1f}ms | "
                f"min={min_ms:>8.1f}ms | "
                f"max={max_ms:>8.1f}ms | "
                f"RTF={rtf:>6.2f} | {status}"
            )

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    section("[3/3] Summary")

    if not results:
        log("  No successful results.")
        return 1

    log("")
    log(f"  {'Samples':>8s} | {'Duration':>10s} | {'Avg Latency':>12s} | {'RTF':>8s} | Status")
    log(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}-+----------")

    realtime_boundary = None
    for r in results:
        status = "REALTIME" if r["rtf"] < 1.0 else "TOO SLOW"
        log(
            f"  {r['chunk']:>8d} | {r['dur_ms']:>8.1f} ms | {r['avg_ms']:>10.1f} ms | {r['rtf']:>7.2f} | {status}"
        )
        if r["rtf"] < 1.0:
            realtime_boundary = r

    log("")
    if realtime_boundary:
        log(f"  >>> Largest real-time chunk: {realtime_boundary['chunk']} samples "
            f"({realtime_boundary['dur_ms']:.1f}ms)  RTF={realtime_boundary['rtf']:.2f}")
    else:
        log(f"  >>> No chunk size achieved real-time (RTF < 1.0)")

    # Find best RTF
    best = min(results, key=lambda r: r["rtf"])
    log(f"  >>> Best RTF: {best['rtf']:.2f} at {best['chunk']} samples ({best['dur_ms']:.1f}ms)")

    total_elapsed = time.perf_counter() - t_total
    log(f"\n  Total benchmark time: {total_elapsed:.1f}s")

    # Cleanup
    log("\n  Unloading model ...")
    engine.unload_model()
    log("  Done.")

    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
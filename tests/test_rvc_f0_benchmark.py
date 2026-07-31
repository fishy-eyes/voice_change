"""RVC F0 Performance Comparison Test

Compares RVC inference with f0=True (normal) vs f0=False (skipping F0 estimation).
Uses the same 10-second input audio for both runs.

Purpose: Evaluate if disabling F0 is worth pursuing for real-time mode.

Usage:
    python -u tests\test_rvc_f0_benchmark.py
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
    section("RVC F0 Performance Comparison")
    t_total = time.perf_counter()

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
        import librosa

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
        log(f"  if_f0={engine._if_f0}  version={engine._version}  device={engine._device}")
    except Exception:
        log("  FAILED: load error")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Prepare 10-second input
    # ------------------------------------------------------------------
    log("\n[2/4] Preparing 10-second test audio ...")
    sr = SAMPLE_RATE
    duration = 10.0
    n_samples = int(sr * duration)
    t_arr = np.linspace(0, duration, n_samples, endpoint=False, dtype=np.float32)
    audio_in = (0.3 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
    log(f"  shape={audio_in.shape}  duration={duration:.1f}s  sr={sr}")

    # ------------------------------------------------------------------
    # 3. Test f0=True (normal)
    # ------------------------------------------------------------------
    section("[3/4] f0=True (normal)")
    N_F0 = 3
    times_f0 = []
    ok_f0 = True

    # warmup
    log("  warmup ...")
    try:
        _ = engine.infer(audio_in)
    except Exception:
        pass

    log(f"  running {N_F0} iterations ...")
    for i in range(N_F0):
        t1 = time.perf_counter()
        try:
            result = engine.infer(audio_in)
            elapsed = time.perf_counter() - t1
            times_f0.append(elapsed)
            rtf = elapsed / duration
            log(f"  #{i+1}  {elapsed*1000:.0f}ms  RTF={rtf:.2f}")
        except Exception as e:
            log(f"  #{i+1}  FAILED: {e}")
            ok_f0 = False
            break

    if ok_f0 and times_f0:
        avg_f0 = sum(times_f0) / len(times_f0)
        rtf_f0 = avg_f0 / duration
        log(f"\n  >>> f0=True  avg={avg_f0*1000:.0f}ms  RTF={rtf_f0:.2f}  status=OK")
    else:
        avg_f0 = None
        log(f"\n  >>> f0=True  FAILED")

    # ------------------------------------------------------------------
    # 4. Test f0=False
    # ------------------------------------------------------------------
    section("[4/4] f0=False (skip F0 estimation)")
    times_nof0 = []
    ok_nof0 = True

    log("  Attempting inference with if_f0=0 ...")
    log("  (calling pipeline directly with if_f0=0)")
    log("")

    # Prepare audio same way as engine.infer() does
    audio_16k = librosa.resample(audio_in, orig_sr=sr, target_sr=16000)
    audio_max = np.abs(audio_16k).max() / 0.95
    if audio_max > 1:
        audio_16k /= audio_max

    N_NOF0 = 3
    for i in range(N_NOF0):
        t1 = time.perf_counter()
        try:
            times_pipe = [0.0, 0.0, 0.0]
            result = engine._pipeline.pipeline(
                model=engine._hubert_model,
                net_g=engine._net_g,
                sid=engine._sid,
                audio=audio_16k,
                times=times_pipe,
                f0_up_key=engine._pitch_shift,
                f0_method=engine._f0_method,
                file_index=engine._index_path,
                index_rate=engine._index_rate,
                if_f0=0,
                tgt_sr=engine._tgt_sr,
                resample_sr=0,
                rms_mix_rate=engine._rms_mix_rate,
                version=engine._version,
                protect=engine._protect,
            )
            elapsed = time.perf_counter() - t1
            times_nof0.append(elapsed)
            rtf = elapsed / duration
            log(f"  #{i+1}  {elapsed*1000:.0f}ms  RTF={rtf:.2f}  OK")
        except Exception as e:
            log(f"  #{i+1}  FAILED: {e}")
            ok_nof0 = False
            break

    if ok_nof0 and times_nof0:
        avg_nof0 = sum(times_nof0) / len(times_nof0)
        rtf_nof0 = avg_nof0 / duration
        log(f"\n  >>> f0=False avg={avg_nof0*1000:.0f}ms  RTF={rtf_nof0:.2f}  status=OK")
    else:
        avg_nof0 = None
        log(f"\n  >>> f0=False FAILED (expected: current model is f0 model, needs non-f0 model)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("SUMMARY")
    log("")
    if avg_f0 is not None:
        log(f"  f0=True:   avg={avg_f0*1000:>8.0f}ms  RTF={avg_f0/duration:.2f}")
    else:
        log(f"  f0=True:   FAILED")

    if avg_nof0 is not None:
        log(f"  f0=False:  avg={avg_nof0*1000:>8.0f}ms  RTF={avg_nof0/duration:.2f}")
    else:
        log(f"  f0=False:  FAILED (model architecture mismatch)")

    log("")
    if avg_f0 is not None and avg_nof0 is not None:
        saved_ms = (avg_f0 - avg_nof0) * 1000
        saved_pct = (avg_f0 - avg_nof0) / avg_f0 * 100
        log(f"  F0 overhead: ~{saved_ms:.0f}ms ({saved_pct:.1f}%)")
        if saved_pct > 20:
            log(f"  >>> F0 is a significant bottleneck. Consider non-f0 model for real-time.")
        else:
            log(f"  >>> F0 overhead is small. Main bottleneck is elsewhere.")
    elif avg_f0 is not None and avg_nof0 is None:
        log(f"  Cannot compare: current model requires F0.")
        log(f"  To test f0=False, need a non-f0 trained model.")
        log(f"  F0 estimation is typically 10-30% of total inference time.")

    log("")
    log(f"  Total benchmark time: {time.perf_counter() - t_total:.1f}s")

    log("\n  Unloading model ...")
    engine.unload_model()
    log("  Done.")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
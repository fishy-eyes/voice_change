"""RVC Integration Smoke Test
验证 voice_change 项目中 RVCEngine 能否成功调用 RVC 推理流程。

Usage:
    python -u tests/test_rvc_integration.py

Requires:
    voice_change conda env with torch, librosa, faiss, parselmouth, etc.
"""

import sys
import os
import time
import traceback

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


def log(msg: str) -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}")


def main() -> int:
    passed = 0
    failed = 0
    total_start = time.perf_counter()

    # ============================================================
    # STAGE 1: Import modules
    # ============================================================
    section("STAGE 1: Import project modules")
    t0 = time.perf_counter()
    try:
        from config.settings import RVC_SOURCE_DIR, RVC_MODELS_DIR, RVC_VOICE_DIR
        from config.settings import RVC_PITCH_SHIFT, RVC_F0_METHOD
        from config.settings import RVC_INDEX_RATE, RVC_RMS_MIX_RATE, RVC_PROTECT
        from config.settings import SAMPLE_RATE
        from ai.rvc_engine import RVCEngine
        log(f"  OK: all modules imported")
        log(f"  RVC_SOURCE_DIR = {RVC_SOURCE_DIR}")
        log(f"  RVC_MODELS_DIR = {RVC_MODELS_DIR}")
        log(f"  RVC_VOICE_DIR  = {RVC_VOICE_DIR}")
        log(f"  SAMPLE_RATE    = {SAMPLE_RATE}")
        passed += 1
    except Exception:
        log("  FAILED: import error")
        traceback.print_exc()
        return 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # STAGE 2: Verify paths exist
    # ============================================================
    section("STAGE 2: Verify RVC paths")
    t0 = time.perf_counter()
    all_ok = True
    for label, path in [
        ("RVC_SOURCE_DIR", RVC_SOURCE_DIR),
        ("RVC_MODELS_DIR", RVC_MODELS_DIR),
        ("RVC_VOICE_DIR", RVC_VOICE_DIR),
    ]:
        exists = os.path.isdir(path)
        status = "OK" if exists else "MISSING"
        log(f"  {status}: {label} = {path}")
        if not exists:
            all_ok = False
    if not all_ok:
        log("  FAILED: required paths missing")
        return 1
    passed += 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # STAGE 3: Instantiate RVCEngine
    # ============================================================
    section("STAGE 3: Instantiate RVCEngine")
    t0 = time.perf_counter()
    try:
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
        log(f"  OK: {engine!r}")
        log(f"  is_loaded = {engine.is_loaded}")
        passed += 1
    except Exception:
        log("  FAILED: instantiation error")
        traceback.print_exc()
        return 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # STAGE 4: Load model (checkpoint + HuBERT + Pipeline)
    # ============================================================
    section("STAGE 4: load_model()")
    t0 = time.perf_counter()
    try:
        engine.load_model()
        log(f"  OK: model loaded")
        log(f"  is_loaded = {engine.is_loaded}")
        passed += 1
    except Exception:
        log("  FAILED: load_model error")
        traceback.print_exc()
        return 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # STAGE 5: Prepare test audio
    # ============================================================
    section("STAGE 5: Prepare test audio")
    t0 = time.perf_counter()
    test_input = os.path.join(PROJECT_ROOT, "tests", "assets", "input.wav")
    use_synthetic = False

    if os.path.isfile(test_input):
        try:
            import soundfile as sf
            import numpy as np
            audio_in, orig_sr = sf.read(test_input, dtype="float32")
            if audio_in.ndim > 1:
                audio_in = audio_in.mean(axis=1)
            if orig_sr != SAMPLE_RATE:
                import librosa
                audio_in = librosa.resample(audio_in, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
            log(f"  OK: loaded {test_input}")
            log(f"  shape={audio_in.shape}  sr={SAMPLE_RATE}  duration={len(audio_in)/SAMPLE_RATE:.2f}s")
            passed += 1
        except Exception:
            log("  FAILED: could not read test audio")
            traceback.print_exc()
            return 1
    else:
        log(f"  INFO: test audio not found: {test_input}")
        log(f"  Generating 3-second synthetic sine wave (440 Hz) ...")
        try:
            import numpy as np
            import soundfile as sf
            duration = 3.0
            t_arr = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
            audio_in = (0.3 * np.sin(2 * np.pi * 440 * t_arr)).astype(np.float32)
            sf.write(test_input, audio_in, SAMPLE_RATE)
            log(f"  OK: synthetic test audio saved to {test_input}")
            log(f"  shape={audio_in.shape}  sr={SAMPLE_RATE}  duration={duration:.1f}s")
            passed += 1
        except Exception:
            log("  FAILED: could not generate synthetic audio")
            traceback.print_exc()
            return 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # STAGE 6: Run inference
    # ============================================================
    section("STAGE 6: engine.infer()")
    t0 = time.perf_counter()
    try:
        audio_out = engine.infer(audio_in)
        elapsed = time.perf_counter() - t0
        log(f"  OK: inference complete")
        log(f"  input  shape={audio_in.shape}  dtype={audio_in.dtype}")
        log(f"  output shape={audio_out.shape}  dtype={audio_out.dtype}")
        log(f"  elapsed: {elapsed:.2f}s")

        # Shape check
        if audio_out.shape != audio_in.shape:
            log(f"  WARNING: shape mismatch (input={audio_in.shape} output={audio_out.shape})")
        else:
            log(f"  OK: shape matches input")
            passed += 1
    except Exception:
        log(f"  FAILED: inference error")
        traceback.print_exc()
        elapsed = time.perf_counter() - t0
        log(f"  elapsed before error: {elapsed:.2f}s")
        return 1

    # ============================================================
    # STAGE 7: Save output
    # ============================================================
    section("STAGE 7: Save output")
    t0 = time.perf_counter()
    output_path = os.path.join(PROJECT_ROOT, "tests", "assets", "rvc_output.wav")
    try:
        import soundfile as sf
        sf.write(output_path, audio_out, SAMPLE_RATE)
        file_size = os.path.getsize(output_path)
        log(f"  OK: saved to {output_path}")
        log(f"  file size: {file_size / 1024:.1f} KB")
        passed += 1
    except Exception:
        log("  FAILED: could not save output")
        traceback.print_exc()
        return 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # STAGE 8: Unload model
    # ============================================================
    section("STAGE 8: unload_model()")
    t0 = time.perf_counter()
    try:
        engine.unload_model()
        log(f"  OK: model unloaded")
        log(f"  is_loaded = {engine.is_loaded}")
        passed += 1
    except Exception:
        log("  FAILED: unload error")
        traceback.print_exc()
        return 1
    log(f"  elapsed: {time.perf_counter() - t0:.2f}s")

    # ============================================================
    # SUMMARY
    # ============================================================
    total_elapsed = time.perf_counter() - total_start
    section("SUMMARY")
    log(f"  Passed: {passed} / {passed + failed}")
    log(f"  Total time: {total_elapsed:.2f}s")
    log(f"  Output: {output_path}")

    if failed == 0:
        log(f"\n  >>> ALL TESTS PASSED <<<")
        return 0
    else:
        log(f"\n  >>> {failed} TEST(S) FAILED <<<")
        return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)

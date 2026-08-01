"""Real-model module test for asynchronous AIVoiceEffect buffering.

No microphone, output device, or VB-CABLE is required.

Usage:
    python -u tests\test_ai_voice_realtime.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def log(message: str) -> None:
    print(message, flush=True)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    log(f"  OK: {message}")


def main() -> int:
    from ai.rvc_engine import RVCEngine
    from config.settings import (
        RVC_F0_METHOD,
        RVC_INDEX_RATE,
        RVC_MODELS_DIR,
        RVC_PITCH_SHIFT,
        RVC_PROTECT,
        RVC_RMS_MIX_RATE,
        RVC_SOURCE_DIR,
        RVC_VOICE_DIR,
        SAMPLE_RATE,
    )
    from effects.ai_voice import AIVoiceEffect

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
    effect = AIVoiceEffect(engine, chunk_size=SAMPLE_RATE, max_queue_size=2)

    load_seconds = 0.0
    wait_seconds = 0.0
    process_times_ms: list[float] = []
    stopped = False

    try:
        log("\n[1/6] Checking unloaded fallback")
        dry_1d = np.linspace(-0.25, 0.25, 256, dtype=np.float32)
        dry_2d = dry_1d.reshape(-1, 1)
        check(effect.start() is False, "start() rejects an unloaded engine")
        check(np.array_equal(effect.process(dry_1d), dry_1d), "1-D unloaded passthrough")
        check(np.array_equal(effect.process(dry_2d), dry_2d), "2-D mono unloaded passthrough")

        log("\n[2/6] Loading real RVC model")
        started_at = time.perf_counter()
        engine.load_model()
        load_seconds = time.perf_counter() - started_at
        check(engine.is_loaded, f"model loaded in {load_seconds:.2f}s")

        log("\n[3/6] Starting asynchronous effect")
        check(effect.start(), "AIVoiceEffect started")
        check(effect.start(), "duplicate start() is safe")
        check(effect.is_running, "worker thread is running")

        log("\n[4/6] Feeding 256-sample callback blocks")
        block_count = (SAMPLE_RATE + 255) // 256
        sample_count = block_count * 256
        expected_remainder = sample_count - SAMPLE_RATE
        timeline = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
        source = (0.3 * np.sin(2 * np.pi * 220.0 * timeline)).astype(np.float32)
        saw_1d = False
        saw_2d = False
        for index in range(block_count):
            block_1d = source[index * 256:(index + 1) * 256]
            original = block_1d.copy()
            block = block_1d if index % 2 == 0 else block_1d.reshape(-1, 1)
            started_at = time.perf_counter()
            output = effect.process(block, 256, None, None)
            process_times_ms.append((time.perf_counter() - started_at) * 1000.0)
            if output.shape != block.shape:
                raise AssertionError(f"block {index + 1} shape was not preserved")
            if output.dtype != np.float32:
                raise AssertionError(f"block {index + 1} dtype was not float32")
            if not np.array_equal(block.reshape(-1), original):
                raise AssertionError(f"block {index + 1} input was modified")
            saw_1d = saw_1d or block.ndim == 1
            saw_2d = saw_2d or block.ndim == 2

        check(
            saw_1d and saw_2d,
            f"all {block_count} callback blocks preserved shape/dtype without input mutation",
        )
        check(saw_1d and saw_2d, "both (frames,) and (frames, 1) inputs handled")
        check(
            effect.input_buffered_samples == expected_remainder,
            f"exact chunk submitted; {expected_remainder} samples retained",
        )
        check(max(process_times_ms) < 100.0, "process() did not wait for full inference")

        log("\n[5/6] Waiting for worker output with a 120s deadline")
        wait_started = time.perf_counter()
        deadline = wait_started + 120.0
        converted = None
        # One-sample polling keeps process() as the only output consumer but
        # cannot accumulate another inference chunk during this bounded wait.
        poll_input = np.zeros(1, dtype=np.float32)
        while time.perf_counter() < deadline:
            started_at = time.perf_counter()
            candidate = effect.process(poll_input)
            process_times_ms.append((time.perf_counter() - started_at) * 1000.0)
            if effect.output_buffered_samples > 0:
                converted = candidate
                break
            time.sleep(0.01)
        wait_seconds = time.perf_counter() - wait_started

        check(converted is not None, f"converted output arrived in {wait_seconds:.2f}s")
        check(converted.shape == poll_input.shape, "polled output shape preserved")
        check(converted.dtype == np.float32, "polled output dtype is float32")
        check(effect.worker.infer_count >= 1, "infer_count >= 1")
        check(effect.worker.error_count == 0, "error_count == 0")
        check(effect.output_buffered_samples <= SAMPLE_RATE * 2, "output buffer is bounded")

        playback = effect.process(np.zeros((256, 1), dtype=np.float32), 256, None, None)
        check(playback.shape == (256, 1), "buffered RVC output sliced to callback shape")
        check(playback.dtype == np.float32, "buffered RVC output remains float32")

        log("\n[6/6] Stopping and unloading")
        stopped = effect.stop(timeout=5.0)
        check(stopped, "worker stopped within timeout")
        check(not effect.worker.thread_alive, "worker thread exited")
        check(effect.stop(timeout=0.1), "duplicate stop() is safe")
        check(effect.input_buffered_samples == 0, "input buffer cleared")
        check(effect.output_buffered_samples == 0, "output buffer cleared")
        engine.unload_model()
        check(not engine.is_loaded, "model unloaded after worker exit")

        log("\nPERFORMANCE")
        log(f"  model load: {load_seconds:.2f}s")
        log(f"  RVC output wait: {wait_seconds:.2f}s")
        log(f"  process() max: {max(process_times_ms):.3f}ms")
        log(f"  process() avg: {sum(process_times_ms) / len(process_times_ms):.3f}ms")
        log(f"  infer_count: {effect.worker.infer_count}")
        log(f"  error_count: {effect.worker.error_count}")
        log("\nALL TESTS PASSED")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if effect.worker.thread_alive:
            # Do not unload a model underneath an inference still in progress.
            stopped = effect.stop(timeout=120.0)
        if engine.is_loaded and not effect.worker.thread_alive:
            engine.unload_model()
        if not stopped and effect.worker.thread_alive:
            log("WARNING: worker still alive; model intentionally left loaded")


if __name__ == "__main__":
    sys.exit(main())

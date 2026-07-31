"""Measure steady-state and real-input-rate RVCWorker performance.

Uses one real model load and one worker. No audio device is required.

Usage:
    python -u tests\test_rvc_worker_stream.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

CHUNK_COUNT = 6
RESULT_TIMEOUT_SECONDS = 120.0
TAIL_TIMEOUT_SECONDS = 90.0


def log(message: str = "") -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_result(result: np.ndarray | None, source: np.ndarray, label: str) -> None:
    require(result is not None, f"{label}: result is None")
    require(result.shape == source.shape, f"{label}: shape mismatch")
    require(result.dtype == np.float32, f"{label}: dtype is not float32")
    require(bool(np.all(np.isfinite(result))), f"{label}: non-finite values")


def load_chunks(sample_rate: int) -> list[np.ndarray]:
    import soundfile as sf

    input_path = PROJECT_ROOT / "tests" / "assets" / "input.wav"
    require(input_path.is_file(), f"missing real input audio: {input_path}")
    audio, source_rate = sf.read(str(input_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        import librosa
        audio = librosa.resample(
            audio,
            orig_sr=source_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    required_samples = 10 * sample_rate
    require(
        audio.size >= required_samples,
        f"input.wav must contain at least 10 seconds; got {audio.size / sample_rate:.2f}s",
    )
    chunks = [
        audio[index * sample_rate:(index + 1) * sample_rate].copy()
        for index in range(10)
    ]
    require(
        len({chunk.tobytes() for chunk in chunks}) == len(chunks),
        "the selected one-second chunks are not content-distinct",
    )
    return chunks


def drain_available(worker, completed_at: list[float], stress_start: float) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    while worker.output_queue_size > 0:
        result = worker.get_nowait()
        if result is None:
            break
        results.append(result)
        completed_at.append(time.perf_counter() - stress_start)
    return results


def main() -> int:
    from ai.rvc_engine import RVCEngine
    from ai.rvc_worker import RVCWorker
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
    worker = RVCWorker(engine, chunk_size=SAMPLE_RATE, max_queue_size=2)
    worker_started = False
    load_seconds = 0.0

    try:
        chunks = load_chunks(SAMPLE_RATE)

        section("Cold/Warmup")
        started_at = time.perf_counter()
        engine.load_model()
        load_seconds = time.perf_counter() - started_at
        require(engine.is_loaded, "engine did not load")
        log(f"  model load: {load_seconds:.3f}s")

        require(worker.start(), "worker did not start")
        worker_started = True
        warmup_started = time.perf_counter()
        require(worker.put(chunks[0], timeout=0.0), "warmup submission failed")
        warmup_result = worker.get(timeout=RESULT_TIMEOUT_SECONDS)
        warmup_seconds = time.perf_counter() - warmup_started
        validate_result(warmup_result, chunks[0], "warmup")
        worker.clear_queues()
        require(worker.input_queue_size == 0, "warmup input was not cleared")
        require(worker.output_queue_size == 0, "warmup output was not cleared")
        log(f"  warmup round-trip: {warmup_seconds:.3f}s")
        log(f"  worker last infer: {worker.last_infer_ms:.1f}ms")

        section("Steady-state: serial one-second chunks")
        serial_chunks = chunks[1:1 + CHUNK_COUNT]
        serial_seconds: list[float] = []
        for index, chunk in enumerate(serial_chunks, start=1):
            started_at = time.perf_counter()
            require(worker.put(chunk, timeout=0.0), f"serial chunk {index} submission failed")
            result = worker.get(timeout=RESULT_TIMEOUT_SECONDS)
            elapsed = time.perf_counter() - started_at
            validate_result(result, chunk, f"serial chunk {index}")
            serial_seconds.append(elapsed)
            log(f"  chunk {index}: {elapsed:.3f}s  RTF={elapsed:.3f}")

        serial_array = np.asarray(serial_seconds, dtype=np.float64)
        average_seconds = float(serial_array.mean())
        fastest_seconds = float(serial_array.min())
        slowest_seconds = float(serial_array.max())
        p95_seconds = float(np.percentile(serial_array, 95))
        average_rtf = average_seconds
        worst_rtf = slowest_seconds
        log("\n  Steady-state summary")
        log(f"    average: {average_seconds:.3f}s")
        log(f"    fastest: {fastest_seconds:.3f}s")
        log(f"    slowest: {slowest_seconds:.3f}s")
        log(f"    P95: {p95_seconds:.3f}s")
        log(f"    average RTF: {average_rtf:.3f}")
        log(f"    worst RTF: {worst_rtf:.3f}")

        if average_rtf < 0.8:
            throughput_verdict = "real-time throughput has useful headroom"
        elif average_rtf < 1.0:
            throughput_verdict = "theoretical real-time throughput, limited headroom"
        else:
            throughput_verdict = "cannot sustain real-time one-second input"
        log(f"    verdict: {throughput_verdict}")
        log("    note: a fixed one-second input chunk still adds about one second")
        log("          of accumulation latency; throughput is not end-to-end latency.")

        section("Real-input-rate pressure: one chunk per second")
        worker.clear_queues()
        baseline_infer = worker.infer_count
        baseline_errors = worker.error_count
        baseline_input_drops = worker.input_drop_count
        baseline_output_drops = worker.output_drop_count

        stress_chunks = chunks[4:4 + CHUNK_COUNT]
        attempted = 0
        submitted = 0
        submission_times: list[float] = []
        put_times_ms: list[float] = []
        completion_times: list[float] = []
        collected_results: list[np.ndarray] = []
        max_input_queue = 0
        max_output_queue = 0
        stress_started = time.perf_counter()

        while attempted < CHUNK_COUNT:
            target = stress_started + attempted
            now = time.perf_counter()
            if now < target:
                collected_results.extend(
                    drain_available(worker, completion_times, stress_started)
                )
                max_input_queue = max(max_input_queue, worker.input_queue_size)
                max_output_queue = max(max_output_queue, worker.output_queue_size)
                time.sleep(min(0.01, target - now))
                continue

            chunk = stress_chunks[attempted]
            put_started = time.perf_counter()
            accepted = worker.put(chunk, timeout=0.0)
            put_times_ms.append((time.perf_counter() - put_started) * 1000.0)
            attempted += 1
            if accepted:
                submitted += 1
                submission_times.append(time.perf_counter() - stress_started)
            log(
                f"  submit {attempted}: accepted={accepted} "
                f"input_q={worker.input_queue_size} "
                f"input_drops={worker.input_drop_count - baseline_input_drops}"
            )
            collected_results.extend(
                drain_available(worker, completion_times, stress_started)
            )
            max_input_queue = max(max_input_queue, worker.input_queue_size)
            max_output_queue = max(max_output_queue, worker.output_queue_size)

        submission_phase_seconds = time.perf_counter() - stress_started
        backlog_at_submit_end = (
            worker.input_queue_size + int(worker.is_inferencing)
        )
        input_drops = worker.input_drop_count - baseline_input_drops
        expected_attempts = submitted - input_drops
        tail_deadline = time.perf_counter() + TAIL_TIMEOUT_SECONDS

        while time.perf_counter() < tail_deadline:
            collected_results.extend(
                drain_available(worker, completion_times, stress_started)
            )
            completed_attempts = (
                worker.infer_count - baseline_infer
                + worker.error_count - baseline_errors
            )
            if (
                completed_attempts >= expected_attempts
                and worker.input_queue_size == 0
                and not worker.is_inferencing
            ):
                collected_results.extend(
                    drain_available(worker, completion_times, stress_started)
                )
                break
            max_input_queue = max(max_input_queue, worker.input_queue_size)
            max_output_queue = max(max_output_queue, worker.output_queue_size)
            time.sleep(0.01)
        else:
            raise TimeoutError("pressure-test tail did not drain within 90 seconds")

        completed = worker.infer_count - baseline_infer
        errors = worker.error_count - baseline_errors
        output_drops = worker.output_drop_count - baseline_output_drops
        require(completed + errors == expected_attempts, "stress completion accounting mismatch")
        require(len(collected_results) + output_drops == completed, "stress output accounting mismatch")
        for index, result in enumerate(collected_results, start=1):
            require(result.shape == (SAMPLE_RATE,), f"stress result {index}: shape mismatch")
            require(result.dtype == np.float32, f"stress result {index}: dtype mismatch")
            require(bool(np.all(np.isfinite(result))), f"stress result {index}: non-finite")

        final_input_queue = worker.input_queue_size
        final_output_queue = worker.output_queue_size
        last_result_latency = (
            completion_times[-1] - submission_times[-1]
            if completion_times and submission_times
            else float("nan")
        )
        keeps_up = (
            average_rtf < 1.0
            and input_drops == 0
            and completed == submitted
            and errors == 0
        )
        sustained_backlog = input_drops > 0 or average_rtf >= 1.0

        log("\n  Pressure summary")
        log(f"    attempted submissions: {attempted}")
        log(f"    accepted submissions: {submitted}")
        log(f"    worker completions: {completed}")
        log(f"    worker errors: {errors}")
        log(f"    input queue drops: {input_drops}")
        log(f"    output queue drops: {output_drops}")
        log(f"    max input queue: {max_input_queue}")
        log(f"    max output queue: {max_output_queue}")
        log(f"    backlog at submission end: {backlog_at_submit_end}")
        log(f"    final input queue: {final_input_queue}")
        log(f"    final output queue: {final_output_queue}")
        log(f"    submission phase: {submission_phase_seconds:.3f}s")
        log(f"    completion timeline: {[round(value, 3) for value in completion_times]}")
        log(f"    last result vs last input: {last_result_latency:.3f}s")
        log(f"    max non-blocking put(): {max(put_times_ms):.3f}ms")
        log(f"    keeps up at one chunk/second: {keeps_up}")
        log(f"    sustained backlog: {sustained_backlog}")
        log(f"    cumulative infer_count: {worker.infer_count}")
        log(f"    cumulative error_count: {worker.error_count}")
        log(f"    cumulative average infer: {worker.average_infer_ms:.1f}ms")

        require(max(put_times_ms) < 100.0, "non-blocking put() was unexpectedly slow")
        require(final_input_queue == 0, "input queue did not drain")
        require(final_output_queue == 0, "output queue did not drain")
        require(errors == 0, "worker reported pressure-test errors")

        section("RESULT")
        log("  ALL TESTS PASSED")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if worker_started or worker.thread_alive:
            worker.stop(timeout=120.0)
        if engine.is_loaded and not worker.thread_alive:
            engine.unload_model()
        elif engine.is_loaded:
            log("WARNING: worker still alive; model intentionally left loaded")


if __name__ == "__main__":
    sys.exit(main())

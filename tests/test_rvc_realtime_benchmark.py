"""Configurable realtime RVCWorker benchmark and JSON diagnostic.

This tool never opens audio devices and never changes production settings. It
loads one engine from an RVC model profile, reuses the existing RVCWorker for
all inference, and creates one worker per chunk shape. Overlap changes only the
input-window hop/cadence; no overlap-add audio implementation is introduced.

Example:
    python -u tests/test_rvc_realtime_benchmark.py \
        --chunk-ms 100 200 325 500 --overlap-ms 0 --input-duration 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    from tests.test_rvc_short_chunk_benchmark import (
        select_distinct_chunks,
        validate_result,
    )
except ModuleNotFoundError:
    from test_rvc_short_chunk_benchmark import (  # type: ignore[no-redef]
        select_distinct_chunks,
        validate_result,
    )


DEFAULT_CHUNK_MS = (100, 200, 325, 500)
DEFAULT_RESULT_TIMEOUT = 120.0
DEFAULT_TAIL_TIMEOUT = 120.0


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def milliseconds_to_samples(milliseconds: float, sample_rate: int) -> int:
    """Convert milliseconds using round-half-up for stable chunk shapes."""

    require(math.isfinite(milliseconds) and milliseconds >= 0, "milliseconds invalid")
    require(sample_rate > 0, "sample rate must be positive")
    return int(math.floor(milliseconds * sample_rate / 1000.0 + 0.5))


def validate_benchmark_settings(
    chunk_ms_values: list[int],
    overlap_ms: float,
    input_duration: float,
    serial_count: int,
    queue_size: int,
) -> None:
    require(bool(chunk_ms_values), "at least one chunk size is required")
    require(all(value > 0 for value in chunk_ms_values), "chunk_ms must be positive")
    require(len(set(chunk_ms_values)) == len(chunk_ms_values), "chunk_ms values must be unique")
    require(math.isfinite(overlap_ms) and overlap_ms >= 0, "overlap_ms must be non-negative")
    require(overlap_ms < min(chunk_ms_values), "overlap_ms must be smaller than every chunk_ms")
    require(math.isfinite(input_duration) and input_duration > 0, "input duration must be positive")
    require(serial_count > 0, "serial_count must be positive")
    require(queue_size > 0, "queue_size must be positive")


def build_overlapping_windows(
    audio: np.ndarray,
    chunk_samples: int,
    overlap_samples: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Build fixed-shape windows; the final partial window is zero padded."""

    source = np.asarray(audio, dtype=np.float32).reshape(-1)
    require(source.size > 0, "input audio is empty")
    require(chunk_samples > 0, "chunk_samples must be positive")
    require(0 <= overlap_samples < chunk_samples, "invalid overlap_samples")
    hop_samples = chunk_samples - overlap_samples
    final_full_start = max(0, source.size - chunk_samples)
    starts = list(range(0, final_full_start + 1, hop_samples))
    if not starts:
        starts = [0]
    if starts[-1] + chunk_samples < source.size:
        starts.append(starts[-1] + hop_samples)

    windows: list[np.ndarray] = []
    for start in starts:
        chunk = source[start:start + chunk_samples]
        if chunk.size < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - chunk.size))
        windows.append(np.asarray(chunk, dtype=np.float32))
    return windows, starts


def summarize_performance(seconds: list[float], chunk_seconds: float) -> dict:
    require(bool(seconds), "no inference timings collected")
    values = [float(value) for value in seconds]
    average = statistics.fmean(values)
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    p95 = ordered[p95_index]
    return {
        "samples": values,
        "total_inference_seconds": float(sum(values)),
        "average_inference_seconds": average,
        "average_latency_ms": average * 1000.0,
        "minimum_inference_seconds": min(values),
        "maximum_inference_seconds": max(values),
        "median_inference_seconds": statistics.median(values),
        "p95_inference_seconds": p95,
        "rtf": average / chunk_seconds,
        "p95_rtf": p95 / chunk_seconds,
    }


def load_audio(path: Path, sample_rate: int, duration_seconds: float) -> tuple[np.ndarray, dict]:
    import soundfile as sf

    require(path.is_file(), f"input audio not found: {path}")
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    source_channels = int(audio.shape[1])
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        import librosa

        mono = librosa.resample(
            mono,
            orig_sr=source_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    required_samples = int(round(duration_seconds * sample_rate))
    require(
        mono.size >= required_samples,
        f"input is shorter than requested duration: {mono.size / sample_rate:.3f}s",
    )
    mono = np.asarray(mono[:required_samples], dtype=np.float32)
    require(bool(np.all(np.isfinite(mono))), "input contains NaN/Inf")
    require(float(np.max(np.abs(mono))) > 1e-6, "input is silent")
    return mono, {
        "path": str(path),
        "source_sample_rate": int(source_rate),
        "source_channels": source_channels,
        "source_frames": int(audio.shape[0]),
        "benchmark_sample_rate": sample_rate,
        "benchmark_samples": int(mono.size),
        "benchmark_duration_seconds": mono.size / sample_rate,
        "rms": float(np.sqrt(np.mean(mono * mono))),
        "peak": float(np.max(np.abs(mono))),
    }


def drain_worker(worker, started_at: float) -> tuple[list[Optional[np.ndarray]], list[float]]:
    results: list[Optional[np.ndarray]] = []
    completion_times: list[float] = []
    while worker.output_queue_size > 0:
        results.append(worker.get_nowait())
        completion_times.append(time.perf_counter() - started_at)
    return results, completion_times


def run_serial_phase(
    worker,
    chunks: list[np.ndarray],
    chunk_seconds: float,
    result_timeout: float,
) -> tuple[dict, int, int]:
    baseline_infer = worker.infer_count
    baseline_errors = worker.error_count
    timings: list[float] = []
    passthrough_count = 0
    for index, chunk in enumerate(chunks, start=1):
        started_at = time.perf_counter()
        require(worker.put(chunk, timeout=0.0), f"serial {index}: put failed")
        result = worker.get(timeout=result_timeout)
        elapsed = time.perf_counter() - started_at
        _, passthrough = validate_result(result, chunk, f"serial {index}")
        passthrough_count += int(passthrough)
        timings.append(elapsed)
        log(
            f"    serial {index}: {elapsed:.3f}s "
            f"RTF={elapsed / chunk_seconds:.3f} passthrough={passthrough}"
        )
    metrics = summarize_performance(timings, chunk_seconds)
    errors = worker.error_count - baseline_errors
    metrics.update(
        infer_count=worker.infer_count - baseline_infer,
        error_count=errors,
        passthrough_count=passthrough_count,
    )
    return metrics, worker.infer_count - baseline_infer, errors


def run_realtime_phase(
    worker,
    windows: list[np.ndarray],
    hop_seconds: float,
    queue_size: int,
    tail_timeout: float,
) -> dict:
    """Submit windows at real wall-clock cadence and observe existing queues."""

    worker.clear_queues()
    baseline_infer = worker.infer_count
    baseline_errors = worker.error_count
    baseline_input_drops = worker.input_drop_count
    baseline_output_drops = worker.output_drop_count
    attempted = 0
    accepted = 0
    submission_times: list[float] = []
    completion_times: list[float] = []
    collected: list[Optional[np.ndarray]] = []
    put_times_ms: list[float] = []
    max_input_queue = 0
    max_output_queue = 0
    backlog_observations = 0
    started_at = time.perf_counter()

    while attempted < len(windows):
        target = started_at + attempted * hop_seconds
        now = time.perf_counter()
        if now < target:
            drained, completed = drain_worker(worker, started_at)
            collected.extend(drained)
            completion_times.extend(completed)
            input_queue = worker.input_queue_size
            output_queue = worker.output_queue_size
            max_input_queue = max(max_input_queue, input_queue)
            max_output_queue = max(max_output_queue, output_queue)
            backlog_observations += int(input_queue > 0)
            time.sleep(min(0.005, target - now))
            continue

        put_started = time.perf_counter()
        submitted = worker.put(windows[attempted], timeout=0.0)
        put_times_ms.append((time.perf_counter() - put_started) * 1000.0)
        attempted += 1
        accepted += int(submitted)
        if submitted:
            submission_times.append(time.perf_counter() - started_at)
        input_queue = worker.input_queue_size
        output_queue = worker.output_queue_size
        max_input_queue = max(max_input_queue, input_queue)
        max_output_queue = max(max_output_queue, output_queue)
        backlog_observations += int(input_queue > 0)
        drained, completed = drain_worker(worker, started_at)
        collected.extend(drained)
        completion_times.extend(completed)

    submission_phase_seconds = time.perf_counter() - started_at
    backlog_at_submit_end = worker.input_queue_size + int(worker.is_inferencing)
    input_drops = worker.input_drop_count - baseline_input_drops
    expected_completions = accepted - input_drops
    tail_started = time.perf_counter()
    tail_deadline = tail_started + tail_timeout
    while time.perf_counter() < tail_deadline:
        drained, completed = drain_worker(worker, started_at)
        collected.extend(drained)
        completion_times.extend(completed)
        input_queue = worker.input_queue_size
        output_queue = worker.output_queue_size
        max_input_queue = max(max_input_queue, input_queue)
        max_output_queue = max(max_output_queue, output_queue)
        completed_or_failed = (
            worker.infer_count - baseline_infer
            + worker.error_count - baseline_errors
        )
        if (
            completed_or_failed >= expected_completions
            and input_queue == 0
            and not worker.is_inferencing
        ):
            drained, completed = drain_worker(worker, started_at)
            collected.extend(drained)
            completion_times.extend(completed)
            break
        time.sleep(0.005)
    else:
        raise TimeoutError(f"realtime tail did not drain within {tail_timeout:.1f}s")

    tail_drain_seconds = time.perf_counter() - tail_started
    completed = worker.infer_count - baseline_infer
    errors = worker.error_count - baseline_errors
    output_drops = worker.output_drop_count - baseline_output_drops
    final_input_queue = worker.input_queue_size
    final_output_queue = worker.output_queue_size
    require(completed + errors == expected_completions, "realtime accounting mismatch")
    require(
        len(collected) + output_drops == completed + errors,
        "realtime output accounting mismatch",
    )

    valid_results = [result for result in collected if result is not None]
    for index, result in enumerate(valid_results, start=1):
        require(result.shape == windows[0].shape, f"realtime {index}: shape mismatch")
        require(result.dtype == np.float32, f"realtime {index}: dtype mismatch")
        require(bool(np.all(np.isfinite(result))), f"realtime {index}: non-finite")

    queue_buildup = bool(
        input_drops
        or max_input_queue >= queue_size
        or final_input_queue
        or backlog_at_submit_end > 1
    )
    sustained_backlog = bool(
        input_drops
        or final_input_queue
        or final_output_queue
        or completed < accepted
        or tail_drain_seconds > hop_seconds
    )
    delivery_latencies: Optional[list[float]] = None
    if (
        input_drops == 0
        and output_drops == 0
        and errors == 0
        and len(completion_times) == len(submission_times)
    ):
        delivery_latencies = [
            completed_at - submitted_at
            for submitted_at, completed_at in zip(submission_times, completion_times)
        ]
    average_delivery_latency = (
        statistics.fmean(delivery_latencies) if delivery_latencies else None
    )
    last_result_vs_last_submit = (
        completion_times[-1] - submission_times[-1]
        if completion_times and submission_times
        else None
    )
    return {
        "simulated_realtime": True,
        "cadence_seconds": hop_seconds,
        "attempted": attempted,
        "accepted": accepted,
        "completed": completed,
        "errors": errors,
        "input_drops": input_drops,
        "output_drops": output_drops,
        "dropped": input_drops + output_drops,
        "max_input_queue": max_input_queue,
        "max_output_queue": max_output_queue,
        "backlog_observations": backlog_observations,
        "backlog_at_submit_end": backlog_at_submit_end,
        "final_input_queue": final_input_queue,
        "final_output_queue": final_output_queue,
        "queue_buildup": queue_buildup,
        "sustained_backlog": sustained_backlog,
        "submission_phase_seconds": submission_phase_seconds,
        "tail_drain_seconds": tail_drain_seconds,
        "completion_timeline_seconds": completion_times,
        "average_delivery_latency_seconds": average_delivery_latency,
        "last_result_vs_last_submit_seconds": last_result_vs_last_submit,
        "maximum_put_ms": max(put_times_ms, default=0.0),
        "worker_running": worker.is_running,
        "worker_thread_alive": worker.thread_alive,
    }


@dataclass
class BenchmarkCase:
    model: str
    chunk_ms: int
    chunk_samples: int
    actual_chunk_ms: float
    overlap_ms: float
    overlap_samples: int
    actual_overlap_ms: float
    hop_ms: float
    hop_samples: int
    window_count: int
    warmup_seconds: float
    warmup_passthrough: bool
    performance: dict
    realtime: dict
    stability: dict
    rtf: float
    avg_latency_ms: float
    errors: int
    dropped: int
    keeps_up: bool
    status: str


def benchmark_case(
    engine,
    model_name: str,
    audio: np.ndarray,
    sample_rate: int,
    chunk_ms: int,
    overlap_ms: float,
    serial_count: int,
    queue_size: int,
    result_timeout: float,
    tail_timeout: float,
) -> BenchmarkCase:
    from ai.rvc_worker import RVCWorker

    chunk_samples = milliseconds_to_samples(chunk_ms, sample_rate)
    overlap_samples = milliseconds_to_samples(overlap_ms, sample_rate)
    require(overlap_samples < chunk_samples, "overlap rounds to a full chunk")
    hop_samples = chunk_samples - overlap_samples
    chunk_seconds = chunk_samples / sample_rate
    hop_seconds = hop_samples / sample_rate
    windows, _starts = build_overlapping_windows(audio, chunk_samples, overlap_samples)
    serial_chunks, _serial_starts, _rms = select_distinct_chunks(
        audio,
        chunk_samples,
        min(serial_count, len(windows)),
        sample_rate,
    )

    worker = RVCWorker(engine, chunk_size=chunk_samples, max_queue_size=queue_size)
    require(worker.start(), f"worker failed to start for {chunk_ms}ms")
    stopped_cleanly = False
    try:
        warmup_started = time.perf_counter()
        require(worker.put(serial_chunks[0], timeout=0.0), "warmup put failed")
        warmup_result = worker.get(timeout=result_timeout)
        warmup_seconds = time.perf_counter() - warmup_started
        _, warmup_passthrough = validate_result(
            warmup_result,
            serial_chunks[0],
            "warmup",
        )
        worker.clear_queues()

        log("  Serial performance")
        performance, serial_infer_count, serial_errors = run_serial_phase(
            worker,
            serial_chunks,
            chunk_seconds,
            result_timeout,
        )
        log(f"  Realtime pressure at {hop_seconds:.6f}s cadence")
        realtime = run_realtime_phase(
            worker,
            windows,
            hop_seconds,
            queue_size,
            tail_timeout,
        )
        total_errors = serial_errors + int(realtime["errors"])
        dropped = int(realtime["dropped"])
        keeps_up = bool(
            performance["average_inference_seconds"] <= hop_seconds
            and total_errors == 0
            and dropped == 0
            and not realtime["queue_buildup"]
            and not realtime["sustained_backlog"]
        )
        if keeps_up:
            status = "keeps_up"
        elif total_errors:
            status = "worker_errors"
        elif dropped:
            status = "queue_drops"
        elif performance["average_inference_seconds"] > hop_seconds:
            status = "inference_slower_than_cadence"
        else:
            status = "backlog_detected"

        stability = {
            "warmup_inference_count": 1,
            "serial_inference_count": serial_infer_count,
            "realtime_inference_count": realtime["completed"],
            "inference_count": worker.infer_count,
            "error_count": worker.error_count,
            "worker_average_inference_ms": worker.average_infer_ms,
            "worker_total_inference_seconds": (
                worker.average_infer_ms * worker.infer_count / 1000.0
            ),
            "worker_running_before_stop": worker.is_running,
            "worker_thread_alive_before_stop": worker.thread_alive,
            "stopped_cleanly": False,
            "worker_thread_alive_after_stop": None,
        }
        return BenchmarkCase(
            model=model_name,
            chunk_ms=chunk_ms,
            chunk_samples=chunk_samples,
            actual_chunk_ms=chunk_seconds * 1000.0,
            overlap_ms=overlap_ms,
            overlap_samples=overlap_samples,
            actual_overlap_ms=overlap_samples * 1000.0 / sample_rate,
            hop_ms=hop_seconds * 1000.0,
            hop_samples=hop_samples,
            window_count=len(windows),
            warmup_seconds=warmup_seconds,
            warmup_passthrough=warmup_passthrough,
            performance=performance,
            realtime=realtime,
            stability=stability,
            rtf=float(performance["rtf"]),
            avg_latency_ms=float(performance["average_latency_ms"]),
            errors=total_errors,
            dropped=dropped,
            keeps_up=keeps_up,
            status=status,
        )
    finally:
        stopped_cleanly = worker.stop(timeout=tail_timeout)
        if "stability" in locals():
            stability["stopped_cleanly"] = stopped_cleanly
            stability["worker_thread_alive_after_stop"] = worker.thread_alive
        require(stopped_cleanly, f"worker stop timed out for {chunk_ms}ms")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    from config.settings import (
        RVC_INPUT_QUEUE_SIZE,
        RVC_MODELS_DIR,
        RVC_SOURCE_DIR,
        SAMPLE_RATE,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROJECT_ROOT / "config" / "rvc_profiles" / "modelF.example.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "tests" / "assets" / "input.wav",
    )
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=list(DEFAULT_CHUNK_MS))
    parser.add_argument("--overlap-ms", type=float, default=0.0)
    parser.add_argument("--input-duration", type=float, default=10.0)
    parser.add_argument("--serial-count", type=int, default=6)
    parser.add_argument("--queue-size", type=int, default=RVC_INPUT_QUEUE_SIZE)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--source-dir", type=Path, default=Path(RVC_SOURCE_DIR))
    parser.add_argument("--models-dir", type=Path, default=Path(RVC_MODELS_DIR))
    parser.add_argument("--result-timeout", type=float, default=DEFAULT_RESULT_TIMEOUT)
    parser.add_argument("--tail-timeout", type=float, default=DEFAULT_TAIL_TIMEOUT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def print_summary(results: list[dict]) -> None:
    log("\nChunk | Hop | Avg infer | RTF | Drops | Errors | Queue | Result")
    log("-" * 82)
    for result in results:
        if result.get("failed"):
            log(f"{result['chunk_ms']:5d} | FAILED: {result['error']}")
            continue
        realtime = result["realtime"]
        log(
            f"{result['chunk_ms']:5d} | {result['hop_ms']:6.1f}ms | "
            f"{result['avg_latency_ms']:9.1f}ms | {result['rtf']:5.2f} | "
            f"{result['dropped']:5d} | {result['errors']:6d} | "
            f"{str(realtime['queue_buildup']):5s} | {result['status']}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    from ai.rvc_engine import RVCEngine
    from config.rvc_profiles import load_rvc_profile

    args = parse_args(argv)
    validate_benchmark_settings(
        args.chunk_ms,
        args.overlap_ms,
        args.input_duration,
        args.serial_count,
        args.queue_size,
    )
    require(args.sample_rate > 0, "sample rate must be positive")
    require(args.result_timeout > 0 and args.tail_timeout > 0, "timeouts must be positive")

    profile = load_rvc_profile(args.profile)
    output_path = args.output or (
        PROJECT_ROOT
        / "tests"
        / "output"
        / "rvc_realtime_benchmark"
        / f"{profile.name}_realtime_benchmark.json"
    )
    audio, input_metadata = load_audio(
        args.input.resolve(),
        args.sample_rate,
        args.input_duration,
    )
    engine = RVCEngine.from_profile(
        profile,
        source_dir=args.source_dir,
        models_dir=args.models_dir,
        sample_rate=args.sample_rate,
    )
    results: list[dict] = []
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(args.profile.resolve()),
        "profile": profile.to_dict(),
        "model": profile.name,
        "voice_dir": str(profile.resolve_voice_dir(args.models_dir)),
        "input": input_metadata,
        "settings": {
            "requested_chunk_ms": args.chunk_ms,
            "overlap_ms": args.overlap_ms,
            "input_duration_seconds": args.input_duration,
            "serial_count": args.serial_count,
            "queue_size": args.queue_size,
            "sample_rate": args.sample_rate,
            "result_timeout_seconds": args.result_timeout,
            "tail_timeout_seconds": args.tail_timeout,
        },
        "model_load_seconds": None,
        "results": results,
    }

    try:
        log(f"Profile: {profile.name} ({args.profile})")
        log(f"Input: {args.input} ({audio.size / args.sample_rate:.3f}s)")
        load_started = time.perf_counter()
        engine.load_model()
        report["model_load_seconds"] = time.perf_counter() - load_started
        require(engine.is_loaded, "engine did not load")
        log(f"Model loaded in {report['model_load_seconds']:.3f}s")

        for chunk_ms in args.chunk_ms:
            log(f"\n=== chunk={chunk_ms}ms overlap={args.overlap_ms:g}ms ===")
            try:
                case = benchmark_case(
                    engine,
                    profile.name,
                    audio,
                    args.sample_rate,
                    chunk_ms,
                    args.overlap_ms,
                    args.serial_count,
                    args.queue_size,
                    args.result_timeout,
                    args.tail_timeout,
                )
                results.append(asdict(case))
            except Exception as exc:
                traceback.print_exc()
                results.append(
                    {
                        "model": profile.name,
                        "chunk_ms": chunk_ms,
                        "overlap_ms": args.overlap_ms,
                        "failed": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            write_report(output_path, report)

        print_summary(results)
        write_report(output_path, report)
        log(f"\nJSON report: {output_path}")
        return 1 if any(result.get("failed") for result in results) else 0
    except Exception:
        traceback.print_exc()
        report["fatal_error"] = traceback.format_exc()
        write_report(output_path, report)
        return 1
    finally:
        if engine.is_loaded:
            engine.unload_model()


if __name__ == "__main__":
    raise SystemExit(main())

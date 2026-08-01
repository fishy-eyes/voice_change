"""Benchmark the configured real RVC model at shorter fixed chunk sizes.

This is an offline evaluation. It does not open audio devices or change the
production realtime configuration. Generated WAV/JSON files live under the
gitignored ``tests/output/rvc_short_chunk`` directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import SAMPLE_RATE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

CHUNK_SIZES = tuple(
    int(round(SAMPLE_RATE * seconds))
    for seconds in (1.0, 0.75, 0.5, 0.375, 0.25)
)
SERIAL_COUNT = 8
STRESS_COUNT = 10
RESULT_TIMEOUT_SECONDS = 120.0
TAIL_TIMEOUT_SECONDS = 120.0
WORKER_STOP_TIMEOUT_SECONDS = 120.0
LISTEN_SECONDS = 8.0
BOUNDARY_WINDOW_SECONDS = 0.010
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "rvc_short_chunk"


def log(message: str = "") -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass
class SerialMetrics:
    seconds: list[float]
    average_seconds: float
    median_seconds: float
    fastest_seconds: float
    slowest_seconds: float
    p95_seconds: float
    average_rtf: float
    p95_rtf: float
    infer_count: int
    error_count: int
    passthrough_count: int


@dataclass
class PressureMetrics:
    attempted: int
    accepted: int
    completed: int
    errors: int
    input_drops: int
    output_drops: int
    max_input_queue: int
    max_output_queue: int
    backlog_at_submit_end: int
    final_input_queue: int
    final_output_queue: int
    completion_timeline: list[float]
    last_result_vs_last_submit: float
    max_put_ms: float
    sustained_backlog: bool
    passthrough_count: int


@dataclass
class BoundaryMetrics:
    boundary_count: int
    average_jump: float
    maximum_jump: float
    average_rms_delta: float
    maximum_rms_delta: float
    output_rms: float
    output_peak: float
    clipped_fraction: float


@dataclass
class ChunkMetrics:
    chunk_size: int
    duration_seconds: float
    selected_starts_seconds: list[float]
    selected_rms: list[float]
    warmup_seconds: float
    warmup_passthrough: bool
    serial: SerialMetrics
    pressure: PressureMetrics
    estimated_first_audio: float
    conservative_first_audio: float
    wav_path: str
    boundary: BoundaryMetrics
    output_valid: bool
    performance_eligible: bool
    status: str


def load_real_audio(sample_rate: int) -> tuple[np.ndarray, int]:
    import soundfile as sf

    input_path = PROJECT_ROOT / "tests" / "assets" / "input.wav"
    require(input_path.is_file(), f"missing real input audio: {input_path}")
    audio, source_rate = sf.read(str(input_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        import librosa

        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=source_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    require(audio.size > sample_rate, "real input audio is too short")
    require(bool(np.all(np.isfinite(audio))), "real input contains non-finite values")
    require(float(np.sqrt(np.mean(audio * audio))) > 1e-5, "real input is silent")
    return audio, int(source_rate)


def select_distinct_chunks(
    audio: np.ndarray,
    chunk_size: int,
    count: int,
    sample_rate: int,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Choose spread-out, non-silent, byte-distinct chunks from real speech."""
    require(audio.size >= chunk_size, "input audio is shorter than one chunk")
    max_start = audio.size - chunk_size
    candidate_count = max(count * 16, 64)
    starts = np.unique(
        np.linspace(0, max_start, candidate_count, dtype=np.int64)
    ).tolist()
    candidates: list[tuple[float, int, np.ndarray]] = []
    for start in starts:
        chunk = audio[start:start + chunk_size].astype(np.float32, copy=True)
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        candidates.append((rms, int(start), chunk))

    positive_rms = [rms for rms, _, _ in candidates if rms > 1e-5]
    require(positive_rms, "no non-silent chunk candidates found")
    threshold = max(1e-5, float(np.percentile(positive_rms, 20)) * 0.5)
    minimum_spacing = max(1, chunk_size // 3)
    chosen: list[tuple[float, int, np.ndarray]] = []
    fingerprints: set[bytes] = set()

    for rms, start, chunk in sorted(candidates, reverse=True):
        if rms < threshold:
            continue
        if any(abs(start - previous[1]) < minimum_spacing for previous in chosen):
            continue
        fingerprint = chunk.tobytes()
        if fingerprint in fingerprints:
            continue
        chosen.append((rms, start, chunk))
        fingerprints.add(fingerprint)
        if len(chosen) == count:
            break

    if len(chosen) < count:
        for rms, start, chunk in sorted(candidates, reverse=True):
            fingerprint = chunk.tobytes()
            if rms < threshold or fingerprint in fingerprints:
                continue
            chosen.append((rms, start, chunk))
            fingerprints.add(fingerprint)
            if len(chosen) == count:
                break

    require(len(chosen) == count, f"could not select {count} distinct speech chunks")
    chosen.sort(key=lambda item: item[1])
    chunks = [item[2] for item in chosen]
    chosen_starts = [item[1] for item in chosen]
    rms_values = [item[0] for item in chosen]
    require(
        len({chunk.tobytes() for chunk in chunks}) == count,
        "selected chunks are not content-distinct",
    )
    log(
        "  selected starts: "
        + str([round(start / sample_rate, 3) for start in chosen_starts])
    )
    log("  selected RMS:    " + str([round(value, 5) for value in rms_values]))
    return chunks, chosen_starts, rms_values


def select_listening_segment(
    audio: np.ndarray,
    sample_rate: int,
    seconds: float,
) -> tuple[np.ndarray, int]:
    desired = min(audio.size, int(round(seconds * sample_rate)))
    require(desired >= sample_rate * 6, "listening source must contain at least 6 seconds")
    if desired == audio.size:
        return audio.copy(), 0
    hop = max(1, sample_rate // 4)
    starts = list(range(0, audio.size - desired + 1, hop))
    final_start = audio.size - desired
    if starts[-1] != final_start:
        starts.append(final_start)
    best_start = max(
        starts,
        key=lambda start: float(np.mean(audio[start:start + desired] ** 2)),
    )
    return audio[best_start:best_start + desired].copy(), best_start


def validate_result(
    result: Optional[np.ndarray],
    source: np.ndarray,
    label: str,
) -> tuple[np.ndarray, bool]:
    require(result is not None, f"{label}: result is None")
    result = np.asarray(result)
    require(result.shape == source.shape, f"{label}: shape mismatch {result.shape}")
    require(result.dtype == np.float32, f"{label}: dtype is not float32")
    require(result.size > 0, f"{label}: result is empty")
    require(bool(np.all(np.isfinite(result))), f"{label}: non-finite output")
    # RVCEngine currently returns audio.copy() when its external pipeline fails.
    passthrough = bool(np.array_equal(result, source))
    return result, passthrough


def run_serial(worker, chunks: list[np.ndarray], duration: float) -> SerialMetrics:
    baseline_infer = worker.infer_count
    baseline_errors = worker.error_count
    elapsed_values: list[float] = []
    passthrough_count = 0
    for index, chunk in enumerate(chunks[:SERIAL_COUNT], start=1):
        started = time.perf_counter()
        require(worker.put(chunk, timeout=0.0), f"serial {index}: put failed")
        result = worker.get(timeout=RESULT_TIMEOUT_SECONDS)
        elapsed = time.perf_counter() - started
        _, passthrough = validate_result(result, chunk, f"serial {index}")
        passthrough_count += int(passthrough)
        elapsed_values.append(elapsed)
        log(
            f"    chunk {index}: {elapsed:.3f}s  RTF={elapsed / duration:.3f}"
            f"  passthrough={passthrough}"
        )

    values = np.asarray(elapsed_values, dtype=np.float64)
    return SerialMetrics(
        seconds=[float(value) for value in values],
        average_seconds=float(values.mean()),
        median_seconds=float(np.median(values)),
        fastest_seconds=float(values.min()),
        slowest_seconds=float(values.max()),
        p95_seconds=float(np.percentile(values, 95)),
        average_rtf=float(values.mean() / duration),
        p95_rtf=float(np.percentile(values, 95) / duration),
        infer_count=worker.infer_count - baseline_infer,
        error_count=worker.error_count - baseline_errors,
        passthrough_count=passthrough_count,
    )


def drain_available(worker, started: float):
    results: list[Optional[np.ndarray]] = []
    completed: list[float] = []
    while worker.output_queue_size > 0:
        results.append(worker.get_nowait())
        completed.append(time.perf_counter() - started)
    return results, completed


def run_pressure(worker, chunks: list[np.ndarray], duration: float) -> PressureMetrics:
    worker.clear_queues()
    baseline_infer = worker.infer_count
    baseline_errors = worker.error_count
    baseline_input_drops = worker.input_drop_count
    baseline_output_drops = worker.output_drop_count
    stress_chunks = chunks[:STRESS_COUNT]
    attempted = 0
    accepted = 0
    submission_times: list[float] = []
    completion_times: list[float] = []
    collected: list[Optional[np.ndarray]] = []
    put_times_ms: list[float] = []
    max_input_queue = 0
    max_output_queue = 0
    stress_started = time.perf_counter()

    while attempted < STRESS_COUNT:
        target = stress_started + attempted * duration
        now = time.perf_counter()
        if now < target:
            drained, times = drain_available(worker, stress_started)
            collected.extend(drained)
            completion_times.extend(times)
            max_input_queue = max(max_input_queue, worker.input_queue_size)
            max_output_queue = max(max_output_queue, worker.output_queue_size)
            time.sleep(min(0.005, target - now))
            continue

        source = stress_chunks[attempted]
        put_started = time.perf_counter()
        submitted = worker.put(source, timeout=0.0)
        put_times_ms.append((time.perf_counter() - put_started) * 1000.0)
        attempted += 1
        accepted += int(submitted)
        if submitted:
            submission_times.append(time.perf_counter() - stress_started)
        max_input_queue = max(max_input_queue, worker.input_queue_size)
        max_output_queue = max(max_output_queue, worker.output_queue_size)
        drained, times = drain_available(worker, stress_started)
        collected.extend(drained)
        completion_times.extend(times)
        log(
            f"    submit {attempted}: accepted={submitted} "
            f"input_q={worker.input_queue_size} "
            f"drops={worker.input_drop_count - baseline_input_drops}"
        )

    backlog_at_submit_end = worker.input_queue_size + int(worker.is_inferencing)
    input_drops = worker.input_drop_count - baseline_input_drops
    expected_completions = accepted - input_drops
    tail_deadline = time.perf_counter() + TAIL_TIMEOUT_SECONDS
    while time.perf_counter() < tail_deadline:
        max_input_queue = max(max_input_queue, worker.input_queue_size)
        max_output_queue = max(max_output_queue, worker.output_queue_size)
        drained, times = drain_available(worker, stress_started)
        collected.extend(drained)
        completion_times.extend(times)
        completed_or_failed = (
            worker.infer_count - baseline_infer
            + worker.error_count - baseline_errors
        )
        if (
            completed_or_failed >= expected_completions
            and worker.input_queue_size == 0
            and not worker.is_inferencing
        ):
            drained, times = drain_available(worker, stress_started)
            collected.extend(drained)
            completion_times.extend(times)
            break
        time.sleep(0.005)
    else:
        raise TimeoutError("pressure tail did not drain within timeout")

    completed = worker.infer_count - baseline_infer
    errors = worker.error_count - baseline_errors
    output_drops = worker.output_drop_count - baseline_output_drops
    final_input_queue = worker.input_queue_size
    final_output_queue = worker.output_queue_size
    require(completed + errors == expected_completions, "pressure accounting mismatch")
    require(len(collected) + output_drops == completed + errors, "output accounting mismatch")

    valid_results = [result for result in collected if result is not None]
    for index, result in enumerate(valid_results, start=1):
        require(result.shape == (chunks[0].size,), f"pressure {index}: shape mismatch")
        require(result.dtype == np.float32, f"pressure {index}: dtype mismatch")
        require(bool(np.all(np.isfinite(result))), f"pressure {index}: non-finite")
    passthrough_count = sum(
        any(np.array_equal(result, source) for source in stress_chunks)
        for result in valid_results
    )
    last_latency = (
        completion_times[-1] - submission_times[-1]
        if completion_times and submission_times
        else float("nan")
    )
    sustained_backlog = bool(
        input_drops
        or final_input_queue
        or final_output_queue
        or completed < accepted
        or (backlog_at_submit_end > 1 and last_latency > duration)
    )
    return PressureMetrics(
        attempted=attempted,
        accepted=accepted,
        completed=completed,
        errors=errors,
        input_drops=input_drops,
        output_drops=output_drops,
        max_input_queue=max_input_queue,
        max_output_queue=max_output_queue,
        backlog_at_submit_end=backlog_at_submit_end,
        final_input_queue=final_input_queue,
        final_output_queue=final_output_queue,
        completion_timeline=[float(value) for value in completion_times],
        last_result_vs_last_submit=float(last_latency),
        max_put_ms=float(max(put_times_ms)),
        sustained_backlog=sustained_backlog,
        passthrough_count=int(passthrough_count),
    )


def compute_boundary_metrics(
    converted_chunks: list[np.ndarray],
    output: np.ndarray,
    sample_rate: int,
) -> BoundaryMetrics:
    window = max(1, int(round(BOUNDARY_WINDOW_SECONDS * sample_rate)))
    jumps: list[float] = []
    rms_deltas: list[float] = []
    for left, right in zip(converted_chunks, converted_chunks[1:]):
        if left.size == 0 or right.size == 0:
            continue
        jumps.append(float(abs(float(right[0]) - float(left[-1]))))
        left_window = left[-window:]
        right_window = right[:window]
        left_rms = float(np.sqrt(np.mean(left_window * left_window)))
        right_rms = float(np.sqrt(np.mean(right_window * right_window)))
        rms_deltas.append(abs(right_rms - left_rms))
    output_rms = float(np.sqrt(np.mean(output * output)))
    output_peak = float(np.max(np.abs(output)))
    clipped_fraction = float(np.mean(np.abs(output) >= 0.999))
    return BoundaryMetrics(
        boundary_count=len(jumps),
        average_jump=float(statistics.fmean(jumps)) if jumps else 0.0,
        maximum_jump=max(jumps, default=0.0),
        average_rms_delta=float(statistics.fmean(rms_deltas)) if rms_deltas else 0.0,
        maximum_rms_delta=max(rms_deltas, default=0.0),
        output_rms=output_rms,
        output_peak=output_peak,
        clipped_fraction=clipped_fraction,
    )


def generate_listening_wav(
    worker,
    source: np.ndarray,
    chunk_size: int,
    sample_rate: int,
) -> tuple[Path, BoundaryMetrics, bool, int]:
    import soundfile as sf

    worker.clear_queues()
    converted_chunks: list[np.ndarray] = []
    passthrough_count = 0
    for start in range(0, source.size, chunk_size):
        real_chunk = source[start:start + chunk_size]
        padded = real_chunk
        if real_chunk.size < chunk_size:
            padded = np.pad(real_chunk, (0, chunk_size - real_chunk.size)).astype(np.float32)
        require(worker.put(padded, timeout=0.0), "listening WAV put failed")
        result = worker.get(timeout=RESULT_TIMEOUT_SECONDS)
        result, passthrough = validate_result(result, padded, "listening WAV")
        passthrough_count += int(passthrough)
        converted_chunks.append(result[:real_chunk.size].copy())

    output = np.concatenate(converted_chunks).astype(np.float32, copy=False)
    require(output.shape == source.shape, "listening output length mismatch")
    require(bool(np.all(np.isfinite(output))), "listening output is non-finite")
    milliseconds = int(round(chunk_size * 1000.0 / sample_rate))
    output_path = OUTPUT_DIR / f"chunk_{milliseconds}ms.wav"
    sf.write(str(output_path), output, sample_rate, subtype="PCM_16")
    require(output_path.is_file() and output_path.stat().st_size > 44, "WAV was not written")
    metrics = compute_boundary_metrics(converted_chunks, output, sample_rate)
    output_valid = bool(
        output.size
        and np.all(np.isfinite(output))
        and metrics.output_rms > 1e-6
        and passthrough_count == 0
    )
    return output_path, metrics, output_valid, passthrough_count


def classify(serial: SerialMetrics, pressure: PressureMetrics, output_valid: bool):
    eligible = bool(
        serial.average_rtf < 0.8
        and serial.p95_rtf < 1.0
        and not pressure.sustained_backlog
        and serial.error_count == 0
        and pressure.errors == 0
        and serial.passthrough_count == 0
        and pressure.passthrough_count == 0
        and output_valid
    )
    if eligible:
        return True, "recommended for overlap/crossfade listening experiment"
    reasons: list[str] = []
    if serial.average_rtf >= 0.8:
        reasons.append("average RTF >= 0.8")
    if serial.p95_rtf >= 1.0:
        reasons.append("P95 RTF >= 1.0")
    if pressure.sustained_backlog:
        reasons.append("pressure backlog/drop")
    if serial.error_count or pressure.errors:
        reasons.append("worker errors")
    if serial.passthrough_count or pressure.passthrough_count or not output_valid:
        reasons.append("invalid or suspected passthrough output")
    return False, "; ".join(reasons) or "not eligible"


def benchmark_size(
    worker,
    audio: np.ndarray,
    listening_source: np.ndarray,
    chunk_size: int,
    sample_rate: int,
) -> ChunkMetrics:
    duration = chunk_size / sample_rate
    section(f"Chunk {chunk_size} samples ({duration:.6f}s)")
    chunks, starts, rms_values = select_distinct_chunks(
        audio,
        chunk_size,
        STRESS_COUNT,
        sample_rate,
    )

    log("\n  Warmup")
    warmup_started = time.perf_counter()
    require(worker.put(chunks[0], timeout=0.0), "warmup put failed")
    warmup_result = worker.get(timeout=RESULT_TIMEOUT_SECONDS)
    warmup_seconds = time.perf_counter() - warmup_started
    _, warmup_passthrough = validate_result(warmup_result, chunks[0], "warmup")
    worker.clear_queues()
    require(worker.input_queue_size == 0, "warmup input queue was not cleared")
    require(worker.output_queue_size == 0, "warmup output queue was not cleared")
    log(f"    round-trip: {warmup_seconds:.3f}s")
    log(f"    worker infer: {worker.last_infer_ms / 1000.0:.3f}s")
    log(f"    suspected passthrough: {warmup_passthrough}")

    log("\n  Steady-state serial")
    serial = run_serial(worker, chunks, duration)
    log(
        f"    avg={serial.average_seconds:.3f}s median={serial.median_seconds:.3f}s "
        f"P95={serial.p95_seconds:.3f}s"
    )
    log(f"    avg RTF={serial.average_rtf:.3f} P95 RTF={serial.p95_rtf:.3f}")
    log(
        f"    infer={serial.infer_count} errors={serial.error_count} "
        f"passthrough={serial.passthrough_count}"
    )

    log(f"\n  Real-rate pressure ({duration:.6f}s cadence)")
    pressure = run_pressure(worker, chunks, duration)
    log(
        f"    attempted={pressure.attempted} accepted={pressure.accepted} "
        f"completed={pressure.completed} errors={pressure.errors}"
    )
    log(
        f"    input drops={pressure.input_drops} output drops={pressure.output_drops} "
        f"max queues={pressure.max_input_queue}/{pressure.max_output_queue}"
    )
    log(f"    completion timeline: {[round(value, 3) for value in pressure.completion_timeline]}")
    log(f"    last result vs last submit: {pressure.last_result_vs_last_submit:.3f}s")
    log(f"    sustained backlog: {pressure.sustained_backlog}")

    log("\n  Listening WAV (fixed non-overlap concatenation)")
    wav_path, boundary, output_valid, wav_passthrough = generate_listening_wav(
        worker,
        listening_source,
        chunk_size,
        sample_rate,
    )
    if wav_passthrough:
        output_valid = False
    log(f"    path: {wav_path}")
    log(f"    suspected passthrough chunks: {wav_passthrough}")
    log(
        f"    boundary jump avg/max={boundary.average_jump:.6f}/{boundary.maximum_jump:.6f} "
        f"RMS delta avg/max={boundary.average_rms_delta:.6f}/{boundary.maximum_rms_delta:.6f}"
    )

    estimated = duration + serial.median_seconds
    conservative = duration + serial.p95_seconds
    eligible, status = classify(serial, pressure, output_valid)
    log(f"\n  Estimated first audio: {estimated:.3f}s")
    log(f"  Conservative estimate: {conservative:.3f}s")
    log(f"  Status: {status}")
    return ChunkMetrics(
        chunk_size=chunk_size,
        duration_seconds=duration,
        selected_starts_seconds=[start / sample_rate for start in starts],
        selected_rms=rms_values,
        warmup_seconds=warmup_seconds,
        warmup_passthrough=warmup_passthrough,
        serial=serial,
        pressure=pressure,
        estimated_first_audio=estimated,
        conservative_first_audio=conservative,
        wav_path=str(wav_path),
        boundary=boundary,
        output_valid=output_valid,
        performance_eligible=eligible,
        status=status,
    )


def print_summary(results: list[ChunkMetrics]) -> None:
    section("SUMMARY")
    log(
        "Chunk | Duration | Avg infer | P95 infer | Avg RTF | P95 RTF | "
        "Drops | Est. first | Status"
    )
    log("-" * 118)
    for result in results:
        drops = result.pressure.input_drops + result.pressure.output_drops
        status = "ELIGIBLE" if result.performance_eligible else "NOT ELIGIBLE"
        log(
            f"{result.chunk_size:5d} | {result.duration_seconds:8.3f} | "
            f"{result.serial.average_seconds:9.3f} | {result.serial.p95_seconds:9.3f} | "
            f"{result.serial.average_rtf:7.3f} | {result.serial.p95_rtf:7.3f} | "
            f"{drops:5d} | {result.estimated_first_audio:10.3f} | {status}"
        )

    eligible = [result for result in results if result.performance_eligible]
    performance_candidate = min(eligible, key=lambda item: item.duration_seconds) if eligible else None
    safer_pool = [result for result in eligible if result.duration_seconds >= 0.5]
    safer_candidate = (
        min(safer_pool, key=lambda item: item.duration_seconds)
        if safer_pool
        else (max(eligible, key=lambda item: item.duration_seconds) if eligible else None)
    )
    log("")
    log(
        "Performance candidate: "
        + (
            f"{performance_candidate.duration_seconds:.3f}s"
            if performance_candidate
            else "none"
        )
    )
    log(
        "Safer candidate: "
        + (f"{safer_candidate.duration_seconds:.3f}s" if safer_candidate else "none")
    )
    for result in results:
        if not result.performance_eligible:
            log(f"Not recommended {result.duration_seconds:.3f}s: {result.status}")
    log("Final selection still requires listening to the generated WAV files.")


def write_json_report(
    results: list[ChunkMetrics],
    metadata: dict,
) -> Path:
    report_path = OUTPUT_DIR / "benchmark_results.json"
    payload = dict(metadata)
    payload["results"] = [asdict(result) for result in results]
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    from loguru import logger

    from ai.rvc_engine import RVCEngine
    from ai.rvc_worker import RVCWorker
    from config.settings import (
        RVC_F0_METHOD,
        RVC_INDEX_RATE,
        RVC_INPUT_QUEUE_SIZE,
        RVC_MODELS_DIR,
        RVC_PITCH_SHIFT,
        RVC_PROTECT,
        RVC_RMS_MIX_RATE,
        RVC_SOURCE_DIR,
        RVC_VOICE_DIR,
        SAMPLE_RATE,
    )

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    output_created = False
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
    worker = None
    results: list[ChunkMetrics] = []
    model_load_seconds = 0.0
    try:
        audio, source_rate = load_real_audio(SAMPLE_RATE)
        listening_source, listening_start = select_listening_segment(
            audio,
            SAMPLE_RATE,
            LISTEN_SECONDS,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_created = True

        voice_dir = Path(RVC_VOICE_DIR)
        pth_files = sorted(voice_dir.glob("*.pth"))
        index_files = sorted(voice_dir.glob("*.index"))
        require(pth_files, f"no .pth model in {voice_dir}")
        section("CONFIGURED MODEL AND INPUT")
        log(f"  voice directory: {voice_dir}")
        log(f"  model: {pth_files[0].name}")
        log(f"  index: {index_files[0].name if index_files else '(none)'}")
        log(
            f"  input: tests/assets/input.wav source_sr={source_rate} "
            f"benchmark_sr={SAMPLE_RATE} duration={audio.size / SAMPLE_RATE:.3f}s"
        )
        log(
            f"  listening segment: {listening_start / SAMPLE_RATE:.3f}s.."
            f"{(listening_start + listening_source.size) / SAMPLE_RATE:.3f}s"
        )

        section("ONE-TIME MODEL LOAD")
        load_started = time.perf_counter()
        engine.load_model()
        model_load_seconds = time.perf_counter() - load_started
        require(engine.is_loaded, "engine did not load")
        log(f"  load time: {model_load_seconds:.3f}s")

        for chunk_size in CHUNK_SIZES:
            require(worker is None, "previous worker reference was not cleared")
            worker = RVCWorker(
                engine,
                chunk_size=chunk_size,
                max_queue_size=RVC_INPUT_QUEUE_SIZE,
            )
            require(worker.start(), f"worker start failed for {chunk_size}")
            try:
                result = benchmark_size(
                    worker,
                    audio,
                    listening_source,
                    chunk_size,
                    SAMPLE_RATE,
                )
                results.append(result)
                metadata = {
                    "voice_dir": str(voice_dir),
                    "model": pth_files[0].name,
                    "index": index_files[0].name if index_files else None,
                    "sample_rate": SAMPLE_RATE,
                    "source_rate": source_rate,
                    "input_seconds": audio.size / SAMPLE_RATE,
                    "listening_start_seconds": listening_start / SAMPLE_RATE,
                    "listening_seconds": listening_source.size / SAMPLE_RATE,
                    "model_load_seconds": model_load_seconds,
                }
                write_json_report(results, metadata)
            finally:
                stopped = worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
                require(stopped, f"worker stop timed out for {chunk_size}")
                require(not worker.thread_alive, f"worker still alive for {chunk_size}")
                worker = None

        print_summary(results)
        report_path = write_json_report(
            results,
            {
                "voice_dir": str(voice_dir),
                "model": pth_files[0].name,
                "index": index_files[0].name if index_files else None,
                "sample_rate": SAMPLE_RATE,
                "source_rate": source_rate,
                "input_seconds": audio.size / SAMPLE_RATE,
                "listening_start_seconds": listening_start / SAMPLE_RATE,
                "listening_seconds": listening_source.size / SAMPLE_RATE,
                "model_load_seconds": model_load_seconds,
            },
        )
        log(f"\nMachine-readable report: {report_path}")
        log("Boundary metrics are relative indicators, not perceptual quality scores.")
        log("FIRST-AUDIO ESTIMATES EXCLUDE Windows monitoring, sounddevice and VB-CABLE buffers.")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if worker is not None and worker.thread_alive:
            worker.stop(timeout=WORKER_STOP_TIMEOUT_SECONDS)
        if engine.is_loaded:
            if worker is None or not worker.thread_alive:
                engine.unload_model()
            else:
                log("WARNING: live worker retained; engine intentionally not unloaded")
        if output_created:
            log(f"Generated artifacts (gitignored): {OUTPUT_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())

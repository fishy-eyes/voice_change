"""Offline RVC chunk/overlap listening experiment.

This is deliberately isolated from AudioStream, AIVoiceEffect and production
RVC inference code. One RVCEngine is loaded once, while each case uses the
existing RVCWorker. Overlapped results are assembled with linear overlap-add
crossfades and trimmed to the exact input length.

Default run covers the requested priority cases. Pass ``--all-combinations``
to run the full Cartesian product of chunk and overlap values.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from tests.test_rvc_realtime_benchmark import (
    build_overlapping_windows,
    milliseconds_to_samples,
    run_realtime_phase,
)
from tests.test_rvc_short_chunk_benchmark import (
    select_distinct_chunks,
    validate_result,
)


DEFAULT_CHUNK_MS = (200, 325, 500, 700)
DEFAULT_OVERLAP_MS = (0, 25, 50, 100)
PRIORITY_CASES = (
    (325, 0),
    (325, 50),
    (325, 100),
    (500, 0),
    (500, 50),
    (500, 100),
    (700, 0),
    (700, 100),
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "rvc_chunk_experiment"
DEFAULT_TIMEOUT = 120.0


def log(message: str = "") -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_cases(
    chunk_values: list[int],
    overlap_values: list[int],
    all_combinations: bool,
) -> list[tuple[int, int]]:
    require(bool(chunk_values), "at least one chunk value is required")
    require(bool(overlap_values), "at least one overlap value is required")
    require(all(value > 0 for value in chunk_values), "chunk_ms must be positive")
    require(all(value >= 0 for value in overlap_values), "overlap_ms must be non-negative")
    allowed = {
        (int(chunk), int(overlap))
        for chunk in chunk_values
        for overlap in overlap_values
        if overlap < chunk
    }
    require(bool(allowed), "no valid chunk/overlap combinations")
    if all_combinations:
        return sorted(allowed)
    priority = [case for case in PRIORITY_CASES if case in allowed]
    return priority if priority else sorted(allowed)


def load_audio(
    path: Path,
    sample_rate: int,
    duration_seconds: float,
) -> tuple[np.ndarray, dict]:
    """Load WAV through soundfile and compressed formats through librosa."""

    require(path.is_file(), f"input audio not found: {path}")
    source_rate: int
    source_channels: int
    decoder: str
    try:
        import soundfile as sf

        values, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        source_channels = int(values.shape[1])
        mono = np.mean(values, axis=1, dtype=np.float32)
        decoder = "soundfile"
    except Exception:
        import librosa

        values, source_rate = librosa.load(
            str(path),
            sr=None,
            mono=False,
            dtype=np.float32,
        )
        source_channels = 1 if values.ndim == 1 else int(values.shape[0])
        mono = values if values.ndim == 1 else np.mean(values, axis=0, dtype=np.float32)
        decoder = "librosa"

    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    source_samples = int(mono.size)
    if source_rate != sample_rate:
        import librosa

        mono = librosa.resample(
            mono,
            orig_sr=source_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    if duration_seconds > 0:
        requested = int(round(duration_seconds * sample_rate))
        require(
            mono.size >= requested,
            f"input is only {mono.size / sample_rate:.3f}s, requested {duration_seconds:.3f}s",
        )
        mono = mono[:requested]
    require(mono.size > 0, "input audio is empty")
    require(bool(np.all(np.isfinite(mono))), "input contains NaN/Inf")
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono * mono)))
    require(peak > 1e-6 and rms > 1e-7, "input audio is silent")
    return mono, {
        "path": str(path.resolve()),
        "decoder": decoder,
        "source_sample_rate": int(source_rate),
        "source_channels": source_channels,
        "source_samples": source_samples,
        "source_duration_seconds": source_samples / source_rate,
        "experiment_sample_rate": sample_rate,
        "experiment_samples": int(mono.size),
        "experiment_duration_seconds": mono.size / sample_rate,
        "dtype": str(mono.dtype),
        "rms": rms,
        "peak": peak,
    }


def linear_overlap_add(
    results: list[np.ndarray],
    starts: list[int],
    output_samples: int,
    overlap_samples: int,
) -> np.ndarray:
    """Assemble fixed windows with complementary linear crossfade weights."""

    require(bool(results), "no inference results to assemble")
    require(len(results) == len(starts), "result/start count mismatch")
    require(output_samples > 0, "output_samples must be positive")
    chunk_samples = int(np.asarray(results[0]).size)
    require(chunk_samples > 0, "empty result chunk")
    require(0 <= overlap_samples < chunk_samples, "invalid overlap_samples")
    total_samples = max(output_samples, starts[-1] + chunk_samples)
    accumulator = np.zeros(total_samples, dtype=np.float64)
    weights = np.zeros(total_samples, dtype=np.float64)
    if overlap_samples:
        fade_in = np.linspace(0.0, 1.0, overlap_samples, endpoint=False, dtype=np.float64)
        fade_out = 1.0 - fade_in
    else:
        fade_in = fade_out = np.empty(0, dtype=np.float64)

    for index, (result, start) in enumerate(zip(results, starts)):
        chunk = np.asarray(result, dtype=np.float32).reshape(-1)
        require(chunk.size == chunk_samples, f"chunk {index}: inconsistent shape")
        require(bool(np.all(np.isfinite(chunk))), f"chunk {index}: non-finite values")
        window = np.ones(chunk_samples, dtype=np.float64)
        if overlap_samples and index > 0:
            window[:overlap_samples] = fade_in
        if overlap_samples and index + 1 < len(results):
            window[-overlap_samples:] = fade_out
        end = start + chunk_samples
        accumulator[start:end] += chunk.astype(np.float64) * window
        weights[start:end] += window

    require(bool(np.all(weights[:output_samples] > 0.0)), "OLA left uncovered samples")
    assembled = accumulator[:output_samples] / weights[:output_samples]
    return np.clip(assembled, -1.0, 1.0).astype(np.float32)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _maximum_silent_run(values: np.ndarray, threshold: float) -> int:
    longest = 0
    current = 0
    for silent in np.abs(values) <= threshold:
        if silent:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def continuity_metrics(
    audio: np.ndarray,
    starts: list[int],
    overlap_samples: int,
    sample_rate: int,
) -> dict:
    """Measure jumps, local RMS discontinuity and silent runs at joins."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    analysis_window = max(1, milliseconds_to_samples(20.0, sample_rate))
    events: list[tuple[int, str]] = [(start, "overlap_start") for start in starts[1:]]
    if overlap_samples:
        events.extend((start + overlap_samples, "overlap_end") for start in starts[1:])
    unique_events = sorted({event for event in events if 0 < event[0] < values.size})
    global_rms = _rms(values)
    silence_threshold = max(1e-5, global_rms * 0.03)
    details: list[dict] = []
    for position, kind in unique_events:
        left = max(0, position - analysis_window)
        right = min(values.size, position + analysis_window)
        pre = values[left:position]
        post = values[position:right]
        if pre.size == 0 or post.size == 0:
            continue
        jump = abs(float(values[position]) - float(values[position - 1]))
        local = values[left:right]
        local_diffs = np.abs(np.diff(local.astype(np.float64)))
        boundary_offset = position - left - 1
        if 0 <= boundary_offset < local_diffs.size:
            local_diffs = np.delete(local_diffs, boundary_offset)
        reference = float(np.percentile(local_diffs, 75)) if local_diffs.size else 0.0
        normalized_jump = jump / max(reference, 1e-7)
        pre_rms = _rms(pre)
        post_rms = _rms(post)
        rms_delta_db = 20.0 * math.log10((post_rms + 1e-9) / (pre_rms + 1e-9))
        silent_run = _maximum_silent_run(local, silence_threshold)
        details.append(
            {
                "sample": position,
                "seconds": position / sample_rate,
                "kind": kind,
                "jump": jump,
                "local_p75_derivative": reference,
                "normalized_jump": normalized_jump,
                "pre_rms": pre_rms,
                "post_rms": post_rms,
                "rms_delta_db": rms_delta_db,
                "absolute_rms_delta_db": abs(rms_delta_db),
                "maximum_silent_run_ms": silent_run * 1000.0 / sample_rate,
            }
        )
    normalized = [item["normalized_jump"] for item in details]
    rms_deltas = [item["absolute_rms_delta_db"] for item in details]
    silent_runs = [item["maximum_silent_run_ms"] for item in details]
    jumps = [item["jump"] for item in details]
    return {
        "boundary_event_count": len(details),
        "analysis_window_ms": analysis_window * 1000.0 / sample_rate,
        "silence_threshold": silence_threshold,
        "average_jump": statistics.fmean(jumps) if jumps else 0.0,
        "maximum_jump": max(jumps, default=0.0),
        "average_normalized_jump": statistics.fmean(normalized) if normalized else 0.0,
        "p95_normalized_jump": (
            float(np.percentile(normalized, 95)) if normalized else 0.0
        ),
        "maximum_normalized_jump": max(normalized, default=0.0),
        "average_absolute_rms_delta_db": statistics.fmean(rms_deltas) if rms_deltas else 0.0,
        "maximum_absolute_rms_delta_db": max(rms_deltas, default=0.0),
        "maximum_silent_run_ms": max(silent_runs, default=0.0),
        "obvious_silence_boundary_count": sum(value >= 8.0 for value in silent_runs),
        "details": details,
    }


def performance_summary(seconds: list[float], chunk_seconds: float, hop_seconds: float) -> dict:
    require(bool(seconds), "no inference timings")
    ordered = sorted(seconds)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    average = statistics.fmean(seconds)
    return {
        "inference_count": len(seconds),
        "total_inference_seconds": sum(seconds),
        "average_inference_ms": average * 1000.0,
        "minimum_inference_ms": min(seconds) * 1000.0,
        "maximum_inference_ms": max(seconds) * 1000.0,
        "p95_inference_ms": ordered[p95_index] * 1000.0,
        "rtf_vs_chunk": average / chunk_seconds,
        "rtf_vs_hop_cadence": average / hop_seconds,
    }


def infer_ordered(worker, windows: list[np.ndarray], timeout: float) -> tuple[list[np.ndarray], dict]:
    baseline_errors = worker.error_count
    baseline_input_drops = worker.input_drop_count
    baseline_output_drops = worker.output_drop_count
    results: list[np.ndarray] = []
    timings: list[float] = []
    passthrough_count = 0
    non_silent_passthrough_count = 0
    for index, window in enumerate(windows):
        started = time.perf_counter()
        require(worker.put(window, timeout=0.0), f"window {index}: submit failed")
        result = worker.get(timeout=timeout)
        timings.append(time.perf_counter() - started)
        converted, passthrough = validate_result(result, window, f"window {index}")
        if passthrough:
            passthrough_count += 1
            if _rms(window) > 1e-4:
                non_silent_passthrough_count += 1
        results.append(converted)
    return results, {
        "timings_seconds": timings,
        "worker_errors": worker.error_count - baseline_errors,
        "input_drops": worker.input_drop_count - baseline_input_drops,
        "output_drops": worker.output_drop_count - baseline_output_drops,
        "passthrough_count": passthrough_count,
        "non_silent_passthrough_count": non_silent_passthrough_count,
    }


def verify_wav(path: Path, expected_samples: int, sample_rate: int) -> dict:
    import soundfile as sf

    values, actual_rate = sf.read(str(path), dtype="float32", always_2d=True)
    require(actual_rate == sample_rate, f"{path.name}: sample-rate mismatch")
    require(values.shape == (expected_samples, 1), f"{path.name}: shape mismatch {values.shape}")
    require(bool(np.all(np.isfinite(values))), f"{path.name}: non-finite samples")
    require(float(np.max(np.abs(values))) > 1e-6, f"{path.name}: silent output")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "samples": int(values.shape[0]),
        "channels": int(values.shape[1]),
        "sample_rate": int(actual_rate),
        "duration_seconds": values.shape[0] / actual_rate,
        "dtype_after_read": str(values.dtype),
    }


def run_case(
    engine,
    audio: np.ndarray,
    sample_rate: int,
    chunk_ms: int,
    overlap_ms: int,
    queue_size: int,
    timeout: float,
    output_dir: Path,
    simulate_realtime: bool,
) -> dict:
    from ai.rvc_worker import RVCWorker
    import soundfile as sf

    chunk_samples = milliseconds_to_samples(chunk_ms, sample_rate)
    overlap_samples = milliseconds_to_samples(overlap_ms, sample_rate)
    require(overlap_samples < chunk_samples, "overlap must be smaller than chunk")
    hop_samples = chunk_samples - overlap_samples
    chunk_seconds = chunk_samples / sample_rate
    hop_seconds = hop_samples / sample_rate
    windows, starts = build_overlapping_windows(audio, chunk_samples, overlap_samples)
    warmup_chunks, _, _ = select_distinct_chunks(audio, chunk_samples, 1, sample_rate)
    worker = RVCWorker(engine, chunk_size=chunk_samples, max_queue_size=queue_size)
    require(worker.start(), f"worker failed to start for {chunk_ms}/{overlap_ms}")
    stopped = False
    try:
        warmup_started = time.perf_counter()
        require(worker.put(warmup_chunks[0], timeout=0.0), "warmup submit failed")
        warmup_result = worker.get(timeout=timeout)
        warmup_seconds = time.perf_counter() - warmup_started
        _, warmup_passthrough = validate_result(warmup_result, warmup_chunks[0], "warmup")
        worker.clear_queues()

        results, ordered = infer_ordered(worker, windows, timeout)
        performance = performance_summary(ordered.pop("timings_seconds"), chunk_seconds, hop_seconds)
        stitched = linear_overlap_add(results, starts, audio.size, overlap_samples)
        require(stitched.shape == audio.shape, "stitched output length mismatch")
        require(stitched.dtype == np.float32, "stitched output dtype mismatch")
        require(bool(np.all(np.isfinite(stitched))), "stitched output contains NaN/Inf")

        output_name = f"chunk{chunk_ms}_overlap{overlap_ms}.wav"
        output_path = output_dir / output_name
        sf.write(str(output_path), stitched, sample_rate, subtype="PCM_16")
        wav = verify_wav(output_path, audio.size, sample_rate)
        continuity = continuity_metrics(stitched, starts, overlap_samples, sample_rate)
        input_continuity = continuity_metrics(audio, starts, overlap_samples, sample_rate)
        continuity["p95_normalized_jump_minus_input"] = (
            continuity["p95_normalized_jump"] - input_continuity["p95_normalized_jump"]
        )
        continuity["average_rms_delta_db_minus_input"] = (
            continuity["average_absolute_rms_delta_db"]
            - input_continuity["average_absolute_rms_delta_db"]
        )
        continuity["obvious_silence_boundary_ratio"] = (
            continuity["obvious_silence_boundary_count"]
            / max(1, continuity["boundary_event_count"])
        )
        input_continuity["obvious_silence_boundary_ratio"] = (
            input_continuity["obvious_silence_boundary_count"]
            / max(1, input_continuity["boundary_event_count"])
        )
        continuity["obvious_silence_ratio_minus_input"] = (
            continuity["obvious_silence_boundary_ratio"]
            - input_continuity["obvious_silence_boundary_ratio"]
        )

        realtime = (
            run_realtime_phase(worker, windows, hop_seconds, queue_size, timeout)
            if simulate_realtime
            else {"simulated_realtime": False, "dropped": 0, "errors": 0}
        )
        worker_errors = int(ordered["worker_errors"]) + int(realtime.get("errors", 0))
        queue_drops = (
            int(ordered["input_drops"])
            + int(ordered["output_drops"])
            + int(realtime.get("dropped", 0))
        )
        valid_audio = bool(
            worker_errors == 0
            and queue_drops == 0
            and ordered["non_silent_passthrough_count"] == 0
        )
        keeps_up = bool(
            performance["rtf_vs_hop_cadence"] <= 1.0
            and worker_errors == 0
            and int(realtime.get("dropped", 0)) == 0
            and not bool(realtime.get("sustained_backlog", False))
        )
        return {
            "chunk_ms": chunk_ms,
            "chunk_samples": chunk_samples,
            "overlap_ms": overlap_ms,
            "overlap_samples": overlap_samples,
            "hop_ms": hop_samples * 1000.0 / sample_rate,
            "hop_samples": hop_samples,
            "window_count": len(windows),
            "warmup_ms": warmup_seconds * 1000.0,
            "warmup_passthrough": warmup_passthrough,
            "stitch_mode": "linear_crossfade" if overlap_samples else "direct_concat",
            "performance": performance,
            "realtime": realtime,
            "stability": {
                **ordered,
                "worker_errors": worker_errors,
                "queue_drops": queue_drops,
                "valid_audio": valid_audio,
                "worker_running_before_stop": worker.is_running,
            },
            "samples": {
                "input": int(audio.size),
                "output": int(stitched.size),
                "difference": int(stitched.size - audio.size),
            },
            "output": {
                "rms": _rms(stitched),
                "peak": float(np.max(np.abs(stitched))),
                "clipping_ratio": float(np.mean(np.abs(stitched) >= 0.999)),
                "finite": bool(np.all(np.isfinite(stitched))),
                "silent": bool(_rms(stitched) <= 1e-7),
                **wav,
            },
            "continuity": continuity,
            "input_continuity_reference": input_continuity,
            "keeps_up": keeps_up,
        }
    finally:
        stopped = worker.stop(timeout=timeout)
        require(stopped and not worker.thread_alive, f"worker did not stop: {chunk_ms}/{overlap_ms}")


def summarize_results(results: list[dict]) -> dict:
    eligible = [
        result
        for result in results
        if result["keeps_up"] and result["stability"]["valid_audio"]
    ]
    performance_best = (
        min(eligible, key=lambda item: item["performance"]["average_inference_ms"])
        if eligible
        else None
    )
    boundary_ranked = sorted(
        eligible,
        key=lambda item: (
            item["continuity"]["p95_normalized_jump"],
            item["continuity"]["average_absolute_rms_delta_db"],
            item["continuity"]["obvious_silence_ratio_minus_input"],
        ),
    )
    lowest_rms = min(
        eligible,
        key=lambda item: item["continuity"]["average_rms_delta_db_minus_input"],
        default=None,
    )
    return {
        "performance_candidate": (
            {
                "chunk_ms": performance_best["chunk_ms"],
                "overlap_ms": performance_best["overlap_ms"],
                "average_inference_ms": performance_best["performance"]["average_inference_ms"],
                "rtf_vs_hop_cadence": performance_best["performance"]["rtf_vs_hop_cadence"],
            }
            if performance_best
            else None
        ),
        "lowest_rms_discontinuity_candidate": (
            {
                "chunk_ms": lowest_rms["chunk_ms"],
                "overlap_ms": lowest_rms["overlap_ms"],
                "average_rms_delta_db_minus_input": lowest_rms["continuity"]["average_rms_delta_db_minus_input"],
                "path": lowest_rms["output"]["path"],
            }
            if lowest_rms
            else None
        ),
        "continuity_candidates_for_listening": [
            {
                "chunk_ms": item["chunk_ms"],
                "overlap_ms": item["overlap_ms"],
                "p95_normalized_jump": item["continuity"]["p95_normalized_jump"],
                "average_absolute_rms_delta_db": item["continuity"]["average_absolute_rms_delta_db"],
                "maximum_silent_run_ms": item["continuity"]["maximum_silent_run_ms"],
                "path": item["output"]["path"],
            }
            for item in boundary_ranked
        ],
        "not_recommended": [
            {
                "chunk_ms": item["chunk_ms"],
                "overlap_ms": item["overlap_ms"],
                "keeps_up": item["keeps_up"],
                "worker_errors": item["stability"]["worker_errors"],
                "queue_drops": item["stability"]["queue_drops"],
                "non_silent_passthrough_count": item["stability"]["non_silent_passthrough_count"],
            }
            for item in results
            if item not in eligible
        ],
        "selection_note": (
            "Continuity metrics only rank technical boundary artifacts. "
            "Final speech clarity and swallowed-phoneme judgment requires listening."
        ),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    from config.settings import (
        RVC_INPUT_QUEUE_SIZE,
        RVC_MODEL_LIBRARY_DIR,
        RVC_MODELS_DIR,
        RVC_SOURCE_DIR,
        SAMPLE_RATE,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="modelF")
    parser.add_argument("--model-root", type=Path, default=Path(RVC_MODEL_LIBRARY_DIR))
    parser.add_argument("--source-dir", type=Path, default=Path(RVC_SOURCE_DIR))
    parser.add_argument("--backend-models-dir", type=Path, default=Path(RVC_MODELS_DIR))
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "tests" / "assets" / "input.wav")
    parser.add_argument("--input-duration", type=float, default=0.0, help="0 uses the full input")
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=list(DEFAULT_CHUNK_MS))
    parser.add_argument("--overlap-ms", type=int, nargs="+", default=list(DEFAULT_OVERLAP_MS))
    parser.add_argument("--all-combinations", action="store_true")
    parser.add_argument("--skip-realtime", action="store_true")
    parser.add_argument("--queue-size", type=int, default=RVC_INPUT_QUEUE_SIZE)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    from ai.rvc_engine import RVCEngine
    from core.rvc_model_manager import RVCModelManager

    args = parse_args(argv)
    require(args.sample_rate > 0, "sample rate must be positive")
    require(args.input_duration >= 0, "input duration must be non-negative")
    require(args.queue_size > 0 and args.timeout > 0, "queue size/timeouts must be positive")
    cases = build_cases(args.chunk_ms, args.overlap_ms, args.all_combinations)
    audio, input_metadata = load_audio(args.input, args.sample_rate, args.input_duration)
    descriptor = RVCModelManager(args.model_root).get_model(args.model)
    engine = RVCEngine.from_profile(
        descriptor.profile,
        source_dir=args.source_dir,
        models_dir=args.backend_models_dir,
        sample_rate=args.sample_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "experiment_results.json"
    results: list[dict] = []
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": descriptor.name,
        "model_paths": {
            "pth": str(descriptor.pth_path),
            "index": str(descriptor.index_path) if descriptor.index_path else None,
            "profile": str(descriptor.profile_path),
        },
        "config": descriptor.profile.inference.to_dict(),
        "input": input_metadata,
        "strategy": {
            "fixed_shape_windows": True,
            "zero_pad_final_window": True,
            "trim_to_input_length": True,
            "overlap_stitch": "complementary linear overlap-add crossfade",
            "production_audio_path_modified": False,
            "simulated_realtime": not args.skip_realtime,
        },
        "requested_cases": [
            {"chunk_ms": chunk, "overlap_ms": overlap}
            for chunk, overlap in cases
        ],
        "results": results,
    }
    try:
        load_started = time.perf_counter()
        engine.load_model()
        report["model_load_seconds"] = time.perf_counter() - load_started
        require(engine.is_loaded, "RVC engine did not load")
        for index, (chunk_ms, overlap_ms) in enumerate(cases, start=1):
            log(f"\n[{index}/{len(cases)}] chunk={chunk_ms}ms overlap={overlap_ms}ms")
            case = run_case(
                engine,
                audio,
                args.sample_rate,
                chunk_ms,
                overlap_ms,
                args.queue_size,
                args.timeout,
                args.output_dir,
                not args.skip_realtime,
            )
            results.append(case)
            report["summary"] = summarize_results(results)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            log(
                f"  avg={case['performance']['average_inference_ms']:.1f}ms "
                f"chunk_rtf={case['performance']['rtf_vs_chunk']:.3f} "
                f"hop_rtf={case['performance']['rtf_vs_hop_cadence']:.3f} "
                f"drops={case['stability']['queue_drops']} "
                f"p95_jump={case['continuity']['p95_normalized_jump']:.2f}"
            )
            log(f"  WAV: {case['output']['path']}")
        report["summary"] = summarize_results(results)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        log(f"\nReport: {report_path.resolve()}")
        log("Final ranking requires listening to the generated WAV files.")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if engine.is_loaded:
            engine.unload_model()


if __name__ == "__main__":
    raise SystemExit(main())

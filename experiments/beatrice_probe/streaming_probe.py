"""Simulate the production 48 kHz / 256-sample callback contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from benchmark import bytes_to_mb, current_rss_bytes, gpu_snapshot
from runtime_support import (
    RESULTS_DIR,
    audio_stats,
    ensure_output_path,
    read_mono_audio,
    resample_audio,
    stats_dict,
)
from streaming_adapter import BeatriceStreamingAdapter


SAMPLE_RATE = 48_000
CALLBACK_SAMPLES = 256


def resampler_mode_probe() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for quality in ("QQ", "LQ", "MQ", "HQ", "VHQ"):
        down = soxr.ResampleStream(
            48_000, 16_000, 1, dtype="float32", quality=quality
        )
        down_sizes = [
            int(down.resample_chunk(np.zeros(256, dtype=np.float32)).size)
            for _ in range(20)
        ]
        up = soxr.ResampleStream(
            24_000, 48_000, 1, dtype="float32", quality=quality
        )
        up_sizes = [
            int(up.resample_chunk(np.zeros(240, dtype=np.float32)).size)
            for _ in range(10)
        ]
        report[quality] = {
            "downsample_output_sizes_for_20_callbacks": down_sizes,
            "downsample_current_delay_samples_at_16khz": float(down.delay()),
            "upsample_output_sizes_for_10_model_blocks": up_sizes,
            "upsample_current_delay_samples_at_48khz": float(up.delay()),
        }
    return report


def pad_to_callback(audio: np.ndarray) -> tuple[np.ndarray, int]:
    padded_size = math.ceil(audio.size / CALLBACK_SAMPLES) * CALLBACK_SAMPLES
    padding = padded_size - audio.size
    return np.pad(audio, (0, padding)).astype(np.float32), int(padding)


def cyclic_block(source: np.ndarray, offset: int, count: int) -> tuple[np.ndarray, int]:
    output = np.empty(count, dtype=np.float32)
    written = 0
    current = offset
    while written < count:
        take = min(count - written, source.size - current)
        output[written : written + take] = source[current : current + take]
        written += take
        current = (current + take) % source.size
    return output, current


def run_short_probe(
    adapter: BeatriceStreamingAdapter,
    input_48khz: np.ndarray,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    padded_input, padding = pad_to_callback(input_48khz)
    outputs: list[np.ndarray] = []
    shape_failures = 0
    dtype_failures = 0
    nonfinite_blocks = 0
    started = perf_counter()
    for start in range(0, padded_input.size, CALLBACK_SAMPLES):
        output = adapter.process(padded_input[start : start + CALLBACK_SAMPLES])
        shape_failures += output.shape != (CALLBACK_SAMPLES,)
        dtype_failures += output.dtype != np.float32
        nonfinite_blocks += not bool(np.isfinite(output).all())
        outputs.append(output)
    wall_seconds = perf_counter() - started
    rendered = np.concatenate(outputs)
    sf.write(output_path, rendered, SAMPLE_RATE, subtype="PCM_16")
    saved, saved_rate = read_mono_audio(output_path)
    adapter_stats = adapter.stats()
    output_metrics = audio_stats(saved, saved_rate)
    validation = {
        "shape_failures": int(shape_failures),
        "dtype_failures": int(dtype_failures),
        "nonfinite_blocks": int(nonfinite_blocks),
        "all_zero": bool(np.all(rendered == 0.0)),
        "callback_output_samples": int(rendered.size),
        "padded_input_samples": int(padded_input.size),
        "input_padding_samples": padding,
        "duration_difference_ms_vs_original": 1000.0
        * (rendered.size - input_48khz.size)
        / SAMPLE_RATE,
        "wav_saved_and_readable": output_path.is_file()
        and saved_rate == SAMPLE_RATE
        and saved.size == rendered.size,
        "input_fifo_bounded": adapter_stats["buffer"][
            "input_fifo_max_samples_at_16khz"
        ]
        < adapter.max_fifo_samples,
        "output_fifo_bounded": adapter_stats["buffer"][
            "output_fifo_max_samples_at_48khz"
        ]
        < adapter.max_fifo_samples,
        "no_uncontrolled_starvation": adapter_stats["buffer"][
            "underflow_count_after_start"
        ]
        == 0,
        "no_overflow": adapter_stats["buffer"]["overflow_count"] == 0,
        "no_dropped_samples": adapter_stats["buffer"]["dropped_samples"] == 0,
        "resampler_drift_below_one_sample": abs(
            adapter_stats["sample_accounting"]["output_resampler_sample_drift"]
        )
        < 1.0,
    }
    if (
        shape_failures
        or dtype_failures
        or nonfinite_blocks
        or validation["all_zero"]
        or not all(
            value
            for key, value in validation.items()
            if isinstance(value, bool) and key != "all_zero"
        )
    ):
        raise RuntimeError(f"Streaming callback validation failed: {validation}")
    report = {
        "input_original_samples_at_48khz": int(input_48khz.size),
        "input_padded_samples_at_48khz": int(padded_input.size),
        "input_padding_samples": padding,
        "callback_count": int(len(outputs)),
        "wall_seconds": wall_seconds,
        "audio_seconds": padded_input.size / SAMPLE_RATE,
        "adapter_wall_rtf": wall_seconds / (padded_input.size / SAMPLE_RATE),
        "output_path": str(output_path),
        "output": stats_dict(output_metrics),
        "validation": validation,
        "adapter": adapter_stats,
    }
    return report, adapter_stats


def run_stress_probe(
    adapter: BeatriceStreamingAdapter,
    source_48khz: np.ndarray,
    duration_seconds: float,
) -> dict[str, Any]:
    requested_samples = round(duration_seconds * SAMPLE_RATE)
    block_count = math.ceil(requested_samples / CALLBACK_SAMPLES)
    actual_samples = block_count * CALLBACK_SAMPLES
    offset = 0
    sum_squares = 0.0
    peak = 0.0
    nonfinite_blocks = 0
    shape_failures = 0
    dtype_failures = 0
    progress_interval = max(1, round(60 * SAMPLE_RATE / CALLBACK_SAMPLES))
    rss_before = current_rss_bytes()
    gpu_before = gpu_snapshot()
    started = perf_counter()
    for index in range(block_count):
        block, offset = cyclic_block(source_48khz, offset, CALLBACK_SAMPLES)
        output = adapter.process(block)
        shape_failures += output.shape != (CALLBACK_SAMPLES,)
        dtype_failures += output.dtype != np.float32
        finite = np.isfinite(output)
        nonfinite_blocks += not bool(finite.all())
        if finite.any():
            finite_output = output[finite].astype(np.float64)
            sum_squares += float(np.dot(finite_output, finite_output))
            peak = max(peak, float(np.max(np.abs(finite_output))))
        if (index + 1) % progress_interval == 0:
            simulated = (index + 1) * CALLBACK_SAMPLES / SAMPLE_RATE
            print(f"stress_progress_simulated_seconds={simulated:.1f}", flush=True)
    wall_seconds = perf_counter() - started
    rss_after = current_rss_bytes()
    gpu_after = gpu_snapshot()
    stats = adapter.stats()
    expected_converts = stats["work"]["input_resampled_samples_at_16khz"] // 160
    validation = {
        "shape_failures": int(shape_failures),
        "dtype_failures": int(dtype_failures),
        "nonfinite_blocks": int(nonfinite_blocks),
        "all_zero": sum_squares == 0.0,
        "convert_count_matches_complete_input_frames": stats["work"]
        ["beatrice_convert_count"]
        == expected_converts,
        "no_uncontrolled_starvation": stats["buffer"][
            "underflow_count_after_start"
        ]
        == 0,
        "no_overflow": stats["buffer"]["overflow_count"] == 0,
        "no_dropped_samples": stats["buffer"]["dropped_samples"] == 0,
        "input_fifo_bounded": stats["buffer"][
            "input_fifo_max_samples_at_16khz"
        ]
        < adapter.max_fifo_samples,
        "output_fifo_bounded": stats["buffer"][
            "output_fifo_max_samples_at_48khz"
        ]
        < adapter.max_fifo_samples,
        "output_resampler_drift_below_one_sample": abs(
            stats["sample_accounting"]["output_resampler_sample_drift"]
        )
        < 1.0,
    }
    if validation["all_zero"] or not all(
        value
        for key, value in validation.items()
        if isinstance(value, bool) and key != "all_zero"
    ):
        raise RuntimeError(f"Stress validation failed: {validation}")
    audio_seconds = actual_samples / SAMPLE_RATE
    inference_total_seconds = (
        stats["timing"]["inference"]["total_ms"] / 1000.0
    )
    return {
        "requested_duration_seconds": duration_seconds,
        "actual_duration_seconds": audio_seconds,
        "input_blocks": block_count,
        "input_samples": actual_samples,
        "wall_seconds": wall_seconds,
        "adapter_wall_rtf": wall_seconds / audio_seconds,
        "beatrice_inference_rtf": inference_total_seconds / audio_seconds,
        "online_output_rms": math.sqrt(sum_squares / actual_samples),
        "online_output_peak": peak,
        "rss_before_mb": bytes_to_mb(rss_before),
        "rss_after_mb": bytes_to_mb(rss_after),
        "rss_delta_mb": (
            bytes_to_mb(rss_after - rss_before)
            if rss_before is not None and rss_after is not None
            else None
        ),
        "gpu_before_snapshot": gpu_before,
        "gpu_after_snapshot": gpu_after,
        "validation": validation,
        "adapter": stats,
    }


def lifecycle_probe(adapter: BeatriceStreamingAdapter) -> dict[str, Any]:
    generation_before = adapter.converter_generation
    started = perf_counter()
    adapter.reset()
    reset_seconds = perf_counter() - started
    reset_stats = adapter.stats()
    generation_after = adapter.converter_generation
    cleared = (
        reset_stats["work"]["input_callbacks"] == 0
        and reset_stats["buffer"]["input_fifo_current_samples_at_16khz"] == 0
        and reset_stats["buffer"]["output_fifo_current_samples_at_48khz"] == 0
    )
    adapter.close()
    adapter.close()
    rejected_process_after_close = False
    try:
        adapter.process(np.zeros(CALLBACK_SAMPLES, dtype=np.float32))
    except RuntimeError:
        rejected_process_after_close = True
    return {
        "native_reset_api_available": False,
        "strategy": "recreate converter and both ResampleStream instances",
        "generation_before": generation_before,
        "generation_after": generation_after,
        "converter_recreated": generation_after == generation_before + 1,
        "reset_seconds": reset_seconds,
        "buffers_and_metrics_cleared": cleared,
        "close_idempotent": adapter.closed,
        "process_after_close_rejected": rejected_process_after_close,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="streaming_beatrice.wav")
    parser.add_argument("--target-speaker", type=int, default=0)
    parser.add_argument("--quality", default="QQ")
    parser.add_argument("--startup-buffer-samples", type=int, default=512)
    parser.add_argument("--stress-seconds", type=float, default=600.0)
    parser.add_argument(
        "--json-out", default=str(RESULTS_DIR / "streaming_probe.json")
    )
    args = parser.parse_args()
    if args.stress_seconds < 0:
        parser.error("stress-seconds must be non-negative")

    input_path = Path(args.input).expanduser().resolve()
    source, source_rate = read_mono_audio(input_path)
    source_48khz = resample_audio(source, source_rate, SAMPLE_RATE)
    output_path = ensure_output_path(args.output)
    common = {
        "model_path": args.model,
        "runtime_root": args.runtime_root,
        "target_speaker": args.target_speaker,
        "resampler_quality": args.quality,
        "startup_buffer_samples": args.startup_buffer_samples,
    }

    short_adapter = BeatriceStreamingAdapter(
        args.model,
        args.runtime_root,
        target_speaker=args.target_speaker,
        resampler_quality=args.quality,
        startup_buffer_samples=args.startup_buffer_samples,
    )
    short_report, short_stats = run_short_probe(
        short_adapter, source_48khz, output_path
    )
    runtime_details = dict(short_adapter.runtime_details)
    lifecycle = lifecycle_probe(short_adapter)

    stress_report = None
    if args.stress_seconds:
        stress_adapter = BeatriceStreamingAdapter(
            args.model,
            args.runtime_root,
            target_speaker=args.target_speaker,
            resampler_quality=args.quality,
            startup_buffer_samples=args.startup_buffer_samples,
        )
        stress_report = run_stress_probe(
            stress_adapter, source_48khz, args.stress_seconds
        )
        stress_adapter.close()

    timing_source = stress_report["adapter"] if stress_report else short_stats
    inference_p50 = timing_source["timing"]["inference"]["p50_ms"] or 0.0
    known_resampler_ms = (
        timing_source["resampler"]["input_max_delay_ms"]
        + timing_source["resampler"]["output_max_delay_ms"]
    )
    report = {
        "configuration": {
            **common,
            "external_sample_rate": SAMPLE_RATE,
            "callback_samples": CALLBACK_SAMPLES,
            "callback_ms": 1000.0 * CALLBACK_SAMPLES / SAMPLE_RATE,
            "input_path": str(input_path),
            "input_original_sample_rate": source_rate,
            "input_original_samples": int(source.size),
            "input_48khz_samples": int(source_48khz.size),
        },
        "runtime": runtime_details,
        "resampler_mode_research": resampler_mode_probe(),
        "short_real_model": short_report,
        "stress_real_model": stress_report,
        "lifecycle": lifecycle,
        "latency_breakdown": {
            "callback_block_ms": 1000.0 * CALLBACK_SAMPLES / SAMPLE_RATE,
            "beatrice_input_accumulation_frame_ms": 10.0,
            "first_complete_frame_callback_quantized_ms": short_stats["latency"]
            ["first_model_frame_input_available_ms"],
            "input_resampler_max_delay_ms": timing_source["resampler"]
            ["input_max_delay_ms"],
            "beatrice_inference_p50_ms": inference_p50,
            "beatrice_inference_p95_ms": timing_source["timing"]["inference"]
            ["p95_ms"],
            "beatrice_inference_p99_ms": timing_source["timing"]["inference"]
            ["p99_ms"],
            "output_resampler_max_delay_ms": timing_source["resampler"]
            ["output_max_delay_ms"],
            "startup_padding_ms": short_stats["buffer"]["startup_padding_ms"],
            "first_valid_output_offset_ms": short_stats["latency"]
            ["first_valid_output_offset_ms"],
            "first_output_latency_ms": short_stats["latency"]
            ["first_output_latency_ms"],
            "steady_state_buffer_latency_ms": timing_source["latency"]
            ["steady_state_buffer_latency_ms"],
            "predicted_minimum_added_latency_ms_excluding_native_receptive_field_worker_and_device": short_stats[
                "buffer"
            ]["startup_padding_ms"]
            + known_resampler_ms
            + inference_p50,
            "note": "Computation time is not end-to-end audio latency.",
        },
        "listening_files": {
            "offline": str(
                (output_path.parent / "beatrice_jvs001.wav").resolve()
            ),
            "streaming": str(output_path),
            "human_ab_completed": False,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    json_path = Path(args.json_out).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    summary = {
        "json_out": str(json_path),
        "streaming_output": str(output_path),
        "short_validation": short_report["validation"],
        "short_timing": short_stats["timing"],
        "short_buffer": short_stats["buffer"],
        "stress": (
            {
                "duration_seconds": stress_report["actual_duration_seconds"],
                "wall_seconds": stress_report["wall_seconds"],
                "validation": stress_report["validation"],
                "timing": stress_report["adapter"]["timing"],
                "buffer": stress_report["adapter"]["buffer"],
                "sample_accounting": stress_report["adapter"]
                ["sample_accounting"],
            }
            if stress_report
            else None
        ),
        "lifecycle": lifecycle,
        "latency_breakdown": report["latency_breakdown"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

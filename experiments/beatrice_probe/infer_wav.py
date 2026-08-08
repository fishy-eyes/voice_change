"""Run real offline WAV-to-WAV conversion with the Beatrice v2 native API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from runtime_support import (
    audio_stats,
    convert_audio_blocks,
    create_converter,
    ensure_output_path,
    percentile,
    read_mono_audio,
    resample_audio,
    stats_dict,
    validate_converted_audio,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Beatrice model directory or TOML")
    parser.add_argument("--input", required=True, help="Read-only mono input WAV")
    parser.add_argument(
        "--output",
        required=True,
        help="Output filename or path beneath local_assets/beatrice/generated/beatrice_probe/outputs",
    )
    parser.add_argument("--runtime-root", help="Directory containing the beatrice package")
    parser.add_argument("--target-speaker", type=int, default=0)
    parser.add_argument("--formant-shift", type=float, default=0.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--min-source-pitch", type=float, default=30.0)
    parser.add_argument("--max-source-pitch", type=float, default=1100.0)
    parser.add_argument("--vq-neighbors", type=int, default=4)
    parser.add_argument("--json-out", help="Optional JSON report path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = ensure_output_path(args.output)
    input_audio, input_sample_rate = read_mono_audio(input_path)
    input_metrics = audio_stats(input_audio, input_sample_rate)
    if input_metrics.nan_count or input_metrics.inf_count or input_metrics.all_zero:
        raise ValueError(f"Input audio is not usable: {stats_dict(input_metrics)}")

    converter, module, _, runtime = create_converter(
        args.model,
        args.runtime_root,
        target_speaker=args.target_speaker,
        formant_shift=args.formant_shift,
        pitch_shift_semitone=args.pitch_shift,
        min_source_pitch=args.min_source_pitch,
        max_source_pitch=args.max_source_pitch,
        vq_num_neighbors=args.vq_neighbors,
    )
    wall_started = perf_counter()
    result = convert_audio_blocks(
        converter, module, input_audio, input_sample_rate
    )
    inference_wall_seconds = perf_counter() - wall_started
    output_metrics = validate_converted_audio(
        result.audio, result.sample_rate, input_metrics.duration_seconds
    )

    resampled_input = resample_audio(
        input_audio, input_sample_rate, result.sample_rate
    )
    comparison_frames = min(resampled_input.size, result.audio.size)
    reference = resampled_input[:comparison_frames]
    converted = result.audio[:comparison_frames]
    is_resampled_identity = bool(np.allclose(converted, reference, atol=1e-6))
    if is_resampled_identity:
        raise RuntimeError("Runtime output is only a resampled copy of the input")
    difference = converted.astype(np.float64) - reference.astype(np.float64)
    waveform_rmse = float(np.sqrt(np.mean(np.square(difference))))
    waveform_correlation = float(np.corrcoef(reference, converted)[0, 1])

    sf.write(output_path, result.audio, result.sample_rate, subtype="PCM_16")
    saved_audio, saved_sample_rate = read_mono_audio(output_path)
    saved_metrics = validate_converted_audio(
        saved_audio, saved_sample_rate, input_metrics.duration_seconds
    )

    block_times = result.block_times_ms
    report = {
        "success": True,
        "model": runtime,
        "parameters": {
            "target_speaker": args.target_speaker,
            "formant_shift": args.formant_shift,
            "pitch_shift_semitone": args.pitch_shift,
            "min_source_pitch": args.min_source_pitch,
            "max_source_pitch": args.max_source_pitch,
            "vq_num_neighbors": args.vq_neighbors,
        },
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input": stats_dict(input_metrics),
        "output_before_save": stats_dict(output_metrics),
        "output_saved": stats_dict(saved_metrics),
        "waveform_vs_resampled_input": {
            "frames": comparison_frames,
            "rmse": waveform_rmse,
            "correlation": waveform_correlation,
            "is_identity": is_resampled_identity,
        },
        "runtime_auxiliary": {
            "meaning": "unknown; undocumented second convert() return value",
            "unique_values": sorted(set(result.runtime_aux_values)),
        },
        "padded_input_frames_at_16khz": result.padded_input_frames,
        "raw_output_frames_at_24khz": result.raw_output_frames,
        "inference_wall_seconds": inference_wall_seconds,
        "inference_sum_block_seconds": sum(block_times) / 1000.0,
        "rtf": inference_wall_seconds / input_metrics.duration_seconds,
        "block_count": len(block_times),
        "block_ms": {
            "mean": sum(block_times) / len(block_times),
            "p50": percentile(block_times, 50),
            "p95": percentile(block_times, 95),
            "p99": percentile(block_times, 99),
            "max": max(block_times),
        },
        "validation": {
            "nan": False,
            "inf": False,
            "all_zero": False,
            "severe_clipping": False,
            "duration_reasonable": True,
            "wav_saved": output_path.is_file(),
            "not_resampled_identity": not is_resampled_identity,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

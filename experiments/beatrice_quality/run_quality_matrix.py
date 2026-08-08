"""Run the real-model Beatrice audio-quality root-cause matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from customization.beatrice import BeatriceParameterSet, analyze_beatrice_voice
from experiments.beatrice_quality.common import (
    DEFAULT_INPUT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_RUNTIME_ROOT,
    OUTPUTS_DIR,
    RESULTS_DIR,
    audio_metrics,
    ensure_directories,
    load_context,
    public_result,
    read_audio,
    render_native,
    render_streaming,
    resample_audio,
    run_worker_diagnostic,
    save_wav,
    write_json,
)
from experiments.beatrice_quality.compare_audio import compare_arrays, periodic_report
from experiments.beatrice_quality.parameter_sweep import run_parameter_sweeps
from experiments.beatrice_quality.speaker_sweep import run_speaker_sweep


def resampler_diagnosis(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qq = comparisons["QQ"]
    hq = comparisons["HQ"]
    correlation_gain = hq["correlation"] - qq["correlation"]
    rmse_reduction = (qq["rmse"] - hq["rmse"]) / max(qq["rmse"], 1e-12)
    if correlation_gain >= 0.02 and rmse_reduction >= 0.15:
        classification = "A"
        conclusion = "QQ has objective degradation relative to HQ on this fixed path."
    elif abs(correlation_gain) < 0.01 and abs(rmse_reduction) < 0.10:
        classification = "B"
        conclusion = "QQ and HQ differ little on lag-aligned objective metrics."
    elif qq["correlation"] >= 0.995 and qq["rmse"] <= 0.005:
        classification = "C"
        conclusion = (
            "QQ tracks Native very closely; MQ/HQ diverge, but divergence is not "
            "evidence that they sound better. No objective evidence currently "
            "identifies QQ as the degradation source."
        )
    else:
        classification = "C"
        conclusion = "Objective differences are mixed; listening is required."
    return {
        "classification": classification,
        "conclusion": conclusion,
        "hq_minus_qq_correlation": correlation_gain,
        "hq_relative_rmse_reduction": rmse_reduction,
        "qq_vs_native_correlation": qq["correlation"],
        "qq_vs_native_rmse": qq["rmse"],
        "hq_vs_native_correlation": hq["correlation"],
        "hq_vs_native_rmse": hq["rmse"],
        "warning": "Correlation and spectral metrics are diagnostic, not listening-quality scores.",
    }


def root_cause_ranking(
    diagnosis: dict[str, Any],
    worker: dict[str, Any],
    parameter_report: dict[str, Any],
    input_metrics: dict[str, Any],
) -> list[dict[str, str]]:
    worker_clean = all(
        worker[key] == 0
        for key in (
            "input_drop_or_overflow_count",
            "output_drop_or_overflow_count",
            "adapter_underflow_count",
            "adapter_overflow_count",
            "reset_count",
            "continuity_error_count",
            "worker_error_count",
        )
    )
    recommended = parameter_report["pitch_range"][0]
    very_wide = parameter_report["pitch_range"][-1]
    parameter_anomaly = (
        recommended["metrics"]["clipping_ratio"] > very_wide["metrics"]["clipping_ratio"]
        or abs(recommended["metrics"]["f0_median"] - input_metrics["f0_median"])
        > abs(very_wide["metrics"]["f0_median"] - input_metrics["f0_median"])
    )
    if parameter_anomaly:
        ordered = [
            (
                "Pitch range and parameters",
                f"The recommended Source Pitch range clipped {recommended['metrics']['clipping_ratio']:.6%} "
                f"and produced estimated median F0 {recommended['metrics']['f0_median']:.2f} Hz, while "
                f"the safe wide range had no clipping and median F0 {very_wide['metrics']['f0_median']:.2f} Hz "
                f"against input {input_metrics['f0_median']:.2f} Hz. This is a technical anomaly, not a listening score.",
            ),
            (
                "JVS / Chinese adaptation",
                f"QQ tracks Native at correlation {diagnosis['qq_vs_native_correlation']:.6f}; Native and the five-speaker files must now determine whether Chinese blur is common to JVS.",
            ),
            ("Beatrice runtime/model itself", "Native output is isolated and must be heard before separating runtime/model behavior from JVS data."),
            ("QQ resampler / StreamingAdapter", diagnosis["conclusion"]),
        ]
    elif diagnosis["classification"] == "A":
        ordered = [
            ("QQ resampler / StreamingAdapter", "HQ materially improves correlation and RMSE against Native."),
            ("JVS / Chinese adaptation", "The supplied model is JVS; Chinese intelligibility still requires Native-vs-streaming listening."),
            ("Pitch range and parameters", "Single-variable candidates were generated; objective validity does not select the best-sounding candidate."),
            ("Beatrice runtime/model itself", "Native output is isolated for direct listening before changing production."),
        ]
    else:
        ordered = [
            (
                "JVS / Chinese adaptation",
                f"Production QQ tracks Native at correlation {diagnosis['qq_vs_native_correlation']:.6f} "
                f"and RMSE {diagnosis['qq_vs_native_rmse']:.6f}; the reported artifacts are therefore "
                "more likely upstream of the outer streaming resampler, pending Native listening.",
            ),
            ("Pitch range and parameters", "F0-based and single-variable candidates can reveal range or compensation errors."),
            ("Beatrice runtime/model itself", "Native output must be heard to separate model/runtime behavior from streaming."),
            (
                "QQ resampler / StreamingAdapter",
                diagnosis["conclusion"],
            ),
        ]
    if not worker_clean:
        ordered.insert(0, ("Worker / buffer", "The simulated production chain recorded queue, continuity, or deadline errors."))
    else:
        ordered.append(("Worker / buffer", "No queue drop, underflow, overflow, reset, continuity error, or worker error was observed."))
    return [
        {"rank": str(index), "suspect": name, "evidence": evidence}
        for index, (name, evidence) in enumerate(ordered[:5], start=1)
    ]


def write_markdown_report(report: dict[str, Any]) -> None:
    native = report["native_vs_streaming"]["Native"]
    modes = report["native_vs_streaming"]
    lines = [
        "# Beatrice Audio Quality Root-Cause Report",
        "",
        "This report is generated from objective diagnostics. Human listening remains required.",
        "",
        "## Input",
        "",
        f"- Duration: {report['input']['metrics']['duration_seconds']:.3f} s",
        f"- Sample rate: {report['input']['source']['sample_rate']} Hz",
        f"- RMS / Peak: {report['input']['metrics']['rms']:.6f} / {report['input']['metrics']['peak']:.6f}",
        f"- F0 P5 / Median / P95: {report['input']['metrics']['f0_p5']:.2f} / {report['input']['metrics']['f0_median']:.2f} / {report['input']['metrics']['f0_p95']:.2f} Hz",
        "",
        "## Native vs Streaming",
        "",
        "| Path | SR | Processing s | Lag ms | RMSE | Correlation | Buffer latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Native | {native['output_sample_rate']} | {native['processing_seconds']:.4f} | 0 | 0 | 1 | 0 |",
    ]
    for quality in ("QQ", "MQ", "HQ"):
        item = modes[quality]
        comparison = report["comparisons_to_native"][quality]
        lines.append(
            f"| {quality} | {item['output_sample_rate']} | {item['processing_seconds']:.4f} | "
            f"{comparison['best_lag_ms']:.3f} | {comparison['rmse']:.6f} | "
            f"{comparison['correlation']:.6f} | {item['buffering_latency_ms']:.3f} |"
        )
    lines.extend([
        "",
        "## Resampler Diagnosis",
        "",
        f"Classification {report['resampler_diagnosis']['classification']}: {report['resampler_diagnosis']['conclusion']}",
        "",
        "## Root Cause Ranking",
        "",
    ])
    for item in report["root_cause_ranking"]:
        lines.append(f"{item['rank']}. **{item['suspect']}** — {item['evidence']}")
    lines.extend([
        "",
        "## Worker / Buffer",
        "",
        "```json",
        str(report["worker_buffer"]),
        "```",
        "",
        "## Listening",
        "",
        "Start with `01_native_offline.wav`, `02_streaming_QQ.wav`, and `04_streaming_HQ.wav`.",
    ])
    (RESULTS_DIR / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--package", default="jvs")
    parser.add_argument("--target-speaker", default="jvs001")
    parser.add_argument("--skip-worker", action="store_true")
    args = parser.parse_args()
    ensure_directories()

    source, source_rate, source_info = read_audio(args.input)
    input_metrics = audio_metrics(source, source_rate)
    reference_48khz = resample_audio(source, source_rate, 48_000)
    save_wav(OUTPUTS_DIR / "00_input.wav", reference_48khz, 48_000)
    descriptor, loader, runtime, capabilities = load_context(
        args.model_root, args.runtime_root, args.package
    )
    if args.target_speaker not in descriptor.speaker_names:
        raise LookupError(f"speaker not found: {args.target_speaker}")
    speaker_index = descriptor.speaker_names.index(args.target_speaker)
    base = BeatriceParameterSet(target_speaker=speaker_index)
    analysis = analyze_beatrice_voice(source, source_rate)

    native_raw = render_native(
        source, source_rate, descriptor, loader, base, OUTPUTS_DIR / "01_native_offline.wav"
    )
    streaming_raw = {
        quality: render_streaming(
            reference_48khz,
            descriptor,
            loader,
            base,
            OUTPUTS_DIR / filename,
            quality=quality,
        )
        for quality, filename in (
            ("QQ", "02_streaming_QQ.wav"),
            ("MQ", "03_streaming_MQ.wav"),
            ("HQ", "04_streaming_HQ.wav"),
        )
    }
    comparisons = {
        quality: compare_arrays(
            native_raw["audio"], 24_000, item["audio"], 48_000
        )
        for quality, item in streaming_raw.items()
    }
    diagnosis = resampler_diagnosis(comparisons)
    parameter_report, candidates = run_parameter_sweeps(
        reference_48khz, descriptor, loader, base, analysis, capabilities
    )
    fixed_speaker_parameters = candidates["pitch"][1][1]
    speaker_report = run_speaker_sweep(
        reference_48khz, descriptor, loader, fixed_speaker_parameters
    )
    worker_report = (
        {"skipped": True}
        if args.skip_worker
        else run_worker_diagnostic(reference_48khz, descriptor, args.runtime_root, base)
    )
    periodic = {
        "Native": periodic_report(native_raw["audio"], 24_000),
        **{
            quality: periodic_report(item["audio"], 48_000)
            for quality, item in streaming_raw.items()
        },
    }
    periodic_max_ratio = max(
        item["boundary_to_global_derivative_ratio"]
        for path_report in periodic.values()
        for item in path_report.values()
    )
    ranking = (
        []
        if args.skip_worker
        else root_cause_ranking(diagnosis, worker_report, parameter_report, input_metrics)
    )
    report = {
        "input": {"source": source_info, "metrics": input_metrics, "reference_48khz_path": str((OUTPUTS_DIR / '00_input.wav').resolve())},
        "configuration": {
            "package": descriptor.package,
            "target_speaker": args.target_speaker,
            "target_speaker_index": speaker_index,
            "base_parameters": base.to_engine_changes(),
            "runtime": runtime,
            "capabilities": capabilities.__dict__,
        },
        "native_vs_streaming": {
            "Native": public_result(native_raw),
            **{quality: public_result(item) for quality, item in streaming_raw.items()},
        },
        "comparisons_to_native": comparisons,
        "resampler_diagnosis": diagnosis,
        "pitch_range": parameter_report["pitch_range"],
        "pitch": parameter_report["pitch"],
        "formant": parameter_report["formant"],
        "vq": parameter_report["vq"],
        "speakers": speaker_report,
        "worker_buffer": worker_report,
        "periodic_artifact": periodic,
        "periodic_conclusion": {
            "max_boundary_to_global_derivative_ratio": periodic_max_ratio,
            "pronounced_boundary_spike_found": periodic_max_ratio >= 2.0,
            "note": "Ratios near 1 mean boundaries resemble ordinary adjacent-sample changes; this is not a perceptual threshold.",
        },
        "root_cause_ranking": ranking,
        "rvc_reference": {"generated": False, "reason": "Known environment _bz2.pyd access problem; not modified in this experiment."},
        "listening_files": [str(path.resolve()) for path in sorted(OUTPUTS_DIR.glob("*.wav"))],
    }
    write_json(RESULTS_DIR / "quality_matrix.json", report)
    write_markdown_report(report)
    print(f"result={RESULTS_DIR / 'quality_matrix.json'}")
    print(f"outputs={len(report['listening_files'])}")
    print(f"resampler_classification={diagnosis['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate four wide-Source-Pitch assisted-tuning listening checkpoints."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

from ai.voice_engine.beatrice import (
    DEFAULT_MAX_SOURCE_PITCH,
    DEFAULT_MIN_SOURCE_PITCH,
)
from customization.beatrice import (
    BeatriceCandidateGenerator,
    BeatriceParameterSearch,
    BeatriceParameterSet,
    analyze_beatrice_voice,
)
from customization.recording_session import RecordingSession
from experiments.beatrice_quality.common import (
    DEFAULT_INPUT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_RUNTIME_ROOT,
    OUTPUTS_DIR,
    RESULTS_DIR,
    ensure_directories,
    load_context,
    write_json,
)


TARGET_SPEAKER = "jvs085"


def result_record(result) -> dict:
    return {
        "parameters": result.parameters.to_engine_changes(),
        "audio_path": result.audio_path,
        "inference_ms": result.inference_ms,
        "error": result.error,
        "raw_safety": asdict(result.raw_safety) if result.raw_safety else None,
        "evaluation": asdict(result.evaluation) if result.evaluation else None,
    }


def safe_result(results, preferred_index: int):
    preferred = results[preferred_index]
    if preferred.audio_path:
        return preferred, preferred_index
    safe = [
        (index, result)
        for index, result in enumerate(results)
        if result.audio_path and result.evaluation is not None
    ]
    if not safe:
        return None, None
    index, result = max(
        safe, key=lambda item: item[1].evaluation.technical_quality
    )
    return result, index


def copy_selected(result, destination: Path) -> None:
    if result is None or not result.audio_path:
        raise RuntimeError(f"No safe candidate available for {destination.name}")
    shutil.copyfile(result.audio_path, destination)


def main() -> int:
    ensure_directories()
    source = RecordingSession.load_file(DEFAULT_INPUT, 48_000)
    descriptor, _loader, runtime, capabilities = load_context(
        DEFAULT_MODEL_ROOT, DEFAULT_RUNTIME_ROOT, "jvs"
    )
    speaker_index = descriptor.speaker_names.index(TARGET_SPEAKER)
    base = BeatriceParameterSet(
        target_speaker=speaker_index,
        min_source_pitch=DEFAULT_MIN_SOURCE_PITCH,
        max_source_pitch=DEFAULT_MAX_SOURCE_PITCH,
    )
    analysis = analyze_beatrice_voice(source, 48_000)
    search = BeatriceParameterSearch(base, analysis, capabilities, descriptor)
    report = {
        "input": str(DEFAULT_INPUT.resolve()),
        "runtime_version": runtime["version"],
        "package": descriptor.package,
        "speaker": TARGET_SPEAKER,
        "analysis": asdict(analysis),
        "source_pitch_initial": [
            base.min_source_pitch,
            base.max_source_pitch,
        ],
        "stages": [],
        "listening_files": [],
    }

    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        baseline_result = BeatriceCandidateGenerator(
            descriptor,
            DEFAULT_RUNTIME_ROOT,
            temp_root / "baseline",
        ).generate(source, [base])[0]
        baseline_path = OUTPUTS_DIR / "wide_source_pitch_baseline.wav"
        copy_selected(baseline_result, baseline_path)
        report["baseline"] = result_record(baseline_result)
        report["listening_files"].append(str(baseline_path.resolve()))

        while search.final_parameters is None:
            round_ = search.current
            results = BeatriceCandidateGenerator(
                descriptor,
                DEFAULT_RUNTIME_ROOT,
                temp_root / round_.stage,
            ).generate(source, round_.candidates)
            preferred_index = len(round_.candidates) // 2
            if round_.stage == "formant":
                preferred_index = min(
                    range(len(round_.candidates)),
                    key=lambda index: abs(round_.candidates[index].formant_shift),
                )
            elif round_.stage == "vq_neighbors":
                preferred_index = min(
                    range(len(round_.candidates)),
                    key=lambda index: abs(
                        round_.candidates[index].vq_num_neighbors
                        - base.vq_num_neighbors
                    ),
                )
            selected, selected_index = safe_result(results, preferred_index)
            report["stages"].append(
                {
                    "stage": round_.stage,
                    "selected_index": selected_index,
                    "candidates": [result_record(result) for result in results],
                }
            )
            if selected_index is None:
                next_round = search.skip_unsafe_round()
            else:
                next_round = search.choose(selected_index)

            destination = None
            if round_.stage == "pitch_fine":
                destination = OUTPUTS_DIR / "wide_after_pitch.wav"
            elif round_.stage == "formant":
                destination = OUTPUTS_DIR / "wide_after_formant.wav"
            elif round_.stage == "vq_neighbors":
                destination = OUTPUTS_DIR / "wide_final_vq.wav"
            if destination is not None:
                copy_selected(selected, destination)
                report["listening_files"].append(str(destination.resolve()))
            if next_round is None:
                break

    report["source_pitch_final"] = [
        search.final_parameters.min_source_pitch,
        search.final_parameters.max_source_pitch,
    ]
    write_json(RESULTS_DIR / "wide_assisted_preview.json", report)
    print(json.dumps({
        "speaker": TARGET_SPEAKER,
        "stages": [item["stage"] for item in report["stages"]],
        "source_pitch_initial": report["source_pitch_initial"],
        "source_pitch_final": report["source_pitch_final"],
        "listening_files": report["listening_files"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

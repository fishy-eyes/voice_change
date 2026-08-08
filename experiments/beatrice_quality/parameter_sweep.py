"""Small, one-variable-at-a-time Beatrice parameter matrix."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from customization.beatrice import (
    BeatriceParameterSet,
    BeatriceTuningCapabilities,
    BeatriceVoiceAnalysis,
    recommend_pitch_shift,
)
from experiments.beatrice_quality.common import (
    OUTPUTS_DIR,
    parameters_dict,
    public_result,
    render_streaming,
)


def diagnostic_source_pitch_range(
    analysis: BeatriceVoiceAnalysis,
    capabilities: BeatriceTuningCapabilities,
    *,
    margin: float,
) -> tuple[float, float] | None:
    """Reproduce the retired narrow-range strategy for regression evidence only."""
    if analysis.f0_p5 is None or analysis.f0_p95 is None:
        return None
    low = analysis.f0_p5 * (1.0 - margin)
    high = analysis.f0_p95 * (1.0 + margin)
    if capabilities.source_pitch_min is not None:
        low = max(low, capabilities.source_pitch_min)
    if capabilities.source_pitch_max is not None:
        high = min(high, capabilities.source_pitch_max)
    return (float(low), float(high)) if high > low else None


def build_candidates(
    base: BeatriceParameterSet,
    analysis: BeatriceVoiceAnalysis,
    capabilities: BeatriceTuningCapabilities,
    descriptor,
) -> dict[str, list[tuple[str, BeatriceParameterSet]]]:
    recommended = diagnostic_source_pitch_range(
        analysis, capabilities, margin=0.20
    )
    if recommended is None:
        recommended = (base.min_source_pitch, base.max_source_pitch)
    wider = (max(1.0, recommended[0] * 0.75), recommended[1] * 1.25)
    if capabilities.source_pitch_min is not None:
        wider = (max(wider[0], capabilities.source_pitch_min), wider[1])
    if capabilities.source_pitch_max is not None:
        wider = (wider[0], min(wider[1], capabilities.source_pitch_max))
    pitch_center = recommend_pitch_shift(
        analysis, descriptor, base.target_speaker, capabilities, fallback=base.pitch_shift_semitone
    )
    pitch_values = [
        float(np.clip(pitch_center + offset, capabilities.pitch_shift_min, capabilities.pitch_shift_max))
        for offset in (-2.0, 0.0, 2.0)
    ]
    formant_step = min(2.0, max(1.0, capabilities.max_formant_shift / 4.0))
    vq_upper = max(1, capabilities.codebook_size)
    vq_values = list(dict.fromkeys([1, min(vq_upper, base.vq_num_neighbors), min(vq_upper, 16)]))
    return {
        "pitch_range": [
            ("recommended", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1])),
            ("wide", replace(base, min_source_pitch=wider[0], max_source_pitch=wider[1])),
            ("very_wide", replace(base, min_source_pitch=30.0, max_source_pitch=1100.0)),
        ],
        "pitch": [
            ("low", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_values[0])),
            ("center", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_values[1])),
            ("high", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_values[2])),
        ],
        "formant": [
            ("negative", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_center, formant_shift=-formant_step)),
            ("zero", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_center, formant_shift=0.0)),
            ("positive", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_center, formant_shift=formant_step)),
        ],
        "vq": [
            ("low" if value == 1 else "default" if value == base.vq_num_neighbors else "high", replace(base, min_source_pitch=recommended[0], max_source_pitch=recommended[1], pitch_shift_semitone=pitch_center, vq_num_neighbors=value))
            for value in vq_values
        ],
    }


def run_parameter_sweeps(
    audio_48khz: np.ndarray,
    descriptor,
    loader,
    base: BeatriceParameterSet,
    analysis: BeatriceVoiceAnalysis,
    capabilities: BeatriceTuningCapabilities,
) -> tuple[dict[str, Any], dict[str, list[tuple[str, BeatriceParameterSet]]]]:
    candidates = build_candidates(base, analysis, capabilities, descriptor)
    prefixes = {"pitch_range": 10, "pitch": 20, "formant": 30, "vq": 40}
    report: dict[str, Any] = {}
    for group, options in candidates.items():
        group_results = []
        for index, (label, parameters) in enumerate(options):
            destination = OUTPUTS_DIR / f"{prefixes[group] + index:02d}_{group}_{label}.wav"
            rendered = render_streaming(
                audio_48khz, descriptor, loader, parameters, destination, quality="QQ"
            )
            group_results.append({
                "label": label,
                "parameters": parameters_dict(parameters),
                **public_result(rendered),
            })
        report[group] = group_results
    return report, candidates


__all__ = ["build_candidates", "run_parameter_sweeps"]

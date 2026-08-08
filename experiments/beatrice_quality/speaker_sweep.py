"""Fixed-parameter sweep across separated JVS speakers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from customization.beatrice import BeatriceParameterSet
from experiments.beatrice_quality.common import (
    OUTPUTS_DIR,
    parameters_dict,
    public_result,
    render_streaming,
)


PREFERRED_SPEAKERS = ("jvs001", "jvs010", "jvs030", "jvs050", "jvs080")


def select_speakers(descriptor) -> list[tuple[int, str]]:
    names = tuple(descriptor.speaker_names)
    selected = [(names.index(name), name) for name in PREFERRED_SPEAKERS if name in names]
    if len(selected) >= 5:
        return selected[:5]
    indexes = np.linspace(0, len(names) - 1, num=min(5, len(names)), dtype=int)
    return [(int(index), names[int(index)]) for index in dict.fromkeys(indexes.tolist())]


def run_speaker_sweep(
    audio_48khz: np.ndarray,
    descriptor,
    loader,
    fixed_parameters: BeatriceParameterSet,
) -> list[dict[str, Any]]:
    report = []
    for output_index, (speaker_index, speaker_name) in enumerate(select_speakers(descriptor), start=50):
        parameters = replace(fixed_parameters, target_speaker=speaker_index)
        destination = OUTPUTS_DIR / f"{output_index:02d}_{speaker_name}.wav"
        rendered = render_streaming(
            audio_48khz, descriptor, loader, parameters, destination, quality="QQ"
        )
        report.append({
            "speaker_index": speaker_index,
            "speaker_name": speaker_name,
            "parameters": parameters_dict(parameters),
            **public_result(rendered),
        })
    return report


__all__ = ["PREFERRED_SPEAKERS", "run_speaker_sweep", "select_speakers"]

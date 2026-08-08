"""Deterministic assisted tuning primitives for the Beatrice backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import string
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf

from ai.voice_engine.beatrice import (
    BeatriceConfig,
    BeatriceVoiceEngine,
    DEFAULT_MAX_SOURCE_PITCH,
    DEFAULT_MIN_SOURCE_PITCH,
)
from customization.candidate_evaluator import (
    CandidateEvaluator,
    RawCandidateSafetyEvaluator,
)
from customization.candidate_generator import match_audition_level
from customization.quality_checker import extract_audio_features
from customization.schemas import CandidateEvaluation, RawCandidateSafetyEvaluation


@dataclass(frozen=True)
class BeatriceParameterSet:
    target_speaker: int = 0
    pitch_shift_semitone: float = 0.0
    min_source_pitch: float = DEFAULT_MIN_SOURCE_PITCH
    max_source_pitch: float = DEFAULT_MAX_SOURCE_PITCH
    formant_shift: float = 0.0
    vq_num_neighbors: int = 4

    def to_engine_changes(self) -> dict[str, int | float]:
        return asdict(self)

    def to_assisted_changes(self) -> dict[str, int | float]:
        """Return only fields owned by assisted tuning."""
        values = self.to_engine_changes()
        values.pop("min_source_pitch")
        values.pop("max_source_pitch")
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BeatriceParameterSet":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: values[key] for key in allowed if key in values})


@dataclass(frozen=True)
class BeatriceTuningCapabilities:
    pitch_shift_min: float = -24.0
    pitch_shift_max: float = 24.0
    source_pitch_min: float | None = None
    source_pitch_max: float | None = None
    max_formant_shift: float = 0.0
    codebook_size: int = 1

    @classmethod
    def from_runtime(cls, values: Mapping[str, Any]) -> "BeatriceTuningCapabilities":
        def optional_float(name: str) -> float | None:
            value = values.get(name)
            return float(value) if value is not None else None

        return cls(
            pitch_shift_min=float(values.get("pitch_shift_min", -24.0)),
            pitch_shift_max=float(values.get("pitch_shift_max", 24.0)),
            source_pitch_min=optional_float("source_pitch_min"),
            source_pitch_max=optional_float("source_pitch_max"),
            max_formant_shift=max(0.0, float(values.get("max_formant_shift", 0.0))),
            codebook_size=max(1, int(values.get("codebook_size", 1))),
        )


@dataclass(frozen=True)
class BeatriceVoiceAnalysis:
    f0_p5: float | None
    f0_p50: float | None
    f0_p95: float | None
    rms: float
    voiced_ratio: float = 0.0
    f0_count: int = 0


@dataclass(frozen=True)
class BeatriceCandidateResult:
    candidate_id: str
    label: str
    parameters: BeatriceParameterSet
    audio_path: str | None
    inference_ms: float
    evaluation: CandidateEvaluation | None = None
    raw_safety: RawCandidateSafetyEvaluation | None = None
    error: str | None = None


@dataclass
class BeatriceSearchRound:
    stage: str
    candidates: list[BeatriceParameterSet]
    fallback: BeatriceParameterSet
    selected_index: int | None = None


def analyze_beatrice_voice(audio: np.ndarray, sample_rate: int) -> BeatriceVoiceAnalysis:
    features = extract_audio_features(audio, sample_rate)
    f0 = features.f0_values
    return BeatriceVoiceAnalysis(
        f0_p5=float(np.percentile(f0, 5)) if len(f0) else None,
        f0_p50=float(np.percentile(f0, 50)) if len(f0) else None,
        f0_p95=float(np.percentile(f0, 95)) if len(f0) else None,
        rms=features.rms,
        voiced_ratio=features.voiced_frame_ratio,
        f0_count=int(len(f0)),
    )


def metadata_pitch_to_hz(value: float | None) -> float | None:
    """Beatrice metadata stores average_pitch on the MIDI-note scale."""
    if value is None or not math.isfinite(value):
        return None
    return float(440.0 * 2.0 ** ((float(value) - 69.0) / 12.0))


def recommend_pitch_shift(
    analysis: BeatriceVoiceAnalysis,
    descriptor,
    target_speaker: int,
    capabilities: BeatriceTuningCapabilities,
    *,
    fallback: float = 0.0,
) -> float:
    source = analysis.f0_p50
    averages = tuple(getattr(descriptor, "speaker_average_pitches", ()))
    metadata_pitch = averages[target_speaker] if 0 <= target_speaker < len(averages) else None
    target = metadata_pitch_to_hz(metadata_pitch)
    value = fallback
    if source is not None and source > 0 and target is not None and target > 0:
        value = 12.0 * math.log2(target / source)
    return float(np.clip(value, capabilities.pitch_shift_min, capabilities.pitch_shift_max))


class BeatriceParameterSearch:
    """Sequential one-parameter-at-a-time Beatrice search state."""

    STAGES = ("pitch_coarse", "pitch_fine", "formant", "vq_neighbors")

    def __init__(
        self,
        base: BeatriceParameterSet,
        analysis: BeatriceVoiceAnalysis,
        capabilities: BeatriceTuningCapabilities,
        descriptor,
    ) -> None:
        self.analysis = analysis
        self.capabilities = capabilities
        self.descriptor = descriptor
        self.history: list[BeatriceSearchRound] = []
        self.final_parameters: BeatriceParameterSet | None = None
        self.cancelled = False
        self.current = self._round("pitch_coarse", base)

    def cancel(self) -> None:
        self.cancelled = True

    def choose(self, index: int) -> BeatriceSearchRound | None:
        if self.cancelled:
            raise RuntimeError("Beatrice parameter search is cancelled")
        if not 0 <= index < len(self.current.candidates):
            raise IndexError("candidate index is out of range")
        self.current.selected_index = index
        selected = self.current.candidates[index]
        self.history.append(self.current)
        stage_index = self.STAGES.index(self.current.stage)
        if stage_index == len(self.STAGES) - 1:
            self.final_parameters = selected
            return None
        self.current = self._round(self.STAGES[stage_index + 1], selected)
        return self.current

    def skip_unsafe_round(self) -> BeatriceSearchRound | None:
        """Advance without changing parameters when every candidate is unsafe."""
        self.history.append(self.current)
        stage_index = self.STAGES.index(self.current.stage)
        if stage_index == len(self.STAGES) - 1:
            self.final_parameters = self.current.fallback
            return None
        self.current = self._round(
            self.STAGES[stage_index + 1], self.current.fallback
        )
        return self.current

    def _round(self, stage: str, base: BeatriceParameterSet) -> BeatriceSearchRound:
        cap = self.capabilities
        if stage == "pitch_coarse":
            center = recommend_pitch_shift(
                self.analysis,
                self.descriptor,
                base.target_speaker,
                cap,
                fallback=base.pitch_shift_semitone,
            )
            span = max(0.0, cap.pitch_shift_max - cap.pitch_shift_min)
            step = max(0.25, min(3.0, span / 8.0))
            values = [
                float(np.clip(center + step * offset, cap.pitch_shift_min, cap.pitch_shift_max))
                for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)
            ]
            candidates = [replace(base, pitch_shift_semitone=value) for value in dict.fromkeys(values)]
        elif stage == "pitch_fine":
            span = max(0.0, cap.pitch_shift_max - cap.pitch_shift_min)
            step = max(0.1, min(1.0, span / 24.0))
            values = [
                float(np.clip(base.pitch_shift_semitone + step * offset, cap.pitch_shift_min, cap.pitch_shift_max))
                for offset in (-1.0, 0.0, 1.0)
            ]
            candidates = [replace(base, pitch_shift_semitone=value) for value in dict.fromkeys(values)]
        elif stage == "formant":
            maximum = cap.max_formant_shift
            step = maximum / 4.0
            values = tuple(
                float(np.clip(base.formant_shift + step * offset, -maximum, maximum))
                for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)
            )
            candidates = [replace(base, formant_shift=float(value)) for value in dict.fromkeys(values)]
        elif stage == "vq_neighbors":
            upper = cap.codebook_size
            current = int(np.clip(base.vq_num_neighbors, 1, upper))
            values = [1, max(1, current // 2), current, min(upper, current * 2)]
            candidates = [replace(base, vq_num_neighbors=value) for value in dict.fromkeys(values)]
        else:
            raise ValueError(f"unknown Beatrice search stage: {stage}")
        return BeatriceSearchRound(stage=stage, candidates=candidates, fallback=base)


class BeatriceCandidateGenerator:
    """Generate every candidate with a fresh isolated converter and FIFO state."""

    def __init__(
        self,
        descriptor,
        runtime_root: str | Path,
        output_directory: str | Path,
        *,
        sample_rate: int = 48_000,
        engine_factory=BeatriceVoiceEngine,
        evaluator: CandidateEvaluator | None = None,
        raw_safety_evaluator: RawCandidateSafetyEvaluator | None = None,
        level_matcher=match_audition_level,
    ) -> None:
        self.descriptor = descriptor
        self.runtime_root = Path(runtime_root)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.engine_factory = engine_factory
        self.evaluator = evaluator or CandidateEvaluator()
        self.raw_safety_evaluator = (
            raw_safety_evaluator or RawCandidateSafetyEvaluator()
        )
        self.level_matcher = level_matcher

    def generate(
        self,
        audio: np.ndarray,
        parameters: Iterable[BeatriceParameterSet],
        *,
        cancel_event: threading.Event | None = None,
        progress: Callable[[int, int, BeatriceCandidateResult], None] | None = None,
    ) -> list[BeatriceCandidateResult]:
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        options = tuple(parameters)
        results: list[BeatriceCandidateResult] = []
        for index, values in enumerate(options):
            if cancel_event is not None and cancel_event.is_set():
                break
            started = perf_counter()
            engine = None
            result: BeatriceCandidateResult
            try:
                engine = self.engine_factory(
                    self.descriptor,
                    runtime_root=self.runtime_root,
                    config=BeatriceConfig(**values.to_engine_changes()),
                )
                engine.load_model()
                converted_parts: list[np.ndarray] = []
                cancelled = False
                for offset in range(0, len(source), 256):
                    block = source[offset : offset + 256]
                    actual = len(block)
                    if actual < 256:
                        block = np.pad(block, (0, 256 - actual))
                    converted_parts.append(engine.process_audio(block)[:actual])
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break
                if cancelled:
                    break
                converted = np.concatenate(converted_parts) if converted_parts else np.empty(0, dtype=np.float32)
                elapsed = (perf_counter() - started) * 1000.0
                raw_safety = self.raw_safety_evaluator.evaluate(
                    source, converted, self.sample_rate
                )
                evaluation = (
                    self.evaluator.evaluate(source, converted, self.sample_rate)
                    if raw_safety.is_safe
                    else raw_safety.technical_evaluation
                )
                destination = self.output_directory / f"beatrice_candidate_{index + 1:02d}.wav"
                audio_path = None
                if raw_safety.is_safe and evaluation is not None and evaluation.is_valid:
                    sf.write(
                        destination,
                        self.level_matcher(source, converted),
                        self.sample_rate,
                        subtype="PCM_16",
                    )
                    audio_path = str(destination)
                result = BeatriceCandidateResult(
                    candidate_id=f"beatrice-candidate-{index + 1}",
                    label="Option " + (string.ascii_uppercase[index] if index < 26 else str(index + 1)),
                    parameters=values,
                    audio_path=audio_path,
                    inference_ms=elapsed,
                    evaluation=evaluation,
                    raw_safety=raw_safety,
                )
            except Exception as exc:
                result = BeatriceCandidateResult(
                    candidate_id=f"beatrice-candidate-{index + 1}",
                    label=f"Option {index + 1}",
                    parameters=values,
                    audio_path=None,
                    inference_ms=(perf_counter() - started) * 1000.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if engine is not None:
                    engine.unload_model()
            results.append(result)
            if progress is not None:
                progress(index + 1, len(options), result)
        return results


def speaker_preset_key(descriptor, speaker_index: int) -> str:
    names = tuple(getattr(descriptor, "speaker_names", ()))
    name = names[speaker_index] if 0 <= speaker_index < len(names) else str(speaker_index)
    return f"{descriptor.identity}:{speaker_index}:{name}"


__all__ = [
    "BeatriceCandidateGenerator",
    "BeatriceCandidateResult",
    "BeatriceParameterSearch",
    "BeatriceParameterSet",
    "BeatriceTuningCapabilities",
    "BeatriceVoiceAnalysis",
    "analyze_beatrice_voice",
    "metadata_pitch_to_hz",
    "recommend_pitch_shift",
    "speaker_preset_key",
]

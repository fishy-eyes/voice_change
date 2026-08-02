"""Typed data exchanged by the customization workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordingQualityResult:
    duration_seconds: float
    effective_voice_seconds: float
    silence_ratio: float
    rms: float
    peak: float
    clipping_ratio: float
    dynamic_range_db: float
    has_valid_pitch: bool
    voiced_frame_ratio: float
    f0_median: float | None
    f0_p10: float | None
    f0_p90: float | None
    pitch_discontinuity_ratio: float
    background_noise_ratio: float
    quality_score: int
    is_acceptable: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class VoiceAnalysisResult:
    duration_seconds: float
    rms_mean: float
    peak: float
    clipping_ratio: float
    voiced_frame_ratio: float
    f0_median: float | None
    f0_p10: float | None
    f0_p90: float | None
    pitch_discontinuity_ratio: float
    dynamic_range_db: float


@dataclass(frozen=True)
class ModelInspectionResult:
    model_hash: str
    model_path: str
    index_path: str | None
    model_version: str
    model_sample_rate: int
    uses_f0: bool
    has_index: bool
    index_loadable: bool
    inspection_time: str
    warning: str | None = None


@dataclass(frozen=True)
class RVCParameterSet:
    pitch_shift: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.0
    protect: float = 0.33
    rms_mix_rate: float = 0.25

    def to_engine_changes(self) -> dict[str, Any]:
        return asdict(self)

    def to_profile_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["pitch"] = values.pop("pitch_shift")
        return values

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "RVCParameterSet":
        data = dict(values)
        if "pitch_shift" not in data and "pitch" in data:
            data["pitch_shift"] = data.pop("pitch")
        return cls(**data)


@dataclass(frozen=True)
class CandidateEvaluation:
    technical_quality: int
    stability_score: int
    volume_score: int
    pitch_continuity_score: int
    is_valid: bool
    rejection_reasons: tuple[str, ...] = ()
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    duration_ratio: float = 1.0
    high_frequency_ratio: float = 0.0
    discontinuity_ratio: float = 0.0


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    label: str
    parameters: RVCParameterSet
    audio_path: str | None
    inference_ms: float
    evaluation: CandidateEvaluation | None = None
    error: str | None = None


@dataclass
class SearchRound:
    stage: str
    candidates: list[RVCParameterSet]
    selected_index: int | None = None
    cancelled: bool = False

    @property
    def can_advance(self) -> bool:
        return not self.cancelled and self.selected_index is not None

    @property
    def selected(self) -> RVCParameterSet | None:
        if self.selected_index is None:
            return None
        return self.candidates[self.selected_index]

    def select(self, index: int) -> RVCParameterSet:
        if self.cancelled:
            raise RuntimeError("search round is cancelled")
        if not 0 <= index < len(self.candidates):
            raise IndexError("candidate index is out of range")
        self.selected_index = index
        return self.candidates[index]

    def cancel(self) -> None:
        self.cancelled = True


@dataclass(frozen=True)
class CustomizationProfile:
    profile_name: str
    model: ModelInspectionResult
    input_device_name: str
    input_sample_rate: int
    voice_analysis: VoiceAnalysisResult
    parameters: RVCParameterSet
    search_summary: dict[str, Any]
    created_at: str
    updated_at: str
    profile_version: int = 1
    adaptation_method: str = "black_box_search_and_user_ab"
    adaptation_confidence: str = "medium"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "model": asdict(self.model),
            "input": {
                "device_name": self.input_device_name,
                "sample_rate": self.input_sample_rate,
            },
            "voice_analysis": asdict(self.voice_analysis),
            "parameters": self.parameters.to_profile_dict(),
            "search_summary": self.search_summary,
            "adaptation": {
                "method": self.adaptation_method,
                "confidence": self.adaptation_confidence,
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "profile_version": self.profile_version,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CustomizationProfile":
        input_values = values.get("input", {})
        adaptation = values.get("adaptation", {})
        return cls(
            profile_name=str(values["profile_name"]),
            model=ModelInspectionResult(**values["model"]),
            input_device_name=str(input_values.get("device_name", "")),
            input_sample_rate=int(input_values.get("sample_rate", 0)),
            voice_analysis=VoiceAnalysisResult(**values["voice_analysis"]),
            parameters=RVCParameterSet.from_mapping(values["parameters"]),
            search_summary=dict(values.get("search_summary", {})),
            created_at=str(values["created_at"]),
            updated_at=str(values["updated_at"]),
            profile_version=int(values.get("profile_version", 1)),
            adaptation_method=str(
                adaptation.get("method", "black_box_search_and_user_ab")
            ),
            adaptation_confidence=str(adaptation.get("confidence", "medium")),
            warnings=tuple(values.get("warnings", ())),
        )

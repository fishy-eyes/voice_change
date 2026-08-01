"""Serializable configuration and model profiles for RVC inference."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping


_F0_METHODS = frozenset({"pm", "rmvpe", "fcpe"})
_INFERENCE_FIELDS = frozenset(
    {"pitch_shift", "f0_method", "index_rate", "rms_mix_rate", "protect"}
)
_PROFILE_FIELDS = frozenset(
    {"name", "voice_dir", "model_file", "index_file", "inference"}
)


@dataclass(frozen=True)
class RVCInferenceConfig:
    """Runtime-safe RVC inference parameters.

    The object deliberately contains no model or application lifecycle state,
    so a future GUI can replace it without rebuilding the audio pipeline.
    """

    pitch_shift: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    rms_mix_rate: float = 0.25
    protect: float = 0.33

    def __post_init__(self) -> None:
        if isinstance(self.pitch_shift, bool) or not isinstance(self.pitch_shift, int):
            raise TypeError("pitch_shift must be an integer number of semitones")

        method = str(self.f0_method).strip().lower()
        if method not in _F0_METHODS:
            choices = ", ".join(sorted(_F0_METHODS))
            raise ValueError(f"f0_method must be one of: {choices}")
        object.__setattr__(self, "f0_method", method)

        self._validate_rate("index_rate", self.index_rate, 0.0, 1.0)
        self._validate_rate("rms_mix_rate", self.rms_mix_rate, 0.0, 1.0)
        self._validate_rate("protect", self.protect, 0.0, 0.5)
        object.__setattr__(self, "index_rate", float(self.index_rate))
        object.__setattr__(self, "rms_mix_rate", float(self.rms_mix_rate))
        object.__setattr__(self, "protect", float(self.protect))

    @staticmethod
    def _validate_rate(name: str, value: object, minimum: float, maximum: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not minimum <= float(value) <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RVCInferenceConfig":
        unknown = set(values) - _INFERENCE_FIELDS
        if unknown:
            raise ValueError(
                "Unknown RVC inference setting(s): " + ", ".join(sorted(unknown))
            )
        return cls(**dict(values))

    def updated(self, **changes: Any) -> "RVCInferenceConfig":
        unknown = set(changes) - _INFERENCE_FIELDS
        if unknown:
            raise ValueError(
                "Unknown RVC inference setting(s): " + ", ".join(sorted(unknown))
            )
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RVCModelProfile:
    """A voice model selection and its independent inference configuration."""

    name: str
    voice_dir: Path
    inference: RVCInferenceConfig
    model_file: Path | None = None
    index_file: Path | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("profile name must not be empty")
        voice_dir = Path(self.voice_dir)
        if not str(voice_dir):
            raise ValueError("voice_dir must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "voice_dir", voice_dir)
        if self.model_file is not None:
            object.__setattr__(self, "model_file", Path(self.model_file))
        if self.index_file is not None:
            object.__setattr__(self, "index_file", Path(self.index_file))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RVCModelProfile":
        unknown = set(values) - _PROFILE_FIELDS
        if unknown:
            raise ValueError("Unknown RVC profile field(s): " + ", ".join(sorted(unknown)))
        if "name" not in values or "voice_dir" not in values:
            raise ValueError("RVC profile requires name and voice_dir")
        inference_values = values.get("inference", {})
        if not isinstance(inference_values, Mapping):
            raise TypeError("profile inference must be an object/table")
        return cls(
            name=str(values["name"]),
            voice_dir=Path(str(values["voice_dir"])),
            model_file=(
                Path(str(values["model_file"]))
                if values.get("model_file") is not None
                else None
            ),
            index_file=(
                Path(str(values["index_file"]))
                if values.get("index_file") is not None
                else None
            ),
            inference=RVCInferenceConfig.from_mapping(inference_values),
        )

    def resolve_voice_dir(self, models_dir: str | Path) -> Path:
        if self.voice_dir.is_absolute():
            return self.voice_dir
        return Path(models_dir) / self.voice_dir

    def resolve_model_file(self, models_dir: str | Path) -> Path | None:
        return self._resolve_voice_file(self.model_file, models_dir)

    def resolve_index_file(self, models_dir: str | Path) -> Path | None:
        return self._resolve_voice_file(self.index_file, models_dir)

    def _resolve_voice_file(
        self,
        value: Path | None,
        models_dir: str | Path,
    ) -> Path | None:
        if value is None:
            return None
        if value.is_absolute():
            return value
        return self.resolve_voice_dir(models_dir) / value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "voice_dir": str(self.voice_dir),
            "model_file": str(self.model_file) if self.model_file is not None else None,
            "index_file": str(self.index_file) if self.index_file is not None else None,
            "inference": self.inference.to_dict(),
        }


def load_rvc_profile(path: str | Path) -> RVCModelProfile:
    """Load an RVC model profile from JSON or TOML without side effects."""

    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"RVC profile not found: {profile_path}")
    suffix = profile_path.suffix.lower()
    with profile_path.open("rb") as handle:
        if suffix == ".json":
            values = json.load(handle)
        elif suffix == ".toml":
            values = tomllib.load(handle)
        else:
            raise ValueError("RVC profile must use .json or .toml")
    if not isinstance(values, Mapping):
        raise TypeError("RVC profile root must be an object/table")
    return RVCModelProfile.from_mapping(values)

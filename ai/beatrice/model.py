"""Discovery and validation for external Beatrice model packages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping
import tomllib

from loguru import logger


EXPECTED_RUNTIME_VERSION = "2.0.0-rc.0"
REQUIRED_MODEL_FILES = (
    "phone_extractor.bin",
    "pitch_estimator.bin",
    "embedding_setter.bin",
    "waveform_generator.bin",
    "speaker_embeddings.bin",
)


@dataclass(frozen=True)
class BeatriceModelDescriptor:
    """Validated metadata for one ``models/beatrice/<package>`` directory."""

    name: str
    package: str
    directory: Path
    metadata_path: Path | None
    model_name: str | None
    version: str | None
    runtime_requirement: str
    speaker_count: int
    speaker_names: tuple[str, ...]
    speaker_average_pitches: tuple[float | None, ...] = ()
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    valid: bool = False
    validation_error: str | None = None

    @property
    def required_files(self) -> Mapping[str, Path]:
        return {name: self.directory / name for name in REQUIRED_MODEL_FILES}

    @property
    def identity(self) -> str:
        """Stable package identity based on small immutable metadata."""
        digest = hashlib.sha256()
        if self.metadata_path is not None and self.metadata_path.is_file():
            digest.update(self.metadata_path.read_bytes())
        else:
            digest.update(f"{self.model_name}|{self.version}".encode("utf-8"))
        return digest.hexdigest()


class BeatriceModelManager:
    """Inspect only Beatrice TOML/bin packages, never RVC ``.pth/.index``."""

    def __init__(self, models_root: str | Path) -> None:
        self.models_root = Path(models_root).expanduser().resolve()

    def inspect_package(self, package: str | Path) -> BeatriceModelDescriptor:
        path = Path(package).expanduser()
        if not path.is_absolute():
            path = self.models_root / path
        directory = path.resolve()
        name = directory.name or str(package)
        try:
            if not directory.is_dir():
                raise NotADirectoryError(f"Beatrice model package not found: {directory}")
            metadata_candidates = tuple(
                sorted(directory.glob("beatrice_paraphernalia_*.toml"))
            )
            if len(metadata_candidates) != 1:
                raise ValueError(
                    "Beatrice package must contain exactly one "
                    f"beatrice_paraphernalia_*.toml; found {len(metadata_candidates)}"
                )
            metadata_path = metadata_candidates[0].resolve()
            with metadata_path.open("rb") as handle:
                metadata = tomllib.load(handle)
            model = metadata.get("model")
            if not isinstance(model, Mapping):
                raise ValueError("Beatrice metadata is missing [model]")
            version = model.get("version")
            if version != EXPECTED_RUNTIME_VERSION:
                raise ValueError(
                    f"Beatrice model requires {version!r}; supported runtime is "
                    f"{EXPECTED_RUNTIME_VERSION!r}"
                )
            model_name = str(model.get("name", "")).strip()
            if not model_name:
                raise ValueError("Beatrice metadata model.name is missing")
            missing = [
                filename
                for filename in REQUIRED_MODEL_FILES
                if not (directory / filename).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Beatrice package is incomplete; missing: " + ", ".join(missing)
                )
            voices = metadata.get("voice")
            if not isinstance(voices, Mapping) or not voices:
                raise ValueError("Beatrice metadata contains no [voice.N] speakers")
            indexed: list[tuple[int, str, float | None]] = []
            for key, value in voices.items():
                if not isinstance(value, Mapping):
                    raise ValueError(f"Invalid Beatrice voice entry: {key}")
                try:
                    index = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid Beatrice voice index: {key}") from exc
                speaker_name = str(value.get("name", "")).strip() or f"Speaker {index}"
                raw_average = value.get("average_pitch")
                try:
                    average_pitch = (
                        float(raw_average) if raw_average is not None else None
                    )
                except (TypeError, ValueError):
                    average_pitch = None
                indexed.append((index, speaker_name, average_pitch))
            indexed.sort()
            expected = list(range(len(indexed)))
            if [index for index, _, _ in indexed] != expected:
                raise ValueError("Beatrice speaker indexes must be contiguous from zero")
            return BeatriceModelDescriptor(
                name=name,
                package=name,
                directory=directory,
                metadata_path=metadata_path,
                model_name=model_name,
                version=str(version),
                runtime_requirement=EXPECTED_RUNTIME_VERSION,
                speaker_count=len(indexed),
                speaker_names=tuple(value for _, value, _ in indexed),
                speaker_average_pitches=tuple(value for _, _, value in indexed),
                valid=True,
            )
        except Exception as exc:
            return BeatriceModelDescriptor(
                name=name,
                package=name,
                directory=directory,
                metadata_path=None,
                model_name=None,
                version=None,
                runtime_requirement=EXPECTED_RUNTIME_VERSION,
                speaker_count=0,
                speaker_names=(),
                speaker_average_pitches=(),
                valid=False,
                validation_error=f"{type(exc).__name__}: {exc}",
            )

    def discover_models(self) -> list[BeatriceModelDescriptor]:
        if not self.models_root.is_dir():
            logger.info("Beatrice model library is not configured: {}", self.models_root)
            return []
        descriptors: list[BeatriceModelDescriptor] = []
        for directory in sorted(self.models_root.iterdir(), key=lambda item: item.name.lower()):
            if not directory.is_dir():
                continue
            descriptor = self.inspect_package(directory)
            if descriptor.valid:
                descriptors.append(descriptor)
            else:
                logger.warning(
                    "Skipping invalid Beatrice package {}: {}",
                    directory.name,
                    descriptor.validation_error,
                )
        return descriptors

    def get_model(self, name: str) -> BeatriceModelDescriptor:
        selected = str(name).strip().lower()
        for descriptor in self.discover_models():
            if descriptor.name.lower() == selected:
                return descriptor
        inspected = self.inspect_package(str(name).strip())
        if not inspected.valid and inspected.directory.exists():
            raise ValueError(inspected.validation_error or "Invalid Beatrice package")
        raise LookupError(f"Beatrice model package not found: {name}")


__all__ = [
    "BeatriceModelDescriptor",
    "BeatriceModelManager",
    "EXPECTED_RUNTIME_VERSION",
    "REQUIRED_MODEL_FILES",
]

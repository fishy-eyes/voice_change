"""Discovery and persistent registration for RVC voice models."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from config.rvc_profiles import RVCInferenceConfig, RVCModelProfile, load_rvc_profile


DEFAULT_IMPORTED_INFERENCE = RVCInferenceConfig(
    pitch_shift=0,
    f0_method="rmvpe",
    index_rate=0.3,
    protect=0.33,
    rms_mix_rate=0.25,
)


@dataclass(frozen=True)
class RVCModelDescriptor:
    """A validated model selection exposed to application and GUI code."""

    name: str
    directory: Path
    pth_path: Path
    index_path: Path | None
    profile_path: Path | None
    profile: RVCModelProfile
    is_external: bool = False
    profile_is_default: bool = False


@dataclass(frozen=True)
class RVCImportInspection:
    """Files discovered directly inside a user-selected directory."""

    directory: Path
    pth_candidates: tuple[Path, ...]
    index_candidates: tuple[Path, ...]
    profile_path: Path | None


class RVCModelSelectionRequired(ValueError):
    """Raised when import cannot safely choose one of several files."""

    def __init__(self, file_type: str, candidates: tuple[Path, ...]) -> None:
        self.file_type = file_type
        self.candidates = candidates
        names = ", ".join(path.name for path in candidates)
        super().__init__(f"multiple .{file_type} files require selection: {names}")


class RVCModelManager:
    """Discover built-in models and persist references to external models."""

    _REGISTRY_VERSION = 1

    def __init__(
        self,
        models_root: str | Path,
        *,
        user_models_path: str | Path | None = None,
    ) -> None:
        self.models_root = Path(models_root).resolve()
        self.user_models_path = (
            Path(user_models_path).resolve()
            if user_models_path is not None
            else None
        )
        self._lock = threading.RLock()

    def discover_models(self) -> list[RVCModelDescriptor]:
        """Return valid built-in and registered models without changing files."""
        with self._lock:
            models = self._discover_library_models()
            names = {descriptor.name.lower() for descriptor in models}
            for record in self._load_registry_records():
                try:
                    descriptor = self._load_registered_descriptor(record)
                except Exception as exc:
                    logger.warning("Skipping unavailable imported RVC model: {}", exc)
                    continue
                key = descriptor.name.lower()
                if key in names:
                    logger.warning(
                        "Skipping duplicate imported RVC model name: {}",
                        descriptor.name,
                    )
                    continue
                names.add(key)
                models.append(descriptor)
            return sorted(models, key=lambda descriptor: descriptor.name.lower())

    def get_model(self, name: str) -> RVCModelDescriptor:
        selected = str(name).strip().lower()
        for descriptor in self.discover_models():
            if descriptor.name.lower() == selected:
                return descriptor
        raise LookupError(f"RVC model not found: {name}")

    def inspect_import_directory(
        self,
        directory: str | Path,
    ) -> RVCImportInspection:
        """Scan only the selected directory; no recursive or random selection."""
        selected = Path(directory).expanduser().resolve()
        if not selected.is_dir():
            raise NotADirectoryError(f"RVC model directory not found: {selected}")
        pth_candidates = self._files_with_suffix(selected, ".pth")
        index_candidates = self._files_with_suffix(selected, ".index")
        profile_path = selected / "profile.json"
        if not profile_path.is_file():
            profile_path = None
        return RVCImportInspection(
            directory=selected,
            pth_candidates=pth_candidates,
            index_candidates=index_candidates,
            profile_path=profile_path,
        )

    def import_model(
        self,
        directory: str | Path,
        *,
        pth_path: str | Path | None = None,
        index_path: str | Path | None = None,
        name: str | None = None,
    ) -> RVCModelDescriptor:
        """Validate and persist an external model reference without copying it."""
        with self._lock:
            inspection = self.inspect_import_directory(directory)
            selected_pth = self._select_candidate(
                "pth",
                inspection.pth_candidates,
                pth_path,
                required=True,
            )
            selected_index = self._select_candidate(
                "index",
                inspection.index_candidates,
                index_path,
                required=False,
            )
            assert selected_pth is not None

            records = self._load_registry_records()
            existing = next(
                (
                    record
                    for record in records
                    if self._same_path(record.get("pth_path"), selected_pth)
                ),
                None,
            )
            if existing is not None:
                model_name = str(existing.get("name", "")).strip()
            else:
                requested_name = (
                    str(name).strip()
                    if name is not None
                    else self._profile_name(inspection.profile_path)
                    or inspection.directory.name
                    or selected_pth.stem
                )
                if not requested_name:
                    requested_name = selected_pth.stem
                occupied = {
                    descriptor.name.lower()
                    for descriptor in self._discover_library_models()
                }
                occupied.update(
                    str(record.get("name", "")).strip().lower()
                    for record in records
                    if record is not existing
                )
                model_name = self._unique_name(requested_name, occupied)

            record = {
                "name": model_name,
                "directory": str(inspection.directory),
                "pth_path": str(selected_pth),
                "index_path": str(selected_index) if selected_index else None,
                "profile_path": (
                    str(inspection.profile_path)
                    if inspection.profile_path is not None
                    else None
                ),
            }
            descriptor = self._load_registered_descriptor(record)
            if existing is None:
                records.append(record)
            else:
                records[records.index(existing)] = record
            self._save_registry_records(records)
            logger.info(
                "Imported external RVC model reference: {} ({})",
                descriptor.name,
                descriptor.pth_path,
            )
            return descriptor

    def _discover_library_models(self) -> list[RVCModelDescriptor]:
        if not self.models_root.is_dir():
            logger.warning("RVC model library not found: {}", self.models_root)
            return []

        models: list[RVCModelDescriptor] = []
        directories = sorted(
            self.models_root.iterdir(),
            key=lambda item: item.name.lower(),
        )
        for directory in directories:
            if not directory.is_dir():
                continue
            profile_path = directory / "profile.json"
            if not profile_path.is_file():
                logger.warning("Skipping RVC model without profile.json: {}", directory)
                continue
            try:
                models.append(self._load_library_descriptor(directory, profile_path))
            except Exception as exc:
                logger.warning("Skipping invalid RVC model {}: {}", directory.name, exc)
        return models

    @staticmethod
    def _files_with_suffix(directory: Path, suffix: str) -> tuple[Path, ...]:
        return tuple(
            sorted(
                (
                    path.resolve()
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() == suffix
                ),
                key=lambda path: path.name.lower(),
            )
        )

    @staticmethod
    def _select_candidate(
        file_type: str,
        candidates: tuple[Path, ...],
        requested: str | Path | None,
        *,
        required: bool,
    ) -> Path | None:
        if requested is not None:
            selected = Path(requested).expanduser().resolve()
            if selected not in candidates:
                raise ValueError(
                    f"selected .{file_type} is not in the imported directory: {selected}"
                )
            return selected
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RVCModelSelectionRequired(file_type, candidates)
        if required:
            raise FileNotFoundError("No RVC .pth model file found")
        return None

    @staticmethod
    def _profile_name(profile_path: Path | None) -> str | None:
        if profile_path is None:
            return None
        try:
            values = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(values, Mapping):
            return None
        name = values.get("name")
        if name is None:
            return None
        selected = str(name).strip()
        return selected or None

    @staticmethod
    def _unique_name(requested: str, occupied: set[str]) -> str:
        if requested.lower() not in occupied:
            return requested
        suffix = 2
        while f"{requested} ({suffix})".lower() in occupied:
            suffix += 1
        return f"{requested} ({suffix})"

    @staticmethod
    def _same_path(stored: object, selected: Path) -> bool:
        if not isinstance(stored, str) or not stored.strip():
            return False
        try:
            return Path(stored).expanduser().resolve() == selected
        except (OSError, RuntimeError):
            return False

    @staticmethod
    def _load_imported_inference(profile_path: Path | None) -> RVCInferenceConfig:
        if profile_path is None:
            return DEFAULT_IMPORTED_INFERENCE
        with profile_path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, Mapping):
            raise TypeError("RVC profile root must be an object")
        inference_values = values.get("inference", values)
        if not isinstance(inference_values, Mapping):
            raise TypeError("profile inference must be an object")

        merged: dict[str, Any] = DEFAULT_IMPORTED_INFERENCE.to_dict()
        for field in (
            "pitch_shift",
            "f0_method",
            "index_rate",
            "rms_mix_rate",
            "protect",
        ):
            if field in inference_values:
                merged[field] = inference_values[field]
        if "pitch_shift" not in inference_values and "pitch" in inference_values:
            merged["pitch_shift"] = inference_values["pitch"]
        return RVCInferenceConfig.from_mapping(merged)

    @classmethod
    def _load_library_descriptor(
        cls,
        directory: Path,
        profile_path: Path,
    ) -> RVCModelDescriptor:
        stored = load_rvc_profile(profile_path)
        voice_dir = (
            stored.voice_dir.resolve()
            if stored.voice_dir.is_absolute()
            else (directory / stored.voice_dir).resolve()
        )
        if not voice_dir.is_dir():
            raise FileNotFoundError(f"voice directory not found: {voice_dir}")

        model_file = stored.model_file
        if model_file is None:
            candidates = cls._files_with_suffix(voice_dir, ".pth")
            if len(candidates) != 1:
                raise ValueError("profile must select model_file when .pth count is not one")
            pth_path = candidates[0]
            model_file = Path(pth_path.name)
        else:
            pth_path = model_file if model_file.is_absolute() else voice_dir / model_file
        pth_path = pth_path.resolve()
        if not pth_path.is_file():
            raise FileNotFoundError(f"RVC .pth not found: {pth_path}")

        index_file = stored.index_file
        if index_file is None:
            candidates = cls._files_with_suffix(voice_dir, ".index")
            index_path = candidates[0] if len(candidates) == 1 else None
            index_file = Path(index_path.name) if index_path is not None else None
        else:
            index_path = index_file if index_file.is_absolute() else voice_dir / index_file
            index_path = index_path.resolve()
            if not index_path.is_file():
                raise FileNotFoundError(f"RVC .index not found: {index_path}")

        runtime_profile = RVCModelProfile(
            name=stored.name,
            voice_dir=voice_dir,
            model_file=model_file,
            index_file=index_file,
            inference=stored.inference,
        )
        return RVCModelDescriptor(
            name=stored.name,
            directory=directory.resolve(),
            pth_path=pth_path,
            index_path=index_path,
            profile_path=profile_path.resolve(),
            profile=runtime_profile,
        )

    @classmethod
    def _load_registered_descriptor(
        cls,
        record: Mapping[str, Any],
    ) -> RVCModelDescriptor:
        name = str(record.get("name", "")).strip()
        if not name:
            raise ValueError("imported model name is missing")
        directory = Path(str(record.get("directory", ""))).expanduser().resolve()
        pth_path = Path(str(record.get("pth_path", ""))).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"model directory not found: {directory}")
        if not pth_path.is_file() or pth_path.suffix.lower() != ".pth":
            raise FileNotFoundError(f"RVC .pth not found: {pth_path}")

        stored_index = record.get("index_path")
        index_path = None
        if isinstance(stored_index, str) and stored_index.strip():
            candidate = Path(stored_index).expanduser().resolve()
            if candidate.is_file() and candidate.suffix.lower() == ".index":
                index_path = candidate
            else:
                logger.warning(
                    "Imported RVC index is unavailable; using no-index mode: {}",
                    candidate,
                )

        stored_profile = record.get("profile_path")
        profile_path = None
        if isinstance(stored_profile, str) and stored_profile.strip():
            candidate = Path(stored_profile).expanduser().resolve()
            if candidate.is_file():
                profile_path = candidate
            else:
                logger.warning(
                    "Imported RVC profile is unavailable; using defaults: {}",
                    candidate,
                )
        inference = cls._load_imported_inference(profile_path)
        model_file = (
            Path(pth_path.name)
            if pth_path.parent == directory
            else pth_path
        )
        index_file = None
        if index_path is not None:
            index_file = (
                Path(index_path.name)
                if index_path.parent == directory
                else index_path
            )
        profile = RVCModelProfile(
            name=name,
            voice_dir=directory,
            model_file=model_file,
            index_file=index_file,
            inference=inference,
        )
        return RVCModelDescriptor(
            name=name,
            directory=directory,
            pth_path=pth_path,
            index_path=index_path,
            profile_path=profile_path,
            profile=profile,
            is_external=True,
            profile_is_default=profile_path is None,
        )

    def _load_registry_records(self) -> list[dict[str, Any]]:
        path = self.user_models_path
        if path is None or not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
            if not isinstance(values, Mapping):
                raise TypeError("registry root must be an object")
            records = values.get("models", [])
            if not isinstance(records, list):
                raise TypeError("registry models must be a list")
            return [dict(record) for record in records if isinstance(record, Mapping)]
        except Exception as exc:
            logger.error("Unable to read RVC user model registry {}: {}", path, exc)
            return []

    def _save_registry_records(self, records: list[dict[str, Any]]) -> None:
        path = self.user_models_path
        if path is None:
            raise RuntimeError("RVC user model registry is not configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._REGISTRY_VERSION,
            "models": records,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

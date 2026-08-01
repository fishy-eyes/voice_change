"""Discovery for application-owned RVC voice model profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config.rvc_profiles import RVCModelProfile, load_rvc_profile


@dataclass(frozen=True)
class RVCModelDescriptor:
    """A validated model directory exposed to application and GUI code."""

    name: str
    directory: Path
    pth_path: Path
    index_path: Path | None
    profile_path: Path
    profile: RVCModelProfile


class RVCModelManager:
    """Scan one level below ``models/rvc`` for usable model profiles."""

    def __init__(self, models_root: str | Path) -> None:
        self.models_root = Path(models_root).resolve()

    def discover_models(self) -> list[RVCModelDescriptor]:
        if not self.models_root.is_dir():
            logger.warning("RVC model library not found: {}", self.models_root)
            return []

        models: list[RVCModelDescriptor] = []
        for directory in sorted(self.models_root.iterdir(), key=lambda item: item.name.lower()):
            if not directory.is_dir():
                continue
            profile_path = directory / "profile.json"
            if not profile_path.is_file():
                logger.warning("Skipping RVC model without profile.json: {}", directory)
                continue
            try:
                models.append(self._load_descriptor(directory, profile_path))
            except Exception as exc:
                logger.warning("Skipping invalid RVC model {}: {}", directory.name, exc)
        return models

    def get_model(self, name: str) -> RVCModelDescriptor:
        selected = str(name).strip().lower()
        for descriptor in self.discover_models():
            if descriptor.name.lower() == selected:
                return descriptor
        raise LookupError(f"RVC model not found: {name}")

    @staticmethod
    def _load_descriptor(directory: Path, profile_path: Path) -> RVCModelDescriptor:
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
            candidates = sorted(voice_dir.glob("*.pth"))
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
            candidates = sorted(voice_dir.glob("*.index"))
            index_path = candidates[0].resolve() if len(candidates) == 1 else None
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

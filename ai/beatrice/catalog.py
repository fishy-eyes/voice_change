"""Multi-location catalog for validated Beatrice model packages."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger

from ai.beatrice.model import BeatriceModelDescriptor, BeatriceModelManager


class BeatriceModelCatalog:
    """Merge libraries and registered package paths without copying files."""

    def __init__(
        self,
        default_root: str | Path,
        *,
        registered_packages: Iterable[str | Path] = (),
        additional_roots: Iterable[str | Path] = (),
        on_registered_paths_changed: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        self.models_root = Path(default_root).expanduser().resolve()
        self._validator = BeatriceModelManager(self.models_root)
        self._registered_packages = self._deduplicate(registered_packages)
        self._additional_roots = self._deduplicate(additional_roots)
        self._on_registered_paths_changed = on_registered_paths_changed

    @staticmethod
    def _key(path: Path) -> str:
        return os.path.normcase(str(path.resolve()))

    @classmethod
    def _deduplicate(cls, values: Iterable[str | Path]) -> list[Path]:
        paths: list[Path] = []
        seen: set[str] = set()
        for value in values:
            if value is None or not str(value).strip():
                continue
            path = Path(value).expanduser().resolve()
            key = cls._key(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
        return paths

    @property
    def registered_paths(self) -> tuple[Path, ...]:
        return tuple(self._registered_packages)

    def inspect_package(self, package: str | Path) -> BeatriceModelDescriptor:
        return self._validator.inspect_package(Path(package).expanduser().resolve())

    def register_package(
        self, package: str | Path
    ) -> tuple[BeatriceModelDescriptor, bool]:
        descriptor = self.inspect_package(package)
        if not descriptor.valid:
            raise ValueError(
                descriptor.validation_error or "Invalid Beatrice model package"
            )
        key = self._key(descriptor.directory)
        if any(self._key(path) == key for path in self._registered_packages):
            return descriptor, False
        self._registered_packages.append(descriptor.directory)
        self._notify_changed()
        return descriptor, True

    def remove_registered_package(self, package: str | Path) -> bool:
        target = self._key(Path(package).expanduser().resolve())
        retained = [
            path for path in self._registered_packages if self._key(path) != target
        ]
        if len(retained) == len(self._registered_packages):
            return False
        self._registered_packages = retained
        self._notify_changed()
        return True

    def _notify_changed(self) -> None:
        callback = self._on_registered_paths_changed
        if callback is not None:
            callback(tuple(str(path) for path in self._registered_packages))

    @staticmethod
    def _library_candidates(root: Path) -> list[Path]:
        if not root.is_dir():
            logger.info("Beatrice model library is not configured: {}", root)
            return []
        if tuple(root.glob("beatrice_paraphernalia_*.toml")):
            return [root]
        return [
            path
            for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
            if path.is_dir()
        ]

    def discover_models(self) -> list[BeatriceModelDescriptor]:
        candidates: list[Path] = []
        for root in (self.models_root, *self._additional_roots):
            candidates.extend(self._library_candidates(root))
        candidates.extend(self._registered_packages)

        descriptors: list[BeatriceModelDescriptor] = []
        seen: set[str] = set()
        for path in candidates:
            key = self._key(path)
            if key in seen:
                continue
            seen.add(key)
            descriptor = self.inspect_package(path)
            if descriptor.valid:
                descriptors.append(descriptor)
            else:
                logger.warning(
                    "Skipping invalid Beatrice package {}: {}",
                    path,
                    descriptor.validation_error,
                )

        by_package: dict[str, int] = {}
        for descriptor in descriptors:
            key = descriptor.package.lower()
            by_package[key] = by_package.get(key, 0) + 1
        return [
            replace(
                descriptor,
                name=f"{descriptor.package} — {descriptor.directory}",
            )
            if by_package[descriptor.package.lower()] > 1
            else descriptor
            for descriptor in descriptors
        ]

    def get_model(self, name: str) -> BeatriceModelDescriptor:
        selected = str(name).strip().lower()
        discovered = self.discover_models()
        for descriptor in discovered:
            if descriptor.name.lower() == selected:
                return descriptor
        matches = [
            descriptor
            for descriptor in discovered
            if descriptor.package.lower() == selected
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple Beatrice packages are named {name!r}; select the displayed path"
            )
        candidate = Path(str(name).strip()).expanduser()
        if candidate.is_absolute():
            descriptor = self.inspect_package(candidate)
            if descriptor.valid:
                return descriptor
            if descriptor.directory.exists():
                raise ValueError(
                    descriptor.validation_error or "Invalid Beatrice package"
                )
        raise LookupError(f"Beatrice model package not found: {name}")


__all__ = ["BeatriceModelCatalog"]

"""Machine-local, non-sensitive application settings."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LOCAL_SETTINGS: dict[str, Any] = {
    "startup": {
        "autoload_last_model": False,
        "last_backend": "",
        "last_model": "",
    },
    "beatrice": {
        "runtime_dir": "",
        "model_roots": [],
        "last_model": "",
        "last_speaker": "",
        "speaker_presets": {},
    }
}


class LocalSettingsStore:
    """Read and atomically update one machine-local JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return deepcopy(DEFAULT_LOCAL_SETTINGS)
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return deepcopy(DEFAULT_LOCAL_SETTINGS)
        if not isinstance(loaded, dict):
            return deepcopy(DEFAULT_LOCAL_SETTINGS)
        data = deepcopy(loaded)
        for section_name, defaults in DEFAULT_LOCAL_SETTINGS.items():
            section = data.get(section_name)
            if not isinstance(section, dict):
                section = {}
                data[section_name] = section
            for key, value in defaults.items():
                if key not in section or not isinstance(section[key], type(value)):
                    section[key] = deepcopy(value)
        return data

    @property
    def data(self) -> Mapping[str, Any]:
        return deepcopy(self._data)

    @property
    def beatrice(self) -> Mapping[str, Any]:
        return deepcopy(self._data["beatrice"])

    @property
    def startup(self) -> Mapping[str, Any]:
        return deepcopy(self._data["startup"])

    def update_beatrice(self, **changes: Any) -> Mapping[str, Any]:
        section = self._data.setdefault("beatrice", {})
        section.update(changes)
        self._save()
        return deepcopy(section)

    def update_startup(self, **changes: Any) -> Mapping[str, Any]:
        section = self._data.setdefault("startup", {})
        section.update(changes)
        self._save()
        return deepcopy(section)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def normalized_existing_paths(values) -> tuple[Path, ...]:
    """Resolve, de-duplicate, and retain only existing directories."""
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if value is None or not str(value).strip():
            continue
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        key = os.path.normcase(str(path))
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        paths.append(path)
    return tuple(paths)


__all__ = ["DEFAULT_LOCAL_SETTINGS", "LocalSettingsStore", "normalized_existing_paths"]

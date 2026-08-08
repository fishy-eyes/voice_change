"""Defensive loader for a user-supplied Beatrice v2 Python runtime."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ai.beatrice.model import (
    BeatriceModelDescriptor,
    MODEL_API_VERSION,
)


EXPECTED_CONSTANTS = {
    "IN_SAMPLE_RATE": 16_000,
    "OUT_SAMPLE_RATE": 24_000,
    "IN_HOP_LENGTH": 160,
    "OUT_HOP_LENGTH": 240,
}
REQUIRED_CONVERTER_METHODS = (
    "convert",
    "is_ready",
    "last_error",
    "last_backend_error",
    "get_num_speakers",
    "set_config",
    "get_codebook_size",
    "get_max_formant_shift",
)
_DLL_HANDLES: list[Any] = []


class RuntimeUnavailableError(RuntimeError):
    """Raised only when the optional external runtime cannot be used."""


def unpack_runtime_output(value: Any) -> tuple[np.ndarray, Any]:
    """Accept the observed tuple API while remaining defensive about drift."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise RuntimeError(
                f"Beatrice convert() returned a {len(value)}-item tuple; expected 2"
            )
        audio, auxiliary = value
    else:
        audio, auxiliary = value, None
    output = np.asarray(audio, dtype=np.float32).reshape(-1)
    return output, auxiliary


class BeatriceRuntimeLoader:
    """Discover from an explicit/user/env path without downloading anything."""

    env_name = "VOICE_CHANGE_BEATRICE_RUNTIME_DIR"

    def __init__(self, runtime_root: str | Path | None = None) -> None:
        configured = runtime_root
        if configured is None:
            configured = os.environ.get(self.env_name)
        self.runtime_root = self._normalize_root(configured)
        self.module: Any | None = None
        self.runtime_implementation_version: str | None = None

    @staticmethod
    def _normalize_root(value: str | Path | None) -> Path | None:
        if value is None or not str(value).strip():
            return None
        root = Path(value).expanduser().resolve()
        for candidate in (root, root / "_internal", root / "dist" / "main" / "_internal"):
            if (candidate / "beatrice" / "__init__.py").is_file():
                return candidate
        return root

    @property
    def available(self) -> bool:
        root = self.runtime_root
        return bool(root is not None and (root / "beatrice" / "__init__.py").is_file())

    def validate(self) -> dict[str, Any]:
        """Load and verify the module contract without creating a converter."""
        module = self.load(MODEL_API_VERSION)
        return {
            "model_api_version": MODEL_API_VERSION,
            # Compatibility alias for older internal callers. This value is a
            # model/API identifier, not the runtime implementation revision.
            "version": MODEL_API_VERSION,
            "runtime_implementation_version": self.runtime_implementation_version,
            "runtime_root": str(self.runtime_root),
            **EXPECTED_CONSTANTS,
            "simple_beatrice": callable(getattr(module, "SimpleBeatrice", None)),
        }

    def load(self, model_api_version: str = MODEL_API_VERSION):
        if model_api_version != MODEL_API_VERSION:
            raise RuntimeUnavailableError(
                f"Unsupported Beatrice model API {model_api_version!r}; "
                f"only {MODEL_API_VERSION!r} is accepted / "
                f"仅支持模型 API {MODEL_API_VERSION}"
            )
        root = self.runtime_root
        if not self.available or root is None:
            raise RuntimeUnavailableError(
                "Beatrice runtime is unavailable. Set "
                f"{self.env_name} or choose a runtime directory / "
                "未找到 Beatrice 运行库，请配置运行库目录"
            )
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(root_text))
        try:
            package = importlib.import_module("beatrice")
            package_file = Path(getattr(package, "__file__", "")).resolve()
            if root not in package_file.parents:
                raise RuntimeUnavailableError(
                    f"A different Beatrice package is already imported: {package_file}"
                )
            loader = getattr(package, "load_beatrice", None)
            if not callable(loader):
                raise AttributeError("beatrice.load_beatrice is missing")
            module = loader(model_api_version)
            self._validate_module(module)
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError(
                f"Beatrice runtime load failed / Beatrice 运行库加载失败: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self.module = module
        self.runtime_implementation_version = self._implementation_version(package)
        return module

    @staticmethod
    def _implementation_version(package: Any) -> str | None:
        """Read an optional package revision without requiring one to exist."""
        for name in ("__version__", "RUNTIME_VERSION", "VERSION"):
            value = getattr(package, name, None)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _validate_module(module: Any) -> None:
        mismatches: dict[str, tuple[Any, int]] = {}
        for name, expected in EXPECTED_CONSTANTS.items():
            actual = getattr(module, name, None)
            try:
                actual_int = int(actual)
            except (TypeError, ValueError):
                actual_int = actual
            if actual_int != expected:
                mismatches[name] = (actual, expected)
        if mismatches:
            raise RuntimeUnavailableError(
                f"Unexpected Beatrice streaming constants: {mismatches}"
            )
        if not callable(getattr(module, "SimpleBeatrice", None)):
            raise RuntimeUnavailableError("Beatrice SimpleBeatrice is missing")
        converter_type = getattr(module, "SimpleBeatrice", None)
        if isinstance(converter_type, type):
            missing = [
                name for name in REQUIRED_CONVERTER_METHODS
                if not callable(getattr(converter_type, name, None))
            ]
            if missing:
                raise RuntimeUnavailableError(
                    "Beatrice SimpleBeatrice API mismatch: " + ", ".join(missing)
                )

    def create_converter(self, descriptor: BeatriceModelDescriptor, config: Any):
        if not descriptor.valid:
            raise ValueError(descriptor.validation_error or "Invalid Beatrice package")
        module = self.load(descriptor.model_api_version)
        files = descriptor.required_files
        converter = module.SimpleBeatrice(
            *(str(files[name]) for name in files)
        )
        missing_methods = [
            name for name in REQUIRED_CONVERTER_METHODS
            if not callable(getattr(converter, name, None))
        ]
        if missing_methods:
            raise RuntimeUnavailableError(
                "Beatrice runtime API mismatch; missing methods: "
                + ", ".join(missing_methods)
            )
        if not bool(converter.is_ready()):
            raise RuntimeUnavailableError(
                "Beatrice converter is not ready: "
                f"last_error={converter.last_error()} "
                f"backend_error={converter.last_backend_error()}"
            )
        speaker_count = int(converter.get_num_speakers())
        if speaker_count != descriptor.speaker_count:
            raise RuntimeUnavailableError(
                "Beatrice speaker count mismatch: "
                f"runtime={speaker_count}, metadata={descriptor.speaker_count}"
            )
        parameters = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        converter.set_config(**parameters)
        details = {
            "model_api_version": descriptor.model_api_version,
            # Compatibility alias; see validate().
            "version": descriptor.model_api_version,
            "runtime_implementation_version": self.runtime_implementation_version,
            "runtime_root": str(self.runtime_root),
            "num_speakers": speaker_count,
            "codebook_size": int(converter.get_codebook_size()),
            "max_formant_shift": int(converter.get_max_formant_shift()),
            "pitch_shift_min": float(
                getattr(module, "MIN_PITCH_SHIFT_SEMITONE", -24.0)
            ),
            "pitch_shift_max": float(
                getattr(module, "MAX_PITCH_SHIFT_SEMITONE", 24.0)
            ),
            "source_pitch_min": getattr(module, "MIN_SOURCE_PITCH", None),
            "source_pitch_max": getattr(module, "MAX_SOURCE_PITCH", None),
            **EXPECTED_CONSTANTS,
        }
        return converter, module, details


__all__ = [
    "BeatriceRuntimeLoader",
    "EXPECTED_CONSTANTS",
    "RuntimeUnavailableError",
    "unpack_runtime_output",
]

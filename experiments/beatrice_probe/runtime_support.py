"""Shared helpers for the isolated Beatrice v2 probe.

This module deliberately has no imports from the production voice_change code.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import tomllib
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[1]
LOCAL_BEATRICE_ROOT = REPO_ROOT / "local_assets" / "beatrice"
GENERATED_DIR = LOCAL_BEATRICE_ROOT / "generated" / "beatrice_probe"
OUTPUTS_DIR = GENERATED_DIR / "outputs"
RESULTS_DIR = GENERATED_DIR / "results"

REQUIRED_MODEL_FILES = (
    "phone_extractor.bin",
    "pitch_estimator.bin",
    "embedding_setter.bin",
    "waveform_generator.bin",
    "speaker_embeddings.bin",
)
SUPPORTED_VERSIONS = {
    "2.0.0-alpha.2": "v20a2",
    "2.0.0-beta.1": "v20b1",
    "2.0.0-rc.0": "v20rc0",
}

_DLL_HANDLES: list[Any] = []


@dataclass(frozen=True)
class AudioStats:
    sample_rate: int
    frames: int
    duration_seconds: float
    peak: float
    rms: float
    clipping_ratio: float
    nan_count: int
    inf_count: int
    all_zero: bool


@dataclass(frozen=True)
class ConversionResult:
    audio: np.ndarray
    sample_rate: int
    block_times_ms: tuple[float, ...]
    runtime_aux_values: tuple[Any, ...]
    padded_input_frames: int
    raw_output_frames: int


def normalize_runtime_root(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    root = Path(value).expanduser().resolve()
    candidates = (
        root,
        root / "_internal",
        root / "dist" / "main" / "_internal",
    )
    for candidate in candidates:
        if (candidate / "beatrice" / "__init__.py").is_file():
            return candidate
    return root


def import_beatrice(
    runtime_root: str | os.PathLike[str] | None = None,
    version: str = "2.0.0-rc.0",
):
    """Import the versioned public Beatrice Python API."""

    root = normalize_runtime_root(runtime_root)
    if root is not None:
        if not root.exists():
            raise FileNotFoundError(f"Runtime root does not exist: {root}")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(root_text))

    package = importlib.import_module("beatrice")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported Beatrice paraphernalia version: {version}")
    module = package.load_beatrice(version)
    return module, root


def find_metadata_file(model_path: str | os.PathLike[str]) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".toml":
            raise ValueError(
                "A model file must be a beatrice_paraphernalia_*.toml file; "
                f"got {path.name}"
            )
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    direct = sorted(path.glob("beatrice_paraphernalia_*.toml"))
    candidates = direct or sorted(path.rglob("beatrice_paraphernalia_*.toml"))
    if not candidates:
        raise FileNotFoundError(
            f"No beatrice_paraphernalia_*.toml found beneath {path}"
        )
    if len(candidates) != 1:
        joined = "\n  ".join(str(candidate) for candidate in candidates)
        raise ValueError(
            f"Model path is ambiguous; found {len(candidates)} metadata files:\n  {joined}"
        )
    return candidates[0]


def load_model_metadata(
    model_path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    metadata_file = find_metadata_file(model_path)
    with metadata_file.open("rb") as stream:
        metadata = tomllib.load(stream)
    return metadata_file, metadata


def validate_model_files(model_dir: Path) -> dict[str, Path]:
    files = {name: model_dir / name for name in REQUIRED_MODEL_FILES}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Beatrice model is incomplete at {model_dir}; missing: {', '.join(missing)}"
        )
    return files


def create_converter(
    model_path: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str] | None = None,
    *,
    target_speaker: int = 0,
    formant_shift: float = 0.0,
    pitch_shift_semitone: float = 0.0,
    min_source_pitch: float = 30.0,
    max_source_pitch: float = 1100.0,
    vq_num_neighbors: int = 4,
):
    metadata_file, metadata = load_model_metadata(model_path)
    model_info = metadata.get("model", {})
    version = model_info.get("version")
    if not isinstance(version, str):
        raise ValueError(f"Model metadata has no string model.version: {metadata_file}")
    module, resolved_runtime_root = import_beatrice(runtime_root, version)
    files = validate_model_files(metadata_file.parent)

    started = perf_counter()
    converter = module.SimpleBeatrice(
        str(files["phone_extractor.bin"]),
        str(files["pitch_estimator.bin"]),
        str(files["embedding_setter.bin"]),
        str(files["waveform_generator.bin"]),
        str(files["speaker_embeddings.bin"]),
    )
    load_seconds = perf_counter() - started
    if not converter.is_ready():
        raise RuntimeError(
            "Beatrice runtime did not become ready: "
            f"last_error={converter.last_error()} "
            f"last_backend_error={converter.last_backend_error()}"
        )

    num_speakers = int(converter.get_num_speakers())
    if not 0 <= target_speaker < num_speakers:
        raise ValueError(
            f"target_speaker must be in [0, {num_speakers - 1}], got {target_speaker}"
        )
    converter.set_config(
        target_speaker=target_speaker,
        formant_shift=float(formant_shift),
        pitch_shift_semitone=float(pitch_shift_semitone),
        min_source_pitch=float(min_source_pitch),
        max_source_pitch=float(max_source_pitch),
        vq_num_neighbors=int(vq_num_neighbors),
    )
    details = {
        "metadata_file": str(metadata_file),
        "model_dir": str(metadata_file.parent),
        "version": version,
        "runtime_root": str(resolved_runtime_root) if resolved_runtime_root else None,
        "load_seconds": load_seconds,
        "num_speakers": num_speakers,
        "in_sample_rate": int(module.IN_SAMPLE_RATE),
        "out_sample_rate": int(module.OUT_SAMPLE_RATE),
        "in_hop_length": int(module.IN_HOP_LENGTH),
        "out_hop_length": int(module.OUT_HOP_LENGTH),
        "phone_channels": int(module.PHONE_CHANNELS),
        "pitch_bins": int(module.PITCH_BINS),
        "pitch_bins_per_octave": int(module.PITCH_BINS_PER_OCTAVE),
        "waveform_generator_hidden_channels": int(
            module.WAVEFORM_GENERATOR_HIDDEN_CHANNELS
        ),
        "codebook_size": int(converter.get_codebook_size()),
        "max_formant_shift": int(converter.get_max_formant_shift()),
    }
    return converter, module, metadata, details


def read_mono_audio(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {audio_path}")
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(
            f"Input must be mono; {audio_path} has {audio.shape[1]} channels"
        )
    mono = np.ascontiguousarray(audio[:, 0], dtype=np.float32)
    return mono, int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    ratio = Fraction(target_rate, source_rate)
    converted = resample_poly(audio, ratio.numerator, ratio.denominator)
    return np.ascontiguousarray(converted, dtype=np.float32)


def audio_stats(audio: np.ndarray, sample_rate: int) -> AudioStats:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    finite_values = values[finite]
    peak = float(np.max(np.abs(finite_values))) if finite_values.size else math.nan
    rms = (
        float(np.sqrt(np.mean(np.square(finite_values.astype(np.float64)))))
        if finite_values.size
        else math.nan
    )
    clipping_ratio = (
        float(np.mean(np.abs(finite_values) >= 0.999))
        if finite_values.size
        else math.nan
    )
    return AudioStats(
        sample_rate=int(sample_rate),
        frames=int(values.size),
        duration_seconds=float(values.size / sample_rate),
        peak=peak,
        rms=rms,
        clipping_ratio=clipping_ratio,
        nan_count=int(np.isnan(values).sum()),
        inf_count=int(np.isinf(values).sum()),
        all_zero=bool(values.size == 0 or np.all(values == 0.0)),
    )


def unpack_runtime_output(value: Any) -> tuple[np.ndarray, Any]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise RuntimeError(
                f"Runtime returned an unexpected {len(value)}-element tuple"
            )
        audio, auxiliary = value
    else:
        audio, auxiliary = value, None
    return np.asarray(audio, dtype=np.float32).reshape(-1), auxiliary


def convert_audio_blocks(
    converter,
    module,
    input_audio: np.ndarray,
    input_sample_rate: int,
) -> ConversionResult:
    native_input = resample_audio(
        input_audio, input_sample_rate, int(module.IN_SAMPLE_RATE)
    )
    hop = int(module.IN_HOP_LENGTH)
    out_hop = int(module.OUT_HOP_LENGTH)
    if hop <= 0 or out_hop <= 0:
        raise RuntimeError(f"Invalid runtime hop sizes: input={hop}, output={out_hop}")

    padded_frames = math.ceil(native_input.size / hop) * hop
    padded = np.pad(native_input, (0, padded_frames - native_input.size))
    outputs: list[np.ndarray] = []
    timings: list[float] = []
    auxiliary_values: list[Any] = []
    for start in range(0, padded_frames, hop):
        block = np.ascontiguousarray(padded[start : start + hop], dtype=np.float32)
        started = perf_counter()
        converted, auxiliary = unpack_runtime_output(converter.convert(block))
        timings.append((perf_counter() - started) * 1000.0)
        auxiliary_values.append(auxiliary)
        if converted.size != out_hop:
            raise RuntimeError(
                f"Runtime returned {converted.size} samples for a {hop}-sample block; "
                f"expected {out_hop}"
            )
        outputs.append(converted)

    raw = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    expected_frames = round(
        native_input.size * int(module.OUT_SAMPLE_RATE) / int(module.IN_SAMPLE_RATE)
    )
    trimmed = np.ascontiguousarray(raw[:expected_frames], dtype=np.float32)
    return ConversionResult(
        audio=trimmed,
        sample_rate=int(module.OUT_SAMPLE_RATE),
        block_times_ms=tuple(timings),
        runtime_aux_values=tuple(auxiliary_values),
        padded_input_frames=int(padded_frames),
        raw_output_frames=int(raw.size),
    )


def validate_converted_audio(
    audio: np.ndarray, sample_rate: int, input_duration: float
) -> AudioStats:
    stats = audio_stats(audio, sample_rate)
    problems: list[str] = []
    if stats.nan_count:
        problems.append(f"contains {stats.nan_count} NaN samples")
    if stats.inf_count:
        problems.append(f"contains {stats.inf_count} Inf samples")
    if stats.all_zero:
        problems.append("is entirely zero")
    if stats.clipping_ratio > 0.01:
        problems.append(f"has severe clipping ({stats.clipping_ratio:.2%})")
    allowed_duration_error = max(0.02, input_duration * 0.01)
    if abs(stats.duration_seconds - input_duration) > allowed_duration_error:
        problems.append(
            "has unreasonable duration "
            f"({stats.duration_seconds:.4f}s vs input {input_duration:.4f}s)"
        )
    if problems:
        raise RuntimeError("Converted audio validation failed: " + "; ".join(problems))
    return stats


def ensure_output_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = OUTPUTS_DIR / path
    resolved = path.expanduser().resolve()
    outputs_root = OUTPUTS_DIR.resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"Output must stay beneath {outputs_root}: {resolved}") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def stats_dict(stats: AudioStats) -> dict[str, Any]:
    return asdict(stats)


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.percentile(array, q)) if array.size else math.nan

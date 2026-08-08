"""Shared primitives for isolated Beatrice quality experiments."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import soxr

from ai.beatrice.model import BeatriceModelDescriptor, BeatriceModelManager
from ai.beatrice.runtime import BeatriceRuntimeLoader, unpack_runtime_output
from ai.beatrice.streaming_adapter import BeatriceStreamingAdapter
from ai.voice_engine.beatrice import BeatriceConfig, BeatriceVoiceEngine
from ai.voice_worker import VoiceConversionWorker
from customization.beatrice import (
    BeatriceParameterSet,
    BeatriceTuningCapabilities,
    analyze_beatrice_voice,
)
from customization.quality_checker import extract_audio_features


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
DEFAULT_INPUT = REPO_ROOT / "tests" / "assets" / "input.wav"
DEFAULT_MODEL_ROOT = REPO_ROOT / "experiments" / "beatrice_probe" / "assets" / "model"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "experiments" / "beatrice_probe" / "assets" / "runtime"
EXTERNAL_SAMPLE_RATE = 48_000
CALLBACK_SAMPLES = 256


def ensure_directories() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_audio(path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    path = path.expanduser().resolve()
    info = sf.info(path)
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    return np.ascontiguousarray(mono), int(sample_rate), {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "source_subtype": info.subtype,
        "analysis_dtype": str(mono.dtype),
    }


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return np.ascontiguousarray(values)
    return np.asarray(
        soxr.resample(values, source_rate, target_rate, quality="HQ"),
        dtype=np.float32,
    ).reshape(-1)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    if not bool(finite.all()):
        safe = np.where(finite, values, 0.0)
    else:
        safe = values
    features = extract_audio_features(safe, sample_rate)
    frame_count = max(0, 1 + (len(safe) - max(64, round(sample_rate * 0.040))) // max(32, round(sample_rate * 0.020)))
    f0 = features.f0_values
    return {
        "sample_rate": int(sample_rate),
        "samples": int(len(values)),
        "duration_seconds": len(values) / float(sample_rate),
        "dtype": str(values.dtype),
        "rms": float(np.sqrt(np.mean(np.square(safe, dtype=np.float64)))) if len(safe) else 0.0,
        "peak": float(np.max(np.abs(safe))) if len(safe) else 0.0,
        "clipping_ratio": float(np.mean(np.abs(safe) >= 0.995)) if len(safe) else 0.0,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "voiced_frame_ratio": float(features.voiced_frame_ratio),
        "valid_f0_ratio": float(len(f0) / frame_count) if frame_count else 0.0,
        "f0_count": int(len(f0)),
        "f0_p5": float(np.percentile(f0, 5)) if len(f0) else None,
        "f0_median": float(np.median(f0)) if len(f0) else None,
        "f0_p95": float(np.percentile(f0, 95)) if len(f0) else None,
        "pitch_discontinuity_ratio": float(features.pitch_discontinuity_ratio),
    }


def load_context(
    model_root: Path,
    runtime_root: Path,
    package: str = "jvs",
) -> tuple[BeatriceModelDescriptor, BeatriceRuntimeLoader, dict[str, Any], BeatriceTuningCapabilities]:
    descriptor = BeatriceModelManager(model_root).get_model(package)
    loader = BeatriceRuntimeLoader(runtime_root)
    converter, _module, runtime = loader.create_converter(descriptor, BeatriceConfig())
    del converter
    capabilities = BeatriceTuningCapabilities.from_runtime(runtime)
    return descriptor, loader, runtime, capabilities


def parameter_config(values: BeatriceParameterSet) -> BeatriceConfig:
    return BeatriceConfig(**values.to_engine_changes())


def render_native(
    audio: np.ndarray,
    source_rate: int,
    descriptor: BeatriceModelDescriptor,
    loader: BeatriceRuntimeLoader,
    parameters: BeatriceParameterSet,
    destination: Path,
) -> dict[str, Any]:
    source = resample_audio(audio, source_rate, 16_000)
    original_samples = len(source)
    padding = (-original_samples) % 160
    if padding:
        source = np.pad(source, (0, padding))
    started = perf_counter()
    converter, module, runtime = loader.create_converter(
        descriptor, parameter_config(parameters)
    )
    load_seconds = perf_counter() - started
    parts: list[np.ndarray] = []
    block_times: list[float] = []
    for offset in range(0, len(source), 160):
        block_started = perf_counter()
        output, _aux = unpack_runtime_output(converter.convert(source[offset : offset + 160]))
        block_times.append((perf_counter() - block_started) * 1000.0)
        if output.size != int(module.OUT_HOP_LENGTH):
            raise RuntimeError(f"native output block has {output.size} samples")
        parts.append(output)
    rendered = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    expected = round(original_samples * 24_000 / 16_000)
    rendered = np.ascontiguousarray(rendered[:expected], dtype=np.float32)
    save_wav(destination, rendered, 24_000)
    times = np.asarray(block_times, dtype=np.float64)
    return {
        "path": str(destination.resolve()),
        "path_kind": "native continuous 160-sample inference",
        "output_sample_rate": 24_000,
        "input_padding_samples_at_16khz": int(padding),
        "load_seconds": load_seconds,
        "processing_seconds": float(times.sum() / 1000.0),
        "processing_rtf": float(times.sum() / 1000.0 / max(len(rendered) / 24_000.0, 1e-9)),
        "block_ms": percentile_summary(times),
        "runtime": runtime,
        "metrics": audio_metrics(rendered, 24_000),
        "audio": rendered,
    }


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(tuple(values), dtype=np.float64)
    if not len(data):
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def render_streaming(
    audio_48khz: np.ndarray,
    descriptor: BeatriceModelDescriptor,
    loader: BeatriceRuntimeLoader,
    parameters: BeatriceParameterSet,
    destination: Path,
    *,
    quality: str = "QQ",
) -> dict[str, Any]:
    source = np.asarray(audio_48khz, dtype=np.float32).reshape(-1)
    padding = (-len(source)) % CALLBACK_SAMPLES
    padded = np.pad(source, (0, padding)) if padding else source
    config = parameter_config(parameters)
    adapter = BeatriceStreamingAdapter(
        lambda: loader.create_converter(descriptor, config),
        callback_samples=CALLBACK_SAMPLES,
        startup_buffer_samples=512,
        resampler_quality=quality,
    )
    parts: list[np.ndarray] = []
    times: list[float] = []
    started = perf_counter()
    try:
        for offset in range(0, len(padded), CALLBACK_SAMPLES):
            block_started = perf_counter()
            parts.append(adapter.process(padded[offset : offset + CALLBACK_SAMPLES]))
            times.append((perf_counter() - block_started) * 1000.0)
        wall_seconds = perf_counter() - started
        stats = adapter.stats()
        runtime = dict(adapter.runtime_details)
    finally:
        adapter.close()
    rendered = np.concatenate(parts)[: len(source)] if parts else np.empty(0, dtype=np.float32)
    rendered = np.ascontiguousarray(rendered, dtype=np.float32)
    save_wav(destination, rendered, EXTERNAL_SAMPLE_RATE)
    callback_ms = 1000.0 * CALLBACK_SAMPLES / EXTERNAL_SAMPLE_RATE
    return {
        "path": str(destination.resolve()),
        "path_kind": "production BeatriceStreamingAdapter",
        "resampler_quality": quality,
        "output_sample_rate": EXTERNAL_SAMPLE_RATE,
        "input_padding_samples_at_48khz": int(padding),
        "processing_seconds": wall_seconds,
        "processing_rtf": wall_seconds / max(len(source) / EXTERNAL_SAMPLE_RATE, 1e-9),
        "callback_deadline_ms": callback_ms,
        "callback_deadline_miss_count": int(np.sum(np.asarray(times) > callback_ms)),
        "callback_ms": percentile_summary(times),
        "buffering_latency_ms": float(stats["startup_silence_ms"]),
        "adapter": stats,
        "runtime": runtime,
        "metrics": audio_metrics(rendered, EXTERNAL_SAMPLE_RATE),
        "audio": rendered,
    }


def run_worker_diagnostic(
    audio_48khz: np.ndarray,
    descriptor: BeatriceModelDescriptor,
    runtime_root: Path,
    parameters: BeatriceParameterSet,
) -> dict[str, Any]:
    source = np.asarray(audio_48khz, dtype=np.float32).reshape(-1)
    padding = (-len(source)) % CALLBACK_SAMPLES
    padded = np.pad(source, (0, padding)) if padding else source
    engine = BeatriceVoiceEngine(
        descriptor,
        runtime_root=runtime_root,
        config=parameter_config(parameters),
    )
    engine.load_model()
    worker = VoiceConversionWorker(engine, chunk_size=CALLBACK_SAMPLES, max_queue_size=8)
    if not worker.start():
        engine.unload_model()
        raise RuntimeError("worker failed to start")
    period = CALLBACK_SAMPLES / EXTERNAL_SAMPLE_RATE
    accepted = 0
    received = 0
    input_max = 0
    output_max = 0
    inference_times: list[float] = []
    observed_infers = 0
    schedule_misses = 0
    started = perf_counter()
    try:
        for index, offset in enumerate(range(0, len(padded), CALLBACK_SAMPLES)):
            target = started + index * period
            remaining = target - perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif -remaining > period:
                schedule_misses += 1
            result = worker.get(timeout=0.0)
            if result is not None:
                received += 1
            accepted += int(worker.put(padded[offset : offset + CALLBACK_SAMPLES]))
            input_max = max(input_max, worker.input_pending)
            output_max = max(output_max, worker.output_pending)
            if worker.infer_count > observed_infers:
                inference_times.append(worker.last_infer_ms)
                observed_infers = worker.infer_count
        deadline = perf_counter() + 3.0
        while received < accepted and perf_counter() < deadline:
            result = worker.get(timeout=0.01)
            if result is not None:
                received += 1
            input_max = max(input_max, worker.input_pending)
            output_max = max(output_max, worker.output_pending)
            if worker.infer_count > observed_infers:
                inference_times.append(worker.last_infer_ms)
                observed_infers = worker.infer_count
        adapter_stats = engine.adapter.stats() if engine.adapter is not None else {}
        report = {
            "input_blocks": int(len(padded) // CALLBACK_SAMPLES),
            "accepted_blocks": int(accepted),
            "received_blocks": int(received),
            "input_queue_max_depth": int(input_max),
            "output_queue_max_depth": int(output_max),
            "input_drop_or_overflow_count": int(worker.input_drop_count),
            "output_drop_or_overflow_count": int(worker.output_drop_count),
            "adapter_underflow_count": int(adapter_stats.get("underflow_count", 0)),
            "adapter_overflow_count": int(adapter_stats.get("overflow_count", 0)),
            "reset_count": int(worker.recovery_count),
            "continuity_error_count": int(worker.continuity_error_count),
            "stale_generation_count": 0 if worker.continuity_generation == 0 else None,
            "worker_error_count": int(worker.error_count),
            "recovery_failure_count": int(worker.recovery_failure_count),
            "inference_ms": percentile_summary(inference_times),
            "callback_deadline_ms": period * 1000.0,
            "inference_deadline_miss_count": int(np.sum(np.asarray(inference_times) > period * 1000.0)),
            "producer_schedule_miss_count": int(schedule_misses),
            "adapter": adapter_stats,
        }
    finally:
        worker.stop(timeout=5.0)
        engine.unload_model()
    return report


def public_result(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "audio"}


def parameters_dict(values: BeatriceParameterSet) -> dict[str, Any]:
    return asdict(values)

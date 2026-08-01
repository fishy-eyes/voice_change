"""Detailed, non-realtime RVC inference profiling and comparison tool.

The production pipeline already reports three coarse timers. During this
standalone diagnostic only, this module temporarily wraps the external
pipeline's callable dependencies to isolate HuBERT, F0, FAISS and generator
wall-clock time without editing the external RVC source tree.

Example:
    python -u tests/test_rvc_inference_profile_benchmark.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    from tests.test_rvc_realtime_benchmark import (
        load_audio,
        milliseconds_to_samples,
    )
    from tests.test_rvc_short_chunk_benchmark import (
        select_distinct_chunks,
        validate_result,
    )
except ModuleNotFoundError:
    from test_rvc_realtime_benchmark import load_audio, milliseconds_to_samples
    from test_rvc_short_chunk_benchmark import select_distinct_chunks, validate_result


DEFAULT_CHUNKS_MS = (325, 500)
DEFAULT_F0_METHODS = ("rmvpe", "pm", "fcpe")
DEFAULT_INDEX_RATES = (0.0, 0.3, 0.5)
EXACT_STAGE_KEYS = (
    "hubert_ms",
    "f0_ms",
    "index_read_ms",
    "index_reconstruct_ms",
    "index_search_ms",
    "synth_ms",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def log(message: str = "") -> None:
    print(message, flush=True)


class _TimedFaissIndex:
    def __init__(self, index: Any, probe: "DetailedStageProbe") -> None:
        self._index = index
        self._probe = probe

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self._probe.measure(
            "index_search_ms", self._index.search, *args, **kwargs,
        )

    def reconstruct_n(self, *args: Any, **kwargs: Any) -> Any:
        return self._probe.measure(
            "index_reconstruct_ms", self._index.reconstruct_n, *args, **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._index, name)


class DetailedStageProbe(AbstractContextManager):
    """Temporarily isolate exact stages around one loaded engine."""

    def __init__(self, engine: Any) -> None:
        require(engine.is_loaded, "engine must be loaded before profiling")
        self.engine = engine
        self.pipeline = engine._pipeline
        self.module = importlib.import_module(self.pipeline.__class__.__module__)
        self._current: Optional[dict[str, float]] = None
        self._original_hubert: Optional[Callable[..., Any]] = None
        self._original_f0: Optional[Callable[..., Any]] = None
        self._original_cuda_graph: Optional[Callable[..., Any]] = None
        self._original_read_index: Optional[Callable[..., Any]] = None
        self._f0_was_instance_attribute = False

    def synchronize(self) -> None:
        if not str(self.engine.device).lower().startswith("cuda"):
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.engine.device)

    def measure(
        self,
        key: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.synchronize()
        started_at = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.synchronize()
            if self._current is not None:
                self._current[key] += (time.perf_counter() - started_at) * 1000.0

    def begin_sample(self) -> None:
        require(self._current is None, "a profiling sample is already active")
        self._current = {key: 0.0 for key in EXACT_STAGE_KEYS}

    def finish_sample(self) -> dict[str, float]:
        require(self._current is not None, "no profiling sample is active")
        result = dict(self._current)
        self._current = None
        return result

    def abort_sample(self) -> None:
        self._current = None

    def __enter__(self) -> "DetailedStageProbe":
        module = self.module
        self._original_hubert = module.extract_hubert_features
        self._original_f0 = self.pipeline.get_f0
        self._original_cuda_graph = module.run_cuda_graph
        self._original_read_index = module.faiss.read_index
        self._f0_was_instance_attribute = "get_f0" in self.pipeline.__dict__

        def timed_hubert(*args: Any, **kwargs: Any) -> Any:
            return self.measure(
                "hubert_ms", self._original_hubert, *args, **kwargs,
            )

        def timed_f0(*args: Any, **kwargs: Any) -> Any:
            return self.measure("f0_ms", self._original_f0, *args, **kwargs)

        def timed_cuda_graph(*args: Any, **kwargs: Any) -> Any:
            return self.measure(
                "synth_ms", self._original_cuda_graph, *args, **kwargs,
            )

        def timed_read_index(*args: Any, **kwargs: Any) -> _TimedFaissIndex:
            index = self.measure(
                "index_read_ms", self._original_read_index, *args, **kwargs,
            )
            return _TimedFaissIndex(index, self)

        module.extract_hubert_features = timed_hubert
        self.pipeline.get_f0 = timed_f0
        module.run_cuda_graph = timed_cuda_graph
        module.faiss.read_index = timed_read_index
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        module = self.module
        if self._original_hubert is not None:
            module.extract_hubert_features = self._original_hubert
        if self._original_cuda_graph is not None:
            module.run_cuda_graph = self._original_cuda_graph
        if self._original_read_index is not None:
            module.faiss.read_index = self._original_read_index
        if self._original_f0 is not None:
            if self._f0_was_instance_attribute:
                self.pipeline.get_f0 = self._original_f0
            else:
                delattr(self.pipeline, "get_f0")
        self.abort_sample()


def build_cases(
    chunk_ms_values: list[int],
    f0_methods: list[str],
    index_rates: list[float],
    baseline_f0: str,
    baseline_index: float,
) -> list[dict[str, Any]]:
    """Build focused A/B cases, avoiding an unnecessary cartesian product."""

    require(bool(chunk_ms_values), "at least one chunk size is required")
    require(all(value > 0 for value in chunk_ms_values), "chunk sizes must be positive")
    require(len(set(chunk_ms_values)) == len(chunk_ms_values), "chunk sizes must be unique")
    require(baseline_f0 in f0_methods, "baseline F0 must be included")
    require(any(math.isclose(baseline_index, value) for value in index_rates),
            "baseline index rate must be included")
    require(all(0.0 <= value <= 1.0 for value in index_rates),
            "index rates must be between 0 and 1")

    cases: list[dict[str, Any]] = []
    seen: set[tuple[int, str, float]] = set()

    def add(chunk_ms: int, f0_method: str, index_rate: float, experiment: str) -> None:
        key = (chunk_ms, f0_method, index_rate)
        if key in seen:
            return
        seen.add(key)
        cases.append(
            {
                "id": f"{experiment}_{chunk_ms}ms_{f0_method}_index_{index_rate:g}",
                "experiment": experiment,
                "chunk_ms": chunk_ms,
                "f0_method": f0_method,
                "index_rate": index_rate,
            }
        )

    for chunk_ms in chunk_ms_values:
        add(chunk_ms, baseline_f0, baseline_index, "chunk")
    reference_chunk = max(chunk_ms_values)
    for method in f0_methods:
        add(reference_chunk, method, baseline_index, "f0")
    for rate in index_rates:
        add(reference_chunk, baseline_f0, rate, "index")
    return cases


def summarize_records(records: list[dict[str, float]], chunk_ms: float) -> dict:
    require(bool(records), "no profiling records")
    keys = tuple(records[0])
    require(all(tuple(record) == keys for record in records), "record keys differ")
    average = {key: statistics.fmean(record[key] for record in records) for key in keys}
    minimum = {key: min(record[key] for record in records) for key in keys}
    maximum = {key: max(record[key] for record in records) for key in keys}
    total_ms = average["total_ms"]
    percentages = {
        key: (value / total_ms * 100.0 if total_ms > 0 else 0.0)
        for key, value in average.items()
        if key.endswith("_ms") and key != "total_ms"
    }
    return {
        "samples": records,
        "average": average,
        "minimum": minimum,
        "maximum": maximum,
        "percentage_of_total": percentages,
        "rtf": average["total_ms"] / chunk_ms,
    }


def combine_timings(
    native: dict[str, float],
    exact: dict[str, float],
) -> dict[str, float]:
    required = {
        "total_ms", "preprocess_ms", "pipeline_ms", "postprocess_ms",
        "content_index_prepare_ms", "f0_ms", "index_synth_ms",
        "pipeline_overhead_ms",
    }
    require(required <= native.keys(), "engine profiling snapshot is incomplete")
    result = {
        "total_ms": native["total_ms"],
        "preprocess_ms": native["preprocess_ms"],
        "pipeline_ms": native["pipeline_ms"],
        "hubert_ms": exact["hubert_ms"],
        "f0_ms": exact["f0_ms"],
        "index_read_ms": exact["index_read_ms"],
        "index_reconstruct_ms": exact["index_reconstruct_ms"],
        "index_search_ms": exact["index_search_ms"],
        "synth_ms": exact["synth_ms"],
        "postprocess_ms": native["postprocess_ms"],
        "native_content_index_prepare_ms": native["content_index_prepare_ms"],
        "native_f0_ms": native["f0_ms"],
        "native_index_synth_ms": native["index_synth_ms"],
        "native_pipeline_overhead_ms": native["pipeline_overhead_ms"],
    }
    attributed = sum(exact.values())
    result["pipeline_unattributed_ms"] = native["pipeline_ms"] - attributed
    result["total_unattributed_ms"] = (
        native["total_ms"]
        - native["preprocess_ms"]
        - native["pipeline_ms"]
        - native["postprocess_ms"]
    )
    return result


def precision_metadata(engine: Any) -> dict[str, Any]:
    def first_dtype(model: Any) -> Optional[str]:
        try:
            return str(next(model.parameters()).dtype)
        except (AttributeError, StopIteration):
            return None

    active = "fp16" if engine.is_half else "fp32"
    return {
        "active": active,
        "is_half": engine.is_half,
        "device": engine.device,
        "generator_parameter_dtype": first_dtype(engine._net_g),
        "hubert_parameter_dtype": first_dtype(engine._hubert_model),
        "fp32_vs_fp16_experiment_run": False,
        "reason": (
            "Current CUDA runtime already uses FP16; no default or model-loading "
            "precision path was changed for this diagnostic."
            if engine.is_half
            else "Current runtime is FP32; this environment has no active CUDA FP16 path."
        ),
    }


def quality_risk(f0_method: str, index_rate: float) -> list[str]:
    risks: list[str] = []
    if f0_method == "pm":
        risks.append("high: PM may track pitch less robustly on speech and transitions")
    elif f0_method == "fcpe":
        risks.append("medium: FCPE timbre/articulation must be verified by listening")
    else:
        risks.append("baseline: RMVPE is the selected modelF quality reference")
    if math.isclose(index_rate, 0.0):
        risks.append("high: disabling retrieval may reduce model identity/timbre similarity")
    elif index_rate > 0.3:
        risks.append("medium: stronger retrieval may increase artifacts on connected speech")
    else:
        risks.append("baseline: modelF index rate is 0.30")
    return risks


def run_case(
    engine: Any,
    probe: DetailedStageProbe,
    case: dict[str, Any],
    chunks: list[np.ndarray],
    sample_rate: int,
) -> dict[str, Any]:
    config = engine.update_config(
        f0_method=case["f0_method"],
        index_rate=case["index_rate"],
    )
    chunk_samples = chunks[0].size
    actual_chunk_ms = chunk_samples * 1000.0 / sample_rate

    probe.begin_sample()
    try:
        warmup_started = time.perf_counter()
        warmup = engine.infer(chunks[0])
        warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        probe.finish_sample()
    except Exception:
        probe.abort_sample()
        raise
    _, warmup_passthrough = validate_result(warmup, chunks[0], "profile warmup")
    require(not warmup_passthrough, "warmup returned passthrough")

    records: list[dict[str, float]] = []
    for number, chunk in enumerate(chunks, start=1):
        probe.begin_sample()
        try:
            output = engine.infer(chunk)
            exact = probe.finish_sample()
        except Exception:
            probe.abort_sample()
            raise
        _, passthrough = validate_result(output, chunk, f"profile sample {number}")
        require(not passthrough, f"sample {number} returned passthrough")
        native = engine.last_inference_profile
        require(native is not None, "engine did not publish a profiling snapshot")
        record = combine_timings(native, exact)
        records.append(record)
        log(
            f"    sample {number}: total={record['total_ms']:.1f}ms "
            f"hubert={record['hubert_ms']:.1f} f0={record['f0_ms']:.1f} "
            f"index_search={record['index_search_ms']:.1f} "
            f"synth={record['synth_ms']:.1f}"
        )

    return {
        **case,
        "failed": False,
        "config": config.to_dict(),
        "chunk_samples": chunk_samples,
        "actual_chunk_ms": actual_chunk_ms,
        "warmup_ms": warmup_ms,
        "warmup_passthrough": warmup_passthrough,
        "quality_risk": quality_risk(case["f0_method"], case["index_rate"]),
        "timing": summarize_records(records, actual_chunk_ms),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    from config.settings import RVC_MODELS_DIR, RVC_SOURCE_DIR, SAMPLE_RATE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path,
        default=PROJECT_ROOT / "config" / "rvc_profiles" / "modelF.example.json",
    )
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "tests" / "assets" / "input.wav",
    )
    parser.add_argument("--input-duration", type=float, default=10.0)
    parser.add_argument("--chunk-ms", nargs="+", type=int, default=list(DEFAULT_CHUNKS_MS))
    parser.add_argument("--f0-methods", nargs="+", default=list(DEFAULT_F0_METHODS))
    parser.add_argument("--index-rates", nargs="+", type=float,
                        default=list(DEFAULT_INDEX_RATES))
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--source-dir", type=Path, default=Path(RVC_SOURCE_DIR))
    parser.add_argument("--models-dir", type=Path, default=Path(RVC_MODELS_DIR))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def print_summary(results: list[dict[str, Any]]) -> None:
    log("\nCase | Total | HuBERT | F0 | Index search | Synth | RTF | Result")
    log("-" * 105)
    for result in results:
        if result.get("skipped"):
            log(f"{result['id']}: SKIPPED {result['reason']}")
            continue
        if result.get("failed"):
            log(f"{result['id']}: FAILED {result['error']}")
            continue
        timing = result["timing"]
        avg = timing["average"]
        log(
            f"{result['id']}: {avg['total_ms']:.1f}ms | {avg['hubert_ms']:.1f} | "
            f"{avg['f0_ms']:.1f} | {avg['index_search_ms']:.1f} | "
            f"{avg['synth_ms']:.1f} | {timing['rtf']:.2f} | OK"
        )


def main(argv: Optional[list[str]] = None) -> int:
    from ai.rvc_engine import RVCEngine
    from config.rvc_profiles import load_rvc_profile

    args = parse_args(argv)
    require(args.input_duration > 0, "input duration must be positive")
    require(args.sample_count > 0, "sample count must be positive")
    profile = load_rvc_profile(args.profile)
    baseline = profile.inference
    cases = build_cases(
        args.chunk_ms,
        [value.lower() for value in args.f0_methods],
        args.index_rates,
        baseline.f0_method,
        baseline.index_rate,
    )
    output_path = args.output or (
        PROJECT_ROOT / "tests" / "output" / "rvc_inference_profile"
        / f"{profile.name}_profile_benchmark.json"
    )
    audio, input_metadata = load_audio(
        args.input.resolve(), args.sample_rate, args.input_duration,
    )
    chunks_by_ms: dict[int, list[np.ndarray]] = {}
    selection_metadata: dict[str, Any] = {}
    for chunk_ms in args.chunk_ms:
        chunk_samples = milliseconds_to_samples(chunk_ms, args.sample_rate)
        chunks, starts, rms_values = select_distinct_chunks(
            audio, chunk_samples, args.sample_count, args.sample_rate,
        )
        chunks_by_ms[chunk_ms] = chunks
        selection_metadata[str(chunk_ms)] = {
            "chunk_samples": chunk_samples,
            "starts_seconds": [start / args.sample_rate for start in starts],
            "rms": rms_values,
        }

    engine = RVCEngine.from_profile(
        profile,
        source_dir=args.source_dir,
        models_dir=args.models_dir,
        sample_rate=args.sample_rate,
    )
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": profile.name,
        "profile_path": str(args.profile.resolve()),
        "profile": profile.to_dict(),
        "input": input_metadata,
        "selection": selection_metadata,
        "settings": {
            "chunk_ms": args.chunk_ms,
            "f0_methods": args.f0_methods,
            "index_rates": args.index_rates,
            "sample_count": args.sample_count,
            "focused_matrix": True,
        },
        "measurement_notes": {
            "native_boundaries": (
                "External times[0] combines content encoding, index work and feature "
                "preparation; times[2] combines later feature work and synthesis."
            ),
            "exact_boundaries": (
                "Temporary wrappers add CUDA synchronization around HuBERT, F0, "
                "FAISS read/reconstruct/search and run_cuda_graph during this tool only."
            ),
            "unattributed": (
                "Pipeline residual includes filtering, padding, feature blending, "
                "interpolation, RMS mixing, allocation and cleanup."
            ),
        },
        "model_load_ms": None,
        "precision": None,
        "results": results,
    }

    try:
        load_started = time.perf_counter()
        engine.load_model()
        report["model_load_ms"] = (time.perf_counter() - load_started) * 1000.0
        report["precision"] = precision_metadata(engine)
        log(
            f"Loaded {profile.name} on {engine.device} "
            f"precision={report['precision']['active']}"
        )
        with DetailedStageProbe(engine) as probe:
            for case in cases:
                log(
                    f"\n=== {case['id']} chunk={case['chunk_ms']}ms "
                    f"f0={case['f0_method']} index={case['index_rate']:g} ==="
                )
                if (
                    case["f0_method"] == "fcpe"
                    and importlib.util.find_spec("torchfcpe") is None
                ):
                    result = {
                        **case,
                        "failed": False,
                        "skipped": True,
                        "reason": "optional dependency 'torchfcpe' is not installed",
                        "quality_risk": quality_risk(
                            case["f0_method"], case["index_rate"],
                        ),
                    }
                    results.append(result)
                    write_report(output_path, report)
                    log(f"    SKIPPED: {result['reason']}")
                    continue
                try:
                    result = run_case(
                        engine,
                        probe,
                        case,
                        chunks_by_ms[case["chunk_ms"]],
                        args.sample_rate,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    result = {
                        **case,
                        "failed": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "quality_risk": quality_risk(
                            case["f0_method"], case["index_rate"],
                        ),
                    }
                results.append(result)
                write_report(output_path, report)
        engine.update_config(baseline)
        print_summary(results)
        write_report(output_path, report)
        log(f"\nJSON report: {output_path}")
        return 1 if any(result.get("failed") for result in results) else 0
    except Exception:
        traceback.print_exc()
        report["fatal_error"] = traceback.format_exc()
        write_report(output_path, report)
        return 1
    finally:
        if engine.is_loaded:
            engine.unload_model()


if __name__ == "__main__":
    raise SystemExit(main())

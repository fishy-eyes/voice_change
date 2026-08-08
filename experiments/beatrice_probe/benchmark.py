"""Benchmark the real Beatrice v2 one-block streaming API on a fixed WAV."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np

from runtime_support import (
    RESULTS_DIR,
    convert_audio_blocks,
    create_converter,
    find_metadata_file,
    percentile,
    read_mono_audio,
    unpack_runtime_output,
)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
        first = result.stdout.splitlines()[0]
        name, utilization, memory_used, memory_total = (
            value.strip() for value in first.split(",")
        )
        return {
            "available": True,
            "name": name,
            "utilization_percent": int(utilization),
            "memory_used_mb": int(memory_used),
            "memory_total_mb": int(memory_total),
        }
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def current_rss_bytes() -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    return int(counters.WorkingSetSize) if succeeded else None


def bytes_to_mb(value: int | None) -> float | None:
    return value / (1024 * 1024) if value is not None else None


def model_size(model_path: str) -> int:
    model_dir = find_metadata_file(model_path).parent
    return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--target-speaker", type=int, default=0)
    parser.add_argument("--formant-shift", type=float, default=0.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--min-source-pitch", type=float, default=30.0)
    parser.add_argument("--max-source-pitch", type=float, default=1100.0)
    parser.add_argument("--vq-neighbors", type=int, default=4)
    parser.add_argument(
        "--warmup-blocks",
        type=int,
        default=100,
        help="Real supported 10 ms blocks; default is one second",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--json-out",
        default=str(RESULTS_DIR / "benchmark.json"),
    )
    args = parser.parse_args()
    if args.warmup_blocks < 0 or args.repeats < 1:
        parser.error("warmup-blocks must be >= 0 and repeats must be >= 1")

    input_audio, input_sample_rate = read_mono_audio(args.input)
    input_duration = input_audio.size / input_sample_rate
    rss_before = current_rss_bytes()
    gpu_before = gpu_snapshot()

    converter, module, _, runtime = create_converter(
        args.model,
        args.runtime_root,
        target_speaker=args.target_speaker,
        formant_shift=args.formant_shift,
        pitch_shift_semitone=args.pitch_shift,
        min_source_pitch=args.min_source_pitch,
        max_source_pitch=args.max_source_pitch,
        vq_num_neighbors=args.vq_neighbors,
    )
    warmup_times = []
    zero_block = np.zeros(int(module.IN_HOP_LENGTH), dtype=np.float32)
    for _ in range(args.warmup_blocks):
        started = perf_counter()
        converted, _ = unpack_runtime_output(converter.convert(zero_block))
        warmup_times.append((perf_counter() - started) * 1000.0)
        if converted.size != int(module.OUT_HOP_LENGTH):
            raise RuntimeError(
                f"Warmup returned {converted.size} samples; expected {module.OUT_HOP_LENGTH}"
            )

    cpu_before = process_time()
    wall_started = perf_counter()
    block_times: list[float] = []
    auxiliary_values: list[Any] = []
    for _ in range(args.repeats):
        result = convert_audio_blocks(
            converter, module, input_audio, input_sample_rate
        )
        block_times.extend(result.block_times_ms)
        auxiliary_values.extend(result.runtime_aux_values)
    wall_seconds = perf_counter() - wall_started
    cpu_after = process_time()
    rss_after = current_rss_bytes()
    gpu_after = gpu_snapshot()

    cpu_seconds = cpu_after - cpu_before
    audio_seconds = input_duration * args.repeats
    inference_sum_seconds = sum(block_times) / 1000.0
    report = {
        "model": runtime,
        "model_size_bytes": model_size(args.model),
        "input": {
            "path": str(Path(args.input).expanduser().resolve()),
            "sample_rate": input_sample_rate,
            "duration_seconds_per_repeat": input_duration,
            "repeats": args.repeats,
            "total_audio_seconds": audio_seconds,
        },
        "load_seconds": runtime["load_seconds"],
        "warmup": {
            "blocks": args.warmup_blocks,
            "audio_seconds": args.warmup_blocks
            * int(module.IN_HOP_LENGTH)
            / int(module.IN_SAMPLE_RATE),
            "total_seconds": sum(warmup_times) / 1000.0,
            "last_block_ms": warmup_times[-1] if warmup_times else None,
        },
        "inference": {
            "wall_seconds": wall_seconds,
            "sum_block_seconds": inference_sum_seconds,
            "rtf_wall": wall_seconds / audio_seconds,
            "rtf_sum_blocks": inference_sum_seconds / audio_seconds,
            "block_count": len(block_times),
            "runtime_auxiliary": {
                "meaning": "unknown; undocumented second convert() return value",
                "unique_values": sorted(set(auxiliary_values)),
            },
            "supported_input_block_samples": int(module.IN_HOP_LENGTH),
            "supported_output_block_samples": int(module.OUT_HOP_LENGTH),
            "supported_block_duration_ms": 1000.0
            * int(module.IN_HOP_LENGTH)
            / int(module.IN_SAMPLE_RATE),
            "mean_block_ms": sum(block_times) / len(block_times),
            "p50_block_ms": percentile(block_times, 50),
            "p95_block_ms": percentile(block_times, 95),
            "p99_block_ms": percentile(block_times, 99),
            "max_block_ms": max(block_times),
        },
        "resources": {
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_percent_one_core_scale": (
                cpu_seconds / wall_seconds * 100.0 if wall_seconds else math.nan
            ),
            "logical_cpu_count": os.cpu_count(),
            "rss_before_mb": bytes_to_mb(rss_before),
            "rss_after_mb": bytes_to_mb(rss_after),
            "rss_delta_mb": bytes_to_mb(rss_after - rss_before)
            if rss_after is not None and rss_before is not None
            else None,
            "gpu_before_snapshot": gpu_before,
            "gpu_after_snapshot": gpu_after,
            "note": "GPU values are snapshots, not profiler peaks; the std runtime is CPU-oriented.",
        },
        "realtime_potential": {
            "block_deadline_ms": 1000.0
            * int(module.IN_HOP_LENGTH)
            / int(module.IN_SAMPLE_RATE),
            "p99_within_deadline": percentile(block_times, 99)
            < 1000.0 * int(module.IN_HOP_LENGTH) / int(module.IN_SAMPLE_RATE),
            "rtf_below_one": wall_seconds / audio_seconds < 1.0,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output = Path(args.json_out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(f"json_out={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

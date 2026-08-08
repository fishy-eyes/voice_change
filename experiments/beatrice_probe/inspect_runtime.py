"""Inspect the current Python environment and optional Beatrice v2 runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_support import import_beatrice


PACKAGES = (
    "torch",
    "torchaudio",
    "numpy",
    "scipy",
    "soxr",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "soundfile",
    "librosa",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            rows.append(
                {
                    "name": values[0],
                    "driver_version": values[1],
                    "memory_total_mb": int(values[2]),
                    "memory_used_mb": int(values[3]),
                    "utilization_percent": int(values[4]),
                }
            )
    return {"available": True, "gpus": rows}


def inspect(runtime_root: str | None, version: str) -> dict[str, Any]:
    packages = {name: package_version(name) for name in PACKAGES}
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "processor": platform.processor() or "unknown",
            "logical_cores": os.cpu_count(),
        },
        "packages": packages,
        "gpu": run_nvidia_smi(),
    }

    if importlib.util.find_spec("torch") is not None:
        import torch

        report["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    else:
        report["pytorch"] = {"available": False}

    providers: list[str] = []
    onnx_error = None
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            import onnxruntime

            providers = list(onnxruntime.get_available_providers())
        except Exception as exc:  # diagnostic boundary
            onnx_error = f"{type(exc).__name__}: {exc}"
    report["onnx_runtime"] = {
        "available": bool(providers),
        "providers": providers,
        "error": onnx_error,
    }

    try:
        module, resolved_root = import_beatrice(runtime_root, version)
        report["beatrice_runtime"] = {
            "importable": True,
            "runtime_root": str(resolved_root) if resolved_root else None,
            "version": version,
            "module": module.__name__,
            "input_sample_rate": int(module.IN_SAMPLE_RATE),
            "output_sample_rate": int(module.OUT_SAMPLE_RATE),
            "input_hop_length": int(module.IN_HOP_LENGTH),
            "output_hop_length": int(module.OUT_HOP_LENGTH),
            "block_duration_ms": 1000.0
            * int(module.IN_HOP_LENGTH)
            / int(module.IN_SAMPLE_RATE),
        }
    except Exception as exc:  # diagnostic boundary
        report["beatrice_runtime"] = {
            "importable": False,
            "runtime_root": runtime_root,
            "version": version,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return report


def pip_freeze() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def format_snapshot(report: dict[str, Any]) -> str:
    lines = [
        "Beatrice v2 probe environment snapshot",
        f"timestamp_utc={report['timestamp_utc']}",
        "",
        json.dumps(report, ensure_ascii=False, indent=2),
        "",
        "pip freeze",
        "----------",
        *pip_freeze(),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", help="Directory containing the beatrice package")
    parser.add_argument("--version", default="2.0.0-rc.0")
    parser.add_argument("--snapshot", help="Write a complete environment snapshot")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    report = inspect(args.runtime_root, args.version)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.snapshot:
        snapshot_path = Path(args.snapshot).expanduser().resolve()
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(format_snapshot(report), encoding="utf-8", newline="\n")
        if not args.json:
            print(f"snapshot={snapshot_path}")
    return 0 if report["beatrice_runtime"]["importable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

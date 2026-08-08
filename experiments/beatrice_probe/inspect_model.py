"""Inspect a Beatrice v2 paraphernalia model without treating it as RVC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from runtime_support import (
    REQUIRED_MODEL_FILES,
    SUPPORTED_VERSIONS,
    create_converter,
    import_beatrice,
    load_model_metadata,
)


def inspect_model(
    model_path: str,
    runtime_root: str | None,
    load_runtime: bool,
) -> dict[str, Any]:
    metadata_file, metadata = load_model_metadata(model_path)
    model_dir = metadata_file.parent
    model_info = metadata.get("model", {})
    version = model_info.get("version")
    voices = metadata.get("voice", {})

    structure = []
    total_size = 0
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_size += size
            structure.append(
                {"path": str(path.relative_to(model_dir)), "size_bytes": size}
            )

    required_files = {
        name: {
            "exists": (model_dir / name).is_file(),
            "size_bytes": (
                (model_dir / name).stat().st_size if (model_dir / name).is_file() else None
            ),
        }
        for name in REQUIRED_MODEL_FILES
    }
    runtime_report: dict[str, Any]
    try:
        module, resolved_root = import_beatrice(runtime_root, str(version))
        runtime_report = {
            "importable": True,
            "runtime_root": str(resolved_root) if resolved_root else None,
            "module": module.__name__,
            "input_sample_rate": int(module.IN_SAMPLE_RATE),
            "output_sample_rate": int(module.OUT_SAMPLE_RATE),
            "input_hop_length": int(module.IN_HOP_LENGTH),
            "output_hop_length": int(module.OUT_HOP_LENGTH),
        }
    except Exception as exc:
        runtime_report = {
            "importable": False,
            "runtime_root": runtime_root,
            "input_sample_rate": "unknown",
            "output_sample_rate": "unknown",
            "input_hop_length": "unknown",
            "output_hop_length": "unknown",
            "error": f"{type(exc).__name__}: {exc}",
        }

    load_report: dict[str, Any] = {"attempted": False}
    if load_runtime:
        load_report["attempted"] = True
        try:
            _, _, _, details = create_converter(model_path, runtime_root)
            load_report.update({"success": True, **details})
        except Exception as exc:
            load_report.update(
                {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            )

    voice_rows = []
    for voice_id, voice in sorted(voices.items(), key=lambda item: int(item[0])):
        voice_rows.append(
            {
                "id": int(voice_id),
                "name": voice.get("name", "unknown"),
                "average_pitch": voice.get("average_pitch", "unknown"),
            }
        )

    complete = all(item["exists"] for item in required_files.values())
    compatible = bool(
        complete
        and isinstance(version, str)
        and version in SUPPORTED_VERSIONS
        and runtime_report["importable"]
        and (not load_runtime or load_report.get("success"))
    )
    return {
        "model_type": "Beatrice v2 paraphernalia directory",
        "model_dir": str(model_dir),
        "metadata_file": str(metadata_file),
        "format": "TOML metadata plus five custom raw binary parameter files",
        "version": version if isinstance(version, str) else "unknown",
        "name": model_info.get("name", "unknown"),
        "description": model_info.get("description", "unknown"),
        "size_bytes": total_size,
        "file_structure": structure,
        "required_files": required_files,
        "runtime_required": "Project Beatrice version-matched native Python runtime",
        "runtime": runtime_report,
        "load": load_report,
        "input_sample_rate": runtime_report["input_sample_rate"],
        "output_sample_rate": runtime_report["output_sample_rate"],
        "metadata_fields": sorted(model_info),
        "speaker_count": len(voice_rows),
        "speakers": voice_rows,
        "adjustable_parameters": [
            "target_speaker",
            "formant_shift",
            "pitch_shift_semitone",
            "min_source_pitch",
            "max_source_pitch",
            "vq_num_neighbors",
        ],
        "uses_pitch_estimator": required_files["pitch_estimator.bin"]["exists"],
        "compatible_with_current_runtime": compatible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Beatrice v2 model directory or metadata TOML")
    parser.add_argument("--runtime-root", help="Directory containing the beatrice package")
    parser.add_argument(
        "--load-runtime",
        action="store_true",
        help="Instantiate the native runtime to verify model loading",
    )
    parser.add_argument("--json-out", help="Optional JSON report path")
    args = parser.parse_args()

    report = inspect_model(args.model, args.runtime_root, args.load_runtime)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    return 0 if report["compatible_with_current_runtime"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

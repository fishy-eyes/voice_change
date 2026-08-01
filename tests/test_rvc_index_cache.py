"""Real modelF lifecycle verification for the RVC index cache."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    from tests.test_rvc_realtime_benchmark import load_audio, milliseconds_to_samples
    from tests.test_rvc_short_chunk_benchmark import select_distinct_chunks, validate_result
except ModuleNotFoundError:
    from test_rvc_realtime_benchmark import load_audio, milliseconds_to_samples
    from test_rvc_short_chunk_benchmark import select_distinct_chunks, validate_result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    from ai.rvc_engine import RVCEngine
    from config.rvc_profiles import load_rvc_profile
    from config.settings import RVC_MODELS_DIR, RVC_SOURCE_DIR, SAMPLE_RATE

    profile_path = PROJECT_ROOT / "config" / "rvc_profiles" / "modelF.example.json"
    input_path = PROJECT_ROOT / "tests" / "assets" / "input.wav"
    output_path = (
        PROJECT_ROOT / "tests" / "output" / "rvc_index_cache"
        / "modelF_index_cache_test.json"
    )
    profile = load_rvc_profile(profile_path)
    audio, input_metadata = load_audio(input_path, SAMPLE_RATE, 10.0)
    chunks, starts, rms_values = select_distinct_chunks(
        audio,
        milliseconds_to_samples(500, SAMPLE_RATE),
        2,
        SAMPLE_RATE,
    )
    engine = RVCEngine.from_profile(
        profile,
        source_dir=RVC_SOURCE_DIR,
        models_dir=RVC_MODELS_DIR,
        sample_rate=SAMPLE_RATE,
    )
    report = {
        "model": profile.name,
        "profile": profile.to_dict(),
        "input": input_metadata,
        "selected_starts_seconds": [start / SAMPLE_RATE for start in starts],
        "selected_rms": rms_values,
        "load_ms": None,
        "after_load": None,
        "inferences": [],
        "after_unload": None,
    }

    try:
        started_at = time.perf_counter()
        engine.load_model()
        report["load_ms"] = (time.perf_counter() - started_at) * 1000.0
        after_load = engine.index_cache_info
        report["after_load"] = after_load
        require(after_load["enabled"], "index cache not enabled after model load")
        require(after_load["initialization_count"] == 1, "index initialized more than once")
        require(after_load["cache_misses"] == 1, "expected exactly one cache miss")
        require(after_load["read_hits"] == 0, "pipeline read before first inference")
        require(after_load["reconstruct_hits"] == 0, "pipeline reconstructed before inference")

        for number, chunk in enumerate(chunks, start=1):
            infer_started = time.perf_counter()
            output = engine.infer(chunk)
            elapsed_ms = (time.perf_counter() - infer_started) * 1000.0
            _, passthrough = validate_result(output, chunk, f"cache inference {number}")
            require(not passthrough, f"inference {number} returned passthrough")
            cache_info = engine.index_cache_info
            require(cache_info["initialization_count"] == 1, "cache reinitialized")
            require(cache_info["read_hits"] == number, "read cache hit count mismatch")
            require(
                cache_info["reconstruct_hits"] == number,
                "reconstruct cache hit count mismatch",
            )
            require(cache_info["fallback_reconstructs"] == 0, "unexpected reconstruction")
            report["inferences"].append(
                {
                    "number": number,
                    "elapsed_ms": elapsed_ms,
                    "passthrough": passthrough,
                    "cache": cache_info,
                }
            )
            print(
                f"infer {number}: {elapsed_ms:.1f}ms "
                f"read_hits={cache_info['read_hits']} "
                f"reconstruct_hits={cache_info['reconstruct_hits']}",
                flush=True,
            )

        engine.unload_model()
        after_unload = engine.index_cache_info
        report["after_unload"] = after_unload
        require(after_unload["released"], "cache was not released")
        require(not after_unload["enabled"], "cache remains enabled after unload")
        require(after_unload["active_registry_entries"] == 0, "cache registry not empty")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(f"PASS: {output_path}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if engine.is_loaded:
            engine.unload_model()


if __name__ == "__main__":
    raise SystemExit(main())

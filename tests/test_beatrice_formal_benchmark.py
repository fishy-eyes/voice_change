"""Opt-in 48 kHz/256 formal Worker -> Engine -> Adapter soak benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import monotonic, perf_counter, sleep
import unittest

import numpy as np

from ai.beatrice.model import BeatriceModelManager
from ai.voice_engine.beatrice import BeatriceVoiceEngine
from effects.ai_voice import AIVoiceEffect


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = Path(
    os.environ.get(
        "VOICE_CHANGE_BEATRICE_RUNTIME_DIR",
        PROJECT_ROOT / "local_assets" / "beatrice" / "runtimes" / "probe-runtime",
    )
)
PACKAGE_ROOT = Path(
    os.environ.get(
        "VOICE_CHANGE_BEATRICE_MODEL_PACKAGE",
        PROJECT_ROOT / "local_assets" / "beatrice" / "models" / "jvs",
    )
)
RUN_BENCHMARK = os.environ.get("RUN_BEATRICE_FORMAL_BENCHMARK") == "1"
HAS_ASSETS = (
    (RUNTIME_ROOT / "beatrice" / "__init__.py").is_file()
    and PACKAGE_ROOT.is_dir()
)


@unittest.skipUnless(
    RUN_BENCHMARK and HAS_ASSETS,
    "set RUN_BEATRICE_FORMAL_BENCHMARK=1 with external assets",
)
class BeatriceFormalBenchmark(unittest.TestCase):
    def test_realtime_worker_soak(self) -> None:
        duration = float(os.environ.get("BEATRICE_BENCHMARK_SECONDS", "600"))
        descriptor = BeatriceModelManager(PACKAGE_ROOT.parent).get_model(
            PACKAGE_ROOT.name
        )
        engine = BeatriceVoiceEngine(descriptor, runtime_root=RUNTIME_ROOT)
        effect = AIVoiceEffect(
            engine,
            chunk_size=256,
            overlap_size=0,
            max_queue_size=8,
        )
        callback_times: list[float] = []
        max_input_queue = 0
        max_output_queue = 0
        callback_count = max(1, round(duration * 48_000 / 256))
        callback_seconds = 256 / 48_000
        engine.load_model()
        try:
            self.assertTrue(effect.start())
            start = monotonic()
            for index in range(callback_count):
                phase = np.arange(256, dtype=np.float32) + index * 256
                block = (0.05 * np.sin(phase * (2 * np.pi * 220 / 48_000))).astype(
                    np.float32
                )
                measured = perf_counter()
                output = effect.process(block, 256, None, None)
                callback_times.append((perf_counter() - measured) * 1000.0)
                self.assertEqual(output.shape, block.shape)
                max_input_queue = max(max_input_queue, effect.worker.input_pending)
                max_output_queue = max(max_output_queue, effect.worker.output_pending)
                deadline = start + (index + 1) * callback_seconds
                remaining = deadline - monotonic()
                if remaining > 0:
                    sleep(remaining)
            sleep(0.1)
            effect.process(np.zeros(256, dtype=np.float32))
            adapter_stats = engine.adapter.stats()
            report = {
                "duration_seconds": duration,
                "callbacks": callback_count,
                "callback_ms": {
                    "p50": float(np.percentile(callback_times, 50)),
                    "p95": float(np.percentile(callback_times, 95)),
                    "p99": float(np.percentile(callback_times, 99)),
                    "max": max(callback_times),
                },
                "worker": {
                    "infer_count": effect.worker.infer_count,
                    "errors": effect.worker.error_count,
                    "continuity_errors": effect.worker.continuity_error_count,
                    "recoveries": effect.worker.recovery_count,
                    "max_input_queue": max_input_queue,
                    "max_output_queue": max_output_queue,
                    "last_infer_ms": effect.worker.last_infer_ms,
                    "average_infer_ms": effect.worker.average_infer_ms,
                },
                "adapter": adapter_stats,
            }
            print("BEATRICE_FORMAL_BENCHMARK=" + json.dumps(report, sort_keys=True))
            self.assertEqual(effect.worker.error_count, 0)
            self.assertEqual(effect.worker.continuity_error_count, 0)
            self.assertEqual(adapter_stats["underflow_count"], 0)
            self.assertEqual(adapter_stats["overflow_count"], 0)
            self.assertLess(abs(adapter_stats["input_resampler_drift"]), 1.0)
            self.assertLess(abs(adapter_stats["output_resampler_drift"]), 1.0)
        finally:
            self.assertTrue(effect.stop(timeout=5.0))
            engine.unload_model()


if __name__ == "__main__":
    unittest.main(verbosity=2)

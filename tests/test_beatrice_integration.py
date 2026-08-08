"""Real Beatrice production-path integration; skips without local assets."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from ai.beatrice.model import BeatriceModelManager
from ai.voice_engine.beatrice import BeatriceVoiceEngine
from ai.voice_worker import VoiceConversionWorker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = PROJECT_ROOT / "local_assets" / "beatrice" / "runtimes" / "probe-runtime"
DEFAULT_PACKAGE = PROJECT_ROOT / "local_assets" / "beatrice" / "models" / "jvs"
RUNTIME_ROOT = Path(os.environ.get("VOICE_CHANGE_BEATRICE_RUNTIME_DIR", DEFAULT_RUNTIME))
PACKAGE_ROOT = Path(os.environ.get("VOICE_CHANGE_BEATRICE_MODEL_PACKAGE", DEFAULT_PACKAGE))
HAS_ASSETS = (
    (RUNTIME_ROOT / "beatrice" / "__init__.py").is_file()
    and PACKAGE_ROOT.is_dir()
)


@unittest.skipUnless(HAS_ASSETS, "external Beatrice runtime/model package unavailable")
class RealBeatriceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptor = BeatriceModelManager(PACKAGE_ROOT.parent).get_model(
            PACKAGE_ROOT.name
        )

    def test_repeated_load_process_reset_unload(self) -> None:
        engine = BeatriceVoiceEngine(self.descriptor, runtime_root=RUNTIME_ROOT)
        try:
            engine.load_model()
            for _ in range(12):
                output = engine.process_audio(np.zeros(256, dtype=np.float32))
                self.assertEqual(output.shape, (256,))
                self.assertTrue(np.isfinite(output).all())
            generation = engine.adapter.converter_generation
            engine.reset_stream()
            self.assertEqual(engine.adapter.converter_generation, generation + 1)
            engine.unload_model()
            engine.load_model()
            self.assertTrue(engine.is_loaded)
        finally:
            engine.unload_model()
        self.assertFalse(engine.is_loaded)

    def test_real_worker_lifecycle_preserves_block_sequence(self) -> None:
        engine = BeatriceVoiceEngine(self.descriptor, runtime_root=RUNTIME_ROOT)
        worker = VoiceConversionWorker(engine, chunk_size=256, max_queue_size=4)
        engine.load_model()
        try:
            self.assertTrue(worker.start())
            for index in range(64):
                phase = np.arange(256, dtype=np.float32) + index * 256
                block = (0.05 * np.sin(phase * (2 * np.pi * 220 / 48_000))).astype(
                    np.float32
                )
                self.assertTrue(worker.put(block, timeout=0.1))
                output = worker.get(timeout=2.0)
                self.assertIsNotNone(output)
                self.assertEqual(output.shape, block.shape)
            self.assertEqual(worker.continuity_error_count, 0)
            self.assertEqual(worker.error_count, 0)
        finally:
            self.assertTrue(worker.stop(timeout=5.0))
            engine.unload_model()
        self.assertFalse(worker.thread_alive)


if __name__ == "__main__":
    unittest.main(verbosity=2)

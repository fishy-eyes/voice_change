"""Real GUI-path configuration flow; skips without local Beatrice assets."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from ai.beatrice.catalog import BeatriceModelCatalog
from config.local_settings import LocalSettingsStore
from core.beatrice_runtime import BeatriceRuntime


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
class RealBeatriceLocalFlowTests(unittest.TestCase):
    def test_persist_discover_load_speaker_worker_and_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            settings_path = Path(temp) / "local_settings.json"
            settings = LocalSettingsStore(settings_path)
            settings.update_beatrice(
                runtime_dir=str(RUNTIME_ROOT.resolve()),
                model_roots=[str(PACKAGE_ROOT.resolve())],
            )
            restarted = LocalSettingsStore(settings_path)
            catalog = BeatriceModelCatalog(
                PROJECT_ROOT / "models" / "beatrice",
                registered_packages=restarted.beatrice["model_roots"],
                on_registered_paths_changed=lambda paths: restarted.update_beatrice(
                    model_roots=list(paths)
                ),
            )
            discovered = catalog.discover_models()
            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].package, "jvs")

            runtime = BeatriceRuntime(
                catalog,
                runtime_root=restarted.beatrice["runtime_dir"],
                local_settings=restarted,
            )
            try:
                details = runtime.validate_runtime()
                self.assertEqual(details["model_api_version"], "2.0.0-rc.0")
                self.assertIsNone(details["runtime_implementation_version"])
                runtime.set_enabled(True)
                state = runtime.load_model(discovered[0].name)
                self.assertTrue(state.ready, state.error)
                target = min(1, discovered[0].speaker_count - 1)
                runtime.update_parameters(target_speaker=target)
                for index in range(48):
                    phase = np.arange(256, dtype=np.float32) + index * 256
                    block = (0.03 * np.sin(phase * (2 * np.pi * 220 / 48_000))).astype(
                        np.float32
                    )
                    output = state.effect.process(block, 256, None, None)
                    self.assertEqual(output.shape, (256,))
                    self.assertTrue(np.isfinite(output).all())
                    time.sleep(256 / 48_000)
                self.assertEqual(state.effect.worker.continuity_error_count, 0)
                self.assertEqual(state.effect.worker.error_count, 0)
                saved = LocalSettingsStore(settings_path).beatrice
                self.assertEqual(saved["last_model"], discovered[0].name)
                self.assertEqual(
                    saved["last_speaker"], discovered[0].speaker_names[target]
                )
            finally:
                self.assertTrue(runtime.shutdown())


if __name__ == "__main__":
    unittest.main(verbosity=2)

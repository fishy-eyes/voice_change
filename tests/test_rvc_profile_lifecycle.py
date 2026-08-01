"""Verify optional model profiles integrate without changing the RVC lifecycle."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.rvc_profiles import RVCInferenceConfig
from core.rvc_lifecycle import cleanup_rvc_application, initialize_rvc_application


class FakeEngine:
    last_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs
        self.is_loaded = False

    def load_model(self) -> None:
        self.is_loaded = True

    def unload_model(self) -> None:
        self.is_loaded = False


class FakeWorker:
    thread_alive = False


class FakeEffect:
    def __init__(self, engine, *, chunk_size: int, max_queue_size: int) -> None:
        self.engine = engine
        self.chunk_size = chunk_size
        self.max_queue_size = max_queue_size
        self.worker = FakeWorker()
        self.last_warmup_ms = 0.0

    def start(self) -> bool:
        return True

    def stop(self, timeout: float) -> bool:
        del timeout
        return True


class RVCProfileLifecycleTests(unittest.TestCase):
    def test_profile_file_overrides_only_model_specific_engine_settings(self) -> None:
        profile_path = PROJECT_ROOT / "config" / "rvc_profiles" / "modelF.example.json"
        state = initialize_rvc_application(
            enabled=True,
            profile=profile_path,
            source_dir="source-root",
            models_dir="models-root",
            sample_rate=48000,
            warmup_enabled=False,
            validate_paths=False,
            engine_factory=FakeEngine,
            effect_factory=FakeEffect,
        )
        self.assertTrue(state.ready, state.error)
        self.assertIsNotNone(FakeEngine.last_kwargs)
        kwargs = FakeEngine.last_kwargs or {}
        self.assertEqual(kwargs["voice_dir"], Path("models-root/voices/modelF"))
        self.assertEqual(kwargs["voice_pth_path"], Path("zhoujie.pth"))
        self.assertEqual(
            kwargs["index_path"],
            Path("added_IVF5660_Flat_nprobe_1.index"),
        )
        self.assertEqual(kwargs["sample_rate"], 48000)
        self.assertEqual(
            kwargs["config"],
            RVCInferenceConfig(
                pitch_shift=12,
                f0_method="rmvpe",
                index_rate=0.30,
                rms_mix_rate=0.25,
                protect=0.33,
            ),
        )
        self.assertTrue(cleanup_rvc_application(state))


if __name__ == "__main__":
    unittest.main(verbosity=2)

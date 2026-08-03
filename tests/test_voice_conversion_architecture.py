"""Unit coverage for the backend-neutral voice-conversion layer."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from ai.voice_conversion_manager import VoiceConversionManager
from ai.voice_engine.rvc import RVCVoiceEngine


class FakeConfig:
    def to_dict(self):
        return {"pitch_shift": 0}


class FakeRVCCore:
    def __init__(self) -> None:
        self.is_loaded = False
        self.device = "cpu"
        self.is_half = False
        self.config = FakeConfig()
        self.index_cache_info = {"loaded": False}

    def load_model(self) -> None:
        self.is_loaded = True

    def unload_model(self) -> None:
        self.is_loaded = False

    def infer(self, audio: np.ndarray) -> np.ndarray:
        return np.asarray(audio, dtype=np.float32) + 0.25


class FakeRuntime:
    def __init__(self) -> None:
        self.model_manager = SimpleNamespace(
            discover_models=lambda: [SimpleNamespace(name="voice-a")]
        )
        self.state = SimpleNamespace(
            ready=False,
            error=None,
            engine=None,
            effect=None,
        )
        self.selected_model = None
        self.enabled = False
        self.load_thread = None
        self.fail_next = False
        self.shutdown_calls = 0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def load_model(self, model: str, *, audio_stream=None):
        del audio_stream
        self.load_thread = threading.get_ident()
        self.selected_model = model
        if self.fail_next:
            self.state.error = "broken model"
            self.state.ready = True
            return self.state
        engine = SimpleNamespace(get_latency=lambda: 12.5)
        worker = SimpleNamespace(last_infer_ms=12.5)
        self.state = SimpleNamespace(
            ready=True,
            error=None,
            engine=engine,
            effect=SimpleNamespace(worker=worker),
        )
        return self.state

    def shutdown(self) -> bool:
        self.shutdown_calls += 1
        self.state.ready = False
        return True


class VoiceConversionArchitectureTests(unittest.TestCase):
    def test_rvc_adapter_preserves_core_and_unified_interface(self) -> None:
        core = FakeRVCCore()
        engine = RVCVoiceEngine(engine=core)
        engine.load_model()
        audio = np.zeros(16, dtype=np.float32)
        output = engine.process_audio(audio)

        self.assertTrue(engine.is_loaded)
        self.assertIs(engine.core_engine, core)
        np.testing.assert_array_equal(output, np.full(16, 0.25, np.float32))
        self.assertGreaterEqual(engine.get_latency(), 0.0)
        self.assertEqual(engine.get_info()["backend"], "rvc")
        capabilities = engine.get_info()["capabilities"]
        self.assertEqual(capabilities.backend_id, "rvc")
        self.assertEqual(capabilities.parameter_names, ("pitch_shift",))
        self.assertEqual(engine.config.to_dict(), {"pitch_shift": 0})
        engine.unload_model()
        self.assertFalse(engine.is_loaded)

    def test_async_switch_reports_status_and_failure_forces_bypass(self) -> None:
        runtime = FakeRuntime()
        manager = VoiceConversionManager(
            {"rvc": runtime},
            default_backend="rvc",
        )
        try:
            manager.set_enabled(True)
            future = manager.switch_model_async("rvc", "voice-a")
            self.assertTrue(future.result(timeout=2.0).ready)
            self.assertNotEqual(runtime.load_thread, threading.get_ident())
            self.assertEqual(manager.available_backends, ("rvc",))
            self.assertEqual(manager.discover_models()[0].name, "voice-a")
            self.assertEqual(manager.get_status().state, "LOADED")
            self.assertAlmostEqual(manager.get_status().latency_ms, 12.5)
            manager.set_enabled(False)
            self.assertEqual(manager.get_status().state, "LOADED")
            self.assertFalse(runtime.enabled)
            manager.set_enabled(True)


            runtime.fail_next = True
            manager.switch_model_async("rvc", "broken").result(timeout=2.0)
            status = manager.get_status()
            self.assertEqual(status.state, "FAILED")
            self.assertFalse(status.enabled)
            self.assertFalse(runtime.enabled)
            self.assertIn("broken model", status.error or "")

            runtime.fail_next = False
            recovered = manager.switch_model_async("rvc", "voice-a").result(timeout=2.0)
            self.assertTrue(recovered.ready)
            self.assertEqual(manager.get_status().state, "LOADED")
            self.assertTrue(manager.get_status().enabled)

            manager.set_enabled(False)
            self.assertFalse(runtime.enabled)
        finally:
            self.assertTrue(manager.shutdown())
        self.assertEqual(runtime.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

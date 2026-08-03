"""Concurrency and shutdown coverage for model switching."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from ai.voice_conversion_manager import VoiceConversionManager


class BlockingRuntime:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.enabled = False
        self.selected_model = None
        self.loads = 0
        self.shutdown_calls = 0
        self.state = SimpleNamespace(ready=False, error=None, engine=None, effect=None)
        self.model_manager = SimpleNamespace(discover_models=lambda: [])

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def load_model(self, model, *, audio_stream=None):
        del audio_stream
        self.loads += 1
        self.started.set()
        self.release.wait(2.0)
        self.selected_model = model
        self.state = SimpleNamespace(ready=True, error=None, engine=None, effect=None)
        return self.state

    def shutdown(self):
        self.shutdown_calls += 1
        self.state.ready = False
        return True


class VoiceConversionConcurrencyTests(unittest.TestCase):
    def test_duplicate_switch_is_rejected_instead_of_queued(self) -> None:
        runtime = BlockingRuntime()
        manager = VoiceConversionManager({"rvc": runtime}, default_backend="rvc")
        try:
            first = manager.switch_model_async("rvc", "voice-a")
            self.assertTrue(runtime.started.wait(1.0))
            self.assertIn(manager.get_status().state, {"LOADING", "SWITCHING"})
            with self.assertRaisesRegex(RuntimeError, "in progress"):
                manager.switch_model_async("rvc", "voice-b")
            runtime.release.set()
            self.assertTrue(first.result(timeout=2.0).ready)
            self.assertEqual(runtime.loads, 1)
            self.assertEqual(manager.get_status().state, "LOADED")
        finally:
            runtime.release.set()
            self.assertTrue(manager.shutdown())

    def test_shutdown_waits_for_active_switch_then_releases_runtime(self) -> None:
        runtime = BlockingRuntime()
        manager = VoiceConversionManager({"rvc": runtime}, default_backend="rvc")
        manager.switch_model_async("rvc", "voice-a")
        self.assertTrue(runtime.started.wait(1.0))
        result = []
        closer = threading.Thread(target=lambda: result.append(manager.shutdown()))
        closer.start()
        time.sleep(0.05)
        self.assertTrue(closer.is_alive())
        runtime.release.set()
        closer.join(2.0)
        self.assertFalse(closer.is_alive())
        self.assertEqual(result, [True])
        self.assertEqual(runtime.shutdown_calls, 1)
        self.assertFalse(
            any(t.name.startswith("voice-conversion-loader") for t in threading.enumerate())
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

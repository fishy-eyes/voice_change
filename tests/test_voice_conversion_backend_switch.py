"""Selection and active-backend ownership must remain distinct."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from ai.voice_conversion_manager import VoiceConversionManager


class TrackingRuntime:
    def __init__(self, name: str) -> None:
        self.name = name
        self.enabled = False
        self.shutdown_calls = 0
        self.load_calls = 0
        self.selected_model = None
        self.state = SimpleNamespace(ready=False, error=None, engine=None, effect=None)
        self.model_manager = SimpleNamespace(discover_models=lambda: [])

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def load_model(self, model, *, audio_stream=None):
        del audio_stream
        self.load_calls += 1
        self.selected_model = model
        self.state = SimpleNamespace(ready=True, error=None, engine=None, effect=None)
        return self.state

    def shutdown(self):
        self.shutdown_calls += 1
        self.state.ready = False
        return True


class VoiceConversionBackendSwitchTests(unittest.TestCase):
    def test_gui_selection_does_not_hide_active_runtime_during_switch(self) -> None:
        rvc = TrackingRuntime("rvc")
        beatrice = TrackingRuntime("beatrice")
        manager = VoiceConversionManager(
            {"rvc": rvc, "beatrice": beatrice}, default_backend="rvc"
        )
        try:
            manager.set_enabled(True)
            self.assertTrue(manager.load_model("rvc", "modelF").ready)
            self.assertEqual(manager.active_backend, "rvc")

            manager.select_backend("beatrice")
            self.assertEqual(manager.current_backend, "beatrice")
            self.assertEqual(manager.active_backend, "rvc")
            manager.set_enabled(False)
            self.assertFalse(rvc.enabled)
            self.assertFalse(beatrice.enabled)

            manager.set_enabled(True)
            self.assertTrue(manager.load_model("beatrice", "jvs").ready)
            self.assertEqual(rvc.shutdown_calls, 1)
            self.assertEqual(manager.active_backend, "beatrice")
            self.assertTrue(beatrice.enabled)
        finally:
            self.assertTrue(manager.shutdown())
        self.assertEqual(beatrice.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

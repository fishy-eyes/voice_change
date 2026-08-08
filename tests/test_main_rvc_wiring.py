"""Verify startup remains lightweight until the user requests a model."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class FakeRuntime:
    instance = None

    def __init__(self, model_manager) -> None:
        type(self).instance = self
        self.model_manager = model_manager
        self.bound = None
        self.enabled = None
        self.loaded = None
        self.shutdown_called = False

    def bind_effect_manager(self, manager) -> None:
        self.bound = manager

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def load_model(self, name: str):
        self.loaded = name
        return SimpleNamespace(ready=True, error=None)

    def shutdown(self) -> bool:
        self.shutdown_called = True
        return True


class FakeStream:
    def stop(self) -> None:
        self.is_running = False

    is_running = False

    def __init__(self, recorder, player, effect_manager=None) -> None:
        self.recorder = recorder
        self.player = player
        self.effect_manager = effect_manager


class FakeApp:
    def exec(self) -> int:
        return 0

    def quit(self) -> None:
        pass


class FakeThread:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def join(self, timeout=None) -> None:
        pass


class FakeSelfMonitor:
    instance = None

    def __init__(self) -> None:
        type(self).instance = self
        self.stop_called = False

    def stop(self) -> None:
        self.stop_called = True


class MainRVCWiringTests(unittest.TestCase):
    @patch(
        "ai.beatrice.runtime.importlib.import_module",
        side_effect=AssertionError("native Beatrice import during startup"),
    )
    @patch("main.SelfMonitor", FakeSelfMonitor)
    @patch("main.threading.Thread", FakeThread)
    @patch("main.create_app", return_value=(FakeApp(), object()))
    @patch("main.RVCRuntime", FakeRuntime)
    @patch("main.RVCModelManager", side_effect=lambda root, **kwargs: SimpleNamespace(root=root, **kwargs))
    @patch("main.AudioStream", FakeStream)
    @patch("main.AudioPlayer", side_effect=lambda device=None: SimpleNamespace(device=device))
    @patch("main.AudioRecorder", side_effect=lambda device=None: SimpleNamespace(device=device))
    @patch("main._select_devices", return_value=(1, 2))
    @patch("main.signal.signal")
    @patch("main.setup_logger")
    def test_main_does_not_load_default_model_and_shuts_runtime_down(self, *mocks) -> None:
        del mocks
        main.main()

        runtime = FakeRuntime.instance
        self.assertIsNotNone(runtime)
        self.assertFalse(runtime.enabled)
        self.assertIsNone(runtime.loaded)
        self.assertIsNotNone(runtime.bound)
        self.assertEqual(
            [effect.name for effect in runtime.bound.effects], ["GainEffect"]
        )
        self.assertTrue(runtime.shutdown_called)
        self.assertIsNotNone(FakeSelfMonitor.instance)
        self.assertTrue(FakeSelfMonitor.instance.stop_called)


if __name__ == "__main__":
    unittest.main(verbosity=2)

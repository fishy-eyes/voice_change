"""Device enumeration and switching tests using fakes only."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  OK: {message}", flush=True)


class FakeRecorder:
    def __init__(self, device=None):
        self.device = device


class FakePlayer:
    def __init__(self, device=None):
        self.device = device


class FakeStream:
    created = []

    def __init__(self, recorder=None, player=None, effect_manager=None, *, running=False, fail_start=False):
        self.recorder = recorder
        self.player = player
        self.effect_manager = effect_manager
        self._running = running
        self.fail_start = fail_start
        self.start_calls = 0
        self.stop_calls = 0
        self._callback_count = 0
        self._total_proc_ms = 0.0
        self._max_proc_ms = 0.0
        self.created.append(self)

    @property
    def is_running(self):
        return self._running

    def start(self):
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("synthetic start failure")
        self._running = True

    def stop(self):
        self.stop_calls += 1
        self._running = False


class GuiDeviceManager:
    inputs = []
    outputs = []

    @classmethod
    def list_input_devices(cls):
        return list(cls.inputs)

    @classmethod
    def list_output_devices(cls):
        return list(cls.outputs)

    @staticmethod
    def get_device_name(index):
        return "System Default" if index is None else f"Device {index}"


def main() -> int:
    from PySide6.QtWidgets import QApplication

    import audio.device_manager as device_module
    from audio.device_manager import DeviceInfo, DeviceManager
    from core.context import AppContext
    from core.device_switching import stop_current_audio_stream, switch_audio_devices
    from effects.manager import EffectManager
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = None
    original_query = device_module.sd.query_devices
    try:
        print("\n[1/5] DeviceManager filters by channel direction", flush=True)
        raw_devices = [
            {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100},
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "CABLE Input", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Primary Sound Driver", "max_input_channels": 2, "max_output_channels": 2, "default_samplerate": 44100},
        ]
        device_module.sd.query_devices = lambda: raw_devices
        require([d.index for d in DeviceManager.list_input_devices()] == [0], "input list contains input-capable devices only")
        require([d.index for d in DeviceManager.list_output_devices()] == [1, 2], "output list includes speakers and CABLE Input")

        print("\n[2/5] GUI combos retain device indices and selections", flush=True)
        GuiDeviceManager.inputs = [DeviceInfo(2, "Mic A", 1, 44100), DeviceInfo(4, "Mic B", 1, 44100)]
        GuiDeviceManager.outputs = [DeviceInfo(7, "Speakers", 2, 48000), DeviceInfo(9, "CABLE Input", 2, 44100)]
        gui_stream = FakeStream(running=False)
        context = AppContext(
            effect_manager=EffectManager(),
            device_manager=GuiDeviceManager,
            audio_stream=gui_stream,
            input_device=2,
            output_device=7,
        )
        window = MainWindow(context)
        require(window._input_combo.currentData() == 2, "input combo selects current stream index")
        require(window._output_combo.currentData() == 7, "output combo selects current stream index")
        require(window._input_combo.itemData(2) == 4, "input item stores index as item data")
        require(window._output_combo.itemData(2) == 9, "output item stores index as item data")
        window._input_combo.setCurrentIndex(2)
        window._output_combo.setCurrentIndex(2)
        window._refresh_device_choices()
        require(window._input_combo.currentData() == 4, "refresh preserves input selection")
        require(window._output_combo.currentData() == 9, "refresh preserves output selection")
        require(context.audio_stream is gui_stream, "refresh does not replace or start the stream")

        print("\n[3/5] Stopped and running stream switches", flush=True)
        manager = EffectManager()
        ai_marker = object()
        manager.ai_marker = ai_marker
        stopped_old = FakeStream(effect_manager=manager, running=False)
        stopped_context = AppContext(manager, None, stopped_old, 1, 3)
        result = switch_audio_devices(
            stopped_context,
            5,
            6,
            recorder_factory=FakeRecorder,
            player_factory=FakePlayer,
            stream_factory=FakeStream,
        )
        require(result.success, "stopped stream switch succeeds")
        require(stopped_old.stop_calls == 1, "inactive old stream is closed before replacement")
        require(not stopped_context.audio_stream.is_running, "stopped stream is not auto-started")
        require(stopped_context.audio_stream.effect_manager is manager, "new stream reuses EffectManager")
        require(manager.ai_marker is ai_marker, "switch leaves AI-owned state untouched")

        running_old = FakeStream(effect_manager=manager, running=True)
        running_context = AppContext(manager, None, running_old, 1, 3)
        result = switch_audio_devices(
            running_context,
            8,
            9,
            recorder_factory=FakeRecorder,
            player_factory=FakePlayer,
            stream_factory=FakeStream,
        )
        first_new = running_context.audio_stream
        require(result.success, "running stream switch succeeds")
        require(running_old.stop_calls == 1, "running old stream is stopped first")
        require(first_new.start_calls == 1 and first_new.is_running, "new stream auto-starts")
        require(first_new.recorder.device == 8 and first_new.player.device == 9, "selected indices reach recorder and player")

        result = switch_audio_devices(
            running_context,
            10,
            11,
            recorder_factory=FakeRecorder,
            player_factory=FakePlayer,
            stream_factory=FakeStream,
        )
        require(result.success, "repeated running switch succeeds")
        require(first_new.stop_calls == 1 and not first_new.is_running, "repeated switch leaves no old stream running")
        require(running_context.audio_stream.is_running, "latest stream remains running")

        print("\n[4/5] Failed switch restores previous stream", flush=True)
        previous = running_context.audio_stream
        result = switch_audio_devices(
            running_context,
            12,
            13,
            recorder_factory=FakeRecorder,
            player_factory=FakePlayer,
            stream_factory=lambda recorder, player, effect_manager: FakeStream(
                recorder,
                player,
                effect_manager,
                fail_start=True,
            ),
        )
        require(not result.success, "start failure is reported")
        require(result.restored_previous_stream, "previous running stream is restored")
        require(running_context.audio_stream is previous, "failed transaction keeps context references")
        require(previous.is_running, "previous stream is running after rollback")
        require(running_context.effect_manager is manager, "rollback preserves EffectManager and AI state")

        print("\n[5/5] Final cleanup stops the latest context stream", flush=True)
        startup_stream = FakeStream(running=True)
        latest_stream = FakeStream(running=True)
        cleanup_context = AppContext(manager, None, latest_stream, 20, 21)
        stop_current_audio_stream(cleanup_context, fallback=startup_stream)
        require(latest_stream.stop_calls == 1, "cleanup stops latest stream")
        require(startup_stream.stop_calls == 0, "cleanup does not stop stale startup reference")

        print("\nDevice switching tests passed.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        device_module.sd.query_devices = original_query
        if window is not None:
            window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
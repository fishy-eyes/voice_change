"""Base-effect configuration and GUI-control tests without audio hardware."""

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


class FakeDeviceManager:
    @staticmethod
    def list_input_devices():
        return []

    @staticmethod
    def list_output_devices():
        return []

    @staticmethod
    def get_device_name(index):
        return "System Default" if index is None else f"Device {index}"


class FakeStream:
    is_running = False
    _callback_count = 0
    _total_proc_ms = 0.0
    _max_proc_ms = 0.0


def main() -> int:
    from PySide6.QtWidgets import QApplication

    import main as app_main
    from core.context import AppContext
    from core.rvc_lifecycle import initialize_rvc_application
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = None
    try:
        print("\n[1/3] All disabled base effects remain registered", flush=True)
        manager = app_main.create_effect_manager(
            gain_enabled=False,
            echo_enabled=False,
            robot_enabled=False,
        )
        names = [effect.name for effect in manager.effects]
        require(
            names == ["GainEffect", "EchoEffect", "RobotEffect"],
            "base effects keep their established order",
        )
        original_effects = {effect.name: effect for effect in manager.effects}
        require(
            all(not effect.enabled for effect in manager.effects),
            "configuration controls initial enabled state only",
        )

        print("\n[2/3] GUI controls mutate shared instances", flush=True)
        context = AppContext(
            effect_manager=manager,
            device_manager=FakeDeviceManager,
            audio_stream=FakeStream(),
        )
        window = MainWindow(context)
        window._toggle_effect("GainEffect")
        require(original_effects["GainEffect"].enabled, "GUI enables shared GainEffect")
        require("GainEffect: ON" in window._effects_label.text(), "refresh shows real ON state")
        window._toggle_effect("GainEffect")
        require(not original_effects["GainEffect"].enabled, "GUI disables shared GainEffect")
        require("GainEffect: OFF" in window._effects_label.text(), "refresh shows real OFF state")

        window._gain_slider.setValue(37)
        require(original_effects["GainEffect"].gain == 3.7, "slider updates shared gain value")
        require(
            manager.get_by_name("GainEffect") is original_effects["GainEffect"],
            "GUI and manager retain one GainEffect instance",
        )
        window._toggle_effect("MissingEffect")
        require(True, "missing effects are logged without crashing the GUI")

        print("\n[3/3] Disabled AI has no engine or worker", flush=True)
        state = initialize_rvc_application(enabled=False)
        require(state.engine is None, "AI-disabled startup creates no engine")
        require(state.effect is None, "AI-disabled startup creates no worker/effect")
        require(manager.get_by_name("AIVoiceEffect") is None, "GUI chain does not fake AI readiness")
        window._update_status_display()
        require("AI model: Not loaded" in window._status_label.text(), "GUI reports AI not loaded")

        print("\nEffect configuration tests passed.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if window is not None:
            window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
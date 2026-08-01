"""Offscreen tests for the developer-only RVC runtime controls."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.rvc_profiles import RVCInferenceConfig


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  OK: {message}", flush=True)


class FakeEngine:
    def __init__(self) -> None:
        self.config = RVCInferenceConfig(
            pitch_shift=12,
            f0_method="rmvpe",
            index_rate=0.30,
            rms_mix_rate=0.25,
            protect=0.33,
        )
        self.is_loaded = True
        self.update_calls: list[dict[str, object]] = []
        self.model_identity = object()
        self.pipeline_identity = object()

    def update_config(self, **changes):
        self.update_calls.append(dict(changes))
        self.config = self.config.updated(**changes)
        return self.config


class FakeAIVoiceEffect:
    name = "AIVoiceEffect"
    enabled = True
    is_running = True

    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine


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


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from core.context import AppContext
    from effects.manager import EffectManager
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    windows: list[MainWindow] = []
    try:
        print("\n[1/3] Missing RVC remains safe", flush=True)
        standalone = MainWindow(None)
        windows.append(standalone)
        require(not standalone._rvc_group.isEnabled(), "standalone controls are disabled")
        standalone._rvc_pitch_slider.setValue(5)
        require(True, "standalone slider change does not raise")

        empty_context = AppContext(
            effect_manager=EffectManager(),
            device_manager=FakeDeviceManager,
        )
        no_rvc = MainWindow(empty_context)
        windows.append(no_rvc)
        require(not no_rvc._rvc_group.isEnabled(), "missing AI effect is handled")

        print("\n[2/3] Controls initialize from the engine snapshot", flush=True)
        engine = FakeEngine()
        manager = EffectManager()
        manager.add(FakeAIVoiceEffect(engine))
        context = AppContext(
            effect_manager=manager,
            device_manager=FakeDeviceManager,
        )
        window = MainWindow(context)
        windows.append(window)
        require(window._rvc_group.isEnabled(), "RVC controls are enabled")
        require(window._rvc_pitch_slider.value() == 12, "pitch reads current config")
        require(window._rvc_index_slider.value() == 30, "index reads current config")
        require(window._rvc_protect_slider.value() == 33, "protect reads current config")
        require(window._rvc_rms_slider.value() == 25, "RMS reads current config")

        print("\n[3/3] Slider changes only call update_config", flush=True)
        model_identity = engine.model_identity
        pipeline_identity = engine.pipeline_identity
        window._rvc_pitch_slider.setValue(11)
        window._rvc_index_slider.setValue(15)
        window._rvc_protect_slider.setValue(20)
        window._rvc_rms_slider.setValue(40)

        require(len(engine.update_calls) == 4, "each slider change updates runtime config")
        require(engine.config.pitch_shift == 11, "pitch update applied")
        require(engine.config.index_rate == 0.15, "index update applied")
        require(engine.config.protect == 0.20, "protect update applied")
        require(engine.config.rms_mix_rate == 0.40, "RMS update applied")
        require(engine.model_identity is model_identity, "model was not replaced")
        require(engine.pipeline_identity is pipeline_identity, "pipeline was not rebuilt")
        require("+11" in window._rvc_pitch_label.text(), "pitch value is displayed")
        require("0.15" in window._rvc_index_label.text(), "index value is displayed")
        require(
            "RVC runtime config updated" in window.statusBar().currentMessage(),
            "successful update is reported",
        )

        print("\nRVC advanced GUI tests passed.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        for window in windows:
            window.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())

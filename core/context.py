"""Application context - shared runtime references."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from audio.stream import AudioStream
    from audio.monitor import SelfMonitor
    from audio.device_manager import DeviceManager
    from effects.manager import EffectManager
    from core.rvc_runtime import RVCRuntime
    from ai.voice_conversion_manager import VoiceConversionManager


class AppContext:
    """Lightweight container for runtime object references.

    Pass this to MainWindow (and later other components) so they
    can access the audio pipeline without import coupling.
    """

    def __init__(
        self,
        effect_manager: Optional[EffectManager] = None,
        device_manager: Optional[DeviceManager] = None,
        audio_stream: Optional[AudioStream] = None,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        rvc_runtime: Optional[RVCRuntime] = None,
        voice_conversion_manager: Optional[VoiceConversionManager] = None,
        self_monitor: Optional[SelfMonitor] = None,
    ) -> None:
        self.effect_manager = effect_manager
        self.device_manager = device_manager
        self.audio_stream = audio_stream
        self.input_device = input_device
        self.output_device = output_device
        self.rvc_runtime = rvc_runtime
        self.voice_conversion_manager = voice_conversion_manager
        self.self_monitor = self_monitor

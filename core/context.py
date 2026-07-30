"""Application context - shared runtime references."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from audio.stream import AudioStream
    from audio.device_manager import DeviceManager
    from effects.manager import EffectManager


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
    ) -> None:
        self.effect_manager = effect_manager
        self.device_manager = device_manager
        self.audio_stream = audio_stream

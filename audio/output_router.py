"""Post-effect audio fan-out without changing AudioStream or effect classes."""

from __future__ import annotations

import numpy as np
from loguru import logger

from effects.manager import EffectManager


class OutputRoutingEffectManager(EffectManager):
    """Run the established effect chain, then copy its output to a monitor."""

    def __init__(self, self_monitor) -> None:
        super().__init__()
        self._self_monitor = self_monitor

    def process(
        self,
        audio_data: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> np.ndarray:
        processed = super().process(audio_data, frames, time_info, status)
        try:
            self._self_monitor.submit(processed)
        except Exception as exc:
            # Monitoring is optional and must never break VB-CABLE output.
            logger.error("self-monitor output routing failed: {}", exc)
        return processed

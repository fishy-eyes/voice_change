"""Effect chain manager.

Holds an ordered list of BaseEffect instances and processes audio
through the entire chain in a single call.
"""

from typing import Optional

import numpy as np
from loguru import logger

from effects.base import BaseEffect


class EffectManager:
    """Manage and execute a chain of audio effects.

    Effects are executed in insertion order:
        data -> effect[0].process -> effect[1].process -> ... -> output

    Disabled effects are skipped automatically.
    """

    def __init__(self) -> None:
        self._effects: list[BaseEffect] = []

    # ------------------------------------------------------------------
    # chain management
    # ------------------------------------------------------------------
    def add(self, effect: BaseEffect) -> None:
        """Append an effect to the end of the chain."""
        self._effects.append(effect)
        logger.info("effect added: {} (chain length: {})", effect.name, len(self._effects))

    def remove(self, effect: BaseEffect) -> bool:
        """Remove an effect from the chain. Returns True if found."""
        try:
            self._effects.remove(effect)
            logger.info("effect removed: {} (chain length: {})", effect.name, len(self._effects))
            return True
        except ValueError:
            logger.warning("effect not found in chain: {}", effect.name)
            return False

    def remove_by_name(self, name: str) -> bool:
        """Remove the first effect matching the given name."""
        for i, e in enumerate(self._effects):
            if e.name == name:
                self._effects.pop(i)
                logger.info("effect removed by name: {} (chain length: {})", name, len(self._effects))
                return True
        logger.warning("effect not found by name: {}", name)
        return False

    def clear(self) -> None:
        """Remove all effects from the chain."""
        count = len(self._effects)
        self._effects.clear()
        logger.info("effect chain cleared ({} effects removed)", count)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    @property
    def effects(self) -> list[BaseEffect]:
        """Read-only view of the current effect chain."""
        return list(self._effects)

    @property
    def is_empty(self) -> bool:
        return len(self._effects) == 0

    def __len__(self) -> int:
        return len(self._effects)

    def get_by_name(self, name: str) -> Optional[BaseEffect]:
        """Return the first effect matching *name*, or None."""
        for e in self._effects:
            if e.name == name:
                return e
        return None

    def enable(self, name: str) -> bool:
        """Enable the effect matching *name*. Returns True if found."""
        effect = self.get_by_name(name)
        if effect is None:
            logger.warning("enable: effect not found: {}", name)
            return False
        effect.enabled = True
        logger.info("effect enabled: {}", name)
        return True

    def disable(self, name: str) -> bool:
        """Disable the effect matching *name*. Returns True if found."""
        effect = self.get_by_name(name)
        if effect is None:
            logger.warning("disable: effect not found: {}", name)
            return False
        effect.enabled = False
        logger.info("effect disabled: {}", name)
        return True

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------
    def process(
        self,
        audio_data: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> np.ndarray:
        """Run audio through the entire effect chain.

        If the chain is empty or all effects are disabled, returns
        audio_data unchanged (zero-copy passthrough).
        """
        data = audio_data
        for effect in self._effects:
            if not effect.enabled:
                continue
            try:
                data = effect.process(data, frames, time_info, status)
            except Exception as e:
                logger.error("effect '{}' raised: {}", effect.name, e)
                # skip this effect, continue with previous data
        return data

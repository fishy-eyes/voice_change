"""Registry for backend-specific settings widget factories."""

from __future__ import annotations

from collections.abc import Callable


class BackendSettingsRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., object]] = {}

    def register(self, backend_id: str, factory: Callable[..., object]) -> None:
        key = str(backend_id).strip().lower()
        if not key or key in self._factories:
            raise ValueError(f"invalid or duplicate backend settings: {backend_id}")
        self._factories[key] = factory

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(self._factories)

    def create(self, backend_id: str, **kwargs):
        key = str(backend_id).strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise LookupError(f"no settings panel for backend: {backend_id}") from exc
        return factory(**kwargs)


def create_default_registry() -> BackendSettingsRegistry:
    from gui.backend_settings.rvc import RVCSettingsPanel

    registry = BackendSettingsRegistry()
    registry.register("rvc", RVCSettingsPanel)
    return registry

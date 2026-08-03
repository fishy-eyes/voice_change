"""Minimal contract for backend-specific settings widgets."""

from __future__ import annotations

from typing import Mapping, Any


class BackendSettingsPanel:
    backend_id: str

    def refresh_from_runtime(self) -> None:
        """Refresh controls after a model switch."""

    def apply_state(self, state: Mapping[str, Any]) -> None:
        """Apply a detached parameter snapshot to controls."""

    def close_panel(self) -> None:
        """Release panel-only resources without touching audio runtime."""

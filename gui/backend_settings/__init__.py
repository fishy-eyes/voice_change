"""Backend-specific settings panels registered outside the main window."""

from gui.backend_settings.registry import BackendSettingsRegistry, create_default_registry

__all__ = ["BackendSettingsRegistry", "create_default_registry"]

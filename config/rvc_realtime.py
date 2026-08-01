"""Validated real-time chunk/overlap presets for RVC streaming."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class RVCRealtimePreset:
    """Latency/continuity settings independent from model inference settings."""

    name: str
    chunk_ms: int
    overlap_ms: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if self.overlap_ms < 0:
            raise ValueError("overlap_ms must not be negative")
        if self.overlap_ms * 2 > self.chunk_ms:
            raise ValueError("overlap_ms must not exceed half of chunk_ms")

    def chunk_samples(self, sample_rate: int) -> int:
        return _milliseconds_to_samples(self.chunk_ms, sample_rate)

    def overlap_samples(self, sample_rate: int) -> int:
        return _milliseconds_to_samples(self.overlap_ms, sample_rate)


def _milliseconds_to_samples(milliseconds: int, sample_rate: int) -> int:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return int(round(sample_rate * milliseconds / 1000.0))


RVC_REALTIME_PRESETS: Mapping[str, RVCRealtimePreset] = MappingProxyType(
    {
        "low_latency": RVCRealtimePreset(
            name="Low Latency",
            chunk_ms=325,
            overlap_ms=50,
        ),
        "balanced": RVCRealtimePreset(
            name="Balanced",
            chunk_ms=500,
            overlap_ms=50,
        ),
        "high_quality": RVCRealtimePreset(
            name="High Quality",
            chunk_ms=500,
            overlap_ms=100,
        ),
    }
)
RVC_DEFAULT_REALTIME_PRESET = "balanced"


def get_rvc_realtime_preset(key: str) -> RVCRealtimePreset:
    """Return a preset by stable key with a useful validation error."""
    try:
        return RVC_REALTIME_PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(RVC_REALTIME_PRESETS)
        raise ValueError(
            f"unknown RVC realtime preset {key!r}; choose: {choices}"
        ) from exc

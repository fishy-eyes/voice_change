"""Continuity contract shared with the backend-neutral Worker."""

CONTIGUOUS_STREAM_ATTRIBUTE = "requires_contiguous_input"
RESET_STREAM_METHOD = "reset_stream"


def requires_contiguous_input(engine: object) -> bool:
    return bool(getattr(engine, CONTIGUOUS_STREAM_ATTRIBUTE, False))


__all__ = [
    "CONTIGUOUS_STREAM_ATTRIBUTE",
    "RESET_STREAM_METHOD",
    "requires_contiguous_input",
]

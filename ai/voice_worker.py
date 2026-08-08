"""Backend-neutral Worker with explicit stateful-continuity recovery."""

from __future__ import annotations

import queue
import threading
from time import perf_counter

from loguru import logger

from ai.beatrice.worker_support import requires_contiguous_input
from ai.rvc_worker import RVCWorker


class VoiceConversionWorker(RVCWorker):
    """Keep RVC drop-oldest semantics, but reset contiguous stream engines."""

    def __init__(self, engine, *args, **kwargs) -> None:
        super().__init__(engine, *args, **kwargs)
        self._requires_continuity = requires_contiguous_input(engine)
        self._recovery_requested = threading.Event()
        self._continuity_generation = 0
        self._continuity_error_count = 0
        self._recovery_count = 0
        self._recovery_failure_count = 0
        self._last_continuity_error: str | None = None

    @property
    def continuity_generation(self) -> int:
        return self._continuity_generation

    @property
    def continuity_error_count(self) -> int:
        return self._continuity_error_count

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    @property
    def recovery_failure_count(self) -> int:
        return self._recovery_failure_count

    @property
    def recovery_pending(self) -> bool:
        return self._recovery_requested.is_set()

    @property
    def last_continuity_error(self) -> str | None:
        return self._last_continuity_error

    def put(self, audio, timeout: float = 0.0) -> bool:
        if not self._requires_continuity:
            return super().put(audio, timeout=timeout)
        if not self._running.is_set() or self._recovery_requested.is_set():
            return False
        try:
            self._input_q.put(audio, timeout=timeout)
            return True
        except queue.Full:
            # Never drop N and then process N+1 against the same converter.
            self._input_drop_count += 1
            self._request_recovery("input queue overflow; newest block rejected")
            return False

    def _request_recovery(self, reason: str) -> None:
        if not self._recovery_requested.is_set():
            self._continuity_generation += 1
            self._continuity_error_count += 1
            self._last_continuity_error = reason
            logger.error("VoiceConversionWorker continuity error: {}", reason)
        self._recovery_requested.set()

    def _recover_stream(self) -> bool:
        self._drain_queue(self._input_q)
        self._drain_queue(self._output_q)
        reset = getattr(self._engine, "reset_stream", None)
        if not callable(reset):
            self._recovery_failure_count += 1
            self._error_count += 1
            logger.error("Contiguous engine has no reset_stream()")
            self._running.clear()
            return False
        try:
            reset()
        except Exception as exc:
            self._recovery_failure_count += 1
            self._error_count += 1
            self._last_continuity_error = f"reset failed: {type(exc).__name__}: {exc}"
            logger.error("VoiceConversionWorker stream reset failed: {}", exc)
            self._running.clear()
            return False
        self._recovery_count += 1
        self._recovery_requested.clear()
        logger.warning(
            "VoiceConversionWorker stream recovered (generation={})",
            self._continuity_generation,
        )
        return True

    def _run_loop(self) -> None:
        if not self._requires_continuity:
            super()._run_loop()
            return

        logger.debug("VoiceConversionWorker contiguous thread started")
        while self._running.is_set():
            if self._recovery_requested.is_set():
                if not self._recover_stream():
                    break
                continue
            try:
                audio = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            result = None
            try:
                self._inferencing.set()
                started = perf_counter()
                process_audio = getattr(self._engine, "process_audio", None)
                result = (
                    process_audio(audio)
                    if callable(process_audio)
                    else self._engine.infer(audio)
                )
                self._infer_count += 1
                infer_ms = (perf_counter() - started) * 1000.0
                self._last_infer_ms = infer_ms
                self._average_infer_ms += (
                    infer_ms - self._average_infer_ms
                ) / self._infer_count
            except Exception as exc:
                self._error_count += 1
                self._request_recovery(
                    f"inference failed: {type(exc).__name__}: {exc}"
                )
            finally:
                self._inferencing.clear()

            # An overflow/error invalidates the active result generation.
            if self._recovery_requested.is_set():
                continue
            try:
                self._output_q.put_nowait(result)
            except queue.Full:
                self._output_drop_count += 1
                self._request_recovery("output queue overflow; result rejected")

        self._drain_queue(self._input_q)
        self._drain_queue(self._output_q)
        logger.debug("VoiceConversionWorker contiguous thread exiting")

__all__ = ["VoiceConversionWorker"]

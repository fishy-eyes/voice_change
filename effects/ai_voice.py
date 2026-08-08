"""Non-blocking voice-conversion adapter for the real-time effect chain."""

from __future__ import annotations

from collections import deque
import threading
from time import monotonic, perf_counter, sleep
from typing import TYPE_CHECKING, Deque

import numpy as np
from loguru import logger

from ai.voice_worker import VoiceConversionWorker
from config.settings import SAMPLE_RATE
from effects.base import BaseEffect

if TYPE_CHECKING:
    from ai.voice_engine.base import VoiceConversionEngine


class AIVoiceEffect(BaseEffect):
    """Buffer callback audio and run backend inference outside the callback.

    Input blocks are collected into overlapping windows for a background worker.
    Completed windows are linearly crossfaded, buffered, and returned in slices
    matching each input callback block's shape.
    """

    def __init__(
        self,
        engine: VoiceConversionEngine,
        chunk_size: int = SAMPLE_RATE,
        overlap_size: int = 0,
        max_queue_size: int = 2,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap_size < 0 or overlap_size * 2 > chunk_size:
            raise ValueError(
                "overlap_size must be between 0 and half the chunk_size"
            )
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        self._engine: VoiceConversionEngine = engine
        self._chunk_size = int(chunk_size)
        self._overlap_size = int(overlap_size)
        self._max_queue_size = int(max_queue_size)
        self._max_output_samples = self._chunk_size * self._max_queue_size
        self._worker = VoiceConversionWorker(
            engine,
            chunk_size=self._chunk_size,
            max_queue_size=max_queue_size,
        )

        # The input buffer is fixed-size, so slow inference can never make
        # callback-side input storage grow without bound.
        self._input_buffer = np.empty(self._chunk_size, dtype=np.float32)
        self._input_size = 0
        self._worker_generation = self._worker.continuity_generation

        self._pending_output_tail: np.ndarray | None = None
        # Output chunks use a deque plus an offset to avoid repeatedly copying
        # an entire one-second result for every 256-sample callback.
        self._output_chunks: Deque[np.ndarray] = deque()
        self._output_offset = 0
        self._output_size = 0

        self._started = False
        self._last_warmup_ms = 0.0
        # process() only attempts this lock; it never waits for lifecycle work.
        self._state_lock = threading.Lock()
        logger.debug("AIVoiceEffect created with engine: {}", engine)

    @property
    def engine(self) -> VoiceConversionEngine:
        """The externally owned voice-conversion engine."""
        return self._engine

    @property
    def worker(self) -> VoiceConversionWorker:
        """The inference worker, exposed for metrics and lifecycle checks."""
        return self._worker

    @property
    def is_running(self) -> bool:
        return self._started and self._worker.is_running

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap_size(self) -> int:
        return self._overlap_size

    @property
    def hop_size(self) -> int:
        return self._chunk_size - self._overlap_size

    @property
    def input_buffered_samples(self) -> int:
        return self._input_size

    @property
    def output_buffered_samples(self) -> int:
        return self._output_size

    @property
    def last_warmup_ms(self) -> float:
        return self._last_warmup_ms

    def start(self) -> bool:
        """Clear stale audio and start the worker if the model is loaded."""
        with self._state_lock:
            if self.is_running:
                return True
            self._reset_buffers()
            if not self._engine.is_loaded:
                self._started = False
                logger.warning("AIVoiceEffect: engine is not loaded; using passthrough")
                return False
            self._started = self._worker.start()
            return self._started

    def warmup(self, timeout: float = 120.0) -> bool:
        """Warm up RVC through the Worker without exposing audio to playback.

        This method is intentionally blocking and must run before AudioStream
        starts. ``process()`` remains non-blocking if called concurrently.
        """
        if timeout <= 0:
            logger.error("AIVoiceEffect: warmup timeout must be positive")
            return False

        with self._state_lock:
            if not self._engine.is_loaded or not self.is_running:
                logger.error("AIVoiceEffect: warmup requires a loaded, running effect")
                return False
            if self._worker.is_inferencing:
                logger.error("AIVoiceEffect: warmup refused while inference is active")
                return False

            self._reset_buffers()
            self._worker.clear_queues()
            sample_rate = int(getattr(self._engine, "sample_rate", SAMPLE_RATE))
            timeline = np.arange(self._chunk_size, dtype=np.float32) / sample_rate
            warmup_audio = (
                0.05 * np.sin(2.0 * np.pi * 220.0 * timeline)
            ).astype(np.float32)
            started_at = perf_counter()

            try:
                if not self._worker.put(warmup_audio, timeout=0.0):
                    logger.error("AIVoiceEffect: warmup submission failed")
                    return False
                result = self._worker.get(timeout=timeout)
                if result is None:
                    logger.error(
                        "AIVoiceEffect: warmup produced no result within {:.3f}s",
                        timeout,
                    )
                    return False
                result = np.asarray(result)
                if (
                    result.shape != warmup_audio.shape
                    or result.dtype != np.float32
                    or not np.all(np.isfinite(result))
                ):
                    logger.error(
                        "AIVoiceEffect: invalid warmup output shape={} dtype={}",
                        result.shape,
                        result.dtype,
                    )
                    return False
                logger.info(
                    "AIVoiceEffect: warmup completed in {:.1f}ms",
                    (perf_counter() - started_at) * 1000.0,
                )
                return True
            except Exception as exc:
                logger.error("AIVoiceEffect: warmup failed: {}", exc)
                return False
            finally:
                self._last_warmup_ms = (perf_counter() - started_at) * 1000.0
                # The warmup result is consumed directly and no warmup or stale
                # callback audio may enter normal playback.
                self._worker.clear_queues()
                self._reset_buffers()

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the worker without unloading its externally owned engine."""
        with self._state_lock:
            self._started = False
            stopped = self._worker.stop(timeout=timeout)
            self._reset_buffers()
            return stopped

    def update_realtime_config(
        self,
        *,
        chunk_size: int,
        overlap_size: int,
        timeout: float = 5.0,
    ) -> None:
        """Resize streaming buffers while retaining the Engine and Worker."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap_size < 0 or overlap_size * 2 > chunk_size:
            raise ValueError(
                "overlap_size must be between 0 and half the chunk_size"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        with self._state_lock:
            if (
                chunk_size == self._chunk_size
                and overlap_size == self._overlap_size
            ):
                return

            # No callback can submit new work while this lock is held. Drain
            # queued work, let at most the active inference finish, then drop
            # its old-shape output before swapping buffer geometry.
            self._worker.clear_queues()
            deadline = monotonic() + timeout
            while self._worker.is_inferencing and monotonic() < deadline:
                sleep(0.005)
            if self._worker.is_inferencing:
                raise TimeoutError("active RVC inference did not finish in time")
            self._worker.clear_queues()

            self._chunk_size = int(chunk_size)
            self._overlap_size = int(overlap_size)
            self._max_output_samples = self._chunk_size * self._max_queue_size
            self._input_buffer = np.empty(self._chunk_size, dtype=np.float32)
            self._reset_buffers()
            logger.info(
                "AIVoiceEffect realtime buffers updated (chunk={}, overlap={})",
                self._chunk_size,
                self._overlap_size,
            )

    def process(
        self,
        audio_data: np.ndarray,
        frames: int | None = None,
        time_info: object = None,
        status: object = None,
    ) -> np.ndarray:
        """Process one callback block without waiting for RVC inference."""
        del frames, time_info, status  # BaseEffect compatibility; not needed here.
        original = np.asarray(audio_data)
        original_shape = original.shape
        if original.ndim == 1:
            sample_count = original.shape[0]
        elif original.ndim == 2 and original.shape[1] == 1:
            sample_count = original.shape[0]
        else:
            logger.warning(
                "AIVoiceEffect: unsupported audio shape {}; passthrough",
                original_shape,
            )
            return original.astype(np.float32, copy=False)

        flat = np.asarray(original, dtype=np.float32).reshape(-1)

        # Preserve the established passthrough behavior until AI is ready.
        if not self._engine.is_loaded or not self.is_running:
            return flat.reshape(original_shape)
        if not self._state_lock.acquire(blocking=False):
            return flat.reshape(original_shape)

        try:
            if self._worker_generation != self._worker.continuity_generation:
                self._reset_buffers()
                self._worker_generation = self._worker.continuity_generation

            self._drain_worker_output()
            self._accumulate_input(flat)

            if self._output_size < sample_count:
                # Active AI with no converted block returns silence. This avoids
                # mixing current dry input with delayed converted speech.
                return np.zeros(original_shape, dtype=np.float32)

            return self._take_output(sample_count).reshape(original_shape)
        except Exception as exc:
            logger.error("AIVoiceEffect: realtime processing failed: {}", exc)
            if self.is_running:
                return np.zeros(original_shape, dtype=np.float32)
            return flat.reshape(original_shape)
        finally:
            self._state_lock.release()

    def _reset_buffers(self) -> None:
        self._input_size = 0
        self._output_chunks.clear()
        self._output_offset = 0
        self._output_size = 0
        self._pending_output_tail = None

    def _accumulate_input(self, audio: np.ndarray) -> None:
        position = 0
        while position < audio.size:
            count = min(
                self._chunk_size - self._input_size,
                audio.size - position,
            )
            end = self._input_size + count
            self._input_buffer[self._input_size:end] = audio[
                position:position + count
            ]
            self._input_size = end
            position += count

            if self._input_size == self._chunk_size:
                chunk = self._input_buffer.copy()
                try:
                    submitted = self._worker.put(chunk, timeout=0.0)
                except Exception as exc:
                    logger.error("AIVoiceEffect: worker submission failed: {}", exc)
                    submitted = False
                if not submitted:
                    if self._worker_generation != self._worker.continuity_generation:
                        self._reset_buffers()
                        self._worker_generation = self._worker.continuity_generation
                        return
                    logger.warning("AIVoiceEffect: conversion chunk was not submitted")
                # Retain the input tail as context for the next window. Queued
                # or dropped, callback-side storage remains fixed-size.
                if self._overlap_size:
                    self._input_buffer[:self._overlap_size] = self._input_buffer[
                        self._chunk_size - self._overlap_size:self._chunk_size
                    ]
                self._input_size = self._overlap_size

    def _drain_worker_output(self) -> None:
        while True:
            result = self._worker.get_nowait()
            if result is None:
                break
            chunk = np.asarray(result, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue
            chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)
            self._append_converted_window(chunk)
            self._trim_output_buffer()

    def _append_converted_window(self, chunk: np.ndarray) -> None:
        """Append one converted window using complementary linear overlap."""
        if chunk.size != self._chunk_size:
            logger.warning(
                "AIVoiceEffect: discarded stale output size {} (expected {})",
                chunk.size,
                self._chunk_size,
            )
            return
        overlap = self._overlap_size
        if overlap == 0:
            self._append_output_chunk(chunk)
            return

        hop = self.hop_size
        if self._pending_output_tail is None:
            self._append_output_chunk(chunk[:hop])
        else:
            fade_in = np.linspace(
                0.0,
                1.0,
                overlap,
                dtype=np.float32,
            )
            blended = (
                self._pending_output_tail * (1.0 - fade_in)
                + chunk[:overlap] * fade_in
            ).astype(np.float32, copy=False)
            if hop > overlap:
                stitched = np.concatenate((blended, chunk[overlap:hop]))
            else:
                stitched = blended
            self._append_output_chunk(stitched)
        self._pending_output_tail = chunk[hop:].copy()

    def _append_output_chunk(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        converted = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._output_chunks.append(converted)
        self._output_size += converted.size

    def _trim_output_buffer(self) -> None:
        """Drop oldest converted samples if callback consumption falls behind."""
        excess = self._output_size - self._max_output_samples
        while excess > 0 and self._output_chunks:
            available = self._output_chunks[0].size - self._output_offset
            dropped = min(excess, available)
            self._output_offset += dropped
            self._output_size -= dropped
            excess -= dropped
            if self._output_offset == self._output_chunks[0].size:
                self._output_chunks.popleft()
                self._output_offset = 0

    def _take_output(self, sample_count: int) -> np.ndarray:
        output = np.empty(sample_count, dtype=np.float32)
        written = 0
        while written < sample_count:
            chunk = self._output_chunks[0]
            count = min(
                sample_count - written,
                chunk.size - self._output_offset,
            )
            output[written:written + count] = chunk[
                self._output_offset:self._output_offset + count
            ]
            written += count
            self._output_offset += count
            self._output_size -= count
            if self._output_offset == chunk.size:
                self._output_chunks.popleft()
                self._output_offset = 0
        return output

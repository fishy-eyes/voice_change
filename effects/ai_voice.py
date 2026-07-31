"""Non-blocking RVC adapter for the real-time effect chain."""

from __future__ import annotations

from collections import deque
import threading
from typing import TYPE_CHECKING, Deque

import numpy as np
from loguru import logger

from ai.rvc_worker import RVCWorker
from effects.base import BaseEffect

if TYPE_CHECKING:
    from ai.rvc_engine import RVCEngine


class AIVoiceEffect(BaseEffect):
    """Buffer callback audio and run RVC inference outside the callback.

    Input blocks are collected into fixed, non-overlapping chunks for a
    :class:`RVCWorker`. Completed chunks are buffered and returned in slices
    matching each input block's shape.
    """

    def __init__(
        self,
        engine: RVCEngine,
        chunk_size: int = 44100,
        max_queue_size: int = 2,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        self._engine: RVCEngine = engine
        self._chunk_size = int(chunk_size)
        self._max_output_samples = self._chunk_size * int(max_queue_size)
        self._worker = RVCWorker(
            engine,
            chunk_size=self._chunk_size,
            max_queue_size=max_queue_size,
        )

        # The input buffer is fixed-size, so slow inference can never make
        # callback-side input storage grow without bound.
        self._input_buffer = np.empty(self._chunk_size, dtype=np.float32)
        self._input_size = 0

        # Output chunks use a deque plus an offset to avoid repeatedly copying
        # an entire one-second result for every 256-sample callback.
        self._output_chunks: Deque[np.ndarray] = deque()
        self._output_offset = 0
        self._output_size = 0

        self._started = False
        # process() only attempts this lock; it never waits for lifecycle work.
        self._state_lock = threading.Lock()
        logger.debug("AIVoiceEffect created with engine: {}", engine)

    @property
    def engine(self) -> RVCEngine:
        """The externally owned RVC engine."""
        return self._engine

    @property
    def worker(self) -> RVCWorker:
        """The inference worker, exposed for metrics and lifecycle checks."""
        return self._worker

    @property
    def is_running(self) -> bool:
        return self._started and self._worker.is_running

    @property
    def input_buffered_samples(self) -> int:
        return self._input_size

    @property
    def output_buffered_samples(self) -> int:
        return self._output_size

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

    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the worker without unloading its externally owned engine."""
        with self._state_lock:
            self._started = False
            stopped = self._worker.stop(timeout=timeout)
            self._reset_buffers()
            return stopped

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
                    logger.warning("AIVoiceEffect: RVC chunk was not submitted")
                # Queued or dropped, a full callback-side chunk is never kept.
                self._input_size = 0

    def _drain_worker_output(self) -> None:
        while True:
            result = self._worker.get_nowait()
            if result is None:
                break
            chunk = np.asarray(result, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue
            chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)
            self._output_chunks.append(chunk)
            self._output_size += chunk.size
            self._trim_output_buffer()

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

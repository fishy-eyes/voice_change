"""RVC real-time inference worker.

Runs RVCEngine.infer() in a dedicated thread with input/output queues.
Decouples audio I/O from GPU inference latency.

Usage:
    from ai.rvc_engine import RVCEngine
    from ai.rvc_worker import RVCWorker

    engine = RVCEngine(...)
    engine.load_model()

    worker = RVCWorker(engine, chunk_size=48000)
    worker.start()

    worker.put(audio_chunk)       # non-blocking
    result = worker.get(timeout=2)  # blocking with timeout

    worker.stop()
"""

from __future__ import annotations

import threading
import queue
from time import perf_counter
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import SAMPLE_RATE


class RVCWorker:
    """Dedicated-thread RVC inference worker.

    Lifecycle:
        1. __init__()  - store engine and config
        2. start()     - launch worker thread
        3. put()       - submit audio chunk to input queue
        4. get()       - retrieve processed audio from output queue
        5. stop()      - signal thread to stop and join

    Parameters
    ----------
    engine : RVCEngine
        A loaded RVC engine instance.
    chunk_size : int
        Expected chunk length in samples (default is one configured second).
        Used for logging/validation, not enforced.
    max_queue_size : int
        Maximum depth of input/output queues (default 2).
        Old items are dropped when full to prevent stale audio.
    """

    def __init__(
        self,
        engine,
        chunk_size: int = SAMPLE_RATE,
        max_queue_size: int = 2,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        self._engine = engine
        self._chunk_size = int(chunk_size)
        self._max_queue = int(max_queue_size)

        self._input_q: queue.Queue = queue.Queue(maxsize=self._max_queue)
        self._output_q: queue.Queue = queue.Queue(maxsize=self._max_queue)

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._inferencing = threading.Event()
        self._infer_count: int = 0
        self._error_count: int = 0
        self._input_drop_count: int = 0
        self._output_drop_count: int = 0
        self._last_infer_ms: float = 0.0
        self._average_infer_ms: float = 0.0

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(
            self._running.is_set()
            and thread is not None
            and thread.is_alive()
        )

    @property
    def thread_alive(self) -> bool:
        """Whether the worker thread still exists, including while stopping."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def input_pending(self) -> int:
        return self._input_q.qsize()

    @property
    def output_pending(self) -> int:
        return self._output_q.qsize()

    @property
    def input_queue_size(self) -> int:
        """Current number of queued input chunks."""
        return self._input_q.qsize()

    @property
    def output_queue_size(self) -> int:
        """Current number of queued output chunks."""
        return self._output_q.qsize()

    @property
    def is_inferencing(self) -> bool:
        """Whether the worker thread is currently inside engine.infer()."""
        return self._inferencing.is_set()

    @property
    def infer_count(self) -> int:
        return self._infer_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def input_drop_count(self) -> int:
        return self._input_drop_count

    @property
    def output_drop_count(self) -> int:
        return self._output_drop_count

    @property
    def last_infer_ms(self) -> float:
        return self._last_infer_ms

    @property
    def average_infer_ms(self) -> float:
        return self._average_infer_ms

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the worker thread, returning whether it is accepting work."""
        if self.is_running:
            logger.warning("RVCWorker: already running")
            return True
        if self.thread_alive:
            logger.warning("RVCWorker: previous thread is still stopping")
            return False

        self._thread = None
        self.clear_queues()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="rvc-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("RVCWorker: started (chunk_size={})", self._chunk_size)
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal the worker to stop and wait at most *timeout* seconds.

        A timed-out inference thread is deliberately retained so a later
        ``stop()`` can join it and ``start()`` cannot create a second worker
        against the same engine.
        """
        thread = self._thread
        if thread is None:
            self._running.clear()
            self.clear_queues()
            logger.warning("RVCWorker: not running")
            return True
        logger.info("RVCWorker: stopping ...")
        self._running.clear()
        thread.join(timeout=max(0.0, timeout))
        if thread.is_alive():
            logger.warning("RVCWorker: thread did not exit within {:.1f}s", timeout)
            return False

        self._thread = None
        self.clear_queues()
        logger.info(
            "RVCWorker: stopped (processed={}, errors={})",
            self._infer_count, self._error_count,
        )
        return True

    # ------------------------------------------------------------------
    # data transfer
    # ------------------------------------------------------------------

    def put(self, audio: np.ndarray, timeout: float = 0.0) -> bool:
        """Submit an audio chunk to the input queue.

        Parameters
        ----------
        audio : np.ndarray
            Float32 mono audio chunk.
        timeout : float
            Block up to this many seconds. 0 = non-blocking (default).

        Returns
        -------
        bool
            True if enqueued, False if queue was full.
        """
        if not self._running.is_set():
            logger.warning("RVCWorker: put() called but worker is not running")
            return False
        try:
            self._input_q.put(audio, timeout=timeout)
            return True
        except queue.Full:
            logger.warning("RVCWorker: input queue full, dropping oldest chunk")
            # Drop oldest and enqueue new
            try:
                self._input_q.get_nowait()
                self._input_drop_count += 1
            except queue.Empty:
                pass
            try:
                self._input_q.put_nowait(audio)
                return True
            except queue.Full:
                return False

    def get(self, timeout: float | None = None) -> Optional[np.ndarray]:
        """Retrieve a processed audio chunk from the output queue.

        Parameters
        ----------
        timeout : float | None
            Block up to this many seconds. None = block forever.

        Returns
        -------
        np.ndarray or None
            Processed audio, or None if infer failed for that chunk.
        """
        try:
            return self._output_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_nowait(self) -> Optional[np.ndarray]:
        """Retrieve one completed result without waiting."""
        return self.get(timeout=0.0)

    def clear_queues(self) -> None:
        """Discard queued input/output without touching an active inference."""
        self._drain_queue(self._input_q)
        self._drain_queue(self._output_q)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _drain_queue(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def _run_loop(self) -> None:
        """Worker thread main loop."""
        logger.debug("RVCWorker: thread started")
        while self._running.is_set():
            try:
                audio = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._inferencing.set()
                infer_started = perf_counter()
                result = self._engine.infer(audio)
                self._infer_count += 1
                infer_ms = (perf_counter() - infer_started) * 1000.0
                self._last_infer_ms = infer_ms
                self._average_infer_ms += (
                    infer_ms - self._average_infer_ms
                ) / self._infer_count
            except Exception as e:
                logger.error("RVCWorker: infer failed: {}", e)
                self._error_count += 1
                result = None
            finally:
                self._inferencing.clear()

            # Enqueue result, drop oldest if full
            try:
                self._output_q.put_nowait(result)
            except queue.Full:
                try:
                    self._output_q.get_nowait()
                    self._output_drop_count += 1
                except queue.Empty:
                    pass
                try:
                    self._output_q.put_nowait(result)
                except queue.Full:
                    pass

        logger.debug("RVCWorker: thread exiting")

    def __repr__(self) -> str:
        state = "running" if self._running.is_set() else "stopped"
        return (
            f"RVCWorker(chunk={self._chunk_size}, {state}, "
            f"processed={self._infer_count}, errors={self._error_count})"
        )

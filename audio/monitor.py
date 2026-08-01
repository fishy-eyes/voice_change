"""Independent playback stream for processed-audio self monitoring."""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from config.settings import BLOCKSIZE, CHANNELS, DTYPE, LATENCY, SAMPLE_RATE


class SelfMonitor:
    """Play copied post-effect audio without owning the primary audio stream."""

    def __init__(
        self,
        *,
        samplerate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        blocksize: int = BLOCKSIZE,
        dtype: str = DTYPE,
        latency: str = LATENCY,
        volume: float = 0.5,
        queue_blocks: int = 4,
        output_stream_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if blocksize <= 0:
            raise ValueError("blocksize must be positive")
        if queue_blocks <= 0:
            raise ValueError("queue_blocks must be positive")
        self._samplerate = int(samplerate)
        self._channels = int(channels)
        self._blocksize = int(blocksize)
        self._dtype = dtype
        self._latency = latency
        self._volume = self._validate_volume(volume)
        self._queue_blocks = int(queue_blocks)
        self._output_stream_factory = output_stream_factory or sd.OutputStream
        self._stream = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self._queue_blocks
        )
        self._pending_block: Optional[np.ndarray] = None
        self._pending_offset = 0
        self._output_device: Optional[int] = None
        self._callback_count = 0
        self._drop_count = 0
        self._underflow_count = 0
        self._lock = threading.RLock()

    @staticmethod
    def _validate_volume(value: float) -> float:
        volume = float(value)
        if not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")
        return volume

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        with self._lock:
            self._volume = self._validate_volume(value)

    @property
    def is_running(self) -> bool:
        stream = self._stream
        return bool(stream is not None and getattr(stream, "active", False))

    @property
    def output_device(self) -> Optional[int]:
        return self._output_device

    @property
    def callback_count(self) -> int:
        return self._callback_count

    @property
    def drop_count(self) -> int:
        return self._drop_count

    @property
    def underflow_count(self) -> int:
        return self._underflow_count

    def submit(self, audio: np.ndarray) -> bool:
        """Queue one copied post-effect block without blocking AudioStream."""
        if not self.is_running:
            return False
        block = np.asarray(audio, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.ndim != 2 or block.shape[1] != self._channels:
            logger.warning("self-monitor ignored unsupported shape {}", block.shape)
            return False
        block = block.copy()
        try:
            self._audio_queue.put_nowait(block)
            return True
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._drop_count += 1
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(block)
                return True
            except queue.Full:
                self._drop_count += 1
                return False

    def start(self, *, output_device: Optional[int]) -> None:
        """Open the independent headphone/output playback stream."""
        with self._lock:
            if self.is_running and self._output_device == output_device:
                return
            self._stop_locked()
            self._audio_queue = queue.Queue(maxsize=self._queue_blocks)
            self._callback_count = 0
            self._drop_count = 0
            self._underflow_count = 0
            self._pending_block = None
            self._pending_offset = 0

            def output_callback(outdata, frames, time_info, status) -> None:
                del time_info
                if status:
                    logger.warning("self-monitor output status: {}", status)
                self._callback_count += 1
                outdata.fill(0)
                requested = min(int(frames), outdata.shape[0])
                written = 0
                while written < requested:
                    block = self._pending_block
                    if block is None:
                        try:
                            block = self._audio_queue.get_nowait()
                        except queue.Empty:
                            self._underflow_count += 1
                            break
                        self._pending_block = block
                        self._pending_offset = 0

                    available = block.shape[0] - self._pending_offset
                    count = min(requested - written, available)
                    source = block[
                        self._pending_offset : self._pending_offset + count
                    ]
                    np.multiply(
                        source,
                        self._volume,
                        out=outdata[written : written + count],
                        casting="unsafe",
                    )
                    written += count
                    self._pending_offset += count
                    if self._pending_offset >= block.shape[0]:
                        self._pending_block = None
                        self._pending_offset = 0
                np.clip(outdata, -1.0, 1.0, out=outdata)

            stream = self._output_stream_factory(
                samplerate=self._samplerate,
                blocksize=self._blocksize,
                dtype=self._dtype,
                channels=self._channels,
                callback=output_callback,
                device=output_device,
                latency=self._latency,
            )
            try:
                stream.start()
            except Exception:
                try:
                    stream.close()
                except Exception as close_exc:
                    logger.error("self-monitor failed stream cleanup: {}", close_exc)
                raise

            self._stream = stream
            self._output_device = output_device
            logger.info(
                "self-monitor started: output={} volume={:.0%}",
                self._output_device,
                self._volume,
            )

    def stop(self) -> None:
        """Stop and close the monitor without touching the primary stream."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        stream = self._stream
        self._stream = None
        self._output_device = None
        self._audio_queue = queue.Queue(maxsize=self._queue_blocks)
        self._pending_block = None
        self._pending_offset = 0
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()
            logger.info("self-monitor stopped")

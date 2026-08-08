"""Stateful 48 kHz callback adapter for Beatrice's 16/24 kHz hop API."""

from __future__ import annotations

from collections import deque
import importlib
from time import perf_counter
from typing import Any, Callable

import numpy as np

from ai.beatrice.runtime import unpack_runtime_output


ConverterFactory = Callable[[], tuple[Any, Any, dict[str, Any]]]
ResamplerFactory = Callable[[int, int], Any]


class Float32FIFO:
    """Chunked FIFO without repeated whole-buffer concatenation."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self.size = 0

    def clear(self) -> None:
        self._chunks.clear()
        self._offset = 0
        self.size = 0

    def append(self, values: np.ndarray) -> None:
        chunk = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        if chunk.size:
            self._chunks.append(chunk)
            self.size += int(chunk.size)

    def pop(self, count: int) -> np.ndarray:
        if count < 0 or count > self.size:
            raise ValueError(f"Cannot pop {count} samples from FIFO depth {self.size}")
        output = np.empty(count, dtype=np.float32)
        written = 0
        while written < count:
            chunk = self._chunks[0]
            available = chunk.size - self._offset
            take = min(count - written, available)
            output[written:written + take] = chunk[self._offset:self._offset + take]
            written += take
            self._offset += take
            self.size -= take
            if self._offset == chunk.size:
                self._chunks.popleft()
                self._offset = 0
        return output


class BeatriceStreamingAdapter:
    """Own converter, both stateful soxr streams, FIFOs and startup state."""

    EXTERNAL_SAMPLE_RATE = 48_000
    INPUT_SAMPLE_RATE = 16_000
    OUTPUT_SAMPLE_RATE = 24_000
    INPUT_HOP = 160
    OUTPUT_HOP = 240

    def __init__(
        self,
        converter_factory: ConverterFactory,
        *,
        callback_samples: int = 256,
        startup_buffer_samples: int = 512,
        max_fifo_samples: int = 8192,
        resampler_quality: str = "QQ",
        resampler_factory: ResamplerFactory | None = None,
    ) -> None:
        if callback_samples <= 0:
            raise ValueError("callback_samples must be positive")
        if startup_buffer_samples < callback_samples:
            raise ValueError("startup buffer must cover at least one callback")
        if max_fifo_samples < startup_buffer_samples + callback_samples:
            raise ValueError("max_fifo_samples is too small")
        self.callback_samples = int(callback_samples)
        self.startup_buffer_samples = int(startup_buffer_samples)
        self.max_fifo_samples = int(max_fifo_samples)
        self.resampler_quality = str(resampler_quality)
        self._converter_factory = converter_factory
        self._resampler_factory = resampler_factory or self._default_resampler
        self._converter: Any | None = None
        self._module: Any | None = None
        self.runtime_details: dict[str, Any] = {}
        self.converter_generation = 0
        self._closed = False
        self._last_process_ms = 0.0
        self._load_converter()
        self._reset_stream_state()

    def _default_resampler(self, source_rate: int, target_rate: int):
        try:
            soxr = importlib.import_module("soxr")
        except ImportError as exc:
            raise RuntimeError(
                "python-soxr is required for Beatrice streaming / "
                "Beatrice 流式处理需要 python-soxr"
            ) from exc
        return soxr.ResampleStream(
            source_rate,
            target_rate,
            1,
            dtype="float32",
            quality=self.resampler_quality,
        )

    def _load_converter(self) -> None:
        converter, module, details = self._converter_factory()
        expected = {
            "IN_SAMPLE_RATE": self.INPUT_SAMPLE_RATE,
            "OUT_SAMPLE_RATE": self.OUTPUT_SAMPLE_RATE,
            "IN_HOP_LENGTH": self.INPUT_HOP,
            "OUT_HOP_LENGTH": self.OUTPUT_HOP,
        }
        mismatches = {
            name: (getattr(module, name, None), value)
            for name, value in expected.items()
            if getattr(module, name, None) != value
        }
        if mismatches:
            raise RuntimeError(f"Unexpected Beatrice streaming contract: {mismatches}")
        self._converter = converter
        self._module = module
        self.runtime_details = dict(details)
        self.converter_generation += 1

    def _reset_stream_state(self) -> None:
        self._input_resampler = self._resampler_factory(
            self.EXTERNAL_SAMPLE_RATE, self.INPUT_SAMPLE_RATE
        )
        self._output_resampler = self._resampler_factory(
            self.OUTPUT_SAMPLE_RATE, self.EXTERNAL_SAMPLE_RATE
        )
        self._input_fifo = Float32FIFO()
        self._output_fifo = Float32FIFO()
        self._started = False
        self._callback_count = 0
        self._convert_count = 0
        self._input_samples = 0
        self._emitted_samples = 0
        self._startup_silence_samples = 0
        self._underflow_count = 0
        self._overflow_count = 0
        self._input_fifo_max = 0
        self._output_fifo_max = 0

    def _append(self, fifo: Float32FIFO, values: np.ndarray) -> None:
        if fifo.size + int(values.size) > self.max_fifo_samples:
            self._overflow_count += 1
            raise BufferError(
                f"Beatrice streaming FIFO exceeds {self.max_fifo_samples} samples"
            )
        fifo.append(values)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self._closed or self._converter is None:
            raise RuntimeError("BeatriceStreamingAdapter is closed")
        values = np.asarray(audio)
        if values.ndim != 1:
            raise ValueError(f"Expected mono 1-D audio, got {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"Expected float32 audio, got {values.dtype}")
        if not np.isfinite(values).all():
            raise ValueError("Beatrice input contains NaN or Inf")
        if values.size == 0:
            return np.empty(0, dtype=np.float32)
        started = perf_counter()
        values = np.ascontiguousarray(values)
        self._callback_count += 1
        self._input_samples += int(values.size)
        downsampled = np.asarray(
            self._input_resampler.resample_chunk(values, last=False),
            dtype=np.float32,
        ).reshape(-1)
        self._append(self._input_fifo, downsampled)
        self._input_fifo_max = max(self._input_fifo_max, self._input_fifo.size)
        while self._input_fifo.size >= self.INPUT_HOP:
            native_input = self._input_fifo.pop(self.INPUT_HOP)
            native_output, _ = unpack_runtime_output(
                self._converter.convert(native_input)
            )
            if native_output.size != self.OUTPUT_HOP:
                raise RuntimeError(
                    f"Beatrice returned {native_output.size} samples; "
                    f"expected {self.OUTPUT_HOP}"
                )
            if not np.isfinite(native_output).all():
                raise RuntimeError("Beatrice output contains NaN or Inf")
            self._convert_count += 1
            upsampled = np.asarray(
                self._output_resampler.resample_chunk(native_output, last=False),
                dtype=np.float32,
            ).reshape(-1)
            self._append(self._output_fifo, upsampled)
            self._output_fifo_max = max(self._output_fifo_max, self._output_fifo.size)

        output = np.zeros(values.size, dtype=np.float32)
        required = max(self.startup_buffer_samples, int(values.size))
        if not self._started and self._output_fifo.size >= required:
            self._started = True
        if not self._started:
            self._startup_silence_samples += int(values.size)
        else:
            available = min(int(values.size), self._output_fifo.size)
            if available:
                output[:available] = self._output_fifo.pop(available)
                self._emitted_samples += available
            if available < values.size:
                self._underflow_count += 1
        self._input_fifo_max = max(self._input_fifo_max, self._input_fifo.size)
        self._output_fifo_max = max(self._output_fifo_max, self._output_fifo.size)
        self._last_process_ms = (perf_counter() - started) * 1000.0
        return output

    def update_config(self, **parameters: Any) -> None:
        if self._closed or self._converter is None:
            raise RuntimeError("BeatriceStreamingAdapter is closed")
        self._converter.set_config(**parameters)

    def reset(self) -> None:
        """Recreate native converter and resamplers; call only off callback."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed Beatrice adapter")
        self._converter = None
        self._module = None
        self._load_converter()
        self._reset_stream_state()

    def close(self) -> None:
        if self._closed:
            return
        self._input_fifo.clear()
        self._output_fifo.clear()
        self._input_resampler = None
        self._output_resampler = None
        self._converter = None
        self._module = None
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def ready(self) -> bool:
        return not self._closed and self._converter is not None and self._module is not None

    @property
    def last_process_ms(self) -> float:
        return self._last_process_ms

    def stats(self) -> dict[str, Any]:
        input_delay = (
            float(self._input_resampler.delay())
            if self._input_resampler is not None else 0.0
        )
        output_delay = (
            float(self._output_resampler.delay())
            if self._output_resampler is not None else 0.0
        )
        expected_down = self._input_samples / 3.0
        down_accounted = self._convert_count * self.INPUT_HOP + self._input_fifo.size + input_delay
        expected_up = self._convert_count * self.OUTPUT_HOP * 2.0
        up_accounted = self._emitted_samples + self._output_fifo.size + output_delay
        return {
            "closed": self._closed,
            "converter_generation": self.converter_generation,
            "callback_count": self._callback_count,
            "convert_count": self._convert_count,
            "startup_silence_samples": self._startup_silence_samples,
            "startup_silence_ms": self._startup_silence_samples / 48.0,
            "underflow_count": self._underflow_count,
            "overflow_count": self._overflow_count,
            "input_fifo_samples": self._input_fifo.size,
            "output_fifo_samples": self._output_fifo.size,
            "input_fifo_max": self._input_fifo_max,
            "output_fifo_max": self._output_fifo_max,
            "input_resampler_drift": down_accounted - expected_down,
            "output_resampler_drift": up_accounted - expected_up,
            "last_process_ms": self._last_process_ms,
        }


__all__ = ["BeatriceStreamingAdapter", "Float32FIFO"]

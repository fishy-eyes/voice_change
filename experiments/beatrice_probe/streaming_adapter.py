"""Isolated 48 kHz streaming adapter for the Beatrice v2 probe.

This module intentionally does not import any production voice_change module.
"""

from __future__ import annotations

from array import array
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import soxr

from runtime_support import create_converter, percentile, unpack_runtime_output


ConverterFactory = Callable[[], tuple[Any, Any, dict[str, Any]]]


class Float32FIFO:
    """Small chunked FIFO that avoids repeated whole-buffer concatenation."""

    def __init__(self) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._head_offset = 0
        self.size = 0

    def clear(self) -> None:
        self._chunks.clear()
        self._head_offset = 0
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
            head = self._chunks[0]
            available = head.size - self._head_offset
            take = min(count - written, available)
            output[written : written + take] = head[
                self._head_offset : self._head_offset + take
            ]
            written += take
            self._head_offset += take
            self.size -= take
            if self._head_offset == head.size:
                self._chunks.popleft()
                self._head_offset = 0
        return output


def _timing_summary(values: array) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    rows = np.frombuffer(values, dtype=np.float64)
    return {
        "count": int(rows.size),
        "total_ms": float(rows.sum()),
        "mean_ms": float(rows.mean()),
        "p50_ms": float(np.percentile(rows, 50)),
        "p95_ms": float(np.percentile(rows, 95)),
        "p99_ms": float(np.percentile(rows, 99)),
        "max_ms": float(rows.max()),
    }


class BeatriceStreamingAdapter:
    """Experimental stateful adapter, not a formal VoiceConversionEngine."""

    EXTERNAL_SAMPLE_RATE = 48_000
    DEFAULT_CALLBACK_SAMPLES = 256
    EXPECTED_INPUT_SAMPLE_RATE = 16_000
    EXPECTED_OUTPUT_SAMPLE_RATE = 24_000
    EXPECTED_INPUT_HOP = 160
    EXPECTED_OUTPUT_HOP = 240

    def __init__(
        self,
        model_path: str | Path | None = None,
        runtime_root: str | Path | None = None,
        *,
        target_speaker: int = 0,
        formant_shift: float = 0.0,
        pitch_shift_semitone: float = 0.0,
        min_source_pitch: float = 30.0,
        max_source_pitch: float = 1100.0,
        vq_num_neighbors: int = 4,
        converter_factory: ConverterFactory | None = None,
        external_sample_rate: int = EXTERNAL_SAMPLE_RATE,
        callback_samples: int = DEFAULT_CALLBACK_SAMPLES,
        resampler_quality: str = "QQ",
        startup_buffer_samples: int = 512,
        max_fifo_samples: int = 8192,
    ) -> None:
        if external_sample_rate != self.EXTERNAL_SAMPLE_RATE:
            raise ValueError("This probe intentionally models only the 48 kHz contract")
        if callback_samples <= 0:
            raise ValueError("callback_samples must be positive")
        if startup_buffer_samples < callback_samples:
            raise ValueError("startup_buffer_samples must cover at least one callback")
        if max_fifo_samples < startup_buffer_samples + callback_samples:
            raise ValueError("max_fifo_samples is too small for the startup policy")
        if converter_factory is None:
            if model_path is None:
                raise ValueError("model_path or converter_factory is required")

            def load_real_converter() -> tuple[Any, Any, dict[str, Any]]:
                converter, module, _, details = create_converter(
                    model_path,
                    runtime_root,
                    target_speaker=target_speaker,
                    formant_shift=formant_shift,
                    pitch_shift_semitone=pitch_shift_semitone,
                    min_source_pitch=min_source_pitch,
                    max_source_pitch=max_source_pitch,
                    vq_num_neighbors=vq_num_neighbors,
                )
                return converter, module, details

            converter_factory = load_real_converter

        self.external_sample_rate = int(external_sample_rate)
        self.callback_samples = int(callback_samples)
        self.resampler_quality = str(resampler_quality)
        self.startup_buffer_samples = int(startup_buffer_samples)
        self.max_fifo_samples = int(max_fifo_samples)
        self._converter_factory = converter_factory
        self._converter: Any | None = None
        self._module: Any | None = None
        self.runtime_details: dict[str, Any] = {}
        self.converter_generation = 0
        self._closed = False
        self._load_converter()
        self._reset_stream_state()

    def _load_converter(self) -> None:
        converter, module, details = self._converter_factory()
        constants = {
            "IN_SAMPLE_RATE": self.EXPECTED_INPUT_SAMPLE_RATE,
            "OUT_SAMPLE_RATE": self.EXPECTED_OUTPUT_SAMPLE_RATE,
            "IN_HOP_LENGTH": self.EXPECTED_INPUT_HOP,
            "OUT_HOP_LENGTH": self.EXPECTED_OUTPUT_HOP,
        }
        mismatches = {
            name: (int(getattr(module, name)), expected)
            for name, expected in constants.items()
            if int(getattr(module, name)) != expected
        }
        if mismatches:
            raise RuntimeError(f"Unexpected Beatrice streaming contract: {mismatches}")
        self._converter = converter
        self._module = module
        self.runtime_details = dict(details)
        self.converter_generation += 1

    def _reset_stream_state(self) -> None:
        self._input_resampler = soxr.ResampleStream(
            self.external_sample_rate,
            self.EXPECTED_INPUT_SAMPLE_RATE,
            1,
            dtype="float32",
            quality=self.resampler_quality,
        )
        self._output_resampler = soxr.ResampleStream(
            self.EXPECTED_OUTPUT_SAMPLE_RATE,
            self.external_sample_rate,
            1,
            dtype="float32",
            quality=self.resampler_quality,
        )
        self._input_fifo = Float32FIFO()
        self._output_fifo = Float32FIFO()
        self._started = False
        self._callback_count = 0
        self._convert_count = 0
        self._callbacks_with_inference = 0
        self._input_samples = 0
        self._input_resampled_samples = 0
        self._model_output_samples = 0
        self._output_resampled_samples = 0
        self._converted_output_emitted = 0
        self._inserted_silence_samples = 0
        self._startup_padding_samples = 0
        self._underflow_count = 0
        self._overflow_count = 0
        self._dropped_samples = 0
        self._input_fifo_max = 0
        self._output_fifo_max = 0
        self._first_valid_callback: int | None = None
        self._first_valid_after_input_samples: int | None = None
        self._max_input_resampler_delay = 0.0
        self._max_output_resampler_delay = 0.0
        self._deadline_miss_count = 0
        self._process_times_ms = array("d")
        self._inference_times_ms = array("d")
        self._input_resampler_times_ms = array("d")
        self._output_resampler_times_ms = array("d")
        self._steady_output_depths = array("I")

    def _append_with_limit(self, fifo: Float32FIFO, values: np.ndarray) -> None:
        if fifo.size + int(values.size) > self.max_fifo_samples:
            self._overflow_count += 1
            raise BufferError(
                f"Streaming FIFO would exceed {self.max_fifo_samples} samples"
            )
        fifo.append(values)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self._closed or self._converter is None or self._module is None:
            raise RuntimeError("BeatriceStreamingAdapter is closed")
        values = np.asarray(audio)
        if values.ndim != 1:
            raise ValueError(f"Expected mono 1-D audio, got shape {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"Expected float32 audio, got {values.dtype}")
        if values.size == 0:
            return np.empty(0, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Input contains NaN or Inf")
        values = np.ascontiguousarray(values)

        process_started = perf_counter()
        self._callback_count += 1
        self._input_samples += int(values.size)

        stage_started = perf_counter()
        resampled_input = np.asarray(
            self._input_resampler.resample_chunk(values, last=False),
            dtype=np.float32,
        ).reshape(-1)
        self._input_resampler_times_ms.append(
            (perf_counter() - stage_started) * 1000.0
        )
        self._input_resampled_samples += int(resampled_input.size)
        self._append_with_limit(self._input_fifo, resampled_input)
        self._input_fifo_max = max(self._input_fifo_max, self._input_fifo.size)

        converts_this_callback = 0
        while self._input_fifo.size >= self.EXPECTED_INPUT_HOP:
            native_input = self._input_fifo.pop(self.EXPECTED_INPUT_HOP)
            inference_started = perf_counter()
            native_output, _ = unpack_runtime_output(
                self._converter.convert(native_input)
            )
            self._inference_times_ms.append(
                (perf_counter() - inference_started) * 1000.0
            )
            if native_output.size != self.EXPECTED_OUTPUT_HOP:
                raise RuntimeError(
                    f"Runtime returned {native_output.size} samples; "
                    f"expected {self.EXPECTED_OUTPUT_HOP}"
                )
            if not np.isfinite(native_output).all():
                raise RuntimeError("Runtime output contains NaN or Inf")
            self._convert_count += 1
            converts_this_callback += 1
            self._model_output_samples += int(native_output.size)

            stage_started = perf_counter()
            resampled_output = np.asarray(
                self._output_resampler.resample_chunk(native_output, last=False),
                dtype=np.float32,
            ).reshape(-1)
            self._output_resampler_times_ms.append(
                (perf_counter() - stage_started) * 1000.0
            )
            self._output_resampled_samples += int(resampled_output.size)
            self._append_with_limit(self._output_fifo, resampled_output)
            self._output_fifo_max = max(self._output_fifo_max, self._output_fifo.size)

        if converts_this_callback:
            self._callbacks_with_inference += 1

        output = np.zeros(values.size, dtype=np.float32)
        required_start_depth = max(self.startup_buffer_samples, int(values.size))
        if not self._started:
            if self._output_fifo.size >= required_start_depth:
                self._started = True
                self._first_valid_callback = self._callback_count
                self._first_valid_after_input_samples = self._input_samples
            else:
                self._startup_padding_samples += int(values.size)
                self._inserted_silence_samples += int(values.size)

        if self._started:
            available = min(int(values.size), self._output_fifo.size)
            if available:
                output[:available] = self._output_fifo.pop(available)
                self._converted_output_emitted += available
            if available < values.size:
                self._underflow_count += 1
                self._inserted_silence_samples += int(values.size) - available
            self._steady_output_depths.append(self._output_fifo.size)

        self._input_fifo_max = max(self._input_fifo_max, self._input_fifo.size)
        self._output_fifo_max = max(self._output_fifo_max, self._output_fifo.size)
        self._max_input_resampler_delay = max(
            self._max_input_resampler_delay, float(self._input_resampler.delay())
        )
        self._max_output_resampler_delay = max(
            self._max_output_resampler_delay, float(self._output_resampler.delay())
        )

        elapsed_ms = (perf_counter() - process_started) * 1000.0
        self._process_times_ms.append(elapsed_ms)
        deadline_ms = 1000.0 * values.size / self.external_sample_rate
        if elapsed_ms > deadline_ms:
            self._deadline_miss_count += 1
        return output

    def reset(self) -> None:
        """Reset streaming state by recreating the converter and resamplers."""

        if self._closed:
            raise RuntimeError("Cannot reset a closed BeatriceStreamingAdapter")
        self._converter = None
        self._module = None
        self._load_converter()
        self._reset_stream_state()

    def close(self) -> None:
        """Release references and buffers; repeated close is safe."""

        if self._closed:
            return
        self._max_input_resampler_delay = max(
            self._max_input_resampler_delay, float(self._input_resampler.delay())
        )
        self._max_output_resampler_delay = max(
            self._max_output_resampler_delay, float(self._output_resampler.delay())
        )
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

    def stats(self) -> dict[str, Any]:
        input_delay = (
            float(self._input_resampler.delay())
            if self._input_resampler is not None
            else self._max_input_resampler_delay
        )
        output_delay = (
            float(self._output_resampler.delay())
            if self._output_resampler is not None
            else self._max_output_resampler_delay
        )
        steady_depths = np.frombuffer(self._steady_output_depths, dtype=np.uint32)
        steady_depth_p50 = (
            float(np.percentile(steady_depths, 50)) if steady_depths.size else None
        )
        expected_downsampled = (
            self._input_samples
            * self.EXPECTED_INPUT_SAMPLE_RATE
            / self.external_sample_rate
        )
        accounted_downsampled = self._input_resampled_samples + input_delay
        expected_upsampled = (
            self._model_output_samples
            * self.external_sample_rate
            / self.EXPECTED_OUTPUT_SAMPLE_RATE
        )
        accounted_upsampled = self._output_resampled_samples + output_delay
        callback_ms = 1000.0 * self.callback_samples / self.external_sample_rate
        model_frame_ms = (
            1000.0
            * self.EXPECTED_INPUT_HOP
            / self.EXPECTED_INPUT_SAMPLE_RATE
        )
        return {
            "closed": self._closed,
            "converter_generation": self.converter_generation,
            "contract": {
                "external_sample_rate": self.external_sample_rate,
                "callback_samples": self.callback_samples,
                "callback_ms": callback_ms,
                "beatrice_input_sample_rate": self.EXPECTED_INPUT_SAMPLE_RATE,
                "beatrice_input_hop": self.EXPECTED_INPUT_HOP,
                "beatrice_output_sample_rate": self.EXPECTED_OUTPUT_SAMPLE_RATE,
                "beatrice_output_hop": self.EXPECTED_OUTPUT_HOP,
            },
            "resampler": {
                "library": "python-soxr",
                "version": soxr.__version__,
                "quality": self.resampler_quality,
                "stateful": True,
                "input_current_delay_samples_at_16khz": input_delay,
                "input_max_delay_samples_at_16khz": self._max_input_resampler_delay,
                "input_max_delay_ms": 1000.0
                * self._max_input_resampler_delay
                / self.EXPECTED_INPUT_SAMPLE_RATE,
                "output_current_delay_samples_at_48khz": output_delay,
                "output_max_delay_samples_at_48khz": self._max_output_resampler_delay,
                "output_max_delay_ms": 1000.0
                * self._max_output_resampler_delay
                / self.external_sample_rate,
                "input_timing": _timing_summary(self._input_resampler_times_ms),
                "output_timing": _timing_summary(self._output_resampler_times_ms),
            },
            "buffer": {
                "input_fifo_current_samples_at_16khz": self._input_fifo.size,
                "input_fifo_max_samples_at_16khz": self._input_fifo_max,
                "output_fifo_current_samples_at_48khz": self._output_fifo.size,
                "output_fifo_max_samples_at_48khz": self._output_fifo_max,
                "output_fifo_steady_p50_samples_at_48khz": steady_depth_p50,
                "steady_state_buffer_latency_ms": (
                    1000.0 * steady_depth_p50 / self.external_sample_rate
                    if steady_depth_p50 is not None
                    else None
                ),
                "underflow_count_after_start": self._underflow_count,
                "overflow_count": self._overflow_count,
                "dropped_samples": self._dropped_samples,
                "inserted_silence_samples": self._inserted_silence_samples,
                "startup_padding_samples": self._startup_padding_samples,
                "startup_padding_ms": 1000.0
                * self._startup_padding_samples
                / self.external_sample_rate,
            },
            "work": {
                "input_callbacks": self._callback_count,
                "input_samples_at_48khz": self._input_samples,
                "callbacks_with_inference": self._callbacks_with_inference,
                "beatrice_convert_count": self._convert_count,
                "input_resampled_samples_at_16khz": self._input_resampled_samples,
                "model_output_samples_at_24khz": self._model_output_samples,
                "output_resampled_samples_at_48khz": self._output_resampled_samples,
                "converted_output_emitted_samples": self._converted_output_emitted,
            },
            "timing": {
                "process": _timing_summary(self._process_times_ms),
                "inference": _timing_summary(self._inference_times_ms),
                "deadline_ms_for_configured_callback": callback_ms,
                "deadline_miss_count": self._deadline_miss_count,
            },
            "latency": {
                "callback_block_ms": callback_ms,
                "beatrice_frame_ms": model_frame_ms,
                "first_model_frame_input_available_ms": (
                    np.ceil(
                        self.EXPECTED_INPUT_HOP
                        * self.external_sample_rate
                        / self.EXPECTED_INPUT_SAMPLE_RATE
                        / self.callback_samples
                    )
                    * callback_ms
                ),
                "first_valid_callback": self._first_valid_callback,
                "first_valid_output_offset_ms": 1000.0
                * self._startup_padding_samples
                / self.external_sample_rate,
                "first_output_latency_ms": (
                    1000.0
                    * self._first_valid_after_input_samples
                    / self.external_sample_rate
                    if self._first_valid_after_input_samples is not None
                    else None
                ),
                "steady_state_buffer_latency_ms": (
                    1000.0 * steady_depth_p50 / self.external_sample_rate
                    if steady_depth_p50 is not None
                    else None
                ),
            },
            "sample_accounting": {
                "expected_downsampled_samples": expected_downsampled,
                "accounted_downsampled_samples_including_delay": accounted_downsampled,
                "input_resampler_sample_drift": accounted_downsampled
                - expected_downsampled,
                "expected_upsampled_samples": expected_upsampled,
                "accounted_upsampled_samples_including_delay": accounted_upsampled,
                "output_resampler_sample_drift": accounted_upsampled
                - expected_upsampled,
                "output_resampler_time_drift_ms": 1000.0
                * (accounted_upsampled - expected_upsampled)
                / self.external_sample_rate,
                "external_input_minus_callback_output_samples": 0,
            },
        }


__all__ = ["BeatriceStreamingAdapter", "Float32FIFO"]

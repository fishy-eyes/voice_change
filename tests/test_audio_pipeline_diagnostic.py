"""Capture and compare the non-RVC audio path without changing production code.

Run from the project root:

    python -u tests\\test_audio_pipeline_diagnostic.py

The default run records ten seconds through the same duplex PortAudio settings
used by :class:`audio.stream.AudioStream`.  Every diagnostic stage is copied
inside one callback, so sample-for-sample comparisons are meaningful.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import threading
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import sounddevice as sd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.device_manager import DeviceManager  # noqa: E402
from audio.recorder import AudioRecorder  # noqa: E402
from config.settings import (  # noqa: E402
    BLOCKSIZE,
    CHANNELS,
    DTYPE,
    INPUT_DEVICE,
    LATENCY,
    SAMPLE_RATE,
)
from main import create_effect_manager  # noqa: E402


DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_OUTPUT_DIR = Path("tests/output/audio_pipeline_diagnostic")
SILENCE_THRESHOLD = 1.0e-4
CLIPPING_THRESHOLD = 0.999

STAGE_FILES = {
    "input_raw": "diagnostic_input_raw.wav",
    "stream_processed": "diagnostic_stream_processed.wav",
    "effect_output": "diagnostic_effect_output.wav",
    "self_monitor_pre": "diagnostic_self_monitor_pre.wav",
}


@dataclass
class _MonitorTap:
    """Capture exactly what OutputRoutingEffectManager submits to monitoring."""

    last_block: np.ndarray | None = None

    def submit(self, audio: np.ndarray) -> bool:
        self.last_block = np.asarray(audio).copy()
        return True


@dataclass
class _CaptureState:
    target_samples: int
    stages: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {name: [] for name in STAGE_FILES}
    )
    callback_frames: list[int] = field(default_factory=list)
    status_counts: Counter[str] = field(default_factory=Counter)
    samples: int = 0
    callback_count: int = 0
    exception: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def append(self, name: str, block: np.ndarray, count: int) -> None:
        array = np.asarray(block)
        self.stages[name].append(array[:count].copy())


def _device_index(value: int | None, kind: str) -> int | None:
    if value is not None:
        return int(value)
    defaults = sd.default.device
    selected = defaults[0 if kind == "input" else 1]
    try:
        return int(selected) if int(selected) >= 0 else None
    except (TypeError, ValueError):
        return None


def _device_record(index: int | None, kind: str) -> dict[str, Any]:
    if index is None:
        return {
            "index": None,
            "name": "unavailable",
            "default_sample_rate": None,
            "max_channels": 0,
        }
    info = sd.query_devices(index, kind)
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    return {
        "index": index,
        "name": str(info["name"]),
        "hostapi": int(info["hostapi"]),
        "default_sample_rate": float(info["default_samplerate"]),
        "max_channels": int(info[channel_key]),
        "default_low_latency": float(info[f"default_low_{kind}_latency"]),
        "default_high_latency": float(info[f"default_high_{kind}_latency"]),
    }


def _status_key(status: Any) -> str:
    text = str(status).strip()
    return text if text else "none"


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Return stable numeric diagnostics for a mono or multichannel array."""
    array = np.asarray(audio)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got {array.shape}")
    flat = array.reshape(-1)
    finite = np.isfinite(flat)
    finite_values = flat[finite].astype(np.float64, copy=False)
    if finite_values.size:
        rms = float(np.sqrt(np.mean(np.square(finite_values))))
        peak = float(np.max(np.abs(finite_values)))
        clipping_ratio = float(
            np.mean(np.abs(finite_values) >= CLIPPING_THRESHOLD)
        )
        silence_ratio = float(
            np.mean(np.abs(finite_values) <= SILENCE_THRESHOLD)
        )
    else:
        rms = peak = clipping_ratio = silence_ratio = 0.0
    result: dict[str, Any] = {
        "sample_rate": int(sample_rate),
        "channels": int(array.shape[1]),
        "dtype": str(array.dtype),
        "sample_count": int(array.shape[0]),
        "scalar_count": int(flat.size),
        "duration_seconds": float(array.shape[0] / sample_rate),
        "rms": rms,
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "silence_threshold": SILENCE_THRESHOLD,
        "nonfinite_count": int(flat.size - np.count_nonzero(finite)),
    }
    if file_path is not None:
        result["file"] = str(file_path.resolve())
        result["file_size_bytes"] = int(file_path.stat().st_size)
    return result


def compare_audio(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compare stages without hiding length, shape, or non-finite errors."""
    left = np.asarray(reference)
    right = np.asarray(candidate)
    same_shape = left.shape == right.shape
    result: dict[str, Any] = {
        "same_shape": same_shape,
        "reference_shape": list(left.shape),
        "candidate_shape": list(right.shape),
        "sample_count_difference": int(right.shape[0] - left.shape[0]),
        "exactly_equal": bool(same_shape and np.array_equal(left, right)),
    }
    if not same_shape or left.size == 0:
        result.update({"max_abs_difference": None, "rmse": None})
        return result
    difference = right.astype(np.float64) - left.astype(np.float64)
    result.update(
        {
            "max_abs_difference": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        }
    )
    return result


def _disabled_chain(manager: Any) -> list[dict[str, Any]]:
    return [
        {"name": effect.name, "enabled": bool(effect.enabled)}
        for effect in manager.effects
    ]


def _open_stream(
    *,
    input_device: int | None,
    output_device: int | None,
    callback: Any,
) -> tuple[sd.Stream, bool, int]:
    """Open a stream with AudioStream's configured-rate fallback semantics."""
    recorder = AudioRecorder(device=input_device)
    params = recorder.get_stream_params()
    requested_rate = int(params["samplerate"])
    common = dict(
        samplerate=requested_rate,
        blocksize=int(params["blocksize"]),
        dtype=str(params["dtype"]),
        channels=int(params["channels"]),
        callback=callback,
        device=(input_device, output_device),
        latency=(str(params["latency"]), str(params["latency"])),
    )
    try:
        return sd.Stream(**common), False, requested_rate
    except sd.PortAudioError:
        fallback_device = input_device if input_device is not None else output_device
        if fallback_device is None:
            raise
        fallback_rate = int(round(sd.query_devices(fallback_device)["default_samplerate"]))
        if fallback_rate == requested_rate:
            raise
        common["samplerate"] = fallback_rate
        return sd.Stream(**common), True, fallback_rate


def _open_input_stream(
    *,
    input_device: int | None,
    resolved_input_device: int | None,
    callback: Any,
) -> tuple[sd.InputStream, bool, int]:
    """Fallback capture when the selected input/output host APIs cannot pair."""
    recorder = AudioRecorder(device=input_device)
    params = recorder.get_stream_params()
    requested_rate = int(params["samplerate"])
    common = dict(
        samplerate=requested_rate,
        blocksize=int(params["blocksize"]),
        dtype=str(params["dtype"]),
        channels=int(params["channels"]),
        callback=callback,
        device=input_device,
        latency=str(params["latency"]),
    )
    try:
        return sd.InputStream(**common), False, requested_rate
    except sd.PortAudioError:
        if resolved_input_device is None:
            raise
        fallback_rate = int(
            round(sd.query_devices(resolved_input_device)["default_samplerate"])
        )
        if fallback_rate == requested_rate:
            raise
        common["samplerate"] = fallback_rate
        return sd.InputStream(**common), True, fallback_rate


def _build_findings(report: dict[str, Any]) -> tuple[list[str], str]:
    findings: list[str] = []
    stream = report["stream"]
    devices = report["devices"]
    requested = int(stream["requested_sample_rate"])
    actual = int(stream["actual_sample_rate"])
    input_default = devices["input"]["default_sample_rate"]

    if input_default is not None and int(round(input_default)) != requested:
        findings.append(
            f"Input device default is {input_default:.0f} Hz while the application "
            f"requests {requested} Hz; host/driver sample-rate conversion may occur."
        )
    if actual != requested:
        findings.append(
            f"The duplex stream opened at {actual} Hz instead of {requested} Hz."
        )
    if stream["mode"] != "duplex":
        findings.append(
            "The configured duplex input/output pair could not be opened; this run "
            "used input-only capture and did not exercise VB-CABLE playback. "
            f"PortAudio error: {stream['duplex_open_error']}"
        )
    if devices["input"]["max_channels"] != CHANNELS:
        findings.append(
            f"Input device exposes {devices['input']['max_channels']} channels; "
            f"the application opens {CHANNELS} channel. PortAudio supplies the "
            "selected mono channel; the Python chain performs no downmix."
        )
    if stream["status_counts"] != {"none": stream["callback_count"]}:
        findings.append(f"PortAudio reported callback status flags: {stream['status_counts']}")

    comparison_names = (
        "input_raw_to_stream_processed",
        "input_raw_to_effect_output",
        "stream_processed_to_self_monitor_pre",
    )
    changed = [
        name
        for name in comparison_names
        if not report["comparisons"][name]["exactly_equal"]
    ]
    if changed:
        findings.append("Sample differences were detected in: " + ", ".join(changed))

    for name, values in report["stages"].items():
        if values["nonfinite_count"]:
            findings.append(f"{name} contains non-finite samples.")
        if values["clipping_ratio"] > 0.0:
            findings.append(
                f"{name} clipping ratio is {values['clipping_ratio']:.6%}."
            )
        if values["rms"] <= SILENCE_THRESHOLD:
            findings.append(f"{name} is effectively silent.")

    if not changed:
        conclusion = (
            "The captured samples are identical across raw callback input, the "
            "disabled EffectManager, AudioStream-style output, and the pre-monitor "
            "tap. No quality loss was introduced inside the Python callback/effect "
            "path during this run. Remaining suspects are device/driver conversion, "
            "VB-CABLE format settings, or monitor/playback hardware."
        )
        if stream["mode"] != "duplex":
            conclusion += (
                " The selected duplex device path failed before capture, so its "
                "configuration is itself a priority suspect and must be retested "
                "with a compatible --input-device/--output-device pair."
            )
    else:
        conclusion = (
            "The first non-identical comparison identifies the earliest software "
            "stage where samples changed; inspect that comparison and its metrics."
        )
    return findings, conclusion


def run_diagnostic(
    *,
    duration: float = DEFAULT_DURATION_SECONDS,
    input_device: int | None = INPUT_DEVICE,
    output_device: int | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Record all stages from one real duplex stream and write WAV/JSON output."""
    if duration <= 0:
        raise ValueError("duration must be positive")

    if output_device is None:
        output_device = DeviceManager.find_virtual_output_device()
    resolved_input = _device_index(input_device, "input")
    resolved_output = _device_index(output_device, "output")
    input_info = _device_record(resolved_input, "input")
    output_info = _device_record(resolved_output, "output")

    plain_manager = create_effect_manager()
    monitor_tap = _MonitorTap()
    routed_manager = create_effect_manager(self_monitor=monitor_tap)

    state_holder: dict[str, _CaptureState | None] = {"state": None}

    def callback(indata, outdata, frames, time_info, status) -> None:
        state = state_holder["state"]
        if state is None:
            outdata.fill(0)
            return
        try:
            raw = np.asarray(indata)
            effect_output = plain_manager.process(
                raw.copy(), frames, time_info, status
            )
            monitor_tap.last_block = None
            stream_processed = routed_manager.process(
                raw.copy(), frames, time_info, status
            )
            monitor_pre = monitor_tap.last_block
            if monitor_pre is None:
                raise RuntimeError("self-monitor pre-output tap received no block")
            if np.asarray(stream_processed).shape != outdata.shape:
                raise ValueError(
                    "processed output shape mismatch: "
                    f"{np.asarray(stream_processed).shape} != {outdata.shape}"
                )
            outdata[:] = stream_processed

            remaining = state.target_samples - state.samples
            count = min(int(frames), remaining)
            if count > 0:
                state.append("input_raw", raw, count)
                state.append("stream_processed", stream_processed, count)
                state.append("effect_output", effect_output, count)
                state.append("self_monitor_pre", monitor_pre, count)
                state.samples += count
            state.callback_count += 1
            state.callback_frames.append(int(frames))
            state.status_counts[_status_key(status)] += 1
            if state.samples >= state.target_samples:
                state.done.set()
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise
        except BaseException as exc:
            state.exception = exc
            state.done.set()
            outdata.fill(0)
            raise sd.CallbackAbort from exc

    stream_mode = "duplex"
    duplex_open_error: str | None = None
    try:
        stream, fallback_used, opened_rate = _open_stream(
            input_device=input_device,
            output_device=output_device,
            callback=callback,
        )
    except sd.PortAudioError as exc:
        duplex_open_error = str(exc)
        stream_mode = "input_only_fallback"

        def input_callback(indata, frames, time_info, status) -> None:
            discarded_output = np.empty_like(indata)
            callback(indata, discarded_output, frames, time_info, status)

        stream, fallback_used, opened_rate = _open_input_stream(
            input_device=input_device,
            resolved_input_device=resolved_input,
            callback=input_callback,
        )
    actual_rate = int(round(stream.samplerate))
    target_samples = int(round(duration * actual_rate))
    state = _CaptureState(target_samples=target_samples)
    state_holder["state"] = state

    started_at = perf_counter()
    try:
        with stream:
            if not state.done.wait(timeout=duration + 10.0):
                raise TimeoutError("audio diagnostic capture timed out")
    finally:
        elapsed = perf_counter() - started_at
    if state.exception is not None:
        raise RuntimeError("audio callback failed") from state.exception
    if state.samples != target_samples:
        raise RuntimeError(
            f"captured {state.samples} samples, expected {target_samples}"
        )

    captured = {
        name: np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
        for name, blocks in state.stages.items()
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_reports: dict[str, dict[str, Any]] = {}
    for name, audio in captured.items():
        path = output_dir / STAGE_FILES[name]
        sf.write(path, audio, actual_rate, subtype="FLOAT")
        file_info = sf.info(path)
        if file_info.samplerate != actual_rate or file_info.channels != audio.shape[1]:
            raise RuntimeError(f"written WAV verification failed: {path}")
        stage_reports[name] = analyze_audio(audio, actual_rate, file_path=path)
        stage_reports[name]["wav_subtype"] = file_info.subtype

    frame_counts = Counter(state.callback_frames)
    report: dict[str, Any] = {
        "diagnostic_version": 1,
        "scope": (
            "PortAudio callback input through disabled base effects and the "
            "pre-self-monitor routing tap; AIVoiceEffect is not present."
        ),
        "devices": {
            "input": input_info,
            "output": output_info,
        },
        "stream": {
            "mode": stream_mode,
            "duplex_open_error": duplex_open_error,
            "output_was_exercised": stream_mode == "duplex",
            "requested_sample_rate": SAMPLE_RATE,
            "opened_sample_rate": opened_rate,
            "actual_sample_rate": actual_rate,
            "fallback_used": fallback_used,
            "requested_channels": CHANNELS,
            "actual_channels": int(captured["input_raw"].shape[1]),
            "dtype": DTYPE,
            "requested_block_size": BLOCKSIZE,
            "callback_count": state.callback_count,
            "callback_frame_sizes": {
                str(size): count for size, count in sorted(frame_counts.items())
            },
            "status_counts": dict(state.status_counts),
            "requested_duration_seconds": duration,
            "wall_time_seconds": elapsed,
        },
        "effect_chain": {
            "ai_effect_present": False,
            "plain_manager": _disabled_chain(plain_manager),
            "routed_manager": _disabled_chain(routed_manager),
            "all_effects_disabled": all(
                not effect.enabled
                for manager in (plain_manager, routed_manager)
                for effect in manager.effects
            ),
        },
        "stages": stage_reports,
        "comparisons": {
            "input_raw_to_stream_processed": compare_audio(
                captured["input_raw"], captured["stream_processed"]
            ),
            "input_raw_to_effect_output": compare_audio(
                captured["input_raw"], captured["effect_output"]
            ),
            "stream_processed_to_self_monitor_pre": compare_audio(
                captured["stream_processed"], captured["self_monitor_pre"]
            ),
        },
    }
    findings, conclusion = _build_findings(report)
    report["findings"] = findings
    report["conclusion"] = conclusion
    report_path = output_dir / "diagnostic_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["report_file"] = str(report_path.resolve())
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("\nAudio pipeline diagnostic complete")
    print("Input :", report["devices"]["input"]["name"])
    print("Output:", report["devices"]["output"]["name"])
    stream = report["stream"]
    print(
        "Rate  : requested={} Hz, device-default={} Hz, actual={} Hz".format(
            stream["requested_sample_rate"],
            report["devices"]["input"]["default_sample_rate"],
            stream["actual_sample_rate"],
        )
    )
    print("\nStages:")
    for name, values in report["stages"].items():
        print(
            "  {:22s} rms={:.6f} peak={:.6f} clip={:.6%} silence={:.2%}".format(
                name,
                values["rms"],
                values["peak"],
                values["clipping_ratio"],
                values["silence_ratio"],
            )
        )
        print("    " + values["file"])
    print("\nComparisons:")
    for name, values in report["comparisons"].items():
        print(
            f"  {name}: exact={values['exactly_equal']} "
            f"max_abs={values['max_abs_difference']} rmse={values['rmse']}"
        )
    if report["findings"]:
        print("\nFindings:")
        for finding in report["findings"]:
            print("  -", finding)
    print("\nConclusion:", report["conclusion"])
    print("Report:", report["report_file"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and compare the non-RVC real-time audio stages."
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--input-device", type=int, default=INPUT_DEVICE)
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Output device index; default auto-detects VB-CABLE, then system default.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input/output devices and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_devices:
        DeviceManager.print_devices()
        return 0
    report = run_diagnostic(
        duration=args.duration,
        input_device=args.input_device,
        output_device=args.output_device,
        output_dir=args.output_dir,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

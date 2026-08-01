"""Run one isolated full-file RVC backend diagnostic.

This script intentionally bypasses the realtime worker/chunk path. It accepts
explicit backend files, converts one source recording to mono float32 at the
project sample rate, runs one complete offline inference, and writes objective
validation metrics next to the listening WAV.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def log(message: str) -> None:
    print(message, flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resampled_frames(resampler, frame) -> list:
    result = resampler.resample(frame)
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def _qt_buffer_to_float(buffer) -> tuple[np.ndarray, int, int]:
    from PySide6.QtMultimedia import QAudioFormat

    audio_format = buffer.format()
    raw = bytes(buffer.constData())
    sample_format = audio_format.sampleFormat()
    if sample_format == QAudioFormat.SampleFormat.Float:
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    elif sample_format == QAudioFormat.SampleFormat.Int16:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_format == QAudioFormat.SampleFormat.Int32:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_format == QAudioFormat.SampleFormat.UInt8:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported Qt audio sample format: {sample_format}")
    channels = int(audio_format.channelCount())
    require(channels > 0, "Qt decoder returned no channels")
    usable = samples.size - samples.size % channels
    return samples[:usable].reshape(-1, channels), int(audio_format.sampleRate()), channels


def _load_audio_qt(path: Path, sample_rate: int) -> tuple[np.ndarray, dict]:
    """Decode through QtMultimedia's bundled FFmpeg backend."""
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer, QUrl
    from PySide6.QtMultimedia import QAudioDecoder

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    decoder = QAudioDecoder()
    loop = QEventLoop()
    chunks: list[np.ndarray] = []
    source_rate = 0
    source_channels = 0
    errors: list[str] = []
    timed_out = False

    def read_available() -> None:
        nonlocal source_rate, source_channels
        while decoder.bufferAvailable():
            buffer = decoder.read()
            if not buffer.isValid():
                continue
            samples, rate, channels = _qt_buffer_to_float(buffer)
            if source_rate and (rate != source_rate or channels != source_channels):
                errors.append("Qt decoder changed format mid-stream")
                loop.quit()
                return
            source_rate, source_channels = rate, channels
            chunks.append(samples)

    def finished() -> None:
        read_available()
        loop.quit()

    def decode_error(error) -> None:
        if error != QAudioDecoder.Error.NoError:
            errors.append(decoder.errorString() or str(error))
            loop.quit()

    def timeout() -> None:
        nonlocal timed_out
        timed_out = True
        decoder.stop()
        loop.quit()

    decoder.bufferReady.connect(read_available)
    decoder.finished.connect(finished)
    decoder.error.connect(decode_error)
    decoder.setSource(QUrl.fromLocalFile(str(path)))
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(timeout)
    timer.start(60_000)
    decoder.start()
    loop.exec()
    timer.stop()
    if timed_out:
        raise TimeoutError(f"Qt audio decode timed out: {path}")
    if errors:
        raise RuntimeError(f"Qt audio decode failed: {'; '.join(errors)}")
    require(bool(chunks), f"Qt decoder produced no samples: {path}")

    decoded = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    decoded_frames = int(decoded.shape[0])
    mono = decoded.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        import librosa

        mono = librosa.resample(
            mono,
            orig_sr=source_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    metadata = {
        "decoder": "PySide6.QtMultimedia.QAudioDecoder (FFmpeg backend)",
        "container_format": path.suffix.lower().lstrip("."),
        "codec": None,
        "original_sample_rate": source_rate,
        "original_channels": source_channels,
        "original_duration_seconds": decoded_frames / source_rate,
        "decoded_original_frames": decoded_frames,
    }
    return mono, metadata


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, dict]:
    """Decode with existing RVC dependencies, using Qt as the installed fallback."""
    require(path.is_file(), f"input not found: {path}")
    try:
        return _load_audio_av(path, sample_rate)
    except ModuleNotFoundError as exc:
        if exc.name != "av":
            raise
        audio, metadata = _load_audio_qt(path, sample_rate)
        return _finalize_audio(path, audio, sample_rate, metadata)


def _load_audio_av(path: Path, sample_rate: int) -> tuple[np.ndarray, dict]:
    import av

    container = av.open(str(path), mode="r")
    try:
        require(bool(container.streams.audio), f"no audio stream: {path}")
        stream = container.streams.audio[0]
        codec = stream.codec_context
        original_rate = int(codec.sample_rate or stream.rate or 0)
        channels = int(codec.channels or 0)
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)

        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=sample_rate,
        )
        chunks: list[np.ndarray] = []
        decoded_frames = 0
        for frame in container.decode(stream):
            decoded_frames += int(frame.samples)
            for converted in _resampled_frames(resampler, frame):
                chunks.append(
                    np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
                )
        for converted in _resampled_frames(resampler, None):
            chunks.append(
                np.asarray(converted.to_ndarray(), dtype=np.float32).reshape(-1)
            )
    finally:
        container.close()

    require(bool(chunks), f"PyAV decoder produced no samples: {path}")
    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    metadata = {
        "decoder": "av",
        "container_format": container.format.name if container.format else None,
        "codec": codec.name,
        "original_sample_rate": original_rate,
        "original_channels": channels,
        "original_duration_seconds": duration,
        "decoded_original_frames": decoded_frames,
    }
    return _finalize_audio(path, audio, sample_rate, metadata)


def _finalize_audio(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    metadata: dict,
) -> tuple[np.ndarray, dict]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    require(audio.ndim == 1 and audio.size > 0, "converted input is not mono")
    require(bool(np.all(np.isfinite(audio))), "converted input contains NaN/Inf")
    input_rms = float(np.sqrt(np.mean(audio * audio)))
    require(input_rms > 1e-6, "converted input is silent")
    metadata.update({
        "path": str(path.resolve()),
        "converted_shape": list(audio.shape),
        "converted_dtype": str(audio.dtype),
        "converted_sample_rate": sample_rate,
        "converted_duration_seconds": audio.size / sample_rate,
        "converted_rms": input_rms,
        "converted_peak": float(np.max(np.abs(audio))),
    })
    return audio, metadata

def silent_window_metrics(
    source: np.ndarray,
    result: np.ndarray,
    sample_rate: int,
) -> dict:
    window = max(1, int(round(0.1 * sample_rate)))
    input_silent = 0
    output_silent = 0
    unexpected = 0
    windows = 0
    for start in range(0, source.size, window):
        src = source[start : start + window]
        out = result[start : start + window]
        src_rms = float(np.sqrt(np.mean(src * src)))
        out_rms = float(np.sqrt(np.mean(out * out)))
        src_is_silent = src_rms <= 1e-6
        out_is_silent = out_rms <= 1e-6
        input_silent += int(src_is_silent)
        output_silent += int(out_is_silent)
        unexpected += int(out_is_silent and src_rms > 1e-4)
        windows += 1
    return {
        "window_seconds": window / sample_rate,
        "window_count": windows,
        "input_silent_windows": input_silent,
        "output_silent_windows": output_silent,
        "unexpected_output_silent_windows": unexpected,
    }


def build_parser() -> argparse.ArgumentParser:
    from config.settings import (
        RVC_F0_METHOD,
        RVC_INDEX_RATE,
        RVC_MODELS_DIR,
        RVC_PITCH_SHIFT,
        RVC_PROTECT,
        RVC_RMS_MIX_RATE,
        RVC_SOURCE_DIR,
        RVC_VOICE_DIR,
        SAMPLE_RATE,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path, default=Path(RVC_VOICE_DIR))
    parser.add_argument("--source-dir", type=Path, default=Path(RVC_SOURCE_DIR))
    parser.add_argument("--models-dir", type=Path, default=Path(RVC_MODELS_DIR))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hubert-path", type=Path)
    parser.add_argument("--rmvpe-path", type=Path)
    parser.add_argument(
        "--f0-method",
        choices=("pm", "rmvpe", "fcpe"),
        default=RVC_F0_METHOD,
        help="Methods accepted by the checked-out external Pipeline.",
    )
    parser.add_argument("--pitch", type=int, default=RVC_PITCH_SHIFT)
    parser.add_argument("--index-rate", type=float, default=RVC_INDEX_RATE)
    parser.add_argument("--rms-mix-rate", type=float, default=RVC_RMS_MIX_RATE)
    parser.add_argument("--protect", type=float, default=RVC_PROTECT)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    import soundfile as sf
    import torch
    from loguru import logger

    from ai.rvc_engine import RVCEngine

    require(0.0 <= args.index_rate <= 1.0, "index-rate must be between 0 and 1")
    require(args.sample_rate > 0, "sample-rate must be positive")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    audio, input_metadata = load_audio(args.input.resolve(), args.sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hubert_path = (
        args.hubert_path.resolve()
        if args.hubert_path is not None
        else (args.models_dir / "hubert").resolve()
    )
    rmvpe_path = (
        args.rmvpe_path.resolve()
        if args.rmvpe_path is not None
        else (args.models_dir / "rmvpe" / "rmvpe.pt").resolve()
    )
    voice_dir = args.voice_dir.resolve()
    pth_files = sorted(voice_dir.glob("*.pth"))
    index_files = sorted(voice_dir.glob("*.index"))
    require(bool(pth_files), f"no .pth model in {voice_dir}")

    log(json.dumps(input_metadata, ensure_ascii=False, indent=2))
    log(f"voice_pth={pth_files[0]}")
    log(f"voice_index={index_files[0] if index_files else '(none)'}")
    log(f"hubert_path={hubert_path}")
    log(f"rmvpe_path={rmvpe_path}")

    engine = RVCEngine(
        voice_dir=voice_dir,
        source_dir=args.source_dir.resolve(),
        models_dir=args.models_dir.resolve(),
        hubert_path=hubert_path,
        rmvpe_path=rmvpe_path,
        pitch_shift=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        sample_rate=args.sample_rate,
    )
    load_seconds = 0.0
    inference_seconds = 0.0
    captured_stderr = ""
    output = None
    backend_loaded = False
    try:
        started = time.perf_counter()
        engine.load_model()
        load_seconds = time.perf_counter() - started

        infer_stderr = io.StringIO()
        started = time.perf_counter()
        with redirect_stderr(infer_stderr):
            output = engine.infer(audio)
        inference_seconds = time.perf_counter() - started
        captured_stderr = infer_stderr.getvalue()
        if captured_stderr:
            sys.stderr.write(captured_stderr)
            sys.stderr.flush()
        backend_loaded = (
            hasattr(engine._pipeline, "model_rmvpe")
            if args.f0_method == "rmvpe"
            else True
        )

        output = np.asarray(output)
        require(output.shape == audio.shape, f"shape mismatch: {output.shape}")
        require(output.dtype == np.float32, f"dtype mismatch: {output.dtype}")
        require(bool(np.all(np.isfinite(output))), "output contains NaN/Inf")
        require(not np.array_equal(output, audio), "suspected source passthrough")
        output_rms = float(np.sqrt(np.mean(output * output)))
        require(output_rms > 1e-6, "output is silent")
        silence = silent_window_metrics(audio, output, args.sample_rate)
        require(
            silence["unexpected_output_silent_windows"] == 0,
            "output contains unexpected silent 100 ms windows",
        )
        require(backend_loaded, "declared RMVPE backend was not loaded")

        sf.write(str(args.output), output, args.sample_rate, subtype="PCM_16")
        reread, reread_rate = sf.read(str(args.output), dtype="float32", always_2d=True)
        require(reread_rate == args.sample_rate, "written WAV sample rate mismatch")
        require(reread.shape == (audio.size, 1), f"written WAV shape: {reread.shape}")
        require(bool(np.all(np.isfinite(reread))), "written WAV contains NaN/Inf")

        report = {
            "success": True,
            "input": input_metadata,
            "output": {
                "path": str(args.output.resolve()),
                "size_bytes": args.output.stat().st_size,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "sample_rate": args.sample_rate,
                "channels": 1,
                "duration_seconds": output.size / args.sample_rate,
                "rms": output_rms,
                "peak": float(np.max(np.abs(output))),
                "clipped_fraction": float(np.mean(np.abs(output) >= 0.999)),
                "finite": True,
                "silent": False,
                "exact_passthrough": False,
                "silent_windows": silence,
            },
            "backend": {
                "voice_dir": str(voice_dir),
                "voice_pth": str(pth_files[0].resolve()),
                "voice_pth_sha256": sha256(pth_files[0]),
                "voice_index": str(index_files[0].resolve()) if index_files else None,
                "hubert_path": str(hubert_path),
                "hubert_weights_sha256": sha256(hubert_path / "pytorch_model.bin"),
                "rmvpe_path": str(rmvpe_path),
                "rmvpe_sha256": sha256(rmvpe_path),
                "f0_method": args.f0_method,
                "pitch": args.pitch,
                "index_rate": args.index_rate,
                "rms_mix_rate": args.rms_mix_rate,
                "protect": args.protect,
                "rvc_version": engine._version,
                "if_f0": engine._if_f0,
                "target_sample_rate": engine._tgt_sr,
                "device": engine._device,
                "half": engine._is_half,
                "declared_backend_loaded": backend_loaded,
            },
            "timing": {
                "model_load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "rtf": inference_seconds / (audio.size / args.sample_rate),
            },
            "diagnostics": {
                "traceback_detected": "Traceback (most recent call last)" in captured_stderr,
                "captured_stderr": captured_stderr,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            },
        }
        require(not report["diagnostics"]["traceback_detected"], "inference traceback detected")
        return report
    finally:
        if engine.is_loaded:
            engine.unload_model()


def main() -> int:
    args = build_parser().parse_args()
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    try:
        report = run(args)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(json.dumps(report, ensure_ascii=False, indent=2))
        log(f"report={report_path.resolve()}")
        return 0
    except Exception as exc:
        traceback.print_exc()
        failure = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "output": str(args.output.resolve()),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

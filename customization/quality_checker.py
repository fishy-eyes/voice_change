"""Traditional signal checks for a customization recording."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from customization.schemas import RecordingQualityResult


@dataclass(frozen=True)
class AudioFeatures:
    duration_seconds: float
    rms: float
    peak: float
    clipping_ratio: float
    dynamic_range_db: float
    voiced_frame_ratio: float
    silence_ratio: float
    f0_values: np.ndarray
    pitch_discontinuity_ratio: float
    background_noise_ratio: float


def normalize_mono(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if values.ndim == 2:
        values = np.mean(values, axis=1)
    if values.ndim != 1:
        raise ValueError("audio must be mono or samples-by-channels")
    if np.issubdtype(values.dtype, np.integer):
        scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        values = values.astype(np.float32) / scale
    else:
        values = values.astype(np.float32, copy=False)
    return values


def _frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(audio) < frame_size:
        return np.empty((0, frame_size), dtype=np.float32)
    count = 1 + (len(audio) - frame_size) // hop_size
    shape = (count, frame_size)
    strides = (audio.strides[0] * hop_size, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()


def _estimate_pitch(frame: np.ndarray, sample_rate: int) -> float | None:
    centered = frame.astype(np.float64) - float(np.mean(frame))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-9:
        return None
    windowed = centered * np.hanning(len(centered))
    fft_size = 1 << (2 * len(windowed) - 1).bit_length()
    spectrum = np.fft.rfft(windowed, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)
    autocorrelation = autocorrelation[: len(windowed)]
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(len(autocorrelation) - 1, int(sample_rate / 50.0))
    if max_lag <= min_lag or autocorrelation[0] <= 0:
        return None
    local = autocorrelation[min_lag : max_lag + 1]
    lag = min_lag + int(np.argmax(local))
    confidence = float(autocorrelation[lag] / autocorrelation[0])
    if confidence < 0.30:
        return None
    return float(sample_rate / lag)


def extract_audio_features(audio: np.ndarray, sample_rate: int) -> AudioFeatures:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = normalize_mono(audio)
    if values.size == 0:
        return AudioFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.array([]), 1.0, 1.0)
    if not np.all(np.isfinite(values)):
        raise ValueError("audio contains NaN or Inf")

    duration = len(values) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    peak = float(np.max(np.abs(values)))
    clipping_ratio = float(np.mean(np.abs(values) >= 0.995))

    frame_size = max(64, int(round(sample_rate * 0.040)))
    hop_size = max(32, int(round(sample_rate * 0.020)))
    frames = _frame_audio(values, frame_size, hop_size)
    if len(frames) == 0:
        frame_rms = np.array([rms], dtype=np.float64)
        frames = values.reshape(1, -1)
    else:
        frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    noise_floor = float(np.percentile(frame_rms, 20))
    typical_loud = float(np.percentile(frame_rms, 80))
    voice_threshold = max(0.003, min(noise_floor * 2.5, typical_loud * 0.5))
    voiced_mask = frame_rms >= voice_threshold
    voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0
    silence_ratio = 1.0 - voiced_ratio

    nonzero = frame_rms[frame_rms > 1e-7]
    if len(nonzero) >= 2:
        low = max(float(np.percentile(nonzero, 10)), 1e-7)
        high = max(float(np.percentile(nonzero, 95)), low)
        dynamic_range_db = float(20.0 * np.log10(high / low))
    else:
        dynamic_range_db = 0.0

    pitches: list[float] = []
    for frame, is_voiced in zip(frames, voiced_mask):
        if is_voiced:
            pitch = _estimate_pitch(frame, sample_rate)
            if pitch is not None:
                pitches.append(pitch)
    f0_values = np.asarray(pitches, dtype=np.float64)
    if len(f0_values) >= 2:
        semitone_steps = np.abs(12.0 * np.log2(f0_values[1:] / f0_values[:-1]))
        discontinuity = float(np.mean(semitone_steps > 7.0))
    else:
        discontinuity = 1.0 if voiced_ratio > 0.1 else 0.0

    voiced_rms = float(np.mean(frame_rms[voiced_mask])) if np.any(voiced_mask) else 0.0
    quiet_rms = float(np.mean(frame_rms[~voiced_mask])) if np.any(~voiced_mask) else 0.0
    noise_ratio = quiet_rms / max(voiced_rms, 1e-8)
    return AudioFeatures(
        duration_seconds=duration,
        rms=rms,
        peak=peak,
        clipping_ratio=clipping_ratio,
        dynamic_range_db=dynamic_range_db,
        voiced_frame_ratio=voiced_ratio,
        silence_ratio=silence_ratio,
        f0_values=f0_values,
        pitch_discontinuity_ratio=discontinuity,
        background_noise_ratio=float(noise_ratio),
    )


class RecordingQualityChecker:
    """Apply conservative, explainable acceptance rules."""

    def __init__(self, *, minimum_duration: float = 5.0) -> None:
        self.minimum_duration = float(minimum_duration)

    def check(self, audio: np.ndarray, sample_rate: int) -> RecordingQualityResult:
        reasons: list[str] = []
        try:
            features = extract_audio_features(audio, sample_rate)
        except ValueError as exc:
            reason = str(exc)
            return RecordingQualityResult(
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, False, 0.0,
                None, None, None, 1.0, 1.0, 0, False, (reason,),
            )

        if features.duration_seconds < self.minimum_duration:
            reasons.append("录音过短")
        if features.voiced_frame_ratio < 0.10:
            reasons.append("未检测到有效语音")
        elif features.silence_ratio > 0.70:
            reasons.append("静音过多")
        if features.rms < 0.008:
            reasons.append("音量过小")
        if features.clipping_ratio > 0.01:
            reasons.append("严重削波")
        if len(features.f0_values) == 0:
            reasons.append("未检测到有效基频")
        if features.background_noise_ratio > 0.45 and features.voiced_frame_ratio > 0.10:
            reasons.append("背景噪声疑似过高")

        score = 100
        score -= 35 if features.duration_seconds < self.minimum_duration else 0
        score -= int(min(35.0, features.silence_ratio * 30.0))
        score -= int(min(40.0, features.clipping_ratio * 1000.0))
        score -= 25 if features.rms < 0.008 else 0
        score -= 20 if len(features.f0_values) == 0 else 0
        score -= int(min(20.0, features.pitch_discontinuity_ratio * 25.0))
        score = max(0, min(100, score))
        f0 = features.f0_values
        return RecordingQualityResult(
            duration_seconds=features.duration_seconds,
            effective_voice_seconds=features.duration_seconds * features.voiced_frame_ratio,
            silence_ratio=features.silence_ratio,
            rms=features.rms,
            peak=features.peak,
            clipping_ratio=features.clipping_ratio,
            dynamic_range_db=features.dynamic_range_db,
            has_valid_pitch=bool(len(f0)),
            voiced_frame_ratio=features.voiced_frame_ratio,
            f0_median=float(np.median(f0)) if len(f0) else None,
            f0_p10=float(np.percentile(f0, 10)) if len(f0) else None,
            f0_p90=float(np.percentile(f0, 90)) if len(f0) else None,
            pitch_discontinuity_ratio=features.pitch_discontinuity_ratio,
            background_noise_ratio=features.background_noise_ratio,
            quality_score=score,
            is_acceptable=not reasons,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )

"""Reject technically damaged candidates without claiming voice similarity."""

from __future__ import annotations

import numpy as np

from customization.quality_checker import extract_audio_features, normalize_mono
from customization.schemas import CandidateEvaluation, RawCandidateSafetyEvaluation


RAW_CANDIDATE_CLIPPING_RATIO_LIMIT = 0.002


def _high_frequency_ratio(audio: np.ndarray, sample_rate: int) -> float:
    values = normalize_mono(audio).astype(np.float64)
    if len(values) < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
    frequencies = np.fft.rfftfreq(len(values), 1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(spectrum[frequencies >= sample_rate * 0.38]) / total)


def _amplitude_discontinuity_ratio(audio: np.ndarray, sample_rate: int) -> float:
    values = np.abs(normalize_mono(audio).astype(np.float64))
    frame_size = max(1, int(sample_rate * 0.02))
    count = len(values) // frame_size
    if count < 2:
        return 0.0
    rms = np.sqrt(
        np.mean(np.square(values[: count * frame_size]).reshape(count, frame_size), axis=1)
    )
    ratios = np.maximum(rms[1:], 1e-7) / np.maximum(rms[:-1], 1e-7)
    return float(np.mean((ratios > 8.0) | (ratios < 0.125)))


class CandidateEvaluator:
    def evaluate(
        self,
        original: np.ndarray,
        candidate: np.ndarray,
        sample_rate: int,
    ) -> CandidateEvaluation:
        reasons: list[str] = []
        source = normalize_mono(original)
        output = normalize_mono(candidate)
        if output.size == 0:
            return CandidateEvaluation(0, 0, 0, 0, False, ("输出为空",), duration_ratio=0.0)
        if not np.all(np.isfinite(output)):
            return CandidateEvaluation(0, 0, 0, 0, False, ("输出包含 NaN 或 Inf",))

        duration_ratio = len(output) / max(1, len(source))
        if not 0.75 <= duration_ratio <= 1.25:
            reasons.append("输出时长异常")

        features = extract_audio_features(output, sample_rate)
        source_rms = float(np.sqrt(np.mean(np.square(source, dtype=np.float64)))) if len(source) else 0.0
        volume_ratio = features.rms / max(source_rms, 1e-8)
        if features.clipping_ratio > 0.02:
            reasons.append("削波严重")
        if features.silence_ratio > 0.88:
            reasons.append("输出静音过多")
        if features.rms < 0.002 or not 0.08 <= volume_ratio <= 8.0:
            reasons.append("输出音量异常")

        high_frequency_ratio = _high_frequency_ratio(output, sample_rate)
        if high_frequency_ratio > 0.40:
            reasons.append("高频异常能量过多")
        discontinuity_ratio = _amplitude_discontinuity_ratio(output, sample_rate)
        if discontinuity_ratio > 0.30:
            reasons.append("输出存在大量不连续帧")

        volume_score = int(max(0.0, 100.0 - min(100.0, abs(np.log2(max(volume_ratio, 1e-6))) * 24.0)))
        pitch_score = int(max(0.0, 100.0 - features.pitch_discontinuity_ratio * 100.0))
        stability_score = int(
            max(0.0, 100.0 - discontinuity_ratio * 120.0 - high_frequency_ratio * 80.0)
        )
        technical = int(
            max(
                0.0,
                min(
                    100.0,
                    # Do not reward candidates merely for being louder.
                    # Volume remains an extreme-value rejection guard.
                    (pitch_score + stability_score) / 2.0
                    - features.clipping_ratio * 500.0,
                ),
            )
        )
        return CandidateEvaluation(
            technical_quality=technical,
            stability_score=stability_score,
            volume_score=volume_score,
            pitch_continuity_score=pitch_score,
            is_valid=not reasons,
            rejection_reasons=tuple(reasons),
            clipping_ratio=features.clipping_ratio,
            silence_ratio=features.silence_ratio,
            duration_ratio=float(duration_ratio),
            high_frequency_ratio=high_frequency_ratio,
            discontinuity_ratio=discontinuity_ratio,
        )


class RawCandidateSafetyEvaluator:
    """Reject unsafe inference output before audition gain or peak limiting."""

    def __init__(
        self,
        *,
        clipping_ratio_limit: float = RAW_CANDIDATE_CLIPPING_RATIO_LIMIT,
        technical_evaluator: CandidateEvaluator | None = None,
    ) -> None:
        self.clipping_ratio_limit = float(clipping_ratio_limit)
        self.technical_evaluator = technical_evaluator or CandidateEvaluator()

    def evaluate(
        self,
        original: np.ndarray,
        candidate: np.ndarray,
        sample_rate: int,
    ) -> RawCandidateSafetyEvaluation:
        source = normalize_mono(original)
        output = normalize_mono(candidate)
        duration_ratio = len(output) / max(1, len(source))
        nan_count = int(np.isnan(output).sum())
        inf_count = int(np.isinf(output).sum())
        finite = np.isfinite(output)
        finite_output = output[finite]
        peak = float(np.max(np.abs(finite_output))) if finite_output.size else 0.0
        rms = (
            float(np.sqrt(np.mean(np.square(finite_output, dtype=np.float64))))
            if finite_output.size
            else 0.0
        )
        clipping_ratio = (
            float(np.mean(np.abs(finite_output) >= 0.995))
            if finite_output.size
            else 0.0
        )
        reasons: list[str] = []
        technical: CandidateEvaluation | None = None
        silence_ratio = 1.0

        if output.size == 0:
            reasons.append("raw output is empty")
        if nan_count or inf_count:
            reasons.append("raw output contains NaN or Inf")
        if output.size and not 0.75 <= duration_ratio <= 1.25:
            reasons.append("raw output length is abnormal")
        if output.size and finite.all():
            technical = self.technical_evaluator.evaluate(source, output, sample_rate)
            silence_ratio = technical.silence_ratio
            reasons.extend(technical.rejection_reasons)
        if rms < 0.002:
            reasons.append("raw output is silent or near-silent")
        if clipping_ratio > self.clipping_ratio_limit:
            reasons.append(
                "raw clipping ratio exceeds "
                f"{self.clipping_ratio_limit:.3%} safety limit"
            )

        unique_reasons = tuple(dict.fromkeys(reasons))
        return RawCandidateSafetyEvaluation(
            is_safe=not unique_reasons,
            rejection_reasons=unique_reasons,
            peak=peak,
            rms=rms,
            clipping_ratio=clipping_ratio,
            nan_count=nan_count,
            inf_count=inf_count,
            duration_ratio=float(duration_ratio),
            silence_ratio=float(silence_ratio),
            would_clip_on_pcm_output=peak > 1.0,
            technical_evaluation=technical,
        )

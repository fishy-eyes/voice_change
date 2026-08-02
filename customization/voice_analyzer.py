"""User voice analysis based on the same deterministic signal features."""

from __future__ import annotations

import numpy as np

from customization.quality_checker import extract_audio_features
from customization.schemas import VoiceAnalysisResult


class VoiceAnalyzer:
    def analyze(self, audio: np.ndarray, sample_rate: int) -> VoiceAnalysisResult:
        features = extract_audio_features(audio, sample_rate)
        f0 = features.f0_values
        return VoiceAnalysisResult(
            duration_seconds=features.duration_seconds,
            rms_mean=features.rms,
            peak=features.peak,
            clipping_ratio=features.clipping_ratio,
            voiced_frame_ratio=features.voiced_frame_ratio,
            f0_median=float(np.median(f0)) if len(f0) else None,
            f0_p10=float(np.percentile(f0, 10)) if len(f0) else None,
            f0_p90=float(np.percentile(f0, 90)) if len(f0) else None,
            pitch_discontinuity_ratio=features.pitch_discontinuity_ratio,
            dynamic_range_db=features.dynamic_range_db,
        )

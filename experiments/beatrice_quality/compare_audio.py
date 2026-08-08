"""Lag-aligned waveform, spectrum and periodic-boundary diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from experiments.beatrice_quality.common import read_audio, resample_audio, write_json


ANALYSIS_RATE = 48_000


def align_by_lag(reference: np.ndarray, candidate: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, int]:
    ref = np.asarray(reference, dtype=np.float64)
    test = np.asarray(candidate, dtype=np.float64)
    correlation = signal.correlate(test - np.mean(test), ref - np.mean(ref), mode="full", method="fft")
    lags = signal.correlation_lags(len(test), len(ref), mode="full")
    allowed = np.abs(lags) <= max_lag
    lag = int(lags[allowed][np.argmax(correlation[allowed])])
    if lag >= 0:
        length = min(len(ref), len(test) - lag)
        return ref[:length], test[lag : lag + length], lag
    length = min(len(ref) + lag, len(test))
    return ref[-lag : -lag + length], test[:length], lag


def spectral_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    values = np.asarray(audio, dtype=np.float64)
    if not len(values):
        return {}
    windowed = (values - np.mean(values)) * np.hanning(len(values))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    total = max(float(np.sum(power)), 1e-20)
    def band(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < high)
        return float(np.sum(power[mask]) / total)
    cumulative = np.cumsum(power)
    rolloff_index = min(int(np.searchsorted(cumulative, total * 0.85)), len(frequencies) - 1)
    return {
        "energy_ratio_0_4khz": band(0, 4_000),
        "energy_ratio_4_8khz": band(4_000, 8_000),
        "energy_ratio_8_12khz": band(8_000, 12_000),
        "high_frequency_ratio_4_24khz": band(4_000, sample_rate / 2 + 1),
        "spectral_centroid_hz": float(np.sum(frequencies * power) / total),
        "spectral_rolloff_85_hz": float(frequencies[rolloff_index]),
    }


def periodic_metrics(audio: np.ndarray, sample_rate: int, period_samples: int) -> dict[str, float]:
    values = np.asarray(audio, dtype=np.float64)
    differences = np.abs(np.diff(values))
    boundaries = np.arange(period_samples, len(values), period_samples)
    boundary_jumps = np.abs(values[boundaries] - values[boundaries - 1]) if len(boundaries) else np.empty(0)
    neighborhood = max(1, round(sample_rate * 0.001))
    local_rms = []
    for boundary in boundaries:
        start = max(0, boundary - neighborhood)
        stop = min(len(values), boundary + neighborhood)
        local_rms.append(float(np.sqrt(np.mean(values[start:stop] ** 2))))
    global_jump_rms = float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0
    boundary_jump_rms = float(np.sqrt(np.mean(boundary_jumps ** 2))) if len(boundary_jumps) else 0.0
    return {
        "period_samples": int(period_samples),
        "period_ms": 1000.0 * period_samples / sample_rate,
        "boundary_count": int(len(boundaries)),
        "boundary_jump_rms": boundary_jump_rms,
        "all_sample_derivative_rms": global_jump_rms,
        "boundary_to_global_derivative_ratio": boundary_jump_rms / max(global_jump_rms, 1e-20),
        "boundary_neighborhood_rms": float(np.mean(local_rms)) if local_rms else 0.0,
    }


def compare_arrays(reference: np.ndarray, reference_rate: int, candidate: np.ndarray, candidate_rate: int) -> dict[str, Any]:
    ref = resample_audio(reference, reference_rate, ANALYSIS_RATE)
    test = resample_audio(candidate, candidate_rate, ANALYSIS_RATE)
    aligned_ref, aligned_test, lag = align_by_lag(ref, test, ANALYSIS_RATE)
    difference = aligned_test - aligned_ref
    denominator = float(np.linalg.norm(aligned_ref) * np.linalg.norm(aligned_test))
    correlation = float(np.dot(aligned_ref, aligned_test) / denominator) if denominator else 0.0
    ref_spectrum = spectral_metrics(aligned_ref, ANALYSIS_RATE)
    test_spectrum = spectral_metrics(aligned_test, ANALYSIS_RATE)
    return {
        "analysis_sample_rate": ANALYSIS_RATE,
        "best_lag_samples": lag,
        "best_lag_ms": 1000.0 * lag / ANALYSIS_RATE,
        "aligned_samples": int(len(difference)),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "mae": float(np.mean(np.abs(difference))),
        "correlation": correlation,
        "peak_difference": float(np.max(np.abs(aligned_test)) - np.max(np.abs(aligned_ref))),
        "rms_difference": float(np.sqrt(np.mean(aligned_test ** 2)) - np.sqrt(np.mean(aligned_ref ** 2))),
        "reference_spectrum": ref_spectrum,
        "candidate_spectrum": test_spectrum,
        "spectral_delta": {key: test_spectrum[key] - ref_spectrum[key] for key in ref_spectrum},
    }


def periodic_report(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    values = resample_audio(audio, sample_rate, ANALYSIS_RATE)
    return {
        "callback_5_333ms": periodic_metrics(values, ANALYSIS_RATE, 256),
        "native_10ms": periodic_metrics(values, ANALYSIS_RATE, 480),
        "native_20ms": periodic_metrics(values, ANALYSIS_RATE, 960),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    reference, reference_rate, _ = read_audio(Path(args.reference))
    candidate, candidate_rate, _ = read_audio(Path(args.candidate))
    report = compare_arrays(reference, reference_rate, candidate, candidate_rate)
    if args.json_out:
        write_json(Path(args.json_out), report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

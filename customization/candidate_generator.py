"""Offline candidate generation using an already loaded, paused RVC engine."""

from __future__ import annotations

import string
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from customization.candidate_evaluator import CandidateEvaluator
from customization.schemas import CandidateResult, RVCParameterSet


ProgressCallback = Callable[[int, int, CandidateResult], None]


def match_audition_level(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Match overall RMS fairly while preserving each file's internal dynamics."""
    source = np.asarray(reference, dtype=np.float32).reshape(-1)
    output = np.asarray(candidate, dtype=np.float32).reshape(-1)
    if not output.size or not np.all(np.isfinite(output)):
        return output.copy()
    source_rms = (
        float(np.sqrt(np.mean(np.square(source, dtype=np.float64))))
        if source.size
        else 0.0
    )
    output_rms = float(np.sqrt(np.mean(np.square(output, dtype=np.float64))))
    if source_rms <= 1e-8 or output_rms <= 1e-8:
        return output.copy()
    gain = float(np.clip(source_rms / output_rms, 0.25, 4.0))
    matched = output * gain
    peak = float(np.max(np.abs(matched)))
    if peak > 0.98:
        matched *= 0.98 / peak
    return matched.astype(np.float32, copy=False)


class CandidateGenerator:
    """Generate files synchronously; callers put this object in a worker thread."""

    def __init__(
        self,
        engine,
        output_directory: str | Path,
        *,
        sample_rate: int = 48000,
        evaluator: CandidateEvaluator | None = None,
    ) -> None:
        self.engine = engine
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.evaluator = evaluator or CandidateEvaluator()

    def generate(
        self,
        audio: np.ndarray,
        parameters: Iterable[RVCParameterSet],
        *,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[CandidateResult]:
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        options = tuple(parameters)
        original_config = self.engine.config
        results: list[CandidateResult] = []
        logger.info("开始候选搜索: candidates={}", len(options))
        try:
            for index, config in enumerate(options):
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("用户取消流程: 已停止后续候选生成")
                    break
                label = "方案 " + (
                    string.ascii_uppercase[index]
                    if index < len(string.ascii_uppercase)
                    else str(index + 1)
                )
                logger.info("候选参数: id={} {}", index, config)
                started = time.perf_counter()
                try:
                    self.engine.update_config(**config.to_engine_changes())
                    converted = np.asarray(self.engine.infer(source.copy()), dtype=np.float32).reshape(-1)
                    inference_ms = (time.perf_counter() - started) * 1000.0
                    evaluation = self.evaluator.evaluate(source, converted, self.sample_rate)
                    destination = self.output_directory / f"candidate_{index + 1:02d}.wav"
                    if converted.size and np.all(np.isfinite(converted)):
                        audition_audio = match_audition_level(source, converted)
                        sf.write(destination, audition_audio, self.sample_rate, subtype="PCM_16")
                        audio_path: str | None = str(destination)
                    else:
                        audio_path = None
                    result = CandidateResult(
                        candidate_id=f"candidate-{index + 1}",
                        label=label,
                        parameters=config,
                        audio_path=audio_path,
                        inference_ms=inference_ms,
                        evaluation=evaluation,
                    )
                    logger.info(
                        "候选推理耗时: id={} ms={:.1f}; 候选评估结果: valid={} score={}",
                        index,
                        inference_ms,
                        evaluation.is_valid,
                        evaluation.technical_quality,
                    )
                    if not evaluation.is_valid:
                        logger.info("候选被淘汰原因: {}", evaluation.rejection_reasons)
                except Exception as exc:
                    inference_ms = (time.perf_counter() - started) * 1000.0
                    result = CandidateResult(
                        candidate_id=f"candidate-{index + 1}",
                        label=label,
                        parameters=config,
                        audio_path=None,
                        inference_ms=inference_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    logger.error("候选推理失败: {}", result.error)
                results.append(result)
                if progress is not None:
                    progress(index + 1, len(options), result)
        finally:
            self.engine.update_config(original_config)
        return results

"""Deterministic layered RVC parameter search state."""

from __future__ import annotations

from dataclasses import replace

from loguru import logger

from customization.schemas import RVCParameterSet, SearchRound


PITCH_COARSE_VALUES = (-12, -8, -4, 0, 4, 8, 12)
PITCH_FINE_OFFSETS = (-2, 0, 2)
INDEX_RATE_VALUES = (0.35, 0.60, 0.80)

# The checked RVC implementation blends more original unvoiced features as
# protect approaches zero.  GUI semantics therefore intentionally run in the
# opposite direction to the numeric value.
PROTECT_OPTIONS = (
    ("辅音保护强", 0.00),
    ("平衡", 0.33),
    ("目标音色优先", 0.50),
)
RMS_MIX_OPTIONS = (
    ("保留原声动态", 0.15),
    ("平衡", 0.50),
    ("输出更稳定", 0.85),
)


def pitch_coarse_round(base: RVCParameterSet) -> SearchRound:
    return SearchRound(
        stage="pitch_coarse",
        candidates=[replace(base, pitch_shift=value) for value in PITCH_COARSE_VALUES],
    )


def pitch_fine_round(center: RVCParameterSet) -> SearchRound:
    values = tuple(
        max(-24, min(24, center.pitch_shift + offset))
        for offset in PITCH_FINE_OFFSETS
    )
    return SearchRound(
        stage="pitch_fine",
        candidates=[replace(center, pitch_shift=value) for value in dict.fromkeys(values)],
    )


def index_rate_round(base: RVCParameterSet, *, has_index: bool) -> SearchRound | None:
    if not has_index:
        return None
    return SearchRound(
        stage="index_rate",
        candidates=[replace(base, index_rate=value) for value in INDEX_RATE_VALUES],
    )


def protect_round(base: RVCParameterSet) -> SearchRound:
    return SearchRound(
        stage="protect",
        candidates=[replace(base, protect=value) for _label, value in PROTECT_OPTIONS],
    )


def rms_mix_round(base: RVCParameterSet) -> SearchRound:
    return SearchRound(
        stage="rms_mix_rate",
        candidates=[replace(base, rms_mix_rate=value) for _label, value in RMS_MIX_OPTIONS],
    )


class ParameterSearch:
    """Small state machine; it never performs inference itself."""

    def __init__(self, *, has_index: bool, base: RVCParameterSet | None = None) -> None:
        self.has_index = bool(has_index)
        initial = base or RVCParameterSet(index_rate=0.60 if has_index else 0.0)
        if not has_index and initial.index_rate != 0.0:
            initial = replace(initial, index_rate=0.0)
        self.current = pitch_coarse_round(initial)
        self.history: list[SearchRound] = []
        self.final_parameters: RVCParameterSet | None = None
        self.cancelled = False
        logger.info("开始候选搜索: stage={}", self.current.stage)

    def cancel(self) -> None:
        self.current.cancel()
        self.cancelled = True
        logger.info("用户取消流程")

    def choose(self, index: int) -> SearchRound | None:
        if self.cancelled:
            raise RuntimeError("parameter search is cancelled")
        selected = self.current.select(index)
        completed = self.current
        self.history.append(completed)
        logger.info("用户选择结果: stage={} candidate={}", completed.stage, index)

        if completed.stage == "pitch_coarse":
            next_round = pitch_fine_round(selected)
        elif completed.stage == "pitch_fine":
            next_round = index_rate_round(selected, has_index=self.has_index)
            if next_round is None:
                next_round = protect_round(replace(selected, index_rate=0.0))
        elif completed.stage == "index_rate":
            next_round = protect_round(selected)
        elif completed.stage == "protect":
            next_round = rms_mix_round(selected)
        elif completed.stage == "rms_mix_rate":
            self.final_parameters = selected
            logger.info("最终参数: {}", selected)
            return None
        else:
            raise ValueError(f"unknown search stage: {completed.stage}")

        self.current = next_round
        logger.info("当前搜索阶段: {}", self.current.stage)
        return self.current

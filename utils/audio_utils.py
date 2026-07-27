"""音频工具函数"""

import numpy as np


def db_to_linear(db: float) -> float:
    """分贝转线性值。"""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """线性值转分贝。"""
    if linear <= 0:
        return -120.0
    return 20.0 * np.log10(linear)


def rms(data: np.ndarray) -> float:
    """计算一帧音频的 RMS 幅值。"""
    return float(np.sqrt(np.mean(data ** 2)))

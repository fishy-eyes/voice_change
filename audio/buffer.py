"""音频缓冲管理 - 线程安全的环形缓冲区"""

import threading
from collections import deque
from typing import Optional

import numpy as np


class AudioBuffer:
    """基于 deque 的线程安全环形音频缓冲区。

    用于 recorder -> player 之间传递 numpy 音频帧。
    后续可扩展为 lock-free ring buffer 以进一步降低延迟。
    """

    def __init__(self, max_frames: int = 8) -> None:
        self._buf: deque = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def push(self, data: np.ndarray) -> None:
        """向缓冲区写入一帧音频数据。满时丢弃最旧帧。"""
        with self._lock:
            self._buf.append(data.copy())

    def pop(self) -> Optional[np.ndarray]:
        """从缓冲区读取并移除一帧。为空时返回 None。"""
        with self._lock:
            if self._buf:
                return self._buf.popleft()
        return None

    def clear(self) -> None:
        """清空缓冲区。"""
        with self._lock:
            self._buf.clear()

    @property
    def size(self) -> int:
        """当前缓冲帧数。"""
        return len(self._buf)

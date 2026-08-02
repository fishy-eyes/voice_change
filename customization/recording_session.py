"""Fixed-script recording and deterministic short-segment selection."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger


STANDARD_RECORDING_TEXT = (
    "今天是星期三，我正在测试实时变声效果。声音从低到高，再轻轻落下；"
    "请问三、七、十五和二十八听起来清楚吗？风吹树梢，闪烁的星光穿过城市，"
    "短暂停顿后，我会自然地说完这句话。"
)


class TemporaryAudioDirectory:
    def __init__(self, root: str | Path | None = None) -> None:
        self._owner = None
        if root is None:
            self._owner = tempfile.TemporaryDirectory(prefix="voice_customization_")
            self.path = Path(self._owner.name)
        else:
            self.path = Path(root)
            self.path.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        if self._owner is not None:
            self._owner.cleanup()
            self._owner = None
        logger.info("定制临时文件清理结果: {}", self.path)


class RecordingSession:
    """Capture mono float32 audio without touching the realtime AudioStream."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        device: int | None = None,
        channels: int = 1,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.device = device
        self.channels = int(channels)
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("recording is already active")
        with self._lock:
            self._chunks.clear()

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                logger.warning("定制录音状态: {}", status)
            with self._lock:
                self._chunks.append(np.array(indata[:, 0], dtype=np.float32, copy=True))

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._stream.start()
        logger.info("开始录音: device={} sample_rate={}", self.device, self.sample_rate)

    def stop(self) -> np.ndarray:
        stream = self._stream
        if stream is None:
            return np.empty(0, dtype=np.float32)
        self._stream = None
        stream.stop()
        stream.close()
        with self._lock:
            chunks = tuple(self._chunks)
            self._chunks.clear()
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    @staticmethod
    def load_file(path: str | Path, target_sample_rate: int = 48000) -> np.ndarray:
        from math import gcd
        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        if np.asarray(audio).ndim == 2:
            audio = np.mean(audio, axis=1)
        if sample_rate != target_sample_rate:
            divisor = gcd(int(sample_rate), int(target_sample_rate))
            audio = resample_poly(
                np.asarray(audio), target_sample_rate // divisor, sample_rate // divisor
            )
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def save_file(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
        return destination

    @staticmethod
    def split_search_segments(audio: np.ndarray) -> dict[str, np.ndarray]:
        """Use fixed-script time proportions; replaceable by future VAD logic."""
        values = np.asarray(audio, dtype=np.float32)
        count = len(values)
        windows = {
            "normal": (0.08, 0.30),
            "consonant": (0.42, 0.64),
            "intonation": (0.70, 0.92),
        }
        return {
            name: values[int(count * start) : int(count * end)].copy()
            for name, (start, end) in windows.items()
        }

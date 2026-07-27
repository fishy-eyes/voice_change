"""麦克风采集模块"""

from typing import Callable, Optional

import sounddevice as sd
from loguru import logger

from config.settings import SAMPLE_RATE, CHANNELS, BLOCKSIZE, DTYPE, LATENCY

# 回调类型：收到一帧 (frames, channels) 的 float32 ndarray
AudioCallback = Callable[["sd.CallbackStop"], None]


class AudioRecorder:
    """封装 sounddevice.InputStream，提供麦克风采集接口。

    与 AudioStream 配合使用，不单独启动独立的输入流。
    """

    def __init__(
        self,
        device: Optional[int] = None,
        samplerate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        blocksize: int = BLOCKSIZE,
        dtype: str = DTYPE,
        latency: str = LATENCY,
    ) -> None:
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self.latency = latency

    @staticmethod
    def list_devices() -> list[dict]:
        """列出系统可用音频输入设备。"""
        devices = sd.query_devices()
        inputs = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                inputs.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
        return inputs

    def get_stream_params(self) -> dict:
        """返回构造 duplex Stream 所需的输入参数。"""
        return dict(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype=self.dtype,
            latency=self.latency,
        )

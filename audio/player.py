"""音频播放模块"""

from typing import Optional

import sounddevice as sd
from loguru import logger

from config.settings import SAMPLE_RATE, CHANNELS, BLOCKSIZE, DTYPE, LATENCY


class AudioPlayer:
    """封装 sounddevice.OutputStream，提供音频播放接口。

    与 AudioStream 配合使用，不单独启动独立的输出流。
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
        """列出系统可用音频输出设备。"""
        devices = sd.query_devices()
        outputs = []
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                outputs.append({"index": i, "name": d["name"], "channels": d["max_output_channels"]})
        return outputs

    def get_stream_params(self) -> dict:
        """返回构造 duplex Stream 所需的输出参数。"""
        return dict(
            device=self.device,
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype=self.dtype,
            latency=self.latency,
        )

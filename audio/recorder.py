"""麦克风采集模块"""

from typing import Optional

from config.settings import SAMPLE_RATE, CHANNELS, BLOCKSIZE, DTYPE, LATENCY


class AudioRecorder:
    """封装麦克风采集参数，供 AudioStream 使用。

    不自行创建 InputStream；设备枚举统一由 DeviceManager 负责。
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

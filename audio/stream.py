"""实时音频流管理 - 双工流（duplex）实现最低延迟回环"""

from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.device_manager import DeviceManager
from config.settings import SAMPLE_RATE, CHANNELS, BLOCKSIZE, DTYPE

# 处理回调签名：接收 (input_data, frame_count, time_info, status)
# 返回处理后的 ndarray，形状与 input_data 相同
ProcessFunc = Callable[[np.ndarray, int, object, object], np.ndarray]


class AudioStream:
    """管理双工音频流：麦克风输入 -> 处理链 -> 扬声器输出。

    架构说明
    --------
    使用 sounddevice 的双工 Stream（input=True, output=True），
    在同一个回调中完成采集和播放，延迟最低。

    插件扩展
    --------
    通过 `set_process_func` 注入处理函数，后续音效模块只需
    实现 `func(data, frames, time_info, status) -> np.ndarray` 即可。
    """

    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        player: Optional[AudioPlayer] = None,
    ) -> None:
        self._recorder = recorder or AudioRecorder()
        self._player = player or AudioPlayer()
        self._process_func: Optional[ProcessFunc] = None
        self._stream: Optional[sd.Stream] = None

    # ------------------------------------------------------------------
    # 处理链
    # ------------------------------------------------------------------
    def set_process_func(self, func: Optional[ProcessFunc]) -> None:
        """设置/清除音频处理回调。设为 None 时直通（loopback）。"""
        self._process_func = func
        logger.info("处理函数已更新: {}", func.__name__ if func else "直通")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动双工音频流。"""
        if self._stream is not None:
            logger.warning("音频流已在运行")
            return

        rec_params = self._recorder.get_stream_params()
        ply_params = self._player.get_stream_params()

        def _callback(indata, outdata, frames, time_info, status):
            if status:
                logger.debug("音频状态: {}", status)
            processed = indata
            if self._process_func is not None:
                try:
                    processed = self._process_func(indata, frames, time_info, status)
                except Exception as e:
                    logger.error("处理回调异常: {}", e)
                    processed = indata
            outdata[:] = processed

        self._stream = sd.Stream(
            samplerate=rec_params["samplerate"],
            blocksize=rec_params["blocksize"],
            dtype=rec_params["dtype"],
            channels=rec_params["channels"],
            callback=_callback,
            device=(rec_params["device"], ply_params["device"]),
            latency=(rec_params["latency"], ply_params["latency"]),
        )
        self._stream.start()

        in_name = DeviceManager.get_device_name(rec_params["device"] or sd.default.device[0])
        out_name = DeviceManager.get_device_name(ply_params["device"] or sd.default.device[1])
        logger.info("音频流已启动 | 采样率 {} | 块大小 {}", rec_params["samplerate"], rec_params["blocksize"])
        logger.info("  输入设备: {}", in_name)
        logger.info("  输出设备: {}", out_name)

    def stop(self) -> None:
        """停止并释放音频流。"""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.info("音频流已停止")

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    # ------------------------------------------------------------------
    # 上下文管理
    # ------------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
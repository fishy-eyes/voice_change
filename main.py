"""Voice Changer - Windows 实时变声器"""

import signal
import sys
import threading

from utils.logger import setup_logger
from loguru import logger
from audio.device_manager import DeviceManager
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.stream import AudioStream

# 全局退出事件，信号处理和主线程共用
_stop_event = threading.Event()


def _on_signal(sig, frame):
    """信号处理：只设置退出标志，不直接调用 stream.stop()。"""
    logger.info("收到退出信号，正在停止...")
    _stop_event.set()


def main() -> None:
    """程序入口：选择设备并启动实时音频回环。"""
    setup_logger()

    DeviceManager.print_devices()

    input_idx = DeviceManager.select_input_device()
    output_idx = DeviceManager.select_output_device()

    recorder = AudioRecorder(device=input_idx)
    player = AudioPlayer(device=output_idx)
    stream = AudioStream(recorder, player)

    stream.start()
    logger.info("实时回环已启动，按 Ctrl+C 停止...")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # 阻塞等待退出，不烧 CPU
    try:
        _stop_event.wait()
    except KeyboardInterrupt:
        pass

    # 只在一处调用 stop()
    stream.stop()
    logger.info("已退出")


if __name__ == "__main__":
    main()

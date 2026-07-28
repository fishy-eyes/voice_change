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
from effects.manager import EffectManager

_stop_event = threading.Event()


def _on_signal(sig, frame):
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

    # 初始化音效链（当前为空，直通模式）
    effect_manager = EffectManager()

    stream = AudioStream(recorder, player, effect_manager=effect_manager)
    stream.start()
    logger.info("实时回环已启动，按 Ctrl+C 停止...")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # 带超时的循环等待，保证 Windows 上 Ctrl+C 可中断
    try:
        while not _stop_event.wait(timeout=0.1):
            pass
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt，正在停止...")
        _stop_event.set()

    stream.stop()
    logger.info("已退出")


if __name__ == "__main__":
    main()

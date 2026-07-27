"""Voice Changer - Windows 实时变声器"""

import signal
import sys

from utils.logger import setup_logger
from loguru import logger
from audio.device_manager import DeviceManager
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.stream import AudioStream


def main() -> None:
    """程序入口：选择设备并启动实时音频回环。"""
    setup_logger()

    # 打印所有可用设备
    DeviceManager.print_devices()

    # 交互式选择输入/输出设备
    input_idx = DeviceManager.select_input_device()
    output_idx = DeviceManager.select_output_device()

    recorder = AudioRecorder(device=input_idx)
    player = AudioPlayer(device=output_idx)
    stream = AudioStream(recorder, player)

    # 直通模式，不挂载任何处理函数
    stream.start()
    logger.info("实时回环已启动，按 Ctrl+C 停止...")

    # 优雅退出
    def _shutdown(sig, frame):
        logger.info("收到退出信号，正在停止...")
        stream.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 阻塞主线程
    try:
        while stream.is_running:
            pass
    except KeyboardInterrupt:
        stream.stop()
        logger.info("已退出")


if __name__ == "__main__":
    main()
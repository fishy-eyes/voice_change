"""Voice Changer - Windows 实时变声器"""

import signal
import sys

from utils.logger import setup_logger
from loguru import logger
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.stream import AudioStream


def main() -> None:
    """程序入口：启动实时音频回环。"""
    setup_logger()

    # 列出可用设备（方便调试）
    logger.info("=== 输入设备 ===")
    for d in AudioRecorder.list_devices():
        logger.info("  [{}] {}", d["index"], d["name"])
    logger.info("=== 输出设备 ===")
    for d in AudioPlayer.list_devices():
        logger.info("  [{}] {}", d["index"], d["name"])

    recorder = AudioRecorder()
    player = AudioPlayer()
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

"""Voice Changer - Windows 实时变声器"""

import signal
import sys
import threading
import time

from utils.logger import setup_logger
from loguru import logger
from audio.device_manager import DeviceManager
from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.stream import AudioStream
from effects.manager import EffectManager
from effects.gain import GainEffect
from effects.echo import EchoEffect
from effects.robot import RobotEffect
from config.settings import (
    INPUT_DEVICE, OUTPUT_DEVICE, AUTO_SELECT_DEVICES, SHOW_DEVICE_LIST,
    ENABLE_GAIN, GAIN_VALUE,
    ENABLE_ECHO, ECHO_DELAY, ECHO_DECAY,
    ENABLE_ROBOT, ROBOT_FREQUENCY,
)

_stop_event = threading.Event()


def _on_signal(sig, frame):
    logger.info("收到退出信号，正在停止...")
    _stop_event.set()


def create_effect_manager() -> EffectManager:
    """创建并配置效果管理器。"""
    effect_manager = EffectManager()
    if ENABLE_GAIN:
        effect_manager.add(GainEffect(gain=GAIN_VALUE))
    if ENABLE_ECHO:
        effect_manager.add(EchoEffect(delay=ECHO_DELAY, decay=ECHO_DECAY))
    if ENABLE_ROBOT:
        effect_manager.add(RobotEffect(frequency=ROBOT_FREQUENCY))
    return effect_manager


def main() -> None:
    """程序入口：选择设备并启动实时音频回环。"""
    setup_logger()

    if SHOW_DEVICE_LIST:
        DeviceManager.print_devices()

    if AUTO_SELECT_DEVICES:
        if INPUT_DEVICE is not None:
            input_idx = INPUT_DEVICE
            logger.info("使用配置指定的输入设备: {}", input_idx)
        else:
            input_idx = None
            logger.info("使用系统默认输入设备")
        output_idx = DeviceManager.find_virtual_output_device()
        if output_idx is not None:
            logger.info("自动使用VB-CABLE输出设备")
        else:
            logger.warning("未检测到VB-CABLE设备，回退到手动选择")
            output_idx = DeviceManager.select_output_device()
    else:
        input_idx = DeviceManager.select_input_device()
        output_idx = DeviceManager.select_output_device()

    recorder = AudioRecorder(device=input_idx)
    player = AudioPlayer(device=output_idx)

    effect_manager = create_effect_manager()

    stream = AudioStream(recorder, player, effect_manager=effect_manager)
    stream.start()
    logger.info("实时回环已启动，按 Ctrl+C 停止...")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # 等待退出信号，保证 Windows 上 Ctrl+C 可中断
    try:
        while not _stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt，正在停止...")
        _stop_event.set()

    stream.stop()
    logger.info("已退出")


if __name__ == "__main__":
    main()

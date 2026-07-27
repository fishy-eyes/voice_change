"""音频设备管理模块

集中管理输入/输出设备的枚举、验证与选择。
AudioRecorder 和 AudioPlayer 不再各自维护 list_devices，
统一从这里获取。
"""

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd
from loguru import logger


@dataclass
class DeviceInfo:
    """单个音频设备的描述信息。"""
    index: int
    name: str
    channels: int          # 输入通道数 或 输出通道数
    default_samplerate: float


class DeviceManager:
    """系统音频设备管理器。"""

    # ------------------------------------------------------------------
    # 枚举
    # ------------------------------------------------------------------
    @staticmethod
    def list_input_devices() -> list[DeviceInfo]:
        """返回所有可用输入设备。"""
        devices = sd.query_devices()
        result: list[DeviceInfo] = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append(DeviceInfo(
                    index=i,
                    name=d["name"],
                    channels=d["max_input_channels"],
                    default_samplerate=d["default_samplerate"],
                ))
        return result

    @staticmethod
    def list_output_devices() -> list[DeviceInfo]:
        """返回所有可用输出设备。"""
        devices = sd.query_devices()
        result: list[DeviceInfo] = []
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                result.append(DeviceInfo(
                    index=i,
                    name=d["name"],
                    channels=d["max_output_channels"],
                    default_samplerate=d["default_samplerate"],
                ))
        return result

    # ------------------------------------------------------------------
    # 格式化打印
    # ------------------------------------------------------------------
    @staticmethod
    def print_devices() -> None:
        """打印当前所有输入/输出设备，供终端调试。"""
        logger.info("========== 输入设备 ==========")
        for d in DeviceManager.list_input_devices():
            logger.info("  [{}] {} (通道: {}, 默认采样率: {} Hz)",
                        d.index, d.name, d.channels, int(d.default_samplerate))
        logger.info("========== 输出设备 ==========")
        for d in DeviceManager.list_output_devices():
            logger.info("  [{}] {} (通道: {}, 默认采样率: {} Hz)",
                        d.index, d.name, d.channels, int(d.default_samplerate))

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------
    @staticmethod
    def select_input_device(prompt: str = "请选择输入设备编号") -> Optional[int]:
        """交互式选择输入设备。返回设备编号，直接回车则使用系统默认。"""
        devices = DeviceManager.list_input_devices()
        if not devices:
            logger.warning("未找到任何输入设备")
            return None

        logger.info("---------- 可用输入设备 ----------")
        for d in devices:
            logger.info("  [{}] {}", d.index, d.name)
        logger.info("  [回车 = 系统默认]")

        try:
            raw = input(f"\n{prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw == "":
            logger.info("使用系统默认输入设备")
            return None

        try:
            idx = int(raw)
        except ValueError:
            logger.warning("输入无效，使用系统默认输入设备")
            return None

        valid_indices = {d.index for d in devices}
        if idx not in valid_indices:
            logger.warning("设备编号 {} 不存在，使用系统默认输入设备", idx)
            return None

        chosen = next(d for d in devices if d.index == idx)
        logger.info("已选择输入设备: [{}] {}", chosen.index, chosen.name)
        return idx

    @staticmethod
    def select_output_device(prompt: str = "请选择输出设备编号") -> Optional[int]:
        """交互式选择输出设备。返回设备编号，直接回车则使用系统默认。"""
        devices = DeviceManager.list_output_devices()
        if not devices:
            logger.warning("未找到任何输出设备")
            return None

        logger.info("---------- 可用输出设备 ----------")
        for d in devices:
            logger.info("  [{}] {}", d.index, d.name)
        logger.info("  [回车 = 系统默认]")

        try:
            raw = input(f"\n{prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw == "":
            logger.info("使用系统默认输出设备")
            return None

        try:
            idx = int(raw)
        except ValueError:
            logger.warning("输入无效，使用系统默认输出设备")
            return None

        valid_indices = {d.index for d in devices}
        if idx not in valid_indices:
            logger.warning("设备编号 {} 不存在，使用系统默认输出设备", idx)
            return None

        chosen = next(d for d in devices if d.index == idx)
        logger.info("已选择输出设备: [{}] {}", chosen.index, chosen.name)
        return idx

    # ------------------------------------------------------------------
    # 查询工具
    # ------------------------------------------------------------------
    @staticmethod
    def get_device_name(index: int) -> str:
        """根据设备编号返回设备名称，找不到返回 '(unknown)'。"""
        try:
            return sd.query_devices(index)["name"]
        except (ValueError, sd.PortAudioError):
            return "(unknown)"

    @staticmethod
    def get_default_devices() -> tuple[Optional[int], Optional[int]]:
        """返回系统默认 (input_device, output_device) 的编号。"""
        return sd.default.device

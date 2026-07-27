"""Audio Device Manager

Centralized audio device enumeration, filtering, and selection.
AudioRecorder and AudioPlayer get device params from here.
"""

from dataclasses import dataclass
from typing import Optional

import sounddevice as sd
from loguru import logger


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float


class DeviceManager:

    @staticmethod
    def _filter_devices(kind: str) -> list[DeviceInfo]:
        devices = sd.query_devices()
        seen: set[str] = set()
        result: list[DeviceInfo] = []
        for i, d in enumerate(devices):
            ch_key = "max_input_channels" if kind == "input" else "max_output_channels"
            if d[ch_key] <= 0:
                continue
            name = d["name"]
            lower = name.lower()
            if "mapper" in lower:
                continue
            if "primary sound" in lower:
                continue
            if name in seen:
                continue
            seen.add(name)
            result.append(DeviceInfo(i, name, d[ch_key], d["default_samplerate"]))
        return result

    @staticmethod
    def list_input_devices() -> list[DeviceInfo]:
        return DeviceManager._filter_devices("input")

    @staticmethod
    def list_output_devices() -> list[DeviceInfo]:
        return DeviceManager._filter_devices("output")

    @staticmethod
    def print_devices() -> None:
        defaults = sd.default.device
        logger.info("")
        logger.info("=" * 50)
        logger.info("  输入设备 (麦克风)")
        logger.info("=" * 50)
        for d in DeviceManager.list_input_devices():
            mark = " [默认]" if d.index == defaults[0] else ""
            logger.info("  编号:{:<3}  名称:{}  输入通道:{}  采样率:{}Hz{}", d.index, d.name, d.channels, int(d.default_samplerate), mark)
        logger.info("")
        logger.info("=" * 50)
        logger.info("  输出设备 (播放)")
        logger.info("=" * 50)
        for d in DeviceManager.list_output_devices():
            mark = " [默认]" if d.index == defaults[1] else ""
            logger.info("  编号:{:<3}  名称:{}  输出通道:{}  采样率:{}Hz{}", d.index, d.name, d.channels, int(d.default_samplerate), mark)
        logger.info("")

    @staticmethod
    def select_input_device() -> Optional[int]:
        devices = DeviceManager.list_input_devices()
        if not devices:
            logger.warning("未找到任何输入设备")
            return None
        defaults = sd.default.device
        for d in devices:
            mark = " [默认]" if d.index == defaults[0] else ""
            logger.info("  [{}] {}{}", d.index, d.name, mark)
        logger.info("  [直接回车 = 系统默认]")
        try:
            raw = input("\n选择输入设备编号: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw == "":
            logger.info("使用系统默认输入设备")
            return None
        try:
            idx = int(raw)
        except ValueError:
            return None
        valid = {d.index for d in devices}
        if idx not in valid:
            logger.warning("设备编号无效，使用系统默认")
            return None
        chosen = next(d for d in devices if d.index == idx)
        logger.info("已选择输入: [{}] {}", chosen.index, chosen.name)
        return idx

    @staticmethod
    def select_output_device() -> Optional[int]:
        devices = DeviceManager.list_output_devices()
        if not devices:
            logger.warning("未找到任何输出设备")
            return None
        defaults = sd.default.device
        for d in devices:
            mark = " [默认]" if d.index == defaults[1] else ""
            logger.info("  [{}] {}{}", d.index, d.name, mark)
        logger.info("  [直接回车 = 系统默认]")
        try:
            raw = input("\n选择输出设备编号: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw == "":
            logger.info("使用系统默认输出设备")
            return None
        try:
            idx = int(raw)
        except ValueError:
            return None
        valid = {d.index for d in devices}
        if idx not in valid:
            logger.warning("设备编号无效，使用系统默认")
            return None
        chosen = next(d for d in devices if d.index == idx)
        logger.info("已选择输出: [{}] {}", chosen.index, chosen.name)
        return idx

    @staticmethod
    def get_device_name(index: Optional[int]) -> str:
        if index is None:
            return "(系统默认)"
        try:
            return sd.query_devices(index)["name"]
        except (ValueError, sd.PortAudioError):
            return "(未知设备 #%d)" % index

    @staticmethod
    def get_default_devices() -> tuple:
        return sd.default.device

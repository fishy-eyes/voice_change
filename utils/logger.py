"""日志工具 - 基于 loguru"""

import sys
from pathlib import Path

from loguru import logger

from config.settings import APP_NAME, APP_VERSION, LOG_LEVEL, PROJECT_ROOT

_configured = False


def setup_logger() -> None:
    """初始化全局日志，重复调用安全。"""
    global _configured
    if _configured:
        return

    logger.remove()                          # 移除默认 handler
    log_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )
    if sys.stderr is not None:
        logger.add(sys.stderr, level=LOG_LEVEL, format=log_format)

    log_directory = Path(PROJECT_ROOT) / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_directory / "voice_change.log",
        level=LOG_LEVEL,
        format=log_format,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )
    logger.info("{} v{} 日志系统已就绪", APP_NAME, APP_VERSION)
    _configured = True

"""日志工具 - 基于 loguru"""

import sys

from loguru import logger

from config.settings import LOG_LEVEL, APP_NAME

_configured = False


def setup_logger() -> None:
    """初始化全局日志，重复调用安全。"""
    global _configured
    if _configured:
        return

    logger.remove()                          # 移除默认 handler
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level:<7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
               "<level>{message}</level>",
    )
    logger.info("{} 日志系统已就绪", APP_NAME)
    _configured = True

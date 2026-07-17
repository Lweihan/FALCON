"""
logger.py — 统一日志
=====================
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str = "FALCON",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    获取统一格式 logger。

    Args:
        name: logger 名称
        level: 日志级别
        log_file: 可选日志文件路径
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

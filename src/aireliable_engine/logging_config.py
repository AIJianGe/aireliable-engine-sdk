"""统一日志配置。

格式：[timestamp] [level] [engine] [message]
输出到 stdout（供主控采集）+ 文件（data/logs/<engine>.log）
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class EngineLogFormatter(logging.Formatter):
    """引擎日志格式化器。

    格式：[2026-07-18 12:34:56] [INFO] [ernie_image] 这是一条日志
    """

    def __init__(self, engine_id: str):
        super().__init__()
        self.engine_id = engine_id

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        return f"[{timestamp}] [{level}] [{self.engine_id}] {record.getMessage()}"


def setup_engine_logging(
    engine_id: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """配置引擎日志。

    Args:
        engine_id: 引擎标识（如 ernie_image）
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_dir: 日志目录，若提供则同时写文件

    Returns:
        logging.Logger: 配置好的根 logger
    """
    level_num = getattr(logging, log_level.upper(), logging.INFO)

    # 配置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_num)

    # 清除既有 handler（避免重复）
    root_logger.handlers.clear()

    formatter = EngineLogFormatter(engine_id)

    # stdout handler（主控通过 stdout 采集日志）
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level_num)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # 文件 handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{engine_id}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level_num)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 配置常用库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return root_logger

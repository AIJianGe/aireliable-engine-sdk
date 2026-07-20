"""日志配置测试。

覆盖 EngineLogFormatter / setup_engine_logging。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aireliable_engine.logging_config import EngineLogFormatter, setup_engine_logging


class TestEngineLogFormatter:
    """EngineLogFormatter 测试。"""

    def test_format_contains_required_parts(self):
        """格式化输出包含 [timestamp] [level] [engine_id] [message]。"""
        formatter = EngineLogFormatter("ernie_image")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="模型加载完成",
            args=None,
            exc_info=None,
        )
        s = formatter.format(record)
        # 应包含 4 个方括号字段
        assert "[INFO]" in s
        assert "[ernie_image]" in s
        assert "模型加载完成" in s
        # timestamp 应是 [YYYY-MM-DD HH:MM:SS] 格式，长度 21 字符
        assert s[0] == "["
        idx = s.find("]")
        assert idx == 20  # "[YYYY-MM-DD HH:MM:SS]" 长度 21，"]" 在索引 20
        assert s[20] == "]"

    def test_format_different_level(self):
        """不同 level 格式化正确。"""
        formatter = EngineLogFormatter("e1")
        record = logging.LogRecord(
            name="t",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warning msg",
            args=None,
            exc_info=None,
        )
        s = formatter.format(record)
        assert "[WARNING]" in s
        assert "[e1]" in s
        assert "warning msg" in s

    def test_format_with_args(self):
        """带参数的 msg 格式化。"""
        formatter = EngineLogFormatter("e1")
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="step %d / %d",
            args=(10, 50),
            exc_info=None,
        )
        s = formatter.format(record)
        assert "step 10 / 50" in s


class TestSetupEngineLogging:
    """setup_engine_logging 测试。"""

    def test_configures_root_logger(self):
        """setup_engine_logging 配置 root logger。"""
        root = setup_engine_logging("test_engine", log_level="DEBUG")
        assert root is logging.getLogger()
        assert root.level == logging.DEBUG

    def test_sets_correct_log_level(self):
        """setup_engine_logging 设置正确日志级别。"""
        for level_name, level_num in [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
        ]:
            root = setup_engine_logging("test_engine", log_level=level_name)
            assert root.level == level_num

    def test_clears_existing_handlers(self):
        """setup_engine_logging 清除既有 handler。"""
        root = logging.getLogger()
        # 添加一个假 handler
        fake_h = logging.NullHandler()
        root.addHandler(fake_h)
        assert len(root.handlers) >= 1

        setup_engine_logging("test_engine")
        # fake_h 应被清除
        assert fake_h not in root.handlers

    def test_writes_to_file(self, tmp_path: Path):
        """setup_engine_logging 写文件。"""
        log_dir = tmp_path / "logs"
        setup_engine_logging("test_engine", log_level="INFO", log_dir=log_dir)

        # 触发一条日志
        logger = logging.getLogger("test_writes_to_file")
        logger.info("hello from test")

        # flush 所有 handler
        for h in logging.getLogger().handlers:
            h.flush()

        log_file = log_dir / "test_engine.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "hello from test" in content
        assert "[test_engine]" in content

    def test_no_file_handler_without_log_dir(self):
        """setup_engine_logging 无 log_dir 时不写文件。"""
        root = setup_engine_logging("test_engine", log_level="INFO")
        # 只应有 stdout handler，不应有 FileHandler
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0
        # 至少有一个 StreamHandler
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_library_loggers_set_to_warning(self):
        """常用库 logger 级别被设置（uvicorn / diffusers / transformers）。"""
        setup_engine_logging("test_engine", log_level="DEBUG")
        assert logging.getLogger("uvicorn").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        assert logging.getLogger("diffusers").level == logging.WARNING
        assert logging.getLogger("transformers").level == logging.WARNING

    def test_log_dir_created_if_not_exists(self, tmp_path: Path):
        """log_dir 不存在时自动创建。"""
        log_dir = tmp_path / "nested" / "deep" / "logs"
        assert not log_dir.exists()
        setup_engine_logging("test_engine", log_level="INFO", log_dir=log_dir)
        assert log_dir.exists()

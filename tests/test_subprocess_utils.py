"""子进程工具测试。

覆盖 make_engine_subprocess_flags / make_uv_run_command /
decode_subprocess_output / find_uv_executable。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aireliable_engine import subprocess_utils
from aireliable_engine.subprocess_utils import (
    decode_subprocess_output,
    find_uv_executable,
    make_engine_subprocess_flags,
    make_uv_run_command,
)


class TestMakeEngineSubprocessFlags:
    """make_engine_subprocess_flags 测试。"""

    def test_returns_dict_with_required_keys(self):
        """返回 dict 含 stdin/stdout/stderr/bufsize/creationflags。"""
        flags = make_engine_subprocess_flags()
        assert isinstance(flags, dict)
        assert "stdin" in flags
        assert "stdout" in flags
        assert "stderr" in flags
        assert "bufsize" in flags
        assert "creationflags" in flags

    def test_stdin_stdout_pipe(self):
        """stdin / stdout 应为 PIPE。"""
        flags = make_engine_subprocess_flags()
        assert flags["stdin"] == subprocess.PIPE
        assert flags["stdout"] == subprocess.PIPE

    def test_stderr_merged_to_stdout(self):
        """stderr 应合并到 stdout。"""
        flags = make_engine_subprocess_flags()
        assert flags["stderr"] == subprocess.STDOUT

    def test_bufsize_zero(self):
        """bufsize 应为 0（无缓冲）。"""
        flags = make_engine_subprocess_flags()
        assert flags["bufsize"] == 0

    def test_creationflags_on_non_windows(self, monkeypatch):
        """非 Windows 平台 creationflags=0。"""
        monkeypatch.setattr(sys, "platform", "linux")
        # 重新加载模块以应用 platform
        import importlib
        importlib.reload(subprocess_utils)
        flags = subprocess_utils.make_engine_subprocess_flags()
        assert flags["creationflags"] == 0
        # 恢复
        monkeypatch.setattr(sys, "platform", "win32" if sys.platform == "win32" else sys.platform)
        importlib.reload(subprocess_utils)

    def test_creationflags_on_windows(self, monkeypatch):
        """Windows 平台 creationflags=subprocess.CREATE_NO_WINDOW。"""
        # 仅在 Windows 上跑得通，因为 CREATE_NO_WINDOW 是 Windows-only 常量
        if sys.platform != "win32":
            pytest.skip("仅在 Windows 上验证 CREATE_NO_WINDOW")
        monkeypatch.setattr(sys, "platform", "win32")
        import importlib
        importlib.reload(subprocess_utils)
        flags = subprocess_utils.make_engine_subprocess_flags()
        assert flags["creationflags"] == subprocess.CREATE_NO_WINDOW
        importlib.reload(subprocess_utils)


class TestMakeUvRunCommand:
    """make_uv_run_command 测试。"""

    def test_basic_command(self):
        """构造完整命令列表。"""
        env_dir = Path("/some/env")
        cmd = make_uv_run_command(env_dir, "engines.test.server")
        assert cmd[0] == "uv"
        assert cmd[1] == "run"
        assert "--project" in cmd
        assert str(env_dir) in cmd  # str(Path) 化（Windows 上为 \some\env）
        assert "python" in cmd
        assert "-m" in cmd
        assert "engines.test.server" in cmd

    def test_with_extra_args(self):
        """含 extra_args。"""
        cmd = make_uv_run_command(
            Path("/env"),
            "m",
            extra_args=["--port", "8000", "--host", "0.0.0.0"],
        )
        assert "--port" in cmd
        assert "8000" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        # extra_args 应在最后
        assert cmd[-1] == "0.0.0.0"

    def test_with_python_version(self):
        """含 python 版本。"""
        cmd = make_uv_run_command(Path("/env"), "m", python="3.11")
        assert "--python" in cmd
        # --python 后面跟 3.11
        idx = cmd.index("--python")
        assert cmd[idx + 1] == "3.11"

    def test_without_python_version(self):
        """不指定 python 版本时不出现 --python。"""
        cmd = make_uv_run_command(Path("/env"), "m")
        assert "--python" not in cmd

    def test_command_structure(self):
        """命令结构正确：uv run --project <env> [python -m module] [extra]。"""
        cmd = make_uv_run_command(
            Path("/env"),
            "m",
            extra_args=["--a"],
            python="3.10",
        )
        # uv run --project /env --python 3.10 python -m m --a
        assert cmd == [
            "uv", "run", "--project", str(Path("/env")),
            "--python", "3.10",
            "python", "-m", "m",
            "--a",
        ]


class TestDecodeSubprocessOutput:
    """decode_subprocess_output 测试。"""

    def test_empty_bytes_returns_empty(self):
        """空字节返回空字符串。"""
        assert decode_subprocess_output(b"") == ""

    def test_force_encoding(self):
        """强制 encoding。"""
        # UTF-8 编码
        data = "你好".encode("utf-8")
        assert decode_subprocess_output(data, encoding="utf-8") == "你好"
        # Latin-1 编码
        data2 = "café".encode("latin-1")
        assert decode_subprocess_output(data2, encoding="latin-1") == "café"

    def test_windows_gbk_decoding(self, monkeypatch):
        """Windows GBK 解码。"""
        monkeypatch.setattr(sys, "platform", "win32")
        # GBK 编码的中文
        data = "你好".encode("gbk")
        result = decode_subprocess_output(data)
        assert result == "你好"

    def test_linux_utf8_decoding(self, monkeypatch):
        """Linux UTF-8 解码。"""
        monkeypatch.setattr(sys, "platform", "linux")
        data = "你好".encode("utf-8")
        result = decode_subprocess_output(data)
        assert result == "你好"

    def test_decode_with_errors_replace(self, monkeypatch):
        """解码错误时用 replace 替代。"""
        monkeypatch.setattr(sys, "platform", "linux")
        # 故意传无效 UTF-8 字节
        data = b"\xff\xfe\xfd"
        result = decode_subprocess_output(data)
        # 应不抛异常，返回带替换字符的字符串
        assert isinstance(result, str)
        assert "\ufffd" in result or len(result) > 0


class TestFindUvExecutable:
    """find_uv_executable 测试。"""

    def test_uses_uv_bin_env_var(self, monkeypatch, tmp_path: Path):
        """用 UV_BIN 环境变量。"""
        # 构造一个假的可执行文件
        fake_uv = tmp_path / "uv.exe"
        fake_uv.write_text("fake")
        monkeypatch.setenv("UV_BIN", str(fake_uv))
        assert find_uv_executable() == str(fake_uv)

    def test_uv_bin_env_var_nonexistent_falls_back(self, monkeypatch):
        """UV_BIN 指向不存在的文件时 fallback 到 uv。"""
        monkeypatch.setenv("UV_BIN", "/nonexistent/path/uv")
        assert find_uv_executable() == "uv"

    def test_no_uv_bin_returns_uv(self, monkeypatch):
        """无 UV_BIN 环境变量时返回 uv。"""
        monkeypatch.delenv("UV_BIN", raising=False)
        assert find_uv_executable() == "uv"

    def test_empty_uv_bin_returns_uv(self, monkeypatch):
        """UV_BIN 为空字符串时返回 uv。"""
        monkeypatch.setenv("UV_BIN", "")
        assert find_uv_executable() == "uv"

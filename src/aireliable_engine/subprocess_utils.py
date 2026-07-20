"""子进程创建工具（跨平台）。

统一封装 subprocess 创建，处理：
- Windows 不弹 cmd 窗口（CREATE_NO_WINDOW）
- 跨平台兼容
- 引擎进程的标准启动参数
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Windows 下避免弹 cmd 窗口的标志
# 在非 Windows 平台为 0（无效果）
_CREATE_NO_WINDOW: int = 0
if sys.platform == "win32":
    _CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW


def make_engine_subprocess_flags() -> dict:
    """构造引擎子进程的标准启动参数。

    返回 dict，可直接 ** 展开到 subprocess.Popen / asyncio.create_subprocess_exec。

    Returns:
        dict: 包含 stdin/stdout/stderr 处理 + creationflags（Windows）
    """
    flags: dict = {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,          # stderr 合并到 stdout，便于日志采集
        # 注意：asyncio.create_subprocess_exec 在 Windows 上仅支持 bufsize=0 或 -1
        # bufsize=1（行缓冲）只在文本模式生效，二进制 PIPE 模式会抛 "bufsize must be 0"
        # 这里用 0（无缓冲），由采集端按行读取
        "bufsize": 0,
        "creationflags": _CREATE_NO_WINDOW,    # Windows 不弹窗
    }
    return flags


def make_uv_run_command(
    uv_env_dir: Path,
    module: str,
    extra_args: list[str] | None = None,
    python: str | None = None,
) -> list[str]:
    """构造 uv run 启动命令。

    Args:
        uv_env_dir: uv 环境目录（含 pyproject.toml）
        module: Python 模块路径（如 engines.ernie_image.server）
        extra_args: 额外命令行参数
        python: 指定 Python 版本（如 "3.11"），None 则用环境默认

    Returns:
        list[str]: 完整命令列表，可直接传给 subprocess
    """
    cmd = ["uv", "run", "--project", str(uv_env_dir)]
    if python:
        cmd.extend(["--python", python])
    cmd.append("python")
    cmd.append("-m")
    cmd.append(module)
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def decode_subprocess_output(data: bytes, encoding: Optional[str] = None) -> str:
    """解码子进程输出。

    优先用指定编码，否则 Windows 用 GBK（系统命令输出），其他平台用 UTF-8。

    Args:
        data: 原始字节
        encoding: 强制编码（None 则自动判断）

    Returns:
        str: 解码后的字符串
    """
    if not data:
        return ""
    if encoding:
        return data.decode(encoding, errors="replace")
    # Windows 系统命令（如 taskkill）输出 GBK
    if sys.platform == "win32":
        try:
            return data.decode("gbk", errors="replace")
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def find_uv_executable() -> str:
    """查找 uv 可执行文件路径。

    优先用环境变量 UV_BIN，其次 PATH 查找，最后返回 "uv"。

    Returns:
        str: uv 可执行文件路径
    """
    # 1. 环境变量指定
    env_uv = os.environ.get("UV_BIN")
    if env_uv and Path(env_uv).exists():
        return env_uv

    # 2. 返回 "uv"，让系统从 PATH 查找
    return "uv"

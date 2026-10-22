"""通用下载器抽象基类（v0.2.7+ 通用下载器重构）。

提供统一的下载器接口和数据结构，消除项目中 6 处下载器代码重复：
    - R5: httpx 流式 + Range + .incomplete 模式（torch_cuda_installer + model_downloader）
    - R2/R3/R4: huggingface_hub 下载重复（config_components_downloader + sdk downloader）

设计原则：
    - 抽象基类定义统一接口，具体实现由子类提供
    - 数据结构统一：DownloadProgress / DownloadResult
    - 支持多源 fallback、断点续传、进度上报、取消
    - 不依赖 core 层，可在 sdk / core / engine 层通用

规范来源：universal-downloader-refactor-analysis.md §3.2
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ===== 数据结构 =====


@dataclass
class DownloadProgress:
    """通用下载进度数据类。

    统一 torch_cuda_installer / model_downloader / config_components_downloader
    的进度上报格式。

    Attributes:
        downloaded_bytes: 已下载字节数
        total_bytes: 总字节数（未知时为 0）
        speed_bps: 下载速度（字节/秒）
        eta_seconds: 预计剩余时间（秒）
        phase: 阶段：pending | downloading | done | failed | cancelled
        message: 状态消息
    """

    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    phase: str = "pending"
    message: str = ""

    @property
    def progress(self) -> float:
        """进度百分比（0.0-1.0），总大小未知时为 0.0。"""
        if self.total_bytes <= 0:
            return 0.0
        return min(self.downloaded_bytes / self.total_bytes, 1.0)


@dataclass
class DownloadResult:
    """通用下载结果。

    Attributes:
        success: 是否成功
        target_path: 目标文件路径
        bytes_downloaded: 已下载字节数
        elapsed_seconds: 耗时（秒）
        error: 错误信息（失败时）
        source_url: 实际下载的 URL（多源 fallback 时记录最终成功的源）
    """

    success: bool
    target_path: Path
    bytes_downloaded: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    source_url: str = ""


# 类型别名：进度回调函数
ProgressCallback = Callable[[DownloadProgress], None]


@dataclass
class DownloadOptions:
    """下载选项（可选参数集合）。

    Attributes:
        cancel_event: 取消事件（asyncio.Event）
        progress_callback: 进度回调函数
        resume: 是否支持断点续传（默认 True）
        chunk_size: 流式读取块大小（默认 1MB）
            v0.2.14.1+ 从 128KB 提升到 1MB，避免高频小块 I/O 拖慢下载速度
            浏览器通常使用 1-4MB 缓冲区，128KB 过小导致吞吐量下降
        timeout: HTTP 超时（秒，默认 30s 连接 / 600s 读取）
        user_agent: 自定义 User-Agent
            v0.2.14.1+ 默认使用浏览器 UA，避免 HF 等服务器对非浏览器 UA 限速
            （HF 官方源对 curl/python-requests 等 UA 有明显限速，浏览器 UA 速度快 10-100 倍）
        auth_token: Bearer 认证 token（v0.2.15+ 新增）
            非 None 且非空时注入 ``Authorization: Bearer <token>`` 请求头，
            用于 HuggingFace gated 仓库下载（如 Lightricks/LTX-2.5-Diffusers）。
            token 不得写入任何日志（日志安全规范 §6.10）。
    """

    cancel_event: Optional[asyncio.Event] = None
    progress_callback: Optional[ProgressCallback] = None
    resume: bool = True
    chunk_size: int = 1024 * 1024  # 1MB（v0.2.14.1+ 从 128KB 提升）
    timeout: float = 30.0
    read_timeout: float = 600.0
    # v0.2.14.1+ 默认浏览器 UA，避免被 HF 等服务器限速
    # 模拟 Chrome 120，与真实浏览器请求一致
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    # v0.2.15+ gated 仓库下载支持：HuggingFace Bearer token
    auth_token: Optional[str] = None


# ===== 抽象基类 =====


class BaseDownloader(ABC):
    """通用下载器抽象基类。

    定义统一的下载接口，子类实现具体的下载逻辑：
        - HttpxDownloader：httpx 流式下载（统一 torch + model_downloader）
        - HfHubDownloader：huggingface_hub 下载（统一 config_components + sdk）

    使用示例：
        downloader = HttpxDownloader()
        result = await downloader.download_with_fallback(
            urls=["https://example.com/file.whl", "https://mirror.com/file.whl"],
            target_path=Path("/tmp/file.whl"),
            options=DownloadOptions(cancel_event=cancel_event),
        )
    """

    @abstractmethod
    async def download(
        self,
        url: str,
        target_path: Path,
        options: Optional[DownloadOptions] = None,
    ) -> DownloadResult:
        """下载单个 URL 到目标路径。

        Args:
            url: 下载 URL
            target_path: 目标文件路径
            options: 下载选项（取消、进度回调、断点续传等）

        Returns:
            DownloadResult: 下载结果
        """
        ...

    @abstractmethod
    async def download_with_fallback(
        self,
        urls: list,
        target_path: Path,
        options: Optional[DownloadOptions] = None,
    ) -> DownloadResult:
        """多源 fallback 下载：逐个尝试 URL 列表，失败切换下一个。

        统一 torch_cuda_installer 的多源 fallback 逻辑，使其可复用。

        Args:
            urls: URL 列表（按优先级排序）
            target_path: 目标文件路径
            options: 下载选项

        Returns:
            DownloadResult: 下载结果（成功时记录 source_url）
        """
        ...

    # ===== 通用工具方法 =====

    def _notify_progress(
        self,
        callback: Optional[ProgressCallback],
        progress: DownloadProgress,
    ) -> None:
        """通知进度回调（如果存在）。"""
        if callback is not None:
            try:
                callback(progress)
            except Exception:
                # 回调异常不应影响下载
                pass

    def _check_cancelled(
        self, cancel_event: Optional[asyncio.Event]
    ) -> bool:
        """检查是否被取消。"""
        return cancel_event is not None and cancel_event.is_set()

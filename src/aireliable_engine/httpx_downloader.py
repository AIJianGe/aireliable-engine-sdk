"""httpx 流式下载器（v0.2.7+ 通用下载器重构）。

统一 torch_cuda_installer 和 model_downloader 的 httpx 流式下载代码，
消除重复 R5：httpx.Client.stream + Range + .incomplete + 原子重命名。

设计要点：
    - 复用 BaseDownloader 抽象接口
    - 统一断点续传：HTTP Range 头 + .incomplete 文件 + 原子重命名
    - 统一进度上报：字节/速度/ETA，每秒采样
    - 统一多源 fallback：URL 列表逐个尝试，失败切换
    - 统一取消支持：asyncio.Event

规范来源：universal-downloader-refactor-analysis.md §3.2.2
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from .base_downloader import (
    BaseDownloader,
    DownloadOptions,
    DownloadProgress,
    DownloadResult,
    ProgressCallback,
)

logger = logging.getLogger(__name__)


class HttpxDownloader(BaseDownloader):
    """httpx 流式下载器。

    统一 torch_cuda_installer::_do_download 和 model_downloader::_download_one
    的重复代码，提供可复用的 httpx 流式下载 + 多源 fallback 能力。

    特性：
        - 断点续传：HTTP Range 头 + .incomplete 文件 + 原子重命名
        - 进度上报：实时字节/速度/ETA，每秒采样
        - 多源 fallback：URL 列表逐个尝试，失败自动切换
        - 取消支持：asyncio.Event
        - 服务器不支持 Range 时自动从头重写
    """

    def __init__(self):
        pass

    async def download(
        self,
        url: str,
        target_path: Path,
        options: Optional[DownloadOptions] = None,
    ) -> DownloadResult:
        """下载单个 URL 到目标路径。

        流程：
            1. 检查目标文件是否已存在（幂等性：已存在则跳过）
            2. 检查 .incomplete 文件是否存在（断点续传）
            3. httpx 流式下载（支持 Range）
            4. 下载完成后原子重命名 .incomplete → target

        Args:
            url: 下载 URL
            target_path: 目标文件路径
            options: 下载选项

        Returns:
            DownloadResult: 下载结果
        """
        if options is None:
            options = DownloadOptions()

        start_time = time.time()

        # 幂等性：目标文件已存在则跳过
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("目标文件已存在，跳过下载: %s", target_path)
            return DownloadResult(
                success=True,
                target_path=target_path,
                bytes_downloaded=target_path.stat().st_size,
                elapsed_seconds=0.0,
                source_url=url,
            )

        # 断点续传：检查 .incomplete 文件
        incomplete_path = target_path.with_suffix(target_path.suffix + ".incomplete")
        start_bytes = 0
        if options.resume and incomplete_path.exists():
            start_bytes = incomplete_path.stat().st_size
            logger.info(
                "断点续传: %s 从 %d 字节开始",
                incomplete_path.name, start_bytes,
            )

        progress = DownloadProgress(
            downloaded_bytes=start_bytes,
            total_bytes=0,
            phase="downloading",
            message=f"正在下载: {url}",
        )
        self._notify_progress(options.progress_callback, progress)

        try:
            # 在线程池中执行同步 httpx 下载
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._download_sync,
                url,
                incomplete_path,
                start_bytes,
                progress,
                options,
            )

            # 检查取消
            if self._check_cancelled(options.cancel_event):
                logger.info("下载被用户取消: %s (保留 %d 字节)", url, progress.downloaded_bytes)
                return DownloadResult(
                    success=False,
                    target_path=target_path,
                    bytes_downloaded=progress.downloaded_bytes,
                    elapsed_seconds=time.time() - start_time,
                    error="下载被用户取消",
                    source_url=url,
                )

            # 原子重命名 .incomplete → target
            incomplete_path.rename(target_path)
            elapsed = time.time() - start_time

            progress.phase = "done"
            progress.message = f"下载完成: {target_path.name}"
            self._notify_progress(options.progress_callback, progress)

            return DownloadResult(
                success=True,
                target_path=target_path,
                bytes_downloaded=progress.downloaded_bytes,
                elapsed_seconds=elapsed,
                source_url=url,
            )

        except Exception as e:
            logger.warning("从 %s 下载失败: %s", url, e)
            progress.phase = "failed"
            progress.message = f"下载失败: {e}"
            self._notify_progress(options.progress_callback, progress)

            return DownloadResult(
                success=False,
                target_path=target_path,
                bytes_downloaded=progress.downloaded_bytes,
                elapsed_seconds=time.time() - start_time,
                error=str(e),
                source_url=url,
            )

    async def download_with_fallback(
        self,
        urls: list,
        target_path: Path,
        options: Optional[DownloadOptions] = None,
    ) -> DownloadResult:
        """多源 fallback 下载：逐个尝试 URL 列表，失败切换下一个。

        统一 torch_cuda_installer 的多源 fallback 逻辑。

        Args:
            urls: URL 列表（按优先级排序）
            target_path: 目标文件路径
            options: 下载选项

        Returns:
            DownloadResult: 下载结果（成功时记录 source_url）
        """
        if options is None:
            options = DownloadOptions()

        if not urls:
            return DownloadResult(
                success=False,
                target_path=target_path,
                error="URL 列表为空",
            )

        last_error = ""
        last_result = None

        for i, url in enumerate(urls):
            # 每次尝试前重置进度（不同 URL 可能不支持 Range）
            progress = DownloadProgress(
                phase="downloading",
                message=f"尝试下载源 {i + 1}/{len(urls)}: {url}",
            )
            self._notify_progress(options.progress_callback, progress)

            # 失败后重置 .incomplete 文件（不同 URL 可能不支持 Range）
            if i > 0 and last_result is not None and not last_result.success:
                incomplete_path = target_path.with_suffix(
                    target_path.suffix + ".incomplete"
                )
                if incomplete_path.exists():
                    try:
                        incomplete_path.unlink()
                        logger.info("切换下载源，删除旧的 .incomplete 文件")
                    except Exception:
                        pass

            result = await self.download(url, target_path, options)
            last_result = result

            if result.success:
                return result

            last_error = result.error
            logger.warning(
                "从 %s 下载失败，尝试下一个 URL: %s",
                url, result.error,
            )

        # 所有 URL 都失败
        return DownloadResult(
            success=False,
            target_path=target_path,
            bytes_downloaded=last_result.bytes_downloaded if last_result else 0,
            error=f"所有下载 URL 都失败，最后错误: {last_error}",
        )

    # ===== 内部同步下载实现 =====

    def _download_sync(
        self,
        url: str,
        incomplete_path: Path,
        start_bytes: int,
        progress: DownloadProgress,
        options: DownloadOptions,
    ) -> None:
        """执行单个 URL 的同步流式下载。

        统一实现 torch_cuda_installer._do_download 和 model_downloader._download_one
        的核心下载逻辑。

        v0.2.14.1+ 性能优化：
            - 默认使用浏览器 User-Agent，避免被 HF 等服务器限速
            - chunk_size 从 128KB 提升到 1MB，减少 I/O 系统调用次数
            - 文件写入使用更大缓冲区（1MB），并定期 flush 避免内存堆积
            - 每秒 progress_callback 上报，但每 5MB 强制 flush 一次磁盘
        """
        import httpx

        mode = "ab" if start_bytes > 0 else "wb"
        incomplete_path.parent.mkdir(parents=True, exist_ok=True)

        # v0.2.14.1+ 使用 1MB 缓冲区打开文件，减少系统调用
        # buffering=0 仅对二进制模式有效，1 为行缓冲，>1 为缓冲区大小
        with open(incomplete_path, mode, buffering=1024 * 1024) as f:
            headers = {}
            if start_bytes > 0:
                headers["Range"] = f"bytes={start_bytes}-"
            # v0.2.14.1+ 默认使用浏览器 UA（在 DownloadOptions 中设置）
            # 避免 HF 服务器对非浏览器 UA 限速
            if options.user_agent:
                headers["User-Agent"] = options.user_agent
            # 增加 Accept-Encoding 头，与浏览器行为一致
            headers["Accept-Encoding"] = "identity"  # 不压缩，避免解压开销
            headers["Accept"] = "*/*"
            # v0.2.15+ gated 仓库下载支持：注入 Bearer token
            # 用于 HuggingFace gated 仓库（huggingface.co 与 hf-mirror.com
            # 镜像均识别 Authorization 头）。token 不得写入日志（§6.10）。
            if options.auth_token:
                headers["Authorization"] = f"Bearer {options.auth_token}"

            # v0.2.14.1+ httpx.Client 配置优化
            # - limits: 提升单连接保持能力，避免重连
            # - http2: 尝试启用 HTTP/2（如果安装了 h2 包），多路复用更高效
            try:
                # 尝试启用 HTTP/2（需要 pip install h2）
                client_kwargs = dict(
                    follow_redirects=True,
                    timeout=httpx.Timeout(
                        options.timeout,
                        read=options.read_timeout,
                        write=30.0,
                        connect=30.0,
                        pool=30.0,
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    ),
                    http2=True,  # 启用 HTTP/2
                )
                client = httpx.Client(**client_kwargs)
            except (ImportError, Exception) as e:
                # h2 未安装或 http2 不可用，回退到 HTTP/1.1
                logger.debug("HTTP/2 不可用，回退到 HTTP/1.1: %s", e)
                client = httpx.Client(
                    follow_redirects=True,
                    timeout=httpx.Timeout(
                        options.timeout,
                        read=options.read_timeout,
                        write=30.0,
                        connect=30.0,
                        pool=30.0,
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    ),
                )

            with client:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(
                            f"下载失败 HTTP {resp.status_code}: {resp.reason_phrase}"
                        )

                    # 服务器不支持 Range 时从头写
                    if resp.status_code == 200 and start_bytes > 0:
                        f.seek(0)
                        f.truncate()
                        start_bytes = 0
                        progress.downloaded_bytes = 0

                    # 从 Content-Range / Content-Length 推断总大小
                    content_range = resp.headers.get("content-range", "")
                    if content_range.startswith("bytes "):
                        try:
                            progress.total_bytes = int(
                                content_range.split("/")[-1]
                            )
                        except (IndexError, ValueError):
                            pass
                    elif "content-length" in resp.headers:
                        try:
                            progress.total_bytes = int(
                                resp.headers["content-length"]
                            ) + start_bytes
                        except ValueError:
                            pass

                    # 进度统计变量
                    last_time = time.time()
                    last_bytes = progress.downloaded_bytes
                    # v0.2.14.1+ 每 5MB 强制 flush 一次磁盘，避免内存堆积
                    last_flush_bytes = progress.downloaded_bytes
                    FLUSH_THRESHOLD = 5 * 1024 * 1024  # 5MB

                    # 流式读取
                    for chunk in resp.iter_bytes(chunk_size=options.chunk_size):
                        if self._check_cancelled(options.cancel_event):
                            logger.info(
                                "下载被用户取消: %s (保留 %d 字节)",
                                incomplete_path.name,
                                progress.downloaded_bytes,
                            )
                            return

                        f.write(chunk)
                        progress.downloaded_bytes += len(chunk)

                        # v0.2.14.1+ 每 5MB flush 一次磁盘
                        # 避免频繁 flush 影响速度，也避免不 flush 导致内存堆积
                        if progress.downloaded_bytes - last_flush_bytes >= FLUSH_THRESHOLD:
                            f.flush()
                            last_flush_bytes = progress.downloaded_bytes

                        # 每 1 秒更新速度和 ETA
                        now = time.time()
                        dt = now - last_time
                        if dt >= 1.0:
                            speed = (progress.downloaded_bytes - last_bytes) / dt
                            if speed > 0:
                                progress.speed_bps = speed
                                remaining = (
                                    progress.total_bytes - progress.downloaded_bytes
                                )
                                if remaining > 0:
                                    progress.eta_seconds = remaining / speed
                            last_time = now
                            last_bytes = progress.downloaded_bytes
                            self._notify_progress(
                                options.progress_callback, progress
                            )

                    # 下载完成后最终 flush
                    f.flush()

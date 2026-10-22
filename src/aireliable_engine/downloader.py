"""引擎级模型下载器抽象基类与通用组件。

职责：
    1. ``BaseEngineDownloader``：引擎级下载器抽象基类
    2. ``HFSnapshotDownloader``：HuggingFace 仓库 snapshot 下载通用组件
       （封装 ``snapshot_download`` + 镜像站选择 + 禁用 xet + 速度监控自动切换）

设计原则：
    - 各引擎实现自己的下载器，注入自己的仓库白名单和镜像 URL
    - 通用组件放 base 层，引擎只做配置注入
    - 避免把某引擎专属的下载逻辑（如百度 hf-mirror）写死在 base 层

关键修复（v0.2 引入）：
    1. **禁用 xet 协议**：huggingface_hub >= 0.30 默认走 xet 协议下载大文件，
       xet 会绕过 ``HF_ENDPOINT`` 镜像直连 ``cas-bridge.xethub.hf.co``，
       国内网络下会触发 WinError 10060。设置 ``HF_HUB_DISABLE_XET=1`` 强制走普通 HTTP。
    2. **单线程下载**：snapshot_download 默认 ``max_workers=4`` 并发拉大文件，
       hf-mirror 镜像单点限速容易触发 RST。改为 ``max_workers=1``。
    3. **子线程执行**：snapshot_download 是同步函数，用 ``asyncio.to_thread``
       包装，避免阻塞引擎进程的 asyncio 事件循环（否则会像推理一样卡死 SSE 流）。

关键修复（v0.1.2 引入）：
    4. **多镜像站列表 + 自动切换**：原实现只支持单一 ``mirror_endpoint``，
       一旦该镜像慢或失败只能报错。改为集成 ``HFMirrorSelector``：
       - 镜像优先级：huggingface.co → hf-mirror.com → 备用镜像
       - 速度持续 < 100KB/s 达 60s → 自动切换到下一个镜像
       - 当前镜像返回 308 / ConnectError → 立即切换
       - HfApi 显式传 endpoint，不依赖 constants.HF_ENDPOINT（import 时固化）
       向后兼容：仍接受 ``mirror_endpoint`` 单值参数（内部包装为单元素列表）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# 默认镜像站列表（与 core.hf_mirrors.DEFAULT_MIRRORS 保持一致）
# 不直接 import core 层（engine 进程独立），在 engine 侧复制一份
DEFAULT_HF_MIRRORS: list[str] = [
    "https://huggingface.co",          # 官方源，优先
    "https://hf-mirror.com",          # 国内社区镜像
]


# 速度监控参数
SLOW_SPEED_THRESHOLD_BPS = 100 * 1024   # 100 KB/s
SLOW_DURATION_SECONDS = 60.0            # 持续 60 秒


class _EngineHFMirrorSelector:
    """engine 进程内的简化版镜像站选择器。

    与 ``core/src/core/hf_mirrors.py::HFMirrorSelector`` 行为一致，
    但放在 base 层避免循环 import。

    特性：
        - 镜像列表按优先级排序
        - 速度持续 < 阈值达指定时长 → 切换
        - 当前镜像失败 → 立即切换
        - 切换后清空速度采样
    """

    UNAVAILABLE_COOLDOWN_SECONDS = 600.0  # 不可用冷却 10 分钟

    def __init__(self, mirrors: Optional[list[str]] = None):
        self.mirrors: list[str] = list(mirrors or DEFAULT_HF_MIRRORS)
        if not self.mirrors:
            raise ValueError("镜像站列表不能为空")
        self._current_idx: int = 0
        # 每个镜像的速度采样 + 状态
        self._speed_samples: dict[str, list[float]] = {u: [] for u in self.mirrors}
        self._slow_since: dict[str, Optional[float]] = {u: None for u in self.mirrors}
        self._marked_unavailable: dict[str, bool] = {u: False for u in self.mirrors}
        self._unavailable_since: dict[str, float] = {u: 0.0 for u in self.mirrors}
        logger.info(
            "EngineHFMirrorSelector 初始化: mirrors=%s, slow_threshold=%d KB/s, slow_duration=%.0fs",
            self.mirrors,
            SLOW_SPEED_THRESHOLD_BPS / 1024,
            SLOW_DURATION_SECONDS,
        )

    def get_current_endpoint(self) -> str:
        """获取当前 endpoint（自动跳过不可用镜像）。"""
        import time as _t
        # 检查当前镜像冷却是否已过
        cur = self.mirrors[self._current_idx]
        if self._marked_unavailable[cur]:
            elapsed = _t.time() - self._unavailable_since[cur]
            if elapsed >= self.UNAVAILABLE_COOLDOWN_SECONDS:
                self._marked_unavailable[cur] = False
                self._slow_since[cur] = None
                self._speed_samples[cur] = []
                logger.info("镜像 %s 冷却已过，恢复可用", cur)
            else:
                self._advance_to_next_available()
        return self.mirrors[self._current_idx]

    def report_speed(self, bytes_per_sec: float) -> None:
        """上报速度（下载循环每秒调一次）。"""
        import time as _t
        cur = self.mirrors[self._current_idx]
        samples = self._speed_samples[cur]
        samples.append(bytes_per_sec)
        if len(samples) > 20:
            self._speed_samples[cur] = samples[-20:]
        avg = sum(self._speed_samples[cur]) / len(self._speed_samples[cur])
        if avg < SLOW_SPEED_THRESHOLD_BPS:
            if self._slow_since[cur] is None:
                self._slow_since[cur] = _t.time()
                logger.info(
                    "镜像 %s 速度 %.1f KB/s 低于阈值，开始计时",
                    cur, avg / 1024,
                )
        else:
            if self._slow_since[cur] is not None:
                logger.info("镜像 %s 速度恢复到 %.1f KB/s", cur, avg / 1024)
            self._slow_since[cur] = None

    def should_switch(self) -> bool:
        """是否应该切换镜像。"""
        import time as _t
        cur = self.mirrors[self._current_idx]
        if self._marked_unavailable[cur]:
            return True
        if self._slow_since[cur] is None or not self._speed_samples[cur]:
            return False
        avg = sum(self._speed_samples[cur]) / len(self._speed_samples[cur])
        if avg < SLOW_SPEED_THRESHOLD_BPS:
            duration = _t.time() - self._slow_since[cur]
            if duration >= SLOW_DURATION_SECONDS:
                logger.warning(
                    "镜像 %s 持续慢速 %.0fs (avg %.1f KB/s)，触发切换",
                    cur, duration, avg / 1024,
                )
                return True
        return False

    def switch_to_next(self) -> str:
        """切换到下一个可用镜像。"""
        import time as _t
        old = self.mirrors[self._current_idx]
        self._marked_unavailable[old] = True
        self._unavailable_since[old] = _t.time()
        logger.warning("切换镜像: %s → (寻找下一个可用)", old)
        self._advance_to_next_available()
        new = self.mirrors[self._current_idx]
        logger.info("已切换到镜像: %s", new)
        return new

    def report_failure(self, error_msg: str = "") -> str:
        """上报当前镜像失败，立即切换。"""
        import time as _t
        cur = self.mirrors[self._current_idx]
        self._marked_unavailable[cur] = True
        self._unavailable_since[cur] = _t.time()
        logger.warning("镜像 %s 失败（%s），标记不可用并切换", cur, error_msg or "未知")
        self._advance_to_next_available()
        return self.mirrors[self._current_idx]

    def report_result(self, success: bool) -> None:
        """上报下载结果。"""
        cur = self.mirrors[self._current_idx]
        if success:
            self._slow_since[cur] = None
            if self._marked_unavailable[cur]:
                self._marked_unavailable[cur] = False
                logger.info("镜像 %s 下载成功，清除不可用标记", cur)

    def _advance_to_next_available(self) -> None:
        """前进到下一个可用镜像。"""
        import time as _t
        n = len(self.mirrors)
        for offset in range(1, n + 1):
            idx = (self._current_idx + offset) % n
            url = self.mirrors[idx]
            if self._marked_unavailable[url]:
                elapsed = _t.time() - self._unavailable_since[url]
                if elapsed < self.UNAVAILABLE_COOLDOWN_SECONDS:
                    continue
                self._marked_unavailable[url] = False
                self._slow_since[url] = None
                self._speed_samples[url] = []
            self._current_idx = idx
            new = self.mirrors[self._current_idx]
            self._speed_samples[new] = []
            self._slow_since[new] = None
            return
        # 全部不可用 → 重置
        logger.error("所有镜像都不可用，重置所有标记重新尝试")
        for u in self.mirrors:
            self._marked_unavailable[u] = False
            self._slow_since[u] = None
            self._speed_samples[u] = []
        self._current_idx = 0


# ===== 数据类 =====


@dataclass
class DownloadSpec:
    """单个下载任务的规格描述。

    用于 ``BaseEngineDownloader.list_download_specs()`` 返回值，
    让 core 层的 ModelDownloader 调度器能统一管理各引擎的下载需求。
    """

    file_id: str                                 # 唯一 ID（如 ernie-image-q4-k-m-gguf）
    filename: str                                # 目标文件名（snapshot 可为空）
    target_dir: Path                              # 目标目录
    source_type: str                              # huggingface_file / huggingface_snapshot / http / manual
    source_config: dict[str, Any] = field(default_factory=dict)
    expected_size: int = 0                        # 0 表示未知
    description: str = ""


@dataclass
class DownloadResult:
    """下载结果。"""

    file_id: str
    local_path: Path                              # 下载完成后的本地路径
    bytes_downloaded: int = 0
    success: bool = True
    error: Optional[str] = None


# 进度回调类型：(downloaded_bytes, total_bytes, speed_bps) -> None
ProgressCallback = Callable[[int, int, float], None]


# ===== 抽象基类 =====


class BaseEngineDownloader(ABC):
    """引擎级模型下载器抽象基类。

    每个引擎实现自己的子类，注入自己的下载源配置（仓库白名单、镜像 URL 等）。
    core 层的 ``ModelDownloader`` 调度器通过 ``list_download_specs()`` 拿到
    所有引擎的下载需求，串行调度，统一进度上报。

    子类必须实现：
        - ``list_download_specs()``：枚举本引擎所有需要下载的文件
        - ``download_one(spec, progress_cb, cancel_event)``：下载单个 spec

    子类可选实现：
        - ``get_hf_endpoint()``：返回本引擎专用的 HF 镜像（None 表示用全局默认）
        - ``ensure_pipeline_config(base_model)``：加载时调用，确保 pipeline 配置就绪
          （仅对需要 diffusers 配置组件的引擎有意义，如 ERNIE-Image）
    """

    @abstractmethod
    async def list_download_specs(self) -> list[DownloadSpec]:
        """枚举本引擎所有需要下载的文件规格。"""

    @abstractmethod
    async def download_one(
        self,
        spec: DownloadSpec,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> DownloadResult:
        """下载单个 spec。"""

    def get_hf_endpoint(self) -> Optional[str]:
        """返回本引擎专用的 HF 镜像（None 表示用全局默认）。"""
        return None

    async def ensure_pipeline_config(self, base_model: str) -> str:
        """确保 pipeline 配置就绪，返回本地路径或 repo_id。

        默认实现：返回空字符串（表示该引擎不需要预下载 pipeline 配置）。
        需要 diffusers 配置组件的引擎（如 ERNIE-Image）重写此方法。

        Args:
            base_model: base_model 标识（如 ernie-image / ernie-image-turbo）

        Returns:
            本地缓存路径（已就绪）或 repo_id（fallback 让 from_pretrained 自己拉）
        """
        return ""


# ===== 通用组件：HFSnapshotDownloader =====


class HFSnapshotDownloader:
    """HuggingFace 仓库 snapshot 下载器（通用组件）。

    封装 ``huggingface_hub.snapshot_download``，解决四个核心问题：
        1. **镜像设置**：国内环境必须走 hf-mirror.com，双保险设置（env + constants）
        2. **禁用 xet**：避免 xet 协议绕过镜像导致 WinError 10060
        3. **子线程执行**：snapshot_download 是同步函数，放到子线程避免阻塞事件循环
        4. **多镜像站列表 + 自动切换**（v0.1.2）：
           - 速度持续 < 100KB/s 达 60s → 自动切换
           - 当前镜像失败 → 立即切换
           - HfApi 显式传 endpoint，不依赖 constants 缓存

    任何需要从 HF 仓库下载多个文件的引擎都可以复用此组件，只需在构造时
    注入 repo_id 和 allow/ignore patterns。

    用法::

        # 推荐：用默认镜像列表（huggingface.co → hf-mirror.com）
        downloader = HFSnapshotDownloader(
            repo_id="Baidu/ERNIE-Image",
            allow_patterns=["*.json", "vae/*", "text_encoder/*"],
            ignore_patterns=["transformer/*", "*.bin"],
        )

        # 兼容 v0.1：仍可传单一 mirror_endpoint
        downloader = HFSnapshotDownloader(
            repo_id="Baidu/ERNIE-Image",
            allow_patterns=["*.json"],
            mirror_endpoint="https://hf-mirror.com",
        )

        local_path = await downloader.download_to_cache()
    """

    def __init__(
        self,
        repo_id: str,
        allow_patterns: list[str],
        ignore_patterns: Optional[list[str]] = None,
        mirror_endpoint: Optional[str] = None,
        mirror_endpoints: Optional[list[str]] = None,
        max_workers: int = 1,
    ):
        """
        Args:
            repo_id: HF 仓库 ID（如 ``Baidu/ERNIE-Image``）
            allow_patterns: 允许下载的文件模式（白名单）
            ignore_patterns: 忽略的文件模式（黑名单，优先级高于 allow_patterns）
            mirror_endpoint: HF 镜像地址（单一值，向后兼容 v0.1）
            mirror_endpoints: HF 镜像地址列表（v0.1.2 推荐，多镜像站自动切换）
                              优先级高于 mirror_endpoint。None 时用 DEFAULT_HF_MIRRORS。
            max_workers: 并发下载数，默认 1（单线程，避免镜像限速触发 RST）
        """
        self.repo_id = repo_id
        self.allow_patterns = list(allow_patterns)
        self.ignore_patterns = list(ignore_patterns or [])
        self.max_workers = max_workers

        # 镜像站列表（v0.1.2 多镜像）
        if mirror_endpoints:
            self.mirror_endpoints = list(mirror_endpoints)
        elif mirror_endpoint:
            # 向后兼容：单一 endpoint 包装为列表
            self.mirror_endpoints = [mirror_endpoint]
        else:
            self.mirror_endpoints = list(DEFAULT_HF_MIRRORS)

        # 兼容字段（部分老代码可能仍读 self.mirror_endpoint）
        self.mirror_endpoint = self.mirror_endpoints[0] if self.mirror_endpoints else None

        # 镜像选择器实例（速度监控 + 自动切换）
        self._selector = _EngineHFMirrorSelector(mirrors=self.mirror_endpoints)

        # v0.2.7+ 通用下载器重构：注入 HfHubDownloader 实例
        # 委托 list_repo_files_with_retry / list_repo_tree_with_retry /
        # download_file_with_retry / fallback_copy_blob_to_snapshot /
        # is_mirror_issue 等通用方法，消除重复代码（R2/R3/R4/R6）
        # 通过 duck typing 让 HfHubDownloader 兼容 _EngineHFMirrorSelector
        # （需提供 mirrors 属性 + get_current_endpoint() + report_failure()）
        from .hf_hub_downloader import HfHubDownloader
        self._hf_hub_downloader = HfHubDownloader(selector=self._selector)

    async def download_to_cache(
        self,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> str:
        """下载到 HF 本地缓存，返回本地路径。

        v0.1.2 改造：多镜像站自动切换重试。
        snapshot_download 失败时切换镜像（设置 HF_ENDPOINT + constants）重试。

        Returns:
            本地缓存目录路径（snapshot_download 返回的 snapshot path）

        Raises:
            RuntimeError: 所有镜像都失败
        """
        # 禁用 xet + 超时
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "未安装 huggingface_hub，请用 "
                "`uv run --project envs/image-ernie-image-gguf pip install huggingface_hub`"
            ) from e

        logger.info(
            "预下载 HF 仓库 snapshot: repo=%s, allow=%d, ignore=%d, mirrors=%s, workers=%d",
            self.repo_id,
            len(self.allow_patterns),
            len(self.ignore_patterns),
            self.mirror_endpoints,
            self.max_workers,
        )

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("下载已取消（启动前）")

        # 多镜像重试
        max_attempts = len(self._selector.mirrors)
        last_err: Exception = Exception("未尝试")

        for attempt in range(max_attempts):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("下载已取消")

            endpoint = self._selector.get_current_endpoint()
            # 双写 HF_ENDPOINT（snapshot_download 内部用 constants）
            os.environ["HF_ENDPOINT"] = endpoint
            try:
                import huggingface_hub.constants
                huggingface_hub.constants.HF_ENDPOINT = endpoint
            except Exception as e:
                logger.warning("设置 constants.HF_ENDPOINT 失败: %s", e)

            logger.info(
                "snapshot_download 尝试 %d/%d (endpoint=%s)",
                attempt + 1, max_attempts, endpoint,
            )

            try:
                local_path = await asyncio.to_thread(
                    snapshot_download,
                    repo_id=self.repo_id,
                    allow_patterns=self.allow_patterns,
                    ignore_patterns=self.ignore_patterns,
                    max_workers=self.max_workers,
                )
                logger.info("预下载完成: %s", local_path)
                self._selector.report_result(success=True)
                return str(local_path)

            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                # v0.2.7+ 委托给 HfHubDownloader.is_mirror_issue 静态方法
                # 统一 6 条件判断（308 / LocalEntryNotFoundError / ConnectError /
                # ConnectTimeout / TimeoutError）
                is_mirror_issue = self._hf_hub_downloader.is_mirror_issue(e)
                if is_mirror_issue and attempt < max_attempts - 1:
                    logger.warning(
                        "snapshot_download 失败（镜像问题），切换镜像重试: %s",
                        err_msg,
                    )
                    self._selector.report_failure(f"snapshot: {err_msg}")
                    continue
                else:
                    logger.error("snapshot_download 失败: %s", err_msg)
                    if not is_mirror_issue:
                        # 非镜像问题，不重试
                        break
                    # 是镜像问题但已重试完所有镜像
                    continue

        raise RuntimeError(
            f"下载 HF 仓库 snapshot 失败（所有镜像均失败，repo={self.repo_id}）: {last_err}"
        )

    async def download_subdir(
        self,
        subdir: str,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> list[str]:
        """下载指定子目录的所有文件到 HF 本地缓存。

        v0.2.7+ 改造：委托给 HfHubDownloader 通用方法。
        - list_repo_files_with_retry / list_repo_tree_with_retry /
          download_file_with_retry / fallback_copy_blob_to_snapshot /
          is_mirror_issue 均统一委托，消除重复代码（R2/R3/R4/R6）
        - 保留 _EngineHFMirrorSelector 进行速度监控 + 自动切换
        - HfApi 显式传 endpoint（不依赖 constants 缓存）

        Args:
            subdir: 子目录名（如 ``vae`` / ``text_encoder`` / ``scheduler``）
            progress_cb: 进度回调 ``(downloaded_bytes, total_bytes, speed_bps)``
            cancel_event: 取消事件，set 后下完当前文件即退出

        Returns:
            已下载文件的本地路径列表

        Raises:
            RuntimeError: 下载失败或被取消
        """
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RuntimeError(
                "未安装 huggingface_hub，请用 "
                "`uv run --project envs/image-ernie-image-gguf pip install huggingface_hub`"
            ) from e

        # v0.2.7+ 委托给 HfHubDownloader 列出仓库文件（含镜像切换重试）
        all_files = await self._hf_hub_downloader.list_repo_files_with_retry(
            repo_id=self.repo_id,
            cancel_event=cancel_event,
        )
        current_endpoint = self._selector.get_current_endpoint()
        api = HfApi(endpoint=current_endpoint)  # 可能已切换

        prefix = subdir.rstrip("/") + "/"
        target_files = [f for f in all_files if f.startswith(prefix)]
        if not target_files:
            logger.warning(
                "仓库 %s 中未找到子目录 %s 下的文件", self.repo_id, subdir,
            )
            return []

        logger.info(
            "子目录 %s 包含 %d 个文件: %s",
            subdir, len(target_files), target_files[:5],
        )

        # v0.2.7+ 委托给 HfHubDownloader 查询文件大小（含镜像切换重试）
        try:
            tree_files = await self._hf_hub_downloader.list_repo_tree_with_retry(
                repo_id=self.repo_id,
                path=subdir,
                recursive=True,
                cancel_event=cancel_event,
            )
            current_endpoint = self._selector.get_current_endpoint()
            api = HfApi(endpoint=current_endpoint)
        except Exception as e:
            logger.warning("list_repo_tree 失败，降级用 0 字节估算: %s", e)
            tree_files = []

        file_sizes: dict[str, int] = {}
        for item in tree_files:
            path = getattr(item, "path", None)
            size = getattr(item, "size", 0)
            if path and size:
                file_sizes[path] = int(size)

        total_bytes = sum(file_sizes.get(f, 0) for f in target_files)
        downloaded_bytes = 0
        downloaded_paths: list[str] = []
        start_time = asyncio.get_event_loop().time()
        last_speed_sample_ts = start_time
        last_sampled_bytes = 0

        logger.info(
            "开始下载子目录 %s/%s: 共 %d 文件，约 %.2f MB, 镜像=%s",
            self.repo_id, subdir, len(target_files),
            total_bytes / (1024 * 1024),
            current_endpoint,
        )

        # 逐个文件下载
        for idx, filename in enumerate(target_files, 1):
            if cancel_event and cancel_event.is_set():
                logger.info("下载被取消（已完成 %d/%d）", idx - 1, len(target_files))
                raise RuntimeError(f"下载被取消（已完成 {idx - 1}/{len(target_files)}）")

            file_size = file_sizes.get(filename, 0)
            logger.info(
                "下载文件 [%d/%d] %s/%s (%.2f MB) 镜像=%s",
                idx, len(target_files), self.repo_id, filename,
                file_size / (1024 * 1024),
                self._selector.get_current_endpoint(),
            )

            # v0.2.7+ 委托给 HfHubDownloader 单文件下载（含镜像切换重试 + fallback copy）
            local_path = await self._hf_hub_downloader.download_file_with_retry(
                repo_id=self.repo_id,
                filename=filename,
                cancel_event=cancel_event,
            )
            # 切换后重建 api
            new_ep = self._selector.get_current_endpoint()
            if new_ep != current_endpoint:
                current_endpoint = new_ep
                api = HfApi(endpoint=current_endpoint)

            downloaded_paths.append(str(local_path))
            downloaded_bytes += file_size
            self._selector.report_result(success=True)

            # 速度上报 + 切换检查（保留 _EngineHFMirrorSelector 的紧耦合逻辑）
            now = asyncio.get_event_loop().time()
            elapsed = now - last_speed_sample_ts
            if elapsed >= 1.0:
                delta = downloaded_bytes - last_sampled_bytes
                bps = delta / elapsed if elapsed > 0 else 0
                self._selector.report_speed(bps)
                last_speed_sample_ts = now
                last_sampled_bytes = downloaded_bytes

                if self._selector.should_switch():
                    new_ep = self._selector.switch_to_next()
                    logger.warning("速度过慢，切换镜像: %s → %s",
                                  current_endpoint, new_ep)
                    current_endpoint = new_ep
                    api = HfApi(endpoint=current_endpoint)

            # 上报进度
            if progress_cb:
                elapsed_total = asyncio.get_event_loop().time() - start_time
                speed = downloaded_bytes / elapsed_total if elapsed_total > 0 else 0
                try:
                    progress_cb(downloaded_bytes, total_bytes, speed)
                except Exception as cb_err:
                    logger.warning("进度回调异常: %s", cb_err)

        logger.info(
            "子目录下载完成: %s/%s, %d 文件, %.2f MB, 最终镜像=%s",
            self.repo_id, subdir, len(downloaded_paths),
            downloaded_bytes / (1024 * 1024),
            self._selector.get_current_endpoint(),
        )
        return downloaded_paths

    async def _list_repo_files_with_retry(self, api, subdir: str) -> list[str]:
        """列出仓库文件（含镜像切换重试）。

        .. deprecated:: v0.2.7+
            改用 ``self._hf_hub_downloader.list_repo_files_with_retry``。
            本方法保留作为兜底，仅当 ``HfHubDownloader`` 不可用时使用。
        """
        max_retries = len(self._selector.mirrors)
        last_err: Exception = Exception("未尝试")
        for attempt in range(max_retries):
            try:
                files = await asyncio.to_thread(api.list_repo_files, self.repo_id)
                return files
            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                logger.warning(
                    "list_repo_files 失败 (attempt=%d/%d, endpoint=%s): %s",
                    attempt + 1, max_retries,
                    self._selector.get_current_endpoint(), err_msg,
                )
                new_ep = self._selector.report_failure(f"list_repo_files: {err_msg}")
                from huggingface_hub import HfApi
                api = HfApi(endpoint=new_ep)
        raise RuntimeError(f"list_repo_files 所有镜像均失败: {last_err}")

    async def _list_repo_tree_with_retry(self, api, subdir: str) -> list:
        """列出仓库子目录树（含镜像切换重试）。

        .. deprecated:: v0.2.7+
            改用 ``self._hf_hub_downloader.list_repo_tree_with_retry``。
            本方法保留作为兜底。
        """
        from huggingface_hub import HfApi
        max_retries = len(self._selector.mirrors)
        last_err: Exception = Exception("未尝试")
        for attempt in range(max_retries):
            try:
                tree = await asyncio.to_thread(
                    lambda: list(api.list_repo_tree(
                        self.repo_id, path_in_repo=subdir, recursive=True
                    ))
                )
                return tree
            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                logger.warning(
                    "list_repo_tree 失败 (attempt=%d/%d, endpoint=%s): %s",
                    attempt + 1, max_retries,
                    self._selector.get_current_endpoint(), err_msg,
                )
                new_ep = self._selector.report_failure(f"list_repo_tree: {err_msg}")
                api = HfApi(endpoint=new_ep)
        raise RuntimeError(f"list_repo_tree 所有镜像均失败: {last_err}")

    async def _download_file_with_retry(
        self,
        hf_hub_download,
        filename: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> str:
        """下载单个文件（含镜像切换重试）。

        .. deprecated:: v0.2.7+
            改用 ``self._hf_hub_downloader.download_file_with_retry``。
            本方法保留作为兜底。
        """
        from huggingface_hub import HfApi
        max_retries = len(self._selector.mirrors)
        last_err: Exception = Exception("未尝试")

        for attempt in range(max_retries):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError(f"下载已取消（文件={filename}）")

            current_endpoint = self._selector.get_current_endpoint()
            # hf_hub_download 内部用 constants.HF_ENDPOINT（不读 HfApi.endpoint）
            # 必须显式更新 constants
            os.environ["HF_ENDPOINT"] = current_endpoint
            try:
                import huggingface_hub.constants
                huggingface_hub.constants.HF_ENDPOINT = current_endpoint
            except Exception as e:
                logger.warning("设置 constants.HF_ENDPOINT 失败: %s", e)

            logger.info(
                "  尝试 %d/%d (endpoint=%s) 下载 %s",
                attempt + 1, max_retries, current_endpoint, filename,
            )

            try:
                local_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=self.repo_id,
                    filename=filename,
                )

                # 验证文件落地（Windows xet/symlink 可能未落地）
                if not os.path.exists(local_path):
                    logger.warning(
                        "hf_hub_download 返回路径但文件不存在: %s，尝试 fallback copy",
                        local_path,
                    )
                    local_path = self._fallback_copy_blob_to_snapshot(filename)
                    logger.info("fallback 复制成功: %s", local_path)

                return str(local_path)

            except (FileNotFoundError, OSError) as symlink_err:
                # Windows symlink 失败 → fallback copy（不切镜像）
                logger.warning(
                    "hf_hub_download symlink 失败，尝试 fallback: %s",
                    symlink_err,
                )
                try:
                    local_path = self._fallback_copy_blob_to_snapshot(filename)
                    logger.info("fallback 复制成功: %s", local_path)
                    return str(local_path)
                except Exception as copy_err:
                    last_err = copy_err
                    logger.error("fallback 复制失败: %s", copy_err)
                    continue  # 切下个镜像重试（不同镜像 blob 可能不同）

            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                is_mirror_issue = (
                    "308" in err_msg
                    or "LocalEntryNotFoundError" in type(e).__name__
                    or "LocalEntryNotFoundError" in err_msg
                    or "ConnectError" in type(e).__name__
                    or "ConnectTimeout" in type(e).__name__
                )
                if is_mirror_issue and attempt < max_retries - 1:
                    logger.warning(
                        "文件下载失败（镜像问题），切换镜像重试: %s",
                        err_msg,
                    )
                    self._selector.report_failure(f"download {filename}: {err_msg}")
                    continue
                else:
                    logger.error("文件下载失败: %s", err_msg)
                    if not is_mirror_issue:
                        break  # 非镜像问题不重试
                    continue

        raise RuntimeError(
            f"下载文件失败（所有镜像均失败，file={filename}）: {last_err}"
        )

    def _fallback_copy_blob_to_snapshot(self, filename: str) -> str:
        """Windows symlink 失败时的 fallback：手动 copy blobs → snapshots。

        .. deprecated:: v0.2.7+
            改用 ``HfHubDownloader.fallback_copy_blob_to_snapshot`` 静态方法。
            本方法保留作为兜底。
        """
        import shutil
        from pathlib import Path

        # 构造 HF 缓存路径
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        cache_repo_dir = cache_root / f"models--{self.repo_id.replace('/', '--')}"
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots"
        refs_main = cache_repo_dir / "refs" / "main"

        if not blobs_dir.exists():
            raise RuntimeError(f"blobs 目录不存在: {blobs_dir}")

        # 找最新 blob
        blob_files = sorted(
            [f for f in blobs_dir.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not blob_files:
            raise RuntimeError(f"blobs 目录为空: {blobs_dir}")
        latest_blob = blob_files[0]

        # 读 commit hash
        commit_hash = ""
        if refs_main.exists():
            commit_hash = refs_main.read_text(encoding="utf-8").strip()
        if not commit_hash and snapshots_dir.exists():
            for entry in snapshots_dir.iterdir():
                if entry.is_dir():
                    commit_hash = entry.name
                    break
        if not commit_hash:
            raise RuntimeError(f"无法确定 commit hash: {refs_main}")

        # 复制
        target_path = snapshots_dir / commit_hash / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and target_path.stat().st_size == latest_blob.stat().st_size:
            return str(target_path)

        shutil.copy2(latest_blob, target_path)
        return str(target_path)

    async def check_reachable(self, timeout: float = 5.0) -> bool:
        """预检 HF 镜像是否可达（遍历镜像列表，找到第一个可达的）。

        v0.1.2 改造：不再只检测单一 mirror_endpoint，而是遍历 selector.mirrors
        列表，找到第一个可达的镜像并把它设为当前镜像。这样调用方就能直接用
        ``get_current_endpoint()`` 拿到可用镜像。

        Args:
            timeout: 单次请求超时秒数

        Returns:
            True 表示至少一个镜像可达，False 表示全部不可达
        """
        import httpx
        for endpoint in self._selector.mirrors:
            url = f"{endpoint}/{self.repo_id}"
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.head(url)
                    if resp.status_code < 500:
                        # 找到可达镜像，把 selector 的当前指针移到它
                        while self._selector.get_current_endpoint() != endpoint:
                            # 还没轮到，切换一次
                            self._selector.switch_to_next()
                            if self._selector.get_current_endpoint() == endpoint:
                                break
                        logger.info("镜像可达: %s (status=%d)", endpoint, resp.status_code)
                        return True
            except Exception as e:
                logger.warning("镜像 %s 不可达: %s", endpoint, e)
                continue
        logger.error("所有镜像均不可达 (repo=%s)", self.repo_id)
        return False

    def _setup_hf_endpoint(self) -> None:
        """设置 HF 镜像（双保险）。

        v0.1.2 改造：使用 selector.get_current_endpoint() 拿到当前镜像，
        不再使用 self.mirror_endpoint 单值。

        1. ``os.environ['HF_ENDPOINT']``：影响 diffusers / transformers / datasets
        2. ``huggingface_hub.constants.HF_ENDPOINT``：影响直接调 huggingface_hub 的代码

        注意：constants 在 import 时固化，运行时改 env 不会重新计算 constants，
        所以这里双写仅作为兜底；显式传 ``HfApi(endpoint=...)`` 才是可靠方案。
        """
        endpoint = self._selector.get_current_endpoint()
        if not endpoint:
            return

        os.environ["HF_ENDPOINT"] = endpoint
        logger.info("设置 HF_ENDPOINT (env): %s", endpoint)

        try:
            import huggingface_hub.constants
            huggingface_hub.constants.HF_ENDPOINT = endpoint
            logger.info(
                "设置 HF_ENDPOINT (huggingface_hub.const): %s",
                endpoint,
            )
        except Exception as e:
            logger.warning(
                "设置 huggingface_hub.constants.HF_ENDPOINT 失败: %s", e
            )

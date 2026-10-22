"""huggingface_hub 下载工具类（v0.2.7+ 通用下载器重构）。

统一 core/.../config_components_downloader.py 和
sdk/.../downloader.py::HFSnapshotDownloader 的 HF 下载重复代码：
    - R2: _fallback_copy_blob_to_snapshot（2 份）
    - R3: _list_repo_files_with_retry / _list_repo_tree_with_retry（2 份）
    - R4/R6: _download_file_with_retry + is_mirror_issue（2 份方法 + 3 份内联表达式）

设计原则：
    - 不继承 BaseDownloader（HF 按 repo_id/filename 下载，与 URL 下载模式不同）
    - 提供通用的 HF 下载方法，core/sdk 层可直接复用
    - 通过 duck typing 兼容 core 的 HFMirrorSelector 和 sdk 的 MirrorSelector
    - 静态方法（is_mirror_issue / fallback_copy_blob_to_snapshot）可独立调用
    - 实例方法（list_repo_files_with_retry 等）依赖 selector 实例

规范来源：universal-downloader-refactor-analysis.md §3.2.4
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from .base_downloader import ProgressCallback

logger = logging.getLogger(__name__)


class HfHubDownloader:
    """huggingface_hub 下载工具类。

    统一 config_components_downloader 和 HFSnapshotDownloader 的 HF 下载逻辑，
    消除 4 处重复代码（R2/R3/R4/R6）。

    特性：
        - 镜像重试：失败时自动切换镜像
        - 镜像问题检测：is_mirror_issue 静态方法统一 6 条件
        - Windows symlink fallback：手动 copy blobs → snapshots
        - 取消支持：asyncio.Event
        - 进度回调：可选的 ProgressCallback

    使用示例::

        # 静态方法（直接调用）
        if HfHubDownloader.is_mirror_issue(error):
            switch_mirror()

        path = HfHubDownloader.fallback_copy_blob_to_snapshot(
            repo_id="Baidu/ERNIE-Image", filename="vae/config.json",
        )

        # 实例方法（带镜像重试）
        from aireliable_engine import get_hf_mirror_selector
        downloader = HfHubDownloader(selector=get_hf_mirror_selector())
        files = await downloader.list_repo_files_with_retry("Baidu/ERNIE-Image")
        local_path = await downloader.download_file_with_retry(
            repo_id="Baidu/ERNIE-Image",
            filename="vae/config.json",
        )
    """

    def __init__(
        self,
        selector=None,
        hf_cache_root: Optional[Path] = None,
    ):
        """
        Args:
            selector: 镜像选择器实例（duck typing）。
                需提供 ``mirrors`` 属性和 ``get_current_endpoint()`` /
                ``report_failure(error_msg)`` 方法。None 时使用 sdk 全局单例。
            hf_cache_root: HF 缓存根目录。None 时使用默认值
                （``~/.cache/huggingface/hub``）。
        """
        # 延迟导入避免循环依赖
        if selector is None:
            from .mirror_selector import get_hf_mirror_selector
            selector = get_hf_mirror_selector()
        self._selector = selector
        self._hf_cache_root = hf_cache_root or self.get_default_hf_cache_root()

    # ===== 静态工具方法 =====

    @staticmethod
    def is_mirror_issue(error: Exception) -> bool:
        """判断异常是否为镜像问题（应切换镜像重试）。

        统一 3 处重复的内联表达式：
            - core config_components_downloader._download_single_file_with_retry
              （5 条件）
            - sdk downloader.download_to_cache（6 条件，含 TimeoutError）
            - sdk downloader._download_file_with_retry（5 条件）

        采用最宽松的 6 条件版本（含 TimeoutError），减少漏判。

        Args:
            error: hf_hub_download / list_repo_files 等抛出的异常

        Returns:
            True 表示是镜像问题，应切换镜像重试；False 表示非镜像问题
            （如磁盘满、权限等），不应重试。
        """
        err_msg = str(error)
        err_type = type(error).__name__
        return (
            "308" in err_msg                              # 永久重定向（镜像配置问题）
            or "LocalEntryNotFoundError" in err_type       # 本地缓存不存在
            or "LocalEntryNotFoundError" in err_msg
            or "ConnectError" in err_type                  # 连接失败
            or "ConnectTimeout" in err_type                # 连接超时
            or "TimeoutError" in err_type                  # 通用超时（sdk 额外条件）
        )

    @staticmethod
    def get_default_hf_cache_root() -> Path:
        """获取默认 HF 缓存根目录。

        优先级：
            1. ``HF_HOME`` 环境变量（如 ``/data/hf``，则缓存根为 ``/data/hf/hub``）
            2. ``TRANSFORMERS_CACHE`` 环境变量（旧版兼容）
            3. ``~/.cache/huggingface/hub``（默认）
        """
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home) / "hub"
        transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
        if transformers_cache:
            return Path(transformers_cache)
        return Path.home() / ".cache" / "huggingface" / "hub"

    @staticmethod
    def repo_id_to_cache_dir(
        repo_id: str,
        hf_cache_root: Optional[Path] = None,
    ) -> Path:
        """根据 repo_id 计算本地缓存目录路径。

        Args:
            repo_id: HF 仓库 ID（如 ``Baidu/ERNIE-Image``）
            hf_cache_root: HF 缓存根目录，None 时用默认值

        Returns:
            缓存目录路径（如 ``~/.cache/huggingface/hub/models--Baidu--ERNIE-Image``）
        """
        root = hf_cache_root or HfHubDownloader.get_default_hf_cache_root()
        return root / f"models--{repo_id.replace('/', '--')}"

    @staticmethod
    def fallback_copy_blob_to_snapshot(
        repo_id: str,
        filename: str,
        hf_cache_root: Optional[Path] = None,
    ) -> str:
        """Windows symlink 失败时的 fallback：手动 copy blobs → snapshots。

        统一 core (config_components_downloader._fallback_copy_blob_to_snapshot)
        和 sdk (HFSnapshotDownloader._fallback_copy_blob_to_snapshot) 两份实现。

        背景：
            Windows 未开 Developer Mode 时不能创建符号链接，
            hf_hub_download 在最后一步创建
            ``symlink(blobs/<etag> → snapshots/<hash>/<file>)`` 会失败抛
            ``FileNotFoundError``。但文件实际已下载到 ``blobs/<etag>``。

        步骤：
            1. 找到 ``blobs/`` 目录下最新的文件（按 mtime 排序）
            2. 从 ``refs/main`` 读 commit hash，fallback 取 snapshots 下第一个子目录
            3. 构造 ``snapshots/<hash>/<filename>`` 路径
            4. 若目标已存在且大小一致，跳过复制
            5. ``shutil.copy2`` 复制 blob → snapshots

        Args:
            repo_id: HF 仓库 ID（如 ``Baidu/ERNIE-Image-Turbo``）
            filename: 仓库内相对路径（如 ``vae/config.json``）
            hf_cache_root: HF 缓存根目录，None 时用默认值

        Returns:
            snapshots 中的目标文件路径（str）

        Raises:
            RuntimeError: blobs 目录不存在/为空，或无法确定 commit hash
        """
        root = hf_cache_root or HfHubDownloader.get_default_hf_cache_root()
        cache_repo_dir = HfHubDownloader.repo_id_to_cache_dir(repo_id, root)
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots"
        refs_main = cache_repo_dir / "refs" / "main"

        if not blobs_dir.exists():
            raise RuntimeError(f"blobs 目录不存在: {blobs_dir}")

        # 1. 找最新的 blob 文件（刚 hf_hub_download 下载完的）
        blob_files = sorted(
            [f for f in blobs_dir.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not blob_files:
            raise RuntimeError(f"blobs 目录为空: {blobs_dir}")
        latest_blob = blob_files[0]
        logger.info(
            "fallback 找到最新 blob: %s (%.2f KB, mtime=%d)",
            latest_blob.name,
            latest_blob.stat().st_size / 1024,
            latest_blob.stat().st_mtime,
        )

        # 2. 从 refs/main 读 commit hash
        commit_hash = ""
        if refs_main.exists():
            commit_hash = refs_main.read_text(encoding="utf-8").strip()
        if not commit_hash:
            # fallback：取 snapshots 下第一个子目录名
            if snapshots_dir.exists():
                for entry in snapshots_dir.iterdir():
                    if entry.is_dir():
                        commit_hash = entry.name
                        break
        if not commit_hash:
            raise RuntimeError(f"无法确定 commit hash: {refs_main}")

        # 3. 构造 snapshots 目标路径
        target_path = snapshots_dir / commit_hash / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. 如果目标已存在且大小一致，跳过复制
        if (
            target_path.exists()
            and target_path.stat().st_size == latest_blob.stat().st_size
        ):
            logger.info(
                "snapshots 目标已存在且大小匹配，跳过复制: %s", target_path
            )
            return str(target_path)

        # 5. 复制 blob → snapshots
        shutil.copy2(latest_blob, target_path)
        logger.info(
            "复制完成: %s (%.2f KB → %s)",
            latest_blob.name,
            latest_blob.stat().st_size / 1024,
            target_path,
        )
        return str(target_path)

    # ===== 实例方法（带镜像重试）=====

    def _get_selector_mirrors(self) -> list:
        """获取 selector 的镜像列表（duck typing 兼容）。"""
        # core HFMirrorSelector.mirrors 是 list[str]
        # sdk MirrorSelector.mirrors 是 list[MirrorSource]
        mirrors = getattr(self._selector, "mirrors", None)
        if mirrors is None:
            return []
        return list(mirrors)

    def _get_current_endpoint(self) -> str:
        """获取当前镜像 endpoint（duck typing 兼容）。"""
        # sdk MirrorSelector 用 get_current_url()
        if hasattr(self._selector, "get_current_endpoint"):
            return self._selector.get_current_endpoint()
        if hasattr(self._selector, "get_current_url"):
            url = self._selector.get_current_url()
            return url or ""
        return ""

    def _report_failure(self, error_msg: str) -> str:
        """上报镜像失败并切换（duck typing 兼容）。"""
        if hasattr(self._selector, "report_failure"):
            result = self._selector.report_failure(error_msg)
            # core HFMirrorSelector.report_failure 返回 str（新 endpoint）
            if isinstance(result, str):
                return result
            # sdk MirrorSelector 无返回值，主动获取新 endpoint
            return self._get_current_endpoint()
        return self._get_current_endpoint()

    async def list_repo_files_with_retry(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> list[str]:
        """列出仓库文件（带镜像重试）。

        统一 core (config_components_downloader._list_repo_files_with_retry)
        和 sdk (HFSnapshotDownloader._list_repo_files_with_retry) 两份实现。

        Args:
            repo_id: HF 仓库 ID
            revision: 仓库版本（None 时用默认 main）
            cancel_event: 取消事件

        Returns:
            仓库文件路径列表

        Raises:
            RuntimeError: 所有镜像都失败
            asyncio.CancelledError: 被取消
        """
        from huggingface_hub import HfApi

        mirrors = self._get_selector_mirrors()
        max_retries = len(mirrors) or 1
        last_err: Exception = Exception("未尝试")

        # 初始 api 实例（用当前 endpoint）
        current_endpoint = self._get_current_endpoint()
        api = HfApi(endpoint=current_endpoint) if current_endpoint else HfApi()

        for attempt in range(max_retries):
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError("list_repo_files 已取消")

            try:
                kwargs = {"repo_id": repo_id}
                if revision:
                    kwargs["revision"] = revision
                files = await asyncio.to_thread(api.list_repo_files, **kwargs)
                return files
            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                logger.warning(
                    "list_repo_files 失败 (attempt=%d/%d, endpoint=%s): %s",
                    attempt + 1, max_retries,
                    self._get_current_endpoint(), err_msg,
                )
                # 切镜像
                new_endpoint = self._report_failure(f"list_repo_files: {err_msg}")
                api = HfApi(endpoint=new_endpoint)

        raise RuntimeError(f"list_repo_files 所有镜像均失败: {last_err}")

    async def list_repo_tree_with_retry(
        self,
        repo_id: str,
        path: str = "",
        recursive: bool = True,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> list:
        """列出仓库子目录树（带镜像重试）。

        统一 core (config_components_downloader._list_repo_tree_with_retry)
        和 sdk (HFSnapshotDownloader._list_repo_tree_with_retry) 两份实现。

        Args:
            repo_id: HF 仓库 ID
            path: 仓库内子路径（空字符串表示根目录）
            recursive: 是否递归列出
            cancel_event: 取消事件

        Returns:
            仓库树条目列表（RepoFile / RepoFolder / RepoSibling）

        Raises:
            RuntimeError: 所有镜像都失败
            asyncio.CancelledError: 被取消
        """
        from huggingface_hub import HfApi

        mirrors = self._get_selector_mirrors()
        max_retries = len(mirrors) or 1
        last_err: Exception = Exception("未尝试")

        current_endpoint = self._get_current_endpoint()
        api = HfApi(endpoint=current_endpoint) if current_endpoint else HfApi()

        for attempt in range(max_retries):
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError("list_repo_tree 已取消")

            try:
                tree_items = await asyncio.to_thread(
                    lambda: list(api.list_repo_tree(
                        repo_id, path_in_repo=path, recursive=recursive,
                    ))
                )
                return tree_items
            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                logger.warning(
                    "list_repo_tree 失败 (attempt=%d/%d, endpoint=%s): %s",
                    attempt + 1, max_retries,
                    self._get_current_endpoint(), err_msg,
                )
                new_endpoint = self._report_failure(f"list_repo_tree: {err_msg}")
                api = HfApi(endpoint=new_endpoint)

        raise RuntimeError(f"list_repo_tree 所有镜像均失败: {last_err}")

    async def download_file_with_retry(
        self,
        repo_id: str,
        filename: str,
        target_dir: Optional[Path] = None,
        cancel_event: Optional[asyncio.Event] = None,
        progress_callback: Optional[ProgressCallback] = None,
        hf_cache_root: Optional[Path] = None,
    ) -> str:
        """下载单个文件（带镜像重试 + fallback copy blob）。

        统一 core (config_components_downloader._download_single_file_with_retry)
        和 sdk (HFSnapshotDownloader._download_file_with_retry) 两份实现。

        简化说明：
            - 本方法不实现 core 版本的速度监控（blobs 目录大小采样）逻辑
              因为速度监控依赖 selector.report_speed / should_switch，
              且 core 版本通过 list 引用包装与外层 _run_download 同步状态，
              抽象后无法保留这种紧耦合。
            - 速度监控可作为后续扩展（通过 progress_callback 注入）。
            - 当前实现：失败时按 is_mirror_issue 决定是否切镜像重试。

        Args:
            repo_id: HF 仓库 ID
            filename: 仓库内文件路径（如 ``vae/config.json``）
            target_dir: 期望的最终落地目录（仅用于日志提示，实际路径由
                hf_hub_download 决定，通常在 HF 缓存目录下）。
            cancel_event: 取消事件
            progress_callback: 进度回调（可选，hf_hub_download 不暴露进度，
                仅在开始/完成/失败时通知）
            hf_cache_root: HF 缓存根目录（None 时用实例默认值）

        Returns:
            下载完成的本地文件路径（str）

        Raises:
            RuntimeError: 所有镜像都失败或被取消
            asyncio.CancelledError: 被取消
        """
        from huggingface_hub import hf_hub_download

        mirrors = self._get_selector_mirrors()
        max_retries = len(mirrors) or 1
        last_err: Exception = Exception("未尝试")
        root = hf_cache_root or self._hf_cache_root

        # 进度通知：开始
        if progress_callback:
            from .base_downloader import DownloadProgress
            progress_callback(DownloadProgress(
                phase="downloading",
                message=f"开始下载 {repo_id}/{filename}",
            ))

        for attempt in range(max_retries):
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError(
                    f"下载已取消（文件={filename}）"
                )

            current_endpoint = self._get_current_endpoint()
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
                    repo_id=repo_id,
                    filename=filename,
                )

                # 验证文件落地（Windows xet/symlink 可能未落地）
                if not os.path.exists(local_path):
                    logger.warning(
                        "hf_hub_download 返回路径但文件不存在: %s，尝试 fallback copy",
                        local_path,
                    )
                    local_path = self.fallback_copy_blob_to_snapshot(
                        repo_id=repo_id,
                        filename=filename,
                        hf_cache_root=root,
                    )
                    logger.info("fallback 复制成功: %s", local_path)

                # 进度通知：完成
                if progress_callback:
                    from .base_downloader import DownloadProgress
                    progress_callback(DownloadProgress(
                        phase="done",
                        message=f"下载完成: {filename}",
                    ))

                return str(local_path)

            except (FileNotFoundError, OSError) as symlink_err:
                # Windows symlink 失败 → fallback copy（不切镜像）
                logger.warning(
                    "hf_hub_download symlink 失败，尝试 fallback: %s",
                    symlink_err,
                )
                try:
                    local_path = self.fallback_copy_blob_to_snapshot(
                        repo_id=repo_id,
                        filename=filename,
                        hf_cache_root=root,
                    )
                    logger.info("fallback 复制成功: %s", local_path)
                    if progress_callback:
                        from .base_downloader import DownloadProgress
                        progress_callback(DownloadProgress(
                            phase="done",
                            message=f"下载完成（fallback）: {filename}",
                        ))
                    return str(local_path)
                except Exception as copy_err:
                    last_err = copy_err
                    logger.error("fallback 复制失败: %s", copy_err)
                    continue  # 切下个镜像重试（不同镜像 blob 可能不同）

            except Exception as e:
                last_err = e
                err_msg = str(e)[:200]
                if self.is_mirror_issue(e) and attempt < max_retries - 1:
                    logger.warning(
                        "文件下载失败（镜像问题），切换镜像重试: endpoint=%s, "
                        "file=%s, err=%s",
                        current_endpoint, filename, err_msg,
                    )
                    self._report_failure(f"download {filename}: {err_msg}")
                    continue
                else:
                    logger.error(
                        "文件下载失败（非镜像问题或已达最大重试）: %s, err=%s",
                        filename, err_msg,
                    )
                    if not self.is_mirror_issue(e):
                        break  # 非镜像问题不重试
                    continue

        # 进度通知：失败
        if progress_callback:
            from .base_downloader import DownloadProgress
            progress_callback(DownloadProgress(
                phase="failed",
                message=f"下载失败: {filename} ({last_err})",
            ))

        raise RuntimeError(
            f"下载文件失败（所有镜像均失败，file={filename}）: {last_err}"
        )

    # ===== 辅助方法 =====

    def get_current_download_bytes(
        self,
        repo_id: str,
        filename: str,
        hf_cache_root: Optional[Path] = None,
    ) -> int:
        """估算当前 hf_hub_download 已下载的字节数。

        通过查看 blobs 目录下最新文件的大小来估算。
        不精确，但够用于速度监控触发切换。

        Args:
            repo_id: HF 仓库 ID
            filename: 仓库内文件路径（仅用于日志）
            hf_cache_root: HF 缓存根目录

        Returns:
            已下载字节数（找不到时返回 0）
        """
        try:
            root = hf_cache_root or self._hf_cache_root
            cache_repo_dir = self.repo_id_to_cache_dir(repo_id, root)
            blobs_dir = cache_repo_dir / "blobs"
            if not blobs_dir.exists():
                return 0
            # 找最新的 blob 文件
            blob_files = [f for f in blobs_dir.iterdir() if f.is_file()]
            if not blob_files:
                return 0
            latest = max(blob_files, key=lambda f: f.stat().st_mtime)
            return latest.stat().st_size
        except Exception:
            return 0

    async def check_reachable(
        self,
        repo_id: str,
        timeout: float = 5.0,
    ) -> bool:
        """预检 HF 镜像是否可达（遍历镜像列表，找到第一个可达的）。

        Args:
            repo_id: HF 仓库 ID（用于构造测试 URL）
            timeout: 单次请求超时秒数

        Returns:
            True 表示至少一个镜像可达，False 表示全部不可达
        """
        import httpx

        mirrors = self._get_selector_mirrors()
        if not mirrors:
            return False

        for mirror in mirrors:
            # sdk MirrorSelector.mirrors 是 list[MirrorSource]
            # core HFMirrorSelector.mirrors 是 list[str]
            url = getattr(mirror, "url", None) or str(mirror)
            test_url = f"{url.rstrip('/')}/{repo_id}"
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True,
                ) as client:
                    resp = await client.head(test_url)
                    if resp.status_code < 500:
                        return True
            except Exception as e:
                logger.warning("镜像 %s 不可达: %s", url, e)
                continue

        return False

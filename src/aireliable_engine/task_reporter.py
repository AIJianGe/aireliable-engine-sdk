"""引擎任务上报 SDK（v0.3.4+ 新增）。

本模块供**引擎子进程**与**平台 gateway** 通信，上报任务状态变化，
让 gateway 维护统一的任务视图（含直连引擎调用 + gateway 转发调用）。

与 `vram_manager.PlatformVramManager` 的区别：
    - `PlatformVramManager`：引擎 → 平台 core，上报模型加载/卸载（显存状态）
    - `PlatformTaskReporter`：引擎 → 平台 gateway，上报任务进度/成功/失败（任务状态）

通信方式：HTTP（引擎子进程 → gateway REST API）
    - POST /v1/tasks/internal                          — upsert 创建/更新任务
    - POST /v1/tasks/internal/{task_id}/progress       — 上报进度
    - POST /v1/tasks/internal/{task_id}/succeeded      — 上报成功 + result
    - POST /v1/tasks/internal/{task_id}/failed         — 上报失败 + error

环境变量依赖：
    - AIRELIABLE_GATEWAY_URL：平台 gateway 地址（如 http://127.0.0.1:8000）
      平台启动引擎子进程时由 engine_manager._build_engine_subprocess_env 注入
      详见 TASK_CENTER_V034_ANALYSIS.md T4
    - AIRELIABLE_ENGINE_ID：当前引擎 ID
      （可选，若未注入则需在构造函数显式传入）

使用示例（4 引擎 progress_reporter.py / engine.py）：

    from aireliable_engine import PlatformTaskReporter

    reporter = PlatformTaskReporter()  # 自动从环境变量读 gateway_url 和 engine_id

    # /infer 入口：上报任务创建
    await reporter.report_created(
        task_id="fc_a1b2c3d4e5f67890",
        request_summary={"prompt": "猫在弹钢琴", "mode": "v2a"},
        variant="foleycrafter-v1",
        task_type="sound_effect",  # v0.2.16+ 可选，与任务中心筛选语义对齐
    )

    # 推理过程中：上报进度
    await reporter.report_progress(
        task_id="fc_a1b2c3d4e5f67890",
        progress=0.5,
        stage="diffusion",
    )

    # 推理完成：上报成功
    await reporter.report_succeeded(
        task_id="fc_a1b2c3d4e5f67890",
        result={"audios": ["/output/audio/fc_a1b2c3d4e5f67890.wav"]},
    )

    # 推理失败：上报失败
    await reporter.report_failed(
        task_id="fc_a1b2c3d4e5f67890",
        error={"type": "RuntimeError", "message": "CUDA OOM"},
    )

容错原则：
    - gateway 不可达时所有方法降级（返回 False），不阻断引擎推理
    - engine_id 未设置时跳过上报（debug 日志）
    - 上报失败只记 warning 日志，不影响主流程

详见：TASK_CENTER_V034_ANALYSIS.md T5
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class PlatformTaskReporter:
    """引擎任务上报器（v0.3.4+ 新增）。

    引擎子进程通过本类向平台 gateway 上报任务状态变化，
    让 gateway 维护统一的任务视图（含直连引擎调用 + gateway 转发调用）。

    设计原则：
        1. **容错**：gateway 不可达时所有方法降级（返回 False），不阻断引擎
        2. **异步**：所有 HTTP 调用都是 async，不阻塞引擎主流程
        3. **幂等**：upsert 语义，重复调用 report_created 安全
        4. **轻量**：单 httpx.AsyncClient 复用连接池

    Attributes:
        gateway_url: 平台 gateway 地址（如 http://127.0.0.1:8000）
        engine_id: 当前引擎 ID
        client: httpx 异步客户端（复用连接池）
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        engine_id: Optional[str] = None,
        timeout: float = 3.0,
    ) -> None:
        """初始化 PlatformTaskReporter。

        Args:
            gateway_url: 平台 gateway 地址；None 时从 AIRELIABLE_GATEWAY_URL 环境变量读取
            engine_id: 引擎 ID；None 时从 AIRELIABLE_ENGINE_ID 环境变量读取
            timeout: HTTP 请求超时秒数（默认 3s，容错优先，避免阻塞 SSE 流）
        """
        self.gateway_url: str = (
            gateway_url
            or os.environ.get("AIRELIABLE_GATEWAY_URL", "")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.engine_id: str = engine_id or os.environ.get("AIRELIABLE_ENGINE_ID", "")
        self._timeout: float = timeout
        self._client: Optional[httpx.AsyncClient] = None

        if not self.engine_id:
            logger.warning(
                "PlatformTaskReporter: engine_id 未设置（AIRELIABLE_ENGINE_ID 环境变量未注入），"
                "上报的任务将缺少 engine_id 字段"
            )
        if not self.gateway_url:
            logger.warning("PlatformTaskReporter: gateway_url 未设置")

        logger.debug(
            "PlatformTaskReporter 初始化: gateway_url=%s engine_id=%s",
            self.gateway_url, self.engine_id,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """获取复用的 httpx 客户端（惰性创建）。"""
        # 捕获主 loop（供工作线程的 fire_report_* 跨线程提交任务）
        # 在 _get_client 中调用，因为本方法一定在 async 上下文中
        global _main_loop
        if _main_loop is None or not _main_loop.is_running():
            try:
                _set_main_loop(asyncio.get_running_loop())
            except RuntimeError:
                pass

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.gateway_url,
                timeout=httpx.Timeout(self._timeout, connect=1.5),
                # trust_env=False：引擎→gateway 平台内部回环，不走系统代理
                trust_env=False,
            )
        return self._client

    async def close(self) -> None:
        """关闭客户端（引擎停止时调用）。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ===== 任务生命周期上报 =====

    async def report_created(
        self,
        task_id: str,
        request_summary: Optional[dict] = None,
        variant: str = "",
        stage: str = "created",
        task_type: str = "",
    ) -> bool:
        """上报任务创建（upsert 语义）（v0.3.4+ 新增）。

        若 task_id 已存在则更新状态为 queued，否则创建新记录。

        Args:
            task_id: 任务 ID（由引擎 /infer 入口生成）
            request_summary: 请求摘要（如 {"prompt": "...", "mode": "..."}）
            variant: variant 名
            stage: 当前阶段（默认 "created"）
            task_type: 任务类型（v0.2.16+ 新增，如 image_generation / image_editing /
                tts / chat 等，与 gateway 任务中心筛选语义对齐）。gateway 侧仅在
                **插入新记录**时写入 task_type（upsert 更新已有记录不覆盖），
                因此 gateway 转发链路（gateway 先建任务带 task_type）与引擎
                上报链路（引擎直连时带 task_type）互不污染。

        Returns:
            bool: True 表示 gateway 已记录；False 表示失败（gateway 不可达）
        """
        payload: dict[str, Any] = {
            "task_id": task_id,
            "status": "queued",
            "stage": stage,
            "engine_id": self.engine_id,
            "variant": variant,
            "source": "engine_subprocess",
        }
        if request_summary is not None:
            payload["request"] = request_summary
        if task_type:
            payload["task_type"] = task_type

        return await self._post("/v1/tasks/internal", payload, op="report_created")

    async def report_running(
        self,
        task_id: str,
        stage: str = "running",
    ) -> bool:
        """上报任务进入运行状态（v0.3.4+ 新增）。

        Args:
            task_id: 任务 ID
            stage: 当前阶段（默认 "running"）

        Returns:
            bool: True 表示 gateway 已记录；False 表示失败
        """
        payload: dict[str, Any] = {
            "task_id": task_id,
            "status": "running",
            "stage": stage,
            "engine_id": self.engine_id,
            "source": "engine_subprocess",
        }
        return await self._post("/v1/tasks/internal", payload, op="report_running")

    async def report_progress(
        self,
        task_id: str,
        progress: float,
        stage: str = "",
    ) -> bool:
        """上报任务进度（v0.3.4+ 新增）。

        Args:
            task_id: 任务 ID
            progress: 进度 0.0 - 1.0
            stage: 当前阶段（可选）

        Returns:
            bool: True 表示 gateway 已记录；False 表示失败
        """
        payload: dict[str, Any] = {
            "progress": float(progress),
        }
        if stage:
            payload["stage"] = stage

        return await self._post(
            f"/v1/tasks/internal/{task_id}/progress",
            payload,
            op="report_progress",
        )

    async def report_succeeded(
        self,
        task_id: str,
        result: Optional[dict] = None,
    ) -> bool:
        """上报任务成功（v0.3.4+ 新增）。

        Args:
            task_id: 任务 ID
            result: 结果字典（含 URLs 等字段，如 {"audios": [...], "audio_url": "..."}）

        Returns:
            bool: True 表示 gateway 已记录；False 表示失败
        """
        payload: dict[str, Any] = {"result": result or {}}
        return await self._post(
            f"/v1/tasks/internal/{task_id}/succeeded",
            payload,
            op="report_succeeded",
        )

    async def report_failed(
        self,
        task_id: str,
        error: Optional[dict] = None,
    ) -> bool:
        """上报任务失败（v0.3.4+ 新增）。

        Args:
            task_id: 任务 ID
            error: 错误字典（如 {"type": "RuntimeError", "message": "..."}）

        Returns:
            bool: True 表示 gateway 已记录；False 表示失败
        """
        payload: dict[str, Any] = {"error": error or {}}
        return await self._post(
            f"/v1/tasks/internal/{task_id}/failed",
            payload,
            op="report_failed",
        )

    # ===== fire-and-forget 包装（不阻塞 SSE 流）=====

    def fire_report_created(
        self,
        task_id: str,
        request_summary: Optional[dict] = None,
        variant: str = "",
        stage: str = "created",
        task_type: str = "",
    ) -> None:
        """fire-and-forget 版本的 report_created（v0.3.4+ 新增）。

        用于在 SSE 流处理中触发上报而不阻塞主流程。
        内部用 asyncio.create_task，调用方无需 await。
        """
        self._fire(
            self.report_created(
                task_id=task_id,
                request_summary=request_summary,
                variant=variant,
                stage=stage,
                task_type=task_type,
            ),
            op="report_created",
        )

    def fire_report_running(self, task_id: str, stage: str = "running") -> None:
        """fire-and-forget 版本的 report_running（v0.3.4+ 新增）。"""
        self._fire(
            self.report_running(task_id=task_id, stage=stage),
            op="report_running",
        )

    def fire_report_progress(
        self,
        task_id: str,
        progress: float,
        stage: str = "",
    ) -> None:
        """fire-and-forget 版本的 report_progress（v0.3.4+ 新增）。"""
        self._fire(
            self.report_progress(task_id=task_id, progress=progress, stage=stage),
            op="report_progress",
        )

    def fire_report_succeeded(
        self,
        task_id: str,
        result: Optional[dict] = None,
    ) -> None:
        """fire-and-forget 版本的 report_succeeded（v0.3.4+ 新增）。"""
        self._fire(
            self.report_succeeded(task_id=task_id, result=result),
            op="report_succeeded",
        )

    def fire_report_failed(
        self,
        task_id: str,
        error: Optional[dict] = None,
    ) -> None:
        """fire-and-forget 版本的 report_failed（v0.3.4+ 新增）。"""
        self._fire(
            self.report_failed(task_id=task_id, error=error),
            op="report_failed",
        )

    # ===== 内部辅助 =====

    async def _post(
        self,
        path: str,
        payload: dict,
        op: str,
    ) -> bool:
        """统一的 POST 调用（含容错）。

        Args:
            path: URL 路径（如 /v1/tasks/internal）
            payload: 请求体 JSON
            op: 操作名（日志用，如 "report_created"）

        Returns:
            bool: True 表示成功；False 表示失败
        """
        try:
            client = await self._get_client()
            resp = await client.post(path, json=payload)
            if resp.status_code == 200:
                logger.debug(
                    "%s 成功: task_id=%s engine=%s",
                    op, payload.get("task_id", "?"), self.engine_id,
                )
                return True
            logger.warning(
                "%s 失败: status=%d body=%s",
                op, resp.status_code, resp.text[:200],
            )
            return False
        except Exception as e:
            logger.warning("%s 异常（不阻断引擎）: %s", op, e)
            return False

    @staticmethod
    def _fire(coro: Any, op: str) -> None:
        """fire-and-forget 包装：创建后台任务，不阻塞调用方。

        支持三种调用上下文：
            1. 主线程 + 有 event loop：用 asyncio.create_task
            2. 工作线程 + 主线程有 event loop：用 run_coroutine_threadsafe 提交到主 loop
            3. 无 event loop：close 协程避免 "never awaited" 警告

        Args:
            coro: 协程对象
            op: 操作名（日志用）
        """
        try:
            # 情况 1：当前线程有运行的 event loop（主线程正常情况）
            asyncio.get_running_loop()
            task = asyncio.create_task(coro)
            task.add_done_callback(lambda t: _log_task_exception(t, op))
            return
        except RuntimeError:
            # 当前线程无 running loop，可能是工作线程
            pass

        # 情况 2：尝试找到主线程的 loop 并提交（工作线程场景）
        # _main_loop 由 _set_main_loop 在 PlatformTaskReporter 初始化时设置
        global _main_loop
        if _main_loop is not None and _main_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(coro, _main_loop)
                # 添加 done 回调，避免异常静默丢失
                future.add_done_callback(lambda f: _log_future_exception(f, op))
                return
            except RuntimeError:
                pass

        # 情况 3：无可用 loop，close 协程避免警告
        try:
            coro.close()
        except Exception:
            pass

    @property
    def is_configured(self) -> bool:
        """是否已配置（gateway_url 非空）。"""
        return bool(self.gateway_url)


def _log_task_exception(task: asyncio.Task, op: str) -> None:
    """任务完成回调：记录异常（避免静默丢失）。"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("%s 后台任务异常: %s", op, exc)


def _log_future_exception(future: Any, op: str) -> None:
    """run_coroutine_threadsafe 的 future 完成回调：记录异常。"""
    if future.cancelled():
        return
    try:
        exc = future.exception()
    except Exception as e:
        logger.warning("%s future.exception() 调用本身异常: %s", op, e)
        return
    if exc is not None:
        logger.warning("%s 跨线程任务异常: %s", op, exc)


# ===== 模块级状态（含跨线程 loop 引用）=====

# 主线程 event loop 引用，供工作线程通过 run_coroutine_threadsafe 提交任务
# 在 PlatformTaskReporter.__init__ 时尝试捕获
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def _set_main_loop(loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """设置主线程 event loop（供工作线程跨线程提交任务用）。"""
    global _main_loop
    _main_loop = loop


def _try_capture_main_loop() -> None:
    """尝试捕获当前 event loop（若存在）作为主 loop。

    在 PlatformTaskReporter 初始化时调用。
    工作线程无法通过本函数捕获主线程 loop，需要外部通过 _set_main_loop 显式设置。
    """
    global _main_loop
    if _main_loop is not None and _main_loop.is_running():
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            _main_loop = loop
    except RuntimeError:
        pass
    except DeprecationWarning:
        pass


# ===== 模块级单例（惰性创建，引擎子进程可直接 import 使用）=====

_singleton: Optional[PlatformTaskReporter] = None
_singleton_lock = asyncio.Lock()


async def get_platform_task_reporter() -> PlatformTaskReporter:
    """获取模块级单例 PlatformTaskReporter（v0.3.4+ 新增）。

    引擎子进程可直接调用：

        from aireliable_engine import get_platform_task_reporter
        reporter = await get_platform_task_reporter()
        await reporter.report_progress(task_id="...", progress=0.5, stage="diffusion")

    Returns:
        PlatformTaskReporter: 单例实例
    """
    global _singleton
    if _singleton is None:
        async with _singleton_lock:
            if _singleton is None:
                _singleton = PlatformTaskReporter()
                # 捕获当前 event loop 作为主 loop（供工作线程跨线程提交任务）
                try:
                    _set_main_loop(asyncio.get_running_loop())
                except RuntimeError:
                    pass
    return _singleton


async def close_platform_task_reporter() -> None:
    """关闭模块级单例（引擎停止时调用）。"""
    global _singleton
    if _singleton is not None:
        await _singleton.close()
        _singleton = None

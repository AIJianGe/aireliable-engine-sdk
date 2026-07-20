"""引擎抽象基类。

所有引擎继承 BaseEngine，实现 8 个标准 HTTP 端点的业务逻辑。
server.py 的 FastAPI 路由直接调用本类的方法。

子类必须实现：
    _load_impl(variant) -> LoadResponse
    _unload_impl() -> UnloadResponse
    _infer_impl(request, sse_stream) -> None   # 通过 sse_stream 推送进度
    _capabilities_impl() -> CapabilitiesResponse
    _status_impl() -> StatusResponse
    _load_lora_impl(name, weight) -> LoraLoadResponse
    _unload_lora_impl(name) -> LoraUnloadResponse

通用能力（状态管理、并发防护、SSE 流）由基类提供。
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Optional

from .schemas import (
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraLoadResponse,
    LoraUnloadRequest,
    LoraUnloadResponse,
    LoraSpec,
    StatusResponse,
    UnloadResponse,
)
from .sse import SSEStream, make_error_frame, make_frame
from .state_machine import RuntimeState, StateMachine

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """所有引擎的抽象基类。

    职责：
        1. 维护引擎运行态状态机
        2. 提供并发防护（同时只能加载/推理一个任务）
        3. 提供 SSE 流管理
        4. 定义 8 个端点的统一调用入口（子类实现具体逻辑）

    不负责：
        - HTTP 路由（由 server.py 处理）
        - 进程管理（由 EngineManager 处理）
        - UI 渲染（由 ui.py 处理）
    """

    def __init__(self, engine_id: str, engine_name: str, capability: str, engine_type: str):
        self.engine_id: str = engine_id
        self.engine_name: str = engine_name
        self.capability: str = capability               # text-to-image / tts / chat ...
        self.engine_type: str = engine_type             # image / voice / video / threed / llm

        self.state_machine: StateMachine = StateMachine(initial=RuntimeState.IDLE)
        self._started_at: float = time.time()

        # 并发锁：load/unload/infer 互斥
        self._op_lock: asyncio.Lock = asyncio.Lock()

        # 输出根目录（子类在 __init__ 中设置，如 self.output_root = config.output_root）
        # 不输出文件的引擎（如纯流式 chat）可不设置，调用 get_output_path 时会抛错
        self.output_root: Optional[Path] = None

    # ============ 输出文件管理 ============

    def get_output_path(self, category: str, filename: str) -> Path:
        """按类型构造输出文件路径。

        统一输出目录结构：``{output_root}/{category}/{filename}``

        Args:
            category: 输出类别（image / video / audio / voice / text / 3d）
            filename: 文件名（如 ``{task_id}.png``）

        Returns:
            完整路径（目录已自动创建）

        Raises:
            AttributeError: 子类未设置 ``output_root``

        规范详见 PR/architecture.md §3.7 输出文件存储规范。
        """
        if self.output_root is None:
            raise AttributeError(
                f"引擎 {self.engine_id} 未设置 output_root，"
                "请在子类 __init__ 中设置（如 self.output_root = config.output_root），"
                "或重写 get_output_path 方法"
            )
        p = self.output_root / category
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    # ============ 8 个 HTTP 端点的统一入口 ============

    async def health(self) -> HealthResponse:
        """GET /health。"""
        return HealthResponse(
            status="ok",
            engine_id=self.engine_id,
            state=self.state_machine.state.value,
            uptime=self.state_machine.uptime,
        )

    async def capabilities(self) -> CapabilitiesResponse:
        """GET /capabilities。"""
        return await self._capabilities_impl()

    async def load(self, request: LoadRequest) -> LoadResponse:
        """POST /load。

        包含并发防护：同一时间只允许一个 load/unload 操作。
        """
        async with self._op_lock:
            if self.state_machine.is_busy():
                raise EngineBusyError(
                    f"引擎忙（state={self.state_machine.state.value}），请稍后再试"
                )
            if self.state_machine.state == RuntimeState.INFERRING:
                raise EngineBusyError("推理进行中，无法加载")

            self.state_machine.transition(RuntimeState.LOADING)
            try:
                response = await self._load_impl(request.variant)
                self.state_machine.transition(RuntimeState.READY)
                return response
            except Exception as e:
                logger.exception("加载失败: variant=%s", request.variant)
                self.state_machine.transition(RuntimeState.ERROR)
                raise

    async def unload(self) -> UnloadResponse:
        """POST /unload。"""
        async with self._op_lock:
            if self.state_machine.state == RuntimeState.INFERRING:
                raise EngineBusyError("推理进行中，无法卸载")

            try:
                response = await self._unload_impl()
                # 从 READY/LOADING 回到 IDLE
                if self.state_machine.state in (RuntimeState.READY, RuntimeState.LOADING):
                    self.state_machine.transition(RuntimeState.IDLE)
                return response
            except Exception as e:
                logger.exception("卸载失败")
                self.state_machine.transition(RuntimeState.ERROR)
                raise

    async def infer(self, request: Any) -> AsyncIterator[str]:
        """POST /infer。

        返回 SSE 流（异步迭代器，FastAPI StreamingResponse 直接消费）。
        """
        # 创建 SSE 流（绑定当前事件循环）
        loop = asyncio.get_event_loop()
        stream = SSEStream(loop=loop)

        async def _run_infer():
            """后台执行推理，把进度帧推到 stream。"""
            async with self._op_lock:
                if self.state_machine.state == RuntimeState.INFERRING:
                    stream.put_threadsafe(
                        make_error_frame("引擎正在推理中，请稍后", "engine_busy")
                    )
                    stream.close()
                    return

                try:
                    # 若模型未加载，先自动加载
                    if self.state_machine.state == RuntimeState.IDLE:
                        # 在锁内调用 load（不通过公开 load()，避免重复加锁）
                        self.state_machine.transition(RuntimeState.LOADING)
                        load_req = LoadRequest(variant=request.variant)
                        await self._load_impl(request.variant)
                        self.state_machine.transition(RuntimeState.READY)

                    # 进入推理态
                    self.state_machine.transition(RuntimeState.INFERRING)
                    try:
                        await self._infer_impl(request, stream)
                    finally:
                        # 推理结束回 READY（保留模型）
                        if self.state_machine.state == RuntimeState.INFERRING:
                            self.state_machine.transition(RuntimeState.READY)
                except Exception as e:
                    logger.exception("推理失败")
                    stream.put_threadsafe(make_error_frame(str(e)))
                    self.state_machine.transition(RuntimeState.ERROR)
                finally:
                    stream.close()

        # 启动后台推理任务
        asyncio.create_task(_run_infer())

        # 返回 SSE 流
        async for frame in stream.events():
            yield frame

    async def status(self) -> StatusResponse:
        """GET /status。"""
        return await self._status_impl()

    async def load_lora(self, request: LoraLoadRequest) -> LoraLoadResponse:
        """POST /lora/load。"""
        return await self._load_lora_impl(request.name, request.weight)

    async def unload_lora(self, request: LoraUnloadRequest) -> LoraUnloadResponse:
        """POST /lora/unload。"""
        return await self._unload_lora_impl(request.name)

    # ============ 子类必须实现的业务逻辑 ============

    @abstractmethod
    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """构造能力声明。"""

    @abstractmethod
    async def _load_impl(self, variant: str) -> LoadResponse:
        """实际加载逻辑。"""

    @abstractmethod
    async def _unload_impl(self) -> UnloadResponse:
        """实际卸载逻辑。"""

    @abstractmethod
    async def _infer_impl(self, request: Any, stream: SSEStream) -> None:
        """实际推理逻辑。

        通过 stream.put_threadsafe / await stream.put 推送进度帧。
        """

    @abstractmethod
    async def _status_impl(self) -> StatusResponse:
        """构造状态响应。"""

    @abstractmethod
    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """实际 LoRA 加载逻辑。"""

    @abstractmethod
    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """实际 LoRA 卸载逻辑。"""


class EngineBusyError(Exception):
    """引擎忙异常。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.error_type = "engine_busy"
        self.message = message


class EngineLoadError(Exception):
    """引擎加载失败异常。"""

    def __init__(self, message: str, error_type: str = "load_failed"):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class EngineInferError(Exception):
    """推理失败异常。"""

    def __init__(self, message: str, error_type: str = "infer_failed"):
        super().__init__(message)
        self.error_type = error_type
        self.message = message

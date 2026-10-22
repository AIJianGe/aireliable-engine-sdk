"""SSE（Server-Sent Events）流式响应工具。

用途：把引擎内部进度事件（加载、反量化、推理步）转成 SSE 帧推给客户端。
被 /infer 端点使用，diffusers 的 callback 在子线程触发，需要跨线程传消息。

设计说明（v0.2.5+）：
    SSEStream 为**单订阅者**模型，每次 /infer 请求创建一个独立 stream，
    请求结束即关闭。适用于推理进度的点对点推送。

    若需**多订阅者共享同一数据源**（如日志平台：多个 SSE 客户端订阅同一来源日志），
    请使用 `aireliable_engine.log_manager.SSELogSubscriber`：
        - 单 `LogBuffer` 持有多个 `SSELogSubscriber`
        - `LogBuffer.append()` 自动 publish 到所有订阅者
        - 订阅者通过 `subscribe()` / `unsubscribe()` 动态加入/退出

    两者均产出 `data: {json}\\n\\n` 格式的 SSE 帧，可被相同的 StreamingResponse 消费。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


class SSEStream:
    """异步 SSE 流缓冲。

    使用模式：
        stream = SSEStream()
        # 子线程（diffusers callback）：
        stream.put({"progress": 0.5, "stage": "denoising", "step": 25})
        # 主协程（FastAPI 端点）：
        async for frame in stream.events():
            yield frame
        stream.close()  # 结束时
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        # 注意：必须在事件循环内创建 Queue
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self._closed: bool = False

    async def put(self, frame: dict[str, Any]) -> None:
        """主协程写入帧。"""
        if self._closed:
            return
        await self._queue.put(frame)

    def _put_nowait(self, frame: dict[str, Any] | None) -> None:
        """把帧投递到队列：同线程直接入队以保序，异线程经事件循环投递。"""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            # 已在本事件循环线程：直接入队，避免延迟到下一轮事件循环
            # （否则先入队的进度帧会排到后续 workflow 帧之后，顺序错乱）
            self._queue.put_nowait(frame)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def put_threadsafe(self, frame: dict[str, Any]) -> None:
        """写入帧，可在子线程或事件循环线程调用（如 diffusers callback）。"""
        if self._closed:
            return
        self._put_nowait(frame)

    def close(self) -> None:
        """关闭流，发送结束信号。"""
        if self._closed:
            return
        self._closed = True
        # 哨兵必须排在已提交的跨线程帧之后：统一走 call_soon_threadsafe，
        # 与 put_threadsafe 的异线程路径保持 FIFO。
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        except RuntimeError:
            # 事件循环已关闭：直接入队，保证 events() 不会永久阻塞
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[str]:
        """异步迭代器：产出 SSE 格式的字符串帧。

        每帧格式：`data: {json}\n\n`
        遇到 None 哨兵时停止。
        """
        while True:
            frame = await self._queue.get()
            if frame is None:
                # 哨兵，结束流
                break
            try:
                yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error("SSE 帧序列化失败: %s, frame=%s", e, frame)
                continue


def make_frame(progress: float, stage: str, **extra: Any) -> dict[str, Any]:
    """构造一个 SSE 进度帧。

    Args:
        progress: 0.0 - 1.0
        stage: 阶段名（loading_model / gguf_dequant / denoising / ...）
        **extra: 附加字段（step / total / result / timing ...）

    Returns:
        dict: {"progress": ..., "stage": ..., **extra}
    """
    frame: dict[str, Any] = {"progress": float(progress), "stage": stage}
    frame.update(extra)
    return frame


def make_final_frame(result: dict[str, Any], timing: dict[str, float]) -> dict[str, Any]:
    """构造最终帧（progress=1.0）。"""
    return {
        "progress": 1.0,
        "stage": "completed",
        "result": result,
        "timing": timing,
    }


def make_error_frame(message: str, error_type: str = "engine_error") -> dict[str, Any]:
    """构造错误帧。"""
    return {
        "progress": 0.0,
        "stage": "error",
        "error": {"type": error_type, "message": message},
    }

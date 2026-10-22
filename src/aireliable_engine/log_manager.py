"""统一日志管理器。

提供跨组件（shell / core / gateway / engine）的统一日志数据结构与管理。

核心组件：
    - LogEntry：统一日志条目（跨所有组件）
    - LogBuffer：单组件日志缓冲区（deque + seq + SSE 订阅者）
    - LogManager：统一日志管理器（多 source 缓冲区 + 全局 seq）

设计参考：core EngineManager._state_change_log（deque + seq + get_since）模式。

使用场景：
    1. core 层：实例化 LogManager，收集引擎/core/gateway 日志，提供 HTTP 查询/SSE 流端点
    2. 引擎 SDK：仅使用 LogEntry 数据结构（供 core 解析 stdout JSON 时反序列化）

v0.2.5+ 引入，配合统一日志平台架构（详见 unified-log-platform-analysis.md）。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """统一日志条目（跨所有组件）。

    字段规范：
        seq: 全局自增序列号（由 LogBuffer.append 填充）
        timestamp: "YYYY-MM-DD HH:MM:SS" 格式
        level: DEBUG / INFO / WARNING / ERROR / CRITICAL
        source: shell / core / gateway / engine:<engine_id>
        message: 日志消息
        engine_id: 仅 source=engine:* 时填充，其余为 None
        task_id: 关联任务 ID（用于链路追踪，v2 规划）
        extra: 扩展字段（如 pid / exit_code / error_type）

    序列化：
        to_dict() 返回 dict（用于 SSE JSON 帧 / HTTP 响应）
    """

    timestamp: str
    level: str
    source: str
    message: str
    seq: int = 0
    engine_id: Optional[str] = None
    task_id: Optional[str] = None
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """转为 dict（用于 SSE JSON 帧 / HTTP 响应）。

        None 字段省略，减少传输体积。
        """
        d: dict[str, Any] = {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "message": self.message,
        }
        if self.engine_id is not None:
            d["engine_id"] = self.engine_id
        if self.task_id is not None:
            d["task_id"] = self.task_id
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogEntry:
        """从 dict 反序列化（用于解析引擎 stdout JSON 日志）。"""
        return cls(
            timestamp=d.get("timestamp", ""),
            level=d.get("level", "INFO"),
            source=d.get("source", ""),
            message=d.get("message", ""),
            seq=d.get("seq", 0),
            engine_id=d.get("engine_id"),
            task_id=d.get("task_id"),
            extra=d.get("extra"),
        )


class LogBuffer:
    """单组件日志缓冲区。

    复制 _state_change_log 模式：
        - deque(maxlen) 环形缓冲
        - 自增 seq
        - 增量查询 get_since(since)
        - SSE 订阅者列表（实时推送）

    线程安全说明：
        append() 可能被 _collect_logs（asyncio 任务）调用，
        subscribe/unsubscribe 可能被 HTTP 端点（asyncio 任务）调用。
        两者在同一事件循环内，无需加锁。
        若跨线程调用，需用 put_threadsafe 推送到 SSEStream。
    """

    def __init__(self, source: str, maxlen: int = 2000):
        self.source: str = source
        self._deque: deque[LogEntry] = deque(maxlen=maxlen)
        self._seq: int = 0
        self._subscribers: list[SSELogSubscriber] = []

    def append(self, entry: LogEntry) -> int:
        """追加日志条目，返回分配的 seq。

        会自动填充 entry.seq，并推送到所有 SSE 订阅者。
        """
        self._seq += 1
        entry.seq = self._seq
        self._deque.append(entry)

        # 推送到所有 SSE 订阅者
        for subscriber in self._subscribers:
            try:
                subscriber.publish(entry)
            except Exception as e:
                logger.error("推送日志到 SSE 订阅者失败: %s", e)

        return self._seq

    def get_since(self, since: int, limit: int = 500) -> list[LogEntry]:
        """返回 seq > since 的日志条目（按 seq 升序），最多 limit 条。"""
        result = [e for e in self._deque if e.seq > since]
        if len(result) > limit:
            result = result[-limit:]
        return result

    def get_recent(self, limit: int = 200) -> list[LogEntry]:
        """返回最近 limit 条日志。"""
        if len(self._deque) <= limit:
            return list(self._deque)
        return list(self._deque)[-limit:]

    @property
    def last_seq(self) -> int:
        """当前最大 seq。"""
        return self._seq

    @property
    def count(self) -> int:
        """当前缓冲区条目数。"""
        return len(self._deque)

    def subscribe(self, subscriber: SSELogSubscriber) -> None:
        """订阅日志流。"""
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: SSELogSubscriber) -> None:
        """取消订阅。"""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def clear(self) -> None:
        """清空缓冲区（不影响 seq）。"""
        self._deque.clear()


class SSELogSubscriber:
    """SSE 日志订阅者。

    包装一个 asyncio.Queue，用于 SSE 端点的 StreamingResponse 消费。
    当 LogBuffer.append 被调用时，通过 publish() 把日志推入队列。

    使用模式：
        subscriber = SSELogSubscriber()
        buffer.subscribe(subscriber)
        # SSE 端点：
        async for frame in subscriber.events():
            yield frame
        # 结束时：
        buffer.unsubscribe(subscriber)
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        self._queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self._closed: bool = False

    def publish(self, entry: LogEntry) -> None:
        """推送日志条目到队列（线程安全）。

        由 LogBuffer.append 调用。若在子线程，通过 run_coroutine_threadsafe 提交。
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(entry.to_dict())
        except asyncio.QueueFull:
            # 队列满（消费者太慢），丢弃最旧的一条
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(entry.to_dict())
            except Exception:
                pass

    def publish_threadsafe(self, entry: LogEntry) -> None:
        """线程安全推送（用于跨线程场景）。"""
        if self._closed:
            return
        asyncio.run_coroutine_threadsafe(self._queue.put(entry.to_dict()), self._loop)

    def close(self) -> None:
        """关闭订阅，发送结束信号。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

    async def events(self):
        """异步迭代器：产出 SSE 格式字符串帧。

        每帧格式：`data: {json}\n\n`
        遇到 None 哨兵时停止。

        v0.2.5+ 心跳保活：空闲时每 15 秒发送 `: heartbeat\\n\\n`（SSE 注释帧），
        防止 reqwest/中间网络设备因空闲超时断开连接。
        """
        import json

        while True:
            try:
                entry_dict = await asyncio.wait_for(self._queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # SSE 注释帧：以 ":" 开头，浏览器 EventSource 和自定义客户端均自动忽略
                # 仅用于保活，防止 reqwest 60 秒超时断开连接
                yield ": heartbeat\n\n"
                continue
            if entry_dict is None:
                break
            try:
                yield f"data: {json.dumps(entry_dict, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.error("SSE 日志帧序列化失败: %s", e)
                continue


class LogManager:
    """统一日志管理器。

    管理所有组件的日志缓冲区：
        - shell / core / gateway：平台组件
        - engine:<engine_id>：各引擎

    提供跨来源的增量查询（get_all_since）和统一 SSE 订阅。

    使用场景：
        core 层实例化一个 LogManager，所有日志收集器（_collect_logs / core 日志 /
        gateway 上报）都写入此实例。HTTP 端点从 LogManager 查询/订阅。
    """

    def __init__(self, max_buffer_size: int = 2000):
        self._buffers: dict[str, LogBuffer] = {}
        self._max_buffer_size: int = max_buffer_size

    def get_or_create(self, source: str) -> LogBuffer:
        """获取或创建某来源的缓冲区。"""
        if source not in self._buffers:
            self._buffers[source] = LogBuffer(source, maxlen=self._max_buffer_size)
        return self._buffers[source]

    def append(
        self,
        source: str,
        level: str,
        message: str,
        engine_id: Optional[str] = None,
        task_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> LogEntry:
        """追加一条日志，返回 LogEntry。

        Args:
            source: 来源（shell / core / gateway / engine:<id>）
            level: 级别（DEBUG / INFO / WARNING / ERROR / CRITICAL）
            message: 日志消息
            engine_id: 仅 source=engine:* 时填充
            task_id: 关联任务 ID（可选）
            extra: 扩展字段（可选）

        Returns:
            LogEntry（含分配的 seq）
        """
        buf = self.get_or_create(source)
        entry = LogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            level=level.upper(),
            source=source,
            message=message,
            engine_id=engine_id,
            task_id=task_id,
            extra=extra,
        )
        buf.append(entry)
        return entry

    def append_entry(self, source: str, entry: LogEntry) -> int:
        """追加已构造的 LogEntry（用于解析引擎 stdout JSON 日志）。

        会重新分配 seq（忽略 entry.seq），并推送到 SSE 订阅者。
        """
        buf = self.get_or_create(source)
        # 重新分配 seq，避免冲突
        buf.append(entry)
        return entry.seq

    def get_since(
        self,
        source: Optional[str] = None,
        since: int = 0,
        limit: int = 500,
    ) -> tuple[list[LogEntry], int, list[str]]:
        """增量查询日志。

        Args:
            source: 过滤来源，None=全部来源
            since: 返回 seq > since 的日志（按来源各自 seq）
            limit: 最大返回条数
            engine_id: 可选，进一步过滤引擎 ID

        Returns:
            (logs, last_seq, sources)
            logs: LogEntry 列表（按 timestamp 升序）
            last_seq: 所有来源中最大的 seq
            sources: 当前可用的来源列表
        """
        if source is not None:
            # 单来源查询
            buf = self._buffers.get(source)
            if buf is None:
                return [], 0, list(self._buffers.keys())
            logs = buf.get_since(since, limit)
            return logs, buf.last_seq, list(self._buffers.keys())

        # 全部来源查询：合并后按 timestamp 排序
        all_logs: list[LogEntry] = []
        max_seq = 0
        for buf in self._buffers.values():
            all_logs.extend(buf.get_since(since, limit))
            if buf.last_seq > max_seq:
                max_seq = buf.last_seq

        # 按 seq 排序（seq 是全局自增，等价于时间顺序）
        all_logs.sort(key=lambda e: e.seq)
        if len(all_logs) > limit:
            all_logs = all_logs[-limit:]

        return all_logs, max_seq, list(self._buffers.keys())

    def get_engine_logs(
        self,
        engine_id: str,
        since: int = 0,
        limit: int = 500,
    ) -> tuple[list[LogEntry], int]:
        """查询单引擎日志。

        Args:
            engine_id: 引擎 ID
            since: 返回 seq > since 的日志
            limit: 最大返回条数

        Returns:
            (logs, last_seq)
        """
        source = f"engine:{engine_id}"
        buf = self._buffers.get(source)
        if buf is None:
            return [], 0
        return buf.get_since(since, limit), buf.last_seq

    def subscribe(
        self,
        source: Optional[str] = None,
    ) -> tuple[SSELogSubscriber, list[LogEntry]]:
        """订阅日志流。

        Args:
            source: 订阅来源，None=全部来源

        Returns:
            (subscriber, initial_logs)
            subscriber: SSELogSubscriber 实例
            initial_logs: 订阅时刻的最近日志（供 SSE 端点先发送历史再发实时）
        """
        subscriber = SSELogSubscriber()

        if source is not None:
            # 单来源订阅
            buf = self.get_or_create(source)
            buf.subscribe(subscriber)
            return subscriber, buf.get_recent(limit=200)
        else:
            # 全部来源订阅：挂到所有缓冲区
            initial_logs: list[LogEntry] = []
            for buf in self._buffers.values():
                buf.subscribe(subscriber)
                initial_logs.extend(buf.get_recent(limit=50))
            initial_logs.sort(key=lambda e: e.seq)
            return subscriber, initial_logs[-200:]

    def unsubscribe(self, subscriber: SSELogSubscriber, source: Optional[str] = None) -> None:
        """取消订阅。"""
        if source is not None:
            buf = self._buffers.get(source)
            if buf:
                buf.unsubscribe(subscriber)
        else:
            for buf in self._buffers.values():
                buf.unsubscribe(subscriber)

    @property
    def sources(self) -> list[str]:
        """当前所有来源列表。"""
        return list(self._buffers.keys())

    def clear(self, source: Optional[str] = None) -> None:
        """清空缓冲区。

        Args:
            source: 指定来源，None=清空全部
        """
        if source is not None:
            buf = self._buffers.get(source)
            if buf:
                buf.clear()
        else:
            for buf in self._buffers.values():
                buf.clear()


def parse_engine_stdout_line(line: str, engine_id: str) -> Optional[LogEntry]:
    """解析引擎 stdout 一行，返回 LogEntry。

    支持两种格式：
        1. JSON 格式（v0.2.5+ 引擎 SDK 输出）：
           {"timestamp":"...","level":"INFO","engine_id":"...","message":"..."}
        2. 人类可读格式（旧格式，兼容）：
           [2026-07-18 12:34:56] [INFO] [engine_id] 消息内容

    无法解析的行返回 None（调用方按原始行处理）。

    Args:
        line: stdout 一行（已去除换行符）
        engine_id: 引擎 ID（用于填充 source / engine_id 字段）

    Returns:
        LogEntry 或 None
    """
    import json

    line = line.rstrip()
    if not line:
        return None

    # 尝试 JSON 格式（以 { 开头）
    if line.startswith("{"):
        try:
            d = json.loads(line)
            entry = LogEntry.from_dict(d)
            # 确保 source 和 engine_id 正确
            entry.source = f"engine:{engine_id}"
            entry.engine_id = engine_id
            # v0.2.5+ T6.1：解析 task_id / request_id（链路追踪）
            # LogEntry.from_dict 已处理 task_id；request_id 放入 extra
            if "request_id" in d and entry.extra is None:
                entry.extra = {}
            if "request_id" in d:
                entry.extra["request_id"] = d["request_id"]
            return entry
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # 不是合法 JSON，继续尝试人类可读格式

    # 尝试人类可读格式：[timestamp] [level] [source] message
    # split("] [") 会消耗分隔符，因此 parts[0]=timestamp, parts[1]=level,
    # parts[2]=source, parts[3]=message（若存在）
    if line.startswith("["):
        inner = line[1:]  # 去掉开头的 [
        parts = inner.split("] [", 3)
        if len(parts) >= 3:
            timestamp = parts[0]
            level = parts[1].upper()
            # parts[2] 形如 "source] message" 或 "source]"（最后一段含 ]）
            last_source_segment = parts[2]
            if "] " in last_source_segment:
                idx = last_source_segment.index("] ")
                source = last_source_segment[:idx]
                message = last_source_segment[idx + 2:]
            elif last_source_segment.endswith("]"):
                source = last_source_segment[:-1]
                message = ""
            else:
                source = last_source_segment
                message = ""
            # 若有第 4 段（说明 message 中本身含 "] [" 字符），拼回 message
            if len(parts) == 4:
                message = (message + "] [" + parts[3]) if message else parts[3]
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=f"engine:{engine_id}",
                message=message,
                engine_id=engine_id,
            )

    # 兜底：整行作为 INFO 消息
    return LogEntry(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        level="INFO",
        source=f"engine:{engine_id}",
        message=line,
        engine_id=engine_id,
    )

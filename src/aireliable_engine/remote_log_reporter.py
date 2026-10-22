"""远程引擎日志回传（v0.2.5+ T6.3）。

为远程部署的引擎提供日志回传能力：把引擎本地产生的日志通过 HTTP 批量
上报到 core 的 ``POST /logs/ingest`` 端点，由 core 的 LogManager 统一消费。

设计要点：
    1. **后台批量上报**：不阻塞引擎主流程，缓冲 + 定时 flush（默认 2s）
    2. **失败静默**：core 不可达时丢弃最旧日志（保住引擎主流程比保日志重要）
    3. **不依赖 SDK 核心**：仅用 stdlib + httpx，避免循环依赖
    4. **协议复用**：直接 POST 到 core 的 ``/logs/ingest`` 端点（与 gateway 一致）
    5. **source 标识**：远程引擎上报时 source 固定为 ``engine:<engine_id>``，
       与本地引擎经由 stdout 采集后的 source 一致，core 端无需区分

数据流：
    engine LogRecord → RemoteLogHandler → RemoteLogReporter.buffer (deque)
        → 后台任务每 flush_interval 秒取一批 → POST /logs/ingest → core LogManager

使用方式（在引擎 ``setup_engine_logging`` 中自动启用）::

    from aireliable_engine import setup_engine_logging, RemoteLogConfig

    setup_engine_logging(
        engine_id="ernie_image",
        log_level="INFO",
        remote_config=RemoteLogConfig(
            core_url="http://core-host:8001",
            auth_token="Bearer xxx",       # 可选
            batch_size=50,
            flush_interval=2.0,
            max_buffer=1000,
        ),
    )

参考：
    - unified-log-platform-analysis.md §5.4 远程引擎日志
    - core/src/core/server.py::ingest_logs  (POST /logs/ingest)
    - gateway/src/gateway/log_handler.py  (类似的批量上报实现)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# 配置
# ===========================================================================

@dataclass
class RemoteLogConfig:
    """远程日志上报配置。

    Attributes:
        core_url: core 服务根 URL（如 "http://core-host:8001"），不带尾斜杠
        auth_token: 鉴权 token（如 "Bearer xxx"）。None 表示无鉴权。
        batch_size: 单次上报最大条数，默认 50
        flush_interval: flush 间隔（秒），默认 2.0
        max_buffer: 缓冲区上限（超出丢弃最旧），默认 1000
        timeout: HTTP 请求超时（秒），默认 5.0
        source: 上报 source 标识。为 None 时自动用 ``engine:<engine_id>``。
            多引擎共享同一 reporter 时可显式指定。
    """

    core_url: str
    auth_token: Optional[str] = None
    batch_size: int = 50
    flush_interval: float = 2.0
    max_buffer: int = 1000
    timeout: float = 5.0
    source: Optional[str] = None


# ===========================================================================
# RemoteLogReporter
# ===========================================================================


class RemoteLogReporter:
    """远程日志上报器（后台批量 POST 到 core）。

    线程安全说明：
        - ``buffer`` 是 deque，单线程 append/pop 是原子的，但跨线程需加锁
        - logging.Handler 可能在任意线程触发 emit，故用 threading.Lock 保护 buffer
        - flush 任务在 asyncio 事件循环中执行，与 buffer 操作通过锁互斥

    生命周期：
        1. ``start()``：启动后台 flush 任务（幂等，多次调用安全）
        2. ``stop()``：停止后台任务并尝试最后一次 flush（幂等）
        3. ``push(entry_dict)``：把日志条目放入缓冲区（线程安全）
    """

    def __init__(
        self,
        engine_id: str,
        config: RemoteLogConfig,
    ) -> None:
        self.engine_id: str = engine_id
        self.config: RemoteLogConfig = config
        # source 默认 engine:<engine_id>
        self.source: str = config.source or f"engine:{engine_id}"
        # 缓冲区（线程安全）
        self._buffer: deque[dict[str, Any]] = deque(maxlen=config.max_buffer)
        self._lock = threading.Lock()
        # 后台任务
        self._task: Optional[asyncio.Task] = None
        self._started: bool = False
        # 统计
        self._total_pushed: int = 0
        self._total_sent: int = 0
        self._total_dropped: int = 0
        self._total_failed: int = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动后台 flush 任务（幂等）。"""
        if self._started:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 无运行中的事件循环（如同步代码中调用），跳过自动启动
            logger.debug(
                "RemoteLogReporter.start: 无运行中的事件循环，跳过自动启动"
            )
            return
        self._task = loop.create_task(self._flush_loop())
        self._started = True
        logger.info(
            "远程日志上报已启动: engine=%s, core_url=%s, source=%s, "
            "flush_interval=%.1fs, batch_size=%d",
            self.engine_id, self.config.core_url, self.source,
            self.config.flush_interval, self.config.batch_size,
        )

    async def stop(self) -> None:
        """停止后台任务并尝试最后一次 flush（幂等）。"""
        if not self._started:
            return
        self._started = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # 最后一次 flush（同步等待，确保剩余日志尽量上报）
        await self._flush_once()
        logger.info(
            "远程日志上报已停止: engine=%s, sent=%d, dropped=%d, failed=%d",
            self.engine_id, self._total_sent, self._total_dropped,
            self._total_failed,
        )

    # ------------------------------------------------------------------
    # 缓冲区操作
    # ------------------------------------------------------------------

    def push(self, entry_dict: dict[str, Any]) -> None:
        """把日志条目放入缓冲区（线程安全）。

        Args:
            entry_dict: LogEntry.to_dict() 格式的日志条目
        """
        with self._lock:
            # deque(maxlen=...) 满了会自动丢弃最旧，这里手动统计丢弃数
            if len(self._buffer) >= self.config.max_buffer:
                self._total_dropped += 1
            self._buffer.append(entry_dict)
            self._total_pushed += 1

    def get_stats(self) -> dict[str, Any]:
        """返回上报统计信息。"""
        with self._lock:
            buffer_len = len(self._buffer)
        return {
            "engine_id": self.engine_id,
            "source": self.source,
            "core_url": self.config.core_url,
            "started": self._started,
            "buffer_len": buffer_len,
            "total_pushed": self._total_pushed,
            "total_sent": self._total_sent,
            "total_dropped": self._total_dropped,
            "total_failed": self._total_failed,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    # 后台 flush 任务
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """后台循环：每 flush_interval 秒 flush 一次。"""
        while self._started:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # 不让异常终止 flush 循环
                self._last_error = f"{type(e).__name__}: {e}"
                logger.debug("RemoteLogReporter flush 异常: %s", e)
                # 短暂退避避免紧密循环
                await asyncio.sleep(self.config.flush_interval)

    async def _flush_once(self) -> None:
        """执行一次 flush：取出一批 buffer 日志并 POST 到 core。"""
        # 取出一批
        batch: list[dict[str, Any]] = []
        with self._lock:
            while self._buffer and len(batch) < self.config.batch_size:
                batch.append(self._buffer.popleft())

        if not batch:
            return

        # 构造请求体（与 core POST /logs/ingest 协议一致）
        payload = {
            "source": self.source,
            "logs": batch,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.auth_token:
            headers["Authorization"] = self.config.auth_token

        url = f"{self.config.core_url.rstrip('/')}/logs/ingest"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.config.timeout, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    self._total_sent += len(batch)
                    self._last_error = None
                else:
                    self._total_failed += len(batch)
                    self._last_error = (
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    logger.debug(
                        "远程日志上报失败: status=%d, url=%s",
                        resp.status_code, url,
                    )
        except Exception as e:
            self._total_failed += len(batch)
            self._last_error = f"{type(e).__name__}: {e}"
            logger.debug("远程日志上报异常: %s, url=%s", e, url)


# ===========================================================================
# RemoteLogHandler
# ===========================================================================


class RemoteLogHandler(logging.Handler):
    """日志 Handler：把 LogRecord 转为 LogEntry dict 并推送到 RemoteLogReporter。

    线程安全：logging.Handler.emit 可能在任意线程调用，
    通过 RemoteLogReporter 内部的锁保护 buffer。

    用法（通常由 setup_engine_logging 自动挂载）::

        handler = RemoteLogHandler(engine_id="my_engine", reporter=reporter)
        root_logger.addHandler(handler)
    """

    def __init__(
        self,
        engine_id: str,
        reporter: RemoteLogReporter,
        level: int = logging.NOTSET,
    ) -> None:
        super().__init__(level=level)
        self.engine_id: str = engine_id
        self.reporter: RemoteLogReporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        """把 LogRecord 转为 LogEntry dict 并推送到 reporter buffer。

        失败时静默（logging.Handler 标准行为），不影响引擎主流程。
        """
        try:
            timestamp = datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            message = record.getMessage()
            if record.exc_info:
                import traceback
                exc_text = "".join(traceback.format_exception(*record.exc_info))
                message = f"{message}\n{exc_text}"

            entry: dict[str, Any] = {
                "timestamp": timestamp,
                "level": record.levelname,
                "source": self.reporter.source,
                "message": message,
                "engine_id": self.engine_id,
            }

            # task_id / request_id 从 contextvars 注入（链路追踪）
            try:
                from .log_context import current_context
                ctx = current_context()
                if ctx.task_id is not None:
                    entry["task_id"] = ctx.task_id
                if ctx.request_id is not None:
                    extra = entry.setdefault("extra", {})
                    extra["request_id"] = ctx.request_id
            except Exception:
                pass

            self.reporter.push(entry)
        except Exception:
            # Handler 不应抛出异常，静默丢弃
            pass


# ===========================================================================
# 全局 reporter 管理（供 setup_engine_logging 使用）
# ===========================================================================

_global_reporters: dict[str, RemoteLogReporter] = {}


def get_remote_reporter(engine_id: str) -> Optional[RemoteLogReporter]:
    """获取已注册的全局 RemoteLogReporter（若存在）。"""
    return _global_reporters.get(engine_id)


def stop_remote_reporter(engine_id: str) -> None:
    """停止并清理指定引擎的远程日志上报器（幂等）。

    通常在引擎 lifespan 关闭时调用。
    """
    reporter = _global_reporters.pop(engine_id, None)
    if reporter is None:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(reporter.stop())
    except RuntimeError:
        # 无事件循环，同步停止（仅清理 _started 标志）
        reporter._started = False


def stop_all_remote_reporters() -> None:
    """停止所有全局 RemoteLogReporter（通常在进程退出时调用）。"""
    for engine_id in list(_global_reporters.keys()):
        stop_remote_reporter(engine_id)

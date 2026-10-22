"""日志链路追踪上下文（v0.2.5+ / v0.2.15+ 日志目录统一）。

基于 `contextvars` 实现线程/协程安全的任务上下文传播，让引擎在推理过程中
产生的所有日志自动带上 `task_id` / `request_id`，便于跨引擎工作流链路追踪。

设计目标：
    - 引擎开发者无需手动给每条 logger.xxx() 调用传 task_id
    - JSONFormatter 自动从 contextvars 读取并输出到 stdout JSON
    - core 的 parse_engine_stdout_line 自动解析 task_id 字段
    - 跨 asyncio 任务隔离（每个推理请求独立上下文）

使用示例：

    from aireliable_engine import log_context, setup_engine_logging

    # v0.2.15+ 推荐方式：依赖平台注入的 AIRELIABLE_LOG_DIR 环境变量
    logger = setup_engine_logging(engine_id="e_xxx")

    # 推理入口处设置上下文（通常由 WorkflowExecutor 自动包裹）
    with log_context(task_id="task_abc", request_id="req_123"):
        logger.info("开始推理")   # 自动带上 task_id / request_id
        # ... 节点执行 ...
        logger.info("推理完成")   # 同样带上 task_id / request_id

    # 上下文外日志不包含 task_id
    logger.info("引擎空闲")

JSON 输出：
    {"timestamp":"...","level":"INFO","engine_id":"e_xxx","message":"开始推理",
     "task_id":"task_abc","request_id":"req_123"}

详见：
    - [project-rules.md §6](../../../../.trae/rules/project-rules.md)（日志规范）
    - [统一日志平台架构](../../../../docs/docs/architecture/logging-platform.md)
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator, Optional


# ===========================================================================
# ContextVar：单次推理请求的日志上下文
# ===========================================================================

# contextvars 保证 asyncio 任务间隔离（每个协程看到独立的值）
_task_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "aireliable_log_task_id", default=None
)
_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "aireliable_log_request_id", default=None
)


@dataclass(frozen=True)
class LogContextSnapshot:
    """日志上下文快照（不可变）。

    由 `current_context()` 返回，供 formatter 读取当前上下文信息。
    """

    task_id: Optional[str] = None
    request_id: Optional[str] = None

    def is_empty(self) -> bool:
        """是否为空上下文（无 task_id / request_id）。"""
        return self.task_id is None and self.request_id is None


def current_context() -> LogContextSnapshot:
    """获取当前日志上下文快照。

    Returns:
        LogContextSnapshot：含当前 contextvars 中的 task_id / request_id
    """
    return LogContextSnapshot(
        task_id=_task_id_var.get(),
        request_id=_request_id_var.get(),
    )


def get_task_id() -> Optional[str]:
    """便捷方法：获取当前 task_id。"""
    return _task_id_var.get()


def get_request_id() -> Optional[str]:
    """便捷方法：获取当前 request_id。"""
    return _request_id_var.get()


# ===========================================================================
# Context Manager：设置/恢复上下文
# ===========================================================================


@contextlib.contextmanager
def log_context(
    task_id: Optional[str] = None,
    request_id: Optional[str] = None,
    *,
    inherit: bool = True,
) -> Iterator[LogContextSnapshot]:
    """临时设置日志上下文（context manager）。

    在 with 块内，所有 logger.xxx() 调用自动带上 task_id / request_id
    （由 JSONFormatter 从 contextvars 读取）。

    Args:
        task_id: 任务 ID（用于跨引擎工作流关联）
        request_id: 请求 ID（用于单次 HTTP 请求关联）
        inherit: 是否继承外层上下文未覆盖的字段（默认 True）
            - True：task_id/request_id 为 None 时保留外层值
            - False：强制覆盖为 None（清除外层值）

    Yields:
        LogContextSnapshot：生效后的上下文快照

    Example:
        with log_context(task_id="task_abc"):
            logger.info("推理开始")  # task_id 自动注入
            with log_context(request_id="req_123"):
                logger.info("加载模型")  # task_id + request_id 都有
            # 内层退出，request_id 恢复为 None
            logger.info("推理完成")  # 仅 task_id
        # 外层退出，task_id 恢复为 None
        logger.info("引擎空闲")  # 无 task_id
    """
    # 计算新值（inherit=True 时保留外层未覆盖字段）
    outer = current_context()
    new_task_id = task_id if task_id is not None else (outer.task_id if inherit else None)
    new_request_id = (
        request_id if request_id is not None else (outer.request_id if inherit else None)
    )

    # 设置 ContextVar（token 用于恢复）
    task_token = _task_id_var.set(new_task_id)
    request_token = _request_id_var.set(new_request_id)

    try:
        yield LogContextSnapshot(task_id=new_task_id, request_id=new_request_id)
    finally:
        _task_id_var.reset(task_token)
        _request_id_var.reset(request_token)


def set_task_id(task_id: Optional[str]) -> contextvars.Token:
    """直接设置 task_id（非 context manager 模式）。

    适用于需要长期生效的场景（如整个引擎生命周期）。建议优先用 `log_context()`。

    Returns:
        Token：用于 `reset_task_id(token)` 恢复
    """
    return _task_id_var.set(task_id)


def reset_task_id(token: contextvars.Token) -> None:
    """恢复 task_id（配合 `set_task_id()` 使用）。"""
    _task_id_var.reset(token)


def set_request_id(request_id: Optional[str]) -> contextvars.Token:
    """直接设置 request_id（非 context manager 模式）。

    Returns:
        Token：用于 `reset_request_id(token)` 恢复
    """
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """恢复 request_id（配合 `set_request_id()` 使用）。"""
    _request_id_var.reset(token)


def clear_context() -> None:
    """清除当前上下文（task_id / request_id 都置 None）。

    注意：直接设置 default 不需要 token，但会污染外层上下文。
    推荐使用 `with log_context(task_id=None, request_id=None, inherit=False):` 替代。
    """
    _task_id_var.set(None)
    _request_id_var.set(None)


__all__ = [
    "LogContextSnapshot",
    "current_context",
    "get_task_id",
    "get_request_id",
    "log_context",
    "set_task_id",
    "reset_task_id",
    "set_request_id",
    "reset_request_id",
    "clear_context",
]

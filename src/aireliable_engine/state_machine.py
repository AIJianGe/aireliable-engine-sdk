"""引擎 6 态状态机。

状态：NotInstalled → Installed → Loading → Ready → Stopped → Error

实际进程级运行态在 Ready / Stopped / Error 之间切换。
NotInstalled / Installed 由 EngineManager 在主控层管理，引擎进程本身从 Loading 起。

进程内细分状态（对外通过 /status 暴露）：
    idle       = Ready 但未加载模型（= Warm）
    loading    = Loading 阶段（加载模型权重）
    ready      = Ready 且模型在显存（= Hot）
    inferring  = 推理中
    error      = 异常
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable, Optional


class EngineState(str, Enum):
    """引擎生命周期 6 态。"""

    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    LOADING = "loading"
    READY = "ready"
    STOPPED = "stopped"
    ERROR = "error"


class RuntimeState(str, Enum):
    """引擎进程内细分运行态（暴露给 /status）。"""

    IDLE = "idle"            # 进程在，模型未加载（Warm）
    LOADING = "loading"      # 加载模型中
    READY = "ready"          # 模型在显存（Hot）
    INFERRING = "inferring"  # 推理中
    ERROR = "error"          # 异常


# 允许的状态转换
_ALLOWED_TRANSITIONS: dict[RuntimeState, set[RuntimeState]] = {
    RuntimeState.IDLE: {RuntimeState.LOADING, RuntimeState.ERROR},
    RuntimeState.LOADING: {RuntimeState.READY, RuntimeState.ERROR, RuntimeState.IDLE},
    RuntimeState.READY: {RuntimeState.INFERRING, RuntimeState.IDLE, RuntimeState.ERROR},
    RuntimeState.INFERRING: {RuntimeState.READY, RuntimeState.ERROR},
    RuntimeState.ERROR: {RuntimeState.IDLE},  # reset 后回到 idle
}


class StateMachine:
    """线程安全的引擎运行态状态机。

    使用场景：单个引擎进程内部维护自身状态。
    主控的 EngineManager 用更高层的 EngineState 管理生命周期。
    """

    def __init__(self, initial: RuntimeState = RuntimeState.IDLE):
        self._state: RuntimeState = initial
        self._lock = threading.RLock()
        self._started_at: float = time.time()
        self._last_used_at: float = time.time()
        self._listeners: list[Callable[[RuntimeState, RuntimeState], None]] = []

    @property
    def state(self) -> RuntimeState:
        """当前状态。"""
        with self._lock:
            return self._state

    @property
    def uptime(self) -> float:
        """进程已运行秒数。"""
        return time.time() - self._started_at

    @property
    def last_used_at(self) -> float:
        """最近一次使用时间戳。"""
        with self._lock:
            return self._last_used_at

    def touch(self) -> None:
        """更新最近使用时间。"""
        with self._lock:
            self._last_used_at = time.time()

    def transition(self, new_state: RuntimeState) -> None:
        """尝试转换状态。

        Raises:
            ValueError: 不允许的状态转换。
        """
        with self._lock:
            old = self._state
            allowed = _ALLOWED_TRANSITIONS.get(old, set())
            if new_state not in allowed and new_state != old:
                raise ValueError(
                    f"非法状态转换: {old.value} → {new_state.value} "
                    f"(允许: {[s.value for s in allowed]})"
                )
            self._state = new_state
            if new_state in (RuntimeState.READY, RuntimeState.INFERRING):
                self._last_used_at = time.time()
            listeners = list(self._listeners)

        # 在锁外触发监听器，避免回调里再调本类方法导致死锁
        for listener in listeners:
            try:
                listener(old, new_state)
            except Exception:
                # 监听器异常不影响状态转换
                pass

    def can_infer(self) -> bool:
        """当前是否可推理（READY 状态）。"""
        return self.state == RuntimeState.READY

    def is_busy(self) -> bool:
        """当前是否忙（LOADING / INFERRING）。"""
        return self.state in (RuntimeState.LOADING, RuntimeState.INFERRING)

    def add_listener(
        self, listener: Callable[[RuntimeState, RuntimeState], None]
    ) -> None:
        """注册状态变化监听器。"""
        with self._lock:
            self._listeners.append(listener)

    def reset(self) -> None:
        """从 ERROR 状态重置回 IDLE。"""
        with self._lock:
            if self._state != RuntimeState.ERROR:
                raise ValueError(f"reset 仅在 ERROR 状态可用，当前: {self._state.value}")
        self.transition(RuntimeState.IDLE)

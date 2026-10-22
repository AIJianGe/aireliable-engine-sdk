"""工作流上下文（v0.2 新增）。

WorkflowContext 是节点执行时的运行时上下文，封装节点需要的所有平台信息：
    - 引擎元数据（engine_id / variant / capability）
    - 设备信息（device / vram_manager）
    - 进度上报（progress_reporter）
    - 任务标识（task_id）
    - 取消信号（cancel_event）
    - 模型根目录（models_root）
    - 原始请求字段（inputs / loras）

设计参考：

使用示例：
    ctx = WorkflowContext(
        engine_id="ernie_image",
        variant="default",
        device="cuda:0",
        models_root=Path("models/"),
        task_id="task_001",
        progress_reporter=SSEProgressReporter(sse_stream),
    )

    # 节点执行时访问 ctx
    async def execute(self, inputs, ctx):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            raise CancelledError()
        await ctx.report_progress(0.5, "loading model")
        model = load_from(ctx.models_root / "checkpoint")
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .sse import SSEStream


logger = logging.getLogger(__name__)


# ===========================================================================
# ProgressReporter：进度上报器抽象
# ===========================================================================

class ProgressReporter(ABC):
    """进度上报器抽象基类。

    节点通过 ctx.report_progress() 上报执行进度，平台将进度推送到 SSE 流。
    不同实现：
        - SSEProgressReporter：通过 SSEStream 推送到 HTTP 客户端
        - ConsoleProgressReporter：打印到 stderr（用于测试 / CLI）
        - NullProgressReporter：丢弃所有进度（用于测试）

    节点应避免直接调用 SSEStream，统一通过 ctx.report_progress() 抽象。
    """

    @abstractmethod
    async def report_progress(
        self,
        progress: float,
        stage: str = "",
        **kwargs: Any,
    ) -> None:
        """上报进度。

        Args:
            progress: 进度值，0.0 ~ 1.0。
            stage: 阶段名（如 "loading_model" / "sampling" / "decoding"）。
            **kwargs: 附加字段（如 step / total / message）。
        """

    @abstractmethod
    async def report_error(
        self,
        message: str,
        error_type: str = "node_error",
        **kwargs: Any,
    ) -> None:
        """上报错误。

        Args:
            message: 错误信息。
            error_type: 错误类型字符串（如 "model_not_found" / "oom"）。
            **kwargs: 附加字段。
        """

    @abstractmethod
    async def report_info(self, message: str, **kwargs: Any) -> None:
        """上报信息（非进度，非错误）。"""


class NullProgressReporter(ProgressReporter):
    """空进度上报器（丢弃所有上报）。

    用于无 SSE 流的场景（如单元测试 / 离线推理 / 命令行工具）。
    """

    async def report_progress(self, progress: float, stage: str = "", **kwargs: Any) -> None:
        pass

    async def report_error(self, message: str, error_type: str = "node_error", **kwargs: Any) -> None:
        pass

    async def report_info(self, message: str, **kwargs: Any) -> None:
        pass


class ConsoleProgressReporter(ProgressReporter):
    """控制台进度上报器（打印到 stderr）。

    用于 CLI / 测试场景，方便观察执行进度。
    """

    async def report_progress(self, progress: float, stage: str = "", **kwargs: Any) -> None:
        pct = progress * 100
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"[{pct:5.1f}%] {stage} {extras}".rstrip())

    async def report_error(self, message: str, error_type: str = "node_error", **kwargs: Any) -> None:
        logger.error(f"[ERROR/{error_type}] {message}")

    async def report_info(self, message: str, **kwargs: Any) -> None:
        logger.info(f"[INFO] {message}")


class SSEProgressReporter(ProgressReporter):
    """基于 SSEStream 的进度上报器。

    将进度推送到引擎的 SSE 流，最终到达 HTTP 客户端。
    """

    def __init__(self, sse_stream: "SSEStream") -> None:
        """初始化。

        Args:
            sse_stream: SSE 流对象，需支持 put / put_threadsafe 方法。
        """
        self.sse_stream = sse_stream

    async def report_progress(self, progress: float, stage: str = "", **kwargs: Any) -> None:
        from .sse import make_frame
        frame = make_frame(progress=progress, stage=stage, **kwargs)
        await self.sse_stream.put(frame)

    async def report_error(self, message: str, error_type: str = "node_error", **kwargs: Any) -> None:
        from .sse import make_error_frame
        frame = make_error_frame(message=message, error_type=error_type, **kwargs)
        await self.sse_stream.put(frame)

    async def report_info(self, message: str, **kwargs: Any) -> None:
        # info 帧不是进度帧，直接构造 dict（避免 make_frame 要求 float progress）
        frame: dict[str, Any] = {"stage": "info", "message": message}
        frame.update(kwargs)
        await self.sse_stream.put(frame)


# ===========================================================================
# VRamManager：显存管理器抽象
# ===========================================================================

class VRamManager(ABC):
    """显存管理器抽象基类。

    提供模型在 CPU / GPU 之间的 offload / prefetch 机制，避免 OOM。

    不同实现：
        - SimpleVRamManager：单模型加载（不 offload）
        - OffloadVRamManager：模型按需 offload 到 CPU
        - SmartVRamManager：智能预取（基于使用模式预测）
    """

    @abstractmethod
    async def load_to_gpu(self, model: Any, device: str = "cuda:0") -> Any:
        """加载模型到 GPU。

        Args:
            model: 模型对象（diffusers pipeline / transformer）。
            device: 目标设备。

        Returns:
            已加载到 GPU 的模型（可能是同一对象或新对象）。
        """

    @abstractmethod
    async def offload_to_cpu(self, model: Any) -> Any:
        """将模型从 GPU offload 到 CPU（释放显存）。

        Args:
            model: GPU 上的模型对象。

        Returns:
            已 offload 到 CPU 的模型。
        """

    @abstractmethod
    async def get_free_vram_bytes(self) -> int:
        """查询当前可用显存（字节）。"""


class SimpleVRamManager(VRamManager):
    """简单的显存管理器（不做 offload）。

    所有模型直接加载到 GPU，不主动 offload。适用于显存充足的场景。
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    async def load_to_gpu(self, model: Any, device: str = "cuda:0") -> Any:
        # 简单实现：尝试调用 model.to(device)
        if hasattr(model, "to"):
            try:
                return model.to(device)
            except Exception as e:
                logger.warning("模型加载到 %s 失败: %s", device, e)
        return model

    async def offload_to_cpu(self, model: Any) -> Any:
        # 简单实现：不实际 offload（保持原状）
        return model

    async def get_free_vram_bytes(self) -> int:
        # 默认返回一个大数（表示不限制）
        return 1 << 60  # 1 EiB


# ===========================================================================
# WorkflowContext：工作流上下文数据类
# ===========================================================================

@dataclass
class WorkflowContext:
    """工作流执行上下文。

    在节点 execute(inputs, ctx) 调用时传入，封装节点运行时所需的全部平台信息。

    设计原则：
        - 不可变数据载体：所有字段在创建后不应修改（除 cancel_event 的状态）
        - 节点只读访问：节点不应修改 ctx，仅通过 ctx 调用平台能力
        - 进度上报抽象：节点通过 ctx.report_progress() 上报，不直接操作 SSE 流

    Attributes:
        engine_id: 引擎 ID（如 "ernie_image"）。
        variant: 引擎变体名（如 "default" / "fp16" / "q4_k_m"）。
        device: 推理设备字符串（如 "cuda:0" / "cpu"）。
        models_root: 模型根目录。节点通过此路径查找模型文件。

            v0.2.1 覆盖契约（决策 H：Optional + 文档化覆盖契约）：
                - 值来源链路（优先级从高到低）：
                    1. 调度层环境变量 AIRELIABLE_MODELS_ROOT
                       （由 CoreConfig.models_root 注入子进程环境）
                    2. 引擎 EngineConfig.models_root（命令行 --models-root 或 MODELS_ROOT 环境变量）
                    3. WorkflowEngine.__init__ 的 models_root 参数
                       （由子类 super().__init__(models_root=config.models_root) 注入）
                    4. 默认值 Path("models/")（向后兼容 v0.1/v0.2 引擎）

                - 不变性：创建后不应修改（除 cancel_event 状态）
                - 节点访问约定：
                    * 主模型路径：ctx.models_root / manifest.models.model_files[?].target_dir / filename
                    * 不应硬编码路径，必须通过 manifest 声明的 target_dir 拼接
                    * 避免与 CHECKPOINT_SUBDIR 常量混用（v0.2.1 起改读环境变量 AIRELIABLE_MODEL_LAYOUT_BASE）

        task_id: 任务 ID（用于日志追踪 / 文件命名）。
        progress_reporter: 进度上报器。节点调用 ctx.report_progress() 推送进度。
        cancel_event: 取消事件。节点应定期检查并主动退出。
            为 None 表示不支持取消。
        vram_manager: 显存管理器。为 None 表示不管理显存（节点自行处理）。
        inputs: 原始请求的 inputs 字段。节点可通过 ctx.inputs 访问请求参数
            （绕过 TextInput 节点，但不推荐，应使用 TextInput）。
        loras: LoRA 列表（来自请求）。为空列表表示无 LoRA。
        extra: 扩展字段（引擎自定义数据）。不同引擎可存放引擎特有对象。
        output_root: 输出文件根目录。ImageSave 等节点使用。
        capability: 引擎能力（如 "text-to-image"）。用于校验工作流合法性。

    Example:
        ctx = WorkflowContext(
            engine_id="ernie_image",
            variant="default",
            device="cuda:0",
            models_root=Path("models/image/checkpoints/ernie-image"),
            task_id="task_abc123",
            progress_reporter=SSEProgressReporter(sse_stream),
        )

        # 节点执行
        async def execute(self, inputs, ctx):
            # 检查取消
            if ctx.is_cancelled():
                raise asyncio.CancelledError()

            # 上报进度
            await ctx.report_progress(0.5, "loading_model")

            # 加载模型
            model = load(ctx.models_root / f"{ctx.variant}.gguf")
    """

    # ===== 引擎元数据 =====
    engine_id: str = ""
    variant: str = "default"
    capability: str = ""

    # ===== 设备与显存 =====
    device: str = "cpu"
    vram_manager: Optional[VRamManager] = None

    # ===== 路径 =====
    models_root: Optional[Path] = None
    output_root: Optional[Path] = None

    # ===== 任务标识 =====
    task_id: Optional[str] = None

    # ===== 进度上报 =====
    progress_reporter: ProgressReporter = field(default_factory=NullProgressReporter)

    # ===== 取消信号 =====
    cancel_event: Optional[asyncio.Event] = None

    # ===== 原始请求字段 =====
    inputs: dict = field(default_factory=dict)
    loras: list = field(default_factory=list)

    # ===== 扩展字段（引擎自定义数据）=====
    extra: dict = field(default_factory=dict)

    # ===== 进度上报便捷方法 =====

    async def report_progress(
        self,
        progress: float,
        stage: str = "",
        **kwargs: Any,
    ) -> None:
        """上报进度（代理到 progress_reporter）。"""
        await self.progress_reporter.report_progress(progress, stage, **kwargs)

    async def report_error(
        self,
        message: str,
        error_type: str = "node_error",
        **kwargs: Any,
    ) -> None:
        """上报错误（代理到 progress_reporter）。"""
        await self.progress_reporter.report_error(message, error_type, **kwargs)

    async def report_info(self, message: str, **kwargs: Any) -> None:
        """上报信息（代理到 progress_reporter）。"""
        await self.progress_reporter.report_info(message, **kwargs)

    # ===== 取消信号便捷方法 =====

    def is_cancelled(self) -> bool:
        """检查是否已取消。

        Returns:
            True 表示工作流已被取消，节点应主动退出。
            False 表示未取消或 cancel_event 为 None。
        """
        return self.cancel_event is not None and self.cancel_event.is_set()

    async def wait_for_cancel(self, timeout: Optional[float] = None) -> bool:
        """等待取消信号。

        用于节点中需要长时间阻塞等待的场景（如等待外部资源）。
        阻塞期间若被取消会立即返回。

        Args:
            timeout: 最大等待时长（秒）。None 表示无限等待。

        Returns:
            True 表示已取消，False 表示超时。
        """
        if self.cancel_event is None:
            return False
        try:
            await asyncio.wait_for(self.cancel_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def raise_if_cancelled(self) -> None:
        """若已取消则抛出 CancelledError。

        节点在循环 / 长时间操作中应定期调用此方法实现协作式取消。
        """
        if self.is_cancelled():
            raise asyncio.CancelledError("工作流已被取消")

    # ===== 路径便捷方法 =====

    def get_model_path(self, *parts: str) -> Path:
        """构造模型文件路径。

        Args:
            *parts: 相对 models_root 的路径片段。

        Returns:
            完整模型路径。

        Raises:
            ValueError: models_root 未设置。
        """
        if self.models_root is None:
            raise ValueError("WorkflowContext.models_root 未设置")
        return self.models_root.joinpath(*parts)

    def get_output_path(self, *parts: str) -> Path:
        """构造输出文件路径。

        Args:
            *parts: 相对 output_root 的路径片段。

        Returns:
            完整输出路径。

        Raises:
            ValueError: output_root 未设置。
        """
        if self.output_root is None:
            raise ValueError("WorkflowContext.output_root 未设置")
        return self.output_root.joinpath(*parts)

    # ===== 输入便捷方法 =====

    def get_input(self, key: str, default: Any = None) -> Any:
        """从 inputs 中读取字段（便捷方法）。

        Args:
            key: 字段名。
            default: 默认值。

        Returns:
            字段值或默认值。
        """
        return self.inputs.get(key, default)

    def has_input(self, key: str) -> bool:
        """检查 inputs 是否包含某字段。"""
        return key in self.inputs

    # ===== 上下文衍生方法 =====

    def with_progress_reporter(self, reporter: ProgressReporter) -> "WorkflowContext":
        """返回带新 progress_reporter 的副本（不可变更新）。

        用于子工作流或子任务使用独立进度上报器。

        Args:
            reporter: 新的进度上报器。

        Returns:
            新的 WorkflowContext（其他字段引用相同）。
        """
        from dataclasses import replace
        return replace(self, progress_reporter=reporter)

    def with_extra(self, **kwargs: Any) -> "WorkflowContext":
        """返回带附加 extra 字段的副本（不可变更新）。"""
        from dataclasses import replace
        new_extra = {**self.extra, **kwargs}
        return replace(self, extra=new_extra)

    def __repr__(self) -> str:
        return (
            f"WorkflowContext(engine_id={self.engine_id!r}, "
            f"variant={self.variant!r}, device={self.device!r}, "
            f"task_id={self.task_id!r})"
        )


# ===========================================================================
# 工厂方法
# ===========================================================================

def make_default_context(
    engine_id: str = "",
    variant: str = "default",
    device: str = "cpu",
    **kwargs: Any,
) -> WorkflowContext:
    """创建默认 WorkflowContext（带 NullProgressReporter / SimpleVRamManager）。

    用于测试 / CLI 场景。

    Args:
        engine_id: 引擎 ID。
        variant: 变体名。
        device: 设备字符串。
        **kwargs: 其他字段。

    Returns:
        WorkflowContext 实例。
    """
    # 若未指定 vram_manager，使用 SimpleVRamManager
    if "vram_manager" not in kwargs:
        kwargs["vram_manager"] = SimpleVRamManager(device=device)
    # 若未指定 progress_reporter，使用 NullProgressReporter
    if "progress_reporter" not in kwargs:
        kwargs["progress_reporter"] = NullProgressReporter()
    return WorkflowContext(
        engine_id=engine_id,
        variant=variant,
        device=device,
        **kwargs,
    )


__all__ = [
    # 上下文
    "WorkflowContext",
    "make_default_context",
    # 进度上报器
    "ProgressReporter",
    "NullProgressReporter",
    "ConsoleProgressReporter",
    "SSEProgressReporter",
    # 显存管理器
    "VRamManager",
    "SimpleVRamManager",
]

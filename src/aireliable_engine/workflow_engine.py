"""WorkflowEngine 工作流引擎基类（v0.2 阶段 5 任务 5.1）。

范式 B 引擎的抽象基类，继承自 BaseEngine。

设计参考：
    - v0.2-tasks.md 任务 5.1

核心特性：
    1. 继承 BaseEngine 8 端点契约（与范式 A/C 完全兼容）
    2. _load_impl / _unload_impl / _status_impl / _load_lora_impl / _unload_lora_impl
       提供工作流引擎默认实现（模型按需加载，无需显式 load）
    3. _build_workflow 抽象方法：子类实现工作流 yaml 加载逻辑
    4. _infer_impl 调 WorkflowExecutor 执行工作流，通过 SSE 流推送进度
    5. 工作流热重载支持（reload_workflow / watch_workflow_file）
    6. 节点注册表（node_registry）+ 进度上报器（progress_reporter）

使用示例：
    class ErnieImageEngine(WorkflowEngine):
        def __init__(self, config):
            super().__init__(
                engine_id="ernie_image",
                engine_name="ERNIE-Image",
                capability="text-to-image",
                engine_type="image",
                workflows_dir=config.workflows_dir,
                default_workflow="default",
            )
            self.output_root = config.output_root

        async def _build_workflow(self, name: Optional[str] = None) -> str:
            workflow_name = name or self.current_workflow
            path = self.workflows_dir / f"{workflow_name}.yaml"
            return path.read_text(encoding="utf-8")

        async def _capabilities_impl(self) -> CapabilitiesResponse:
            ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from .engine_base import BaseEngine
from .node_registry import NodeRegistry
from .schemas import (
    CapabilitiesResponse,
    InferRequest,
    LoadResponse,
    LoraLoadResponse,
    LoraUnloadResponse,
    StatusResponse,
    UnloadResponse,
)
from .sse import SSEStream, make_error_frame, make_final_frame, make_frame
from .workflow_context import (
    ProgressReporter,
    SSEProgressReporter,
    WorkflowContext,
)
from .workflow_executor import (
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowExecutorError,
)


logger = logging.getLogger(__name__)


# ===========================================================================
# 异常类
# ===========================================================================

class WorkflowEngineError(Exception):
    """WorkflowEngine 相关异常基类。"""


class WorkflowNotFoundError(WorkflowEngineError):
    """工作流未找到异常。"""

    def __init__(self, workflow_name: str, available: Optional[list[str]] = None):
        self.workflow_name = workflow_name
        self.available = available or []
        msg = f"工作流 {workflow_name!r} 未找到"
        if self.available:
            msg += f"（可用工作流：{self.available}）"
        super().__init__(msg)


class WorkflowLoadError(WorkflowEngineError):
    """工作流加载失败异常。"""

    def __init__(self, workflow_name: str, cause: Exception):
        self.workflow_name = workflow_name
        self.cause = cause
        super().__init__(f"加载工作流 {workflow_name!r} 失败：{cause}")


# ===========================================================================
# WorkflowEngine 基类
# ===========================================================================

class WorkflowEngine(BaseEngine):
    """工作流引擎基类（范式 B）。

    继承 BaseEngine 8 端点契约，子类只需实现：
        - _build_workflow(name) -> str：加载工作流 yaml 文本
        - _capabilities_impl()：能力声明（variants 等）

    工作流引擎特点：
        1. 无需显式 load：模型加载由工作流中的 ModelLoader 节点按需完成
        2. infer 时构建 WorkflowExecutor 执行工作流
        3. 通过 SSE 流推送节点执行进度
        4. 支持工作流热重载（重读 yaml 即可，无需重启进程）
        5. 多工作流切换（current_workflow 字段，可通过 request.workflow 指定）

    Args:
        engine_id: 引擎 ID（如 "ernie_image"）。
        engine_name: 引擎显示名（如 "ERNIE-Image"）。
        capability: 能力（如 "text-to-image"）。
        engine_type: 类型（如 "image" / "voice" / "video" / "threed" / "llm"）。
        workflows_dir: 工作流 yaml 文件目录。为 None 时子类自行管理路径。
        default_workflow: 默认工作流名（不含 .yaml 扩展名）。
        node_registry: 节点注册表。为 None 时使用全局默认注册表。
        progress_reporter: 进度上报器。为 None 时由 _infer_impl 内部构造。

    Attributes:
        workflows_dir: 工作流文件目录。
        default_workflow: 默认工作流名。
        current_workflow: 当前工作流名（推理时使用，可被 request.workflow 覆盖）。
        node_registry: 节点注册表。
        loaded_variant: 已加载的 variant 名（load 后记录，unload 后清空）。
        loaded_loras: 已加载的 LoRA 列表（load_lora 后记录）。
        last_used_at: 上次推理时间戳（Unix 秒）。
        _workflow_cache: 工作流 yaml 缓存 {name: (mtime, yaml_text)}（热重载时检查 mtime）。
    """

    def __init__(
        self,
        engine_id: str,
        engine_name: str,
        capability: str,
        engine_type: str,
        *,
        workflows_dir: Optional[Path] = None,
        default_workflow: str = "default",
        node_registry: Optional[NodeRegistry] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        # v0.2.1 新增：models_root 注入（决策 H：Optional + 文档化覆盖契约）
        # 子类应从 EngineConfig.models_root 注入（由调度层 CoreConfig.models_root 决定）
        # 为 None 时 fallback 到 Path("models/")（向后兼容旧引擎）
        models_root: Optional[Path] = None,
    ) -> None:
        super().__init__(
            engine_id=engine_id,
            engine_name=engine_name,
            capability=capability,
            engine_type=engine_type,
        )

        # 工作流相关
        self.workflows_dir: Optional[Path] = workflows_dir
        self.default_workflow: str = default_workflow
        self.current_workflow: str = default_workflow

        # 节点注册表（默认使用全局默认注册表，与 register_node 装饰器一致）
        self.node_registry: NodeRegistry = node_registry or NodeRegistry._get_default()

        # 进度上报器（可选，子类可注入；为 None 时 _infer_impl 内部构造）
        self._progress_reporter: Optional[ProgressReporter] = progress_reporter

        # v0.2.1：models_root（决策 H：Optional + 文档化覆盖契约）
        # - 子类从 EngineConfig.models_root 注入
        # - EngineConfig.models_root 由调度层 CoreConfig.models_root → 子进程环境变量 AIRELIABLE_MODELS_ROOT
        # - 为 None 时 fallback 到 Path("models/")（向后兼容旧 v0.1/v0.2 引擎）
        self.models_root: Path = models_root if models_root is not None else Path("models/")

        # 运行态记录
        self.loaded_variant: Optional[str] = None
        self.loaded_loras: list[str] = []
        self.last_used_at: Optional[float] = None

        # 工作流 yaml 缓存：{name: (mtime_float, yaml_text)}
        # 用于热重载检测：每次读取时检查文件 mtime，变化则重新读取
        self._workflow_cache: dict[str, tuple[float, str]] = {}

    # ===========================================================================
    # 工作流管理
    # ===========================================================================

    @abstractmethod
    async def _build_workflow(self, name: Optional[str] = None) -> str:
        """加载工作流 yaml 文本。

        子类必须实现：从文件 / 远程 / 内置模板加载 yaml。

        Args:
            name: 工作流名。None 表示使用 self.current_workflow。

        Returns:
            yaml 文本字符串。

        Raises:
            WorkflowNotFoundError: 工作流不存在。
            WorkflowLoadError: 加载失败。

        Note:
            实现可使用 self._read_workflow_file(name) 简化文件读取。
            热重载支持由 _read_workflow_file 自动处理（检查 mtime）。
        """

    @abstractmethod
    def _get_default_workflow_template(self) -> str:
        """返回最小可运行的默认工作流 yaml 模板（v0.2.5 节点体系增强 L6）。

        子类必须实现：返回一个包含该引擎核心节点（如 ModelLoader + TextInput
        + Sampler + ImageSave）的最小工作流 yaml 字符串，供 _ensure_default_workflow()
        在首次启动时创建 default.yaml。

        模板要求：
            - 仅包含引擎必备节点（最少 3-4 个）
            - 节点间连线已正确配置
            - params 使用合理默认值
            - yaml 注释说明用途

        Returns:
            默认工作流 yaml 文本（UTF-8 编码字符串）。
        """

    def _ensure_default_workflow(self) -> None:
        """确保 workflows_dir 中存在 default.yaml（v0.2.5 节点体系增强 L5）。

        若 default.yaml 不存在，从 _get_default_workflow_template() 模板创建。
        若 workflows_dir 未设置，则跳过（子类自行管理路径）。
        首次运行时自动创建 workflows_dir 目录。

        Note:
            - 应在引擎 __init__ 末尾调用（workflows_dir 已就绪后）
            - 仅在文件不存在时创建，不覆盖已有文件
            - 创建后记录 info 日志
        """
        if self.workflows_dir is None:
            logger.debug("_ensure_default_workflow: workflows_dir 未设置，跳过")
            return

        default_path = self.workflows_dir / f"{self.default_workflow}.yaml"
        if default_path.exists():
            logger.debug("_ensure_default_workflow: default.yaml 已存在: %s", default_path)
            return

        try:
            self.workflows_dir.mkdir(parents=True, exist_ok=True)
            template = self._get_default_workflow_template()
            default_path.write_text(template, encoding="utf-8")
            logger.info("已创建默认工作流：%s", default_path)
        except Exception as e:
            logger.warning("创建默认工作流失败：%s（不影响引擎启动）", e)

    def _read_workflow_file(self, name: Optional[str] = None) -> str:
        """从 workflows_dir 读取工作流 yaml 文件（支持热重载）。

        通过文件 mtime 实现热重载：
            - 首次读取：加载文件内容并缓存
            - 后续读取：比较 mtime，变化时重新读取

        Args:
            name: 工作流名。None 表示使用 self.current_workflow。

        Returns:
            yaml 文本字符串。

        Raises:
            WorkflowNotFoundError: workflows_dir 未设置 / 文件不存在。
        """
        workflow_name = name or self.current_workflow
        if self.workflows_dir is None:
            raise WorkflowNotFoundError(
                workflow_name,
                available=self.list_workflows(),
            )
        path = self.workflows_dir / f"{workflow_name}.yaml"
        if not path.exists():
            raise WorkflowNotFoundError(
                workflow_name,
                available=self.list_workflows(),
            )

        # 热重载检测：比较 mtime
        try:
            current_mtime = path.stat().st_mtime
        except OSError as e:
            raise WorkflowLoadError(workflow_name, e) from e

        cached = self._workflow_cache.get(workflow_name)
        if cached is not None:
            cached_mtime, cached_text = cached
            # mtime 未变化，返回缓存
            if cached_mtime == current_mtime:
                logger.debug("工作流 %r 命中缓存（mtime=%s）", workflow_name, current_mtime)
                return cached_text

        # mtime 变化或首次读取，重新加载
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise WorkflowLoadError(workflow_name, e) from e

        self._workflow_cache[workflow_name] = (current_mtime, text)
        if cached is not None:
            logger.info("工作流 %r 已热重载（mtime: %s → %s）", workflow_name, cached_mtime, current_mtime)
        else:
            logger.debug("工作流 %r 首次加载（mtime=%s）", workflow_name, current_mtime)
        return text

    def list_workflows(self) -> list[str]:
        """列出可用工作流名（不含 .yaml 扩展名）。

        扫描 self.workflows_dir 下的 *.yaml 文件。

        Returns:
            工作流名列表。workflows_dir 未设置时返回空列表。
        """
        if self.workflows_dir is None or not self.workflows_dir.exists():
            return []
        return sorted(
            p.stem for p in self.workflows_dir.glob("*.yaml")
        )

    def reload_workflow(self, name: Optional[str] = None) -> None:
        """强制重新加载工作流（清空缓存）。

        Args:
            name: 工作流名。None 表示清空所有缓存。
        """
        if name is None:
            self._workflow_cache.clear()
            logger.info("清空所有工作流缓存")
        else:
            self._workflow_cache.pop(name, None)
            logger.info("清空工作流 %r 缓存", name)

    def set_current_workflow(self, name: str) -> None:
        """切换当前工作流。

        Args:
            name: 工作流名。

        Note:
            仅切换 self.current_workflow 字段，不立即加载工作流。
            实际加载在下次 _build_workflow 调用时发生。
        """
        self.current_workflow = name
        logger.info("切换当前工作流为 %r", name)

    # ===========================================================================
    # 8 端点实现（覆写 BaseEngine 的默认实现）
    # ===========================================================================

    async def _load_impl(self, variant: str) -> LoadResponse:
        """工作流引擎的 load 实现。

        工作流引擎无需显式加载模型：
            - 模型加载由工作流中的 ModelLoader 节点按需完成
            - load 仅记录 variant 名，状态切到 READY

        Args:
            variant: variant 名。

        Returns:
            LoadResponse，state=ready。
        """
        self.loaded_variant = variant
        logger.info(
            "工作流引擎 %s 加载 variant=%s（实际加载由 ModelLoader 节点执行）",
            self.engine_id, variant,
        )
        return LoadResponse(
            variant=variant,
            state="ready",
            vram_used="0GB",
            load_time=0.0,
        )

    async def _unload_impl(self) -> UnloadResponse:
        """工作流引擎的 unload 实现。

        清空 loaded_variant 记录。模型显存释放由工作流节点的 cleanup() 负责。

        Returns:
            UnloadResponse，state=idle。
        """
        released_variant = self.loaded_variant
        self.loaded_variant = None
        self.loaded_loras.clear()
        logger.info(
            "工作流引擎 %s 卸载 variant=%s",
            self.engine_id, released_variant,
        )
        return UnloadResponse(state="idle", vram_freed=None)

    async def _status_impl(self) -> StatusResponse:
        """工作流引擎的状态查询。"""
        state = self.state_machine.state.value
        return StatusResponse(
            state=state,
            loaded_variant=self.loaded_variant,
            vram_used="0GB",
            vram_total=None,
            loaded_loras=list(self.loaded_loras),
            last_used_at=self.last_used_at,
            uptime=self.state_machine.uptime,
            extra={
                "current_workflow": self.current_workflow,
                "available_workflows": self.list_workflows(),
            },
        )

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """LoRA 加载（记录到 loaded_loras 列表）。

        实际 LoRA 加载由工作流中的 LoRALoader 节点按需完成。
        """
        if name not in self.loaded_loras:
            self.loaded_loras.append(name)
        logger.info("工作流引擎 %s 记录 LoRA %s（weight=%s）", self.engine_id, name, weight)
        return LoraLoadResponse(
            name=name,
            weight=weight,
            loaded_loras=list(self.loaded_loras),
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """LoRA 卸载（从 loaded_loras 列表移除）。"""
        if name in self.loaded_loras:
            self.loaded_loras.remove(name)
        logger.info("工作流引擎 %s 移除 LoRA %s", self.engine_id, name)
        return LoraUnloadResponse(
            name=name,
            loaded_loras=list(self.loaded_loras),
        )

    # ===========================================================================
    # 推理实现（核心：构建 WorkflowExecutor 执行工作流）
    # ===========================================================================

    async def _infer_impl(self, request: InferRequest, stream: SSEStream) -> None:
        """工作流推理实现。

        流程：
            1. 确定工作流名（request.workflow 或 self.current_workflow）
            2. 调 _build_workflow 加载 yaml（含热重载）
            3. 构建 WorkflowExecutor
            4. 构造 WorkflowContext（绑定 SSE 进度上报器 + cancel_event）
            5. 执行工作流
            6. 推送进度帧 + 最终结果帧
            7. 失败推送错误帧

        Args:
            request: InferRequest，包含 variant / inputs / loras / workflow 等。
            stream: SSE 流，用于推送进度帧。
        """
        # 1. 确定工作流名
        workflow_name = request.workflow or self.current_workflow
        if workflow_name != self.current_workflow:
            logger.info(
                "推理请求指定工作流 %r（当前默认 %r）",
                workflow_name, self.current_workflow,
            )

        # 推送开始帧
        await stream.put(make_frame(
            progress=0.0,
            stage="workflow_start",
            workflow=workflow_name,
            engine_id=self.engine_id,
            variant=request.variant,
        ))

        # 2. 加载工作流 yaml
        try:
            workflow_yaml = await self._build_workflow(workflow_name)
        except WorkflowEngineError as e:
            await stream.put(make_error_frame(
                f"加载工作流失败：{e}",
                error_type="workflow_load_failed",
            ))
            return
        except Exception as e:
            await stream.put(make_error_frame(
                f"加载工作流异常：{e}",
                error_type="workflow_load_error",
            ))
            return

        # 3. 构建 WorkflowExecutor
        executor = WorkflowExecutor(workflow_yaml, self.node_registry)

        # 推送工作流加载完成帧（含执行顺序）
        try:
            execution_order = executor.get_execution_order()
        except Exception:
            # 拓扑排序失败（循环 / 引用错误）等会抛错
            execution_order = []
        await stream.put(make_frame(
            progress=0.05,
            stage="workflow_loaded",
            workflow=workflow_name,
            node_count=len(execution_order) if execution_order else 0,
            execution_order=execution_order,
        ))

        # 4. 构造 WorkflowContext
        cancel_event = asyncio.Event()
        progress_reporter = self._progress_reporter or SSEProgressReporter(stream)
        ctx = WorkflowContext(
            engine_id=self.engine_id,
            variant=request.variant,
            device="cpu",  # 子类应在 _infer_impl 之前设置 self.device 或重写
            # v0.2.1：models_root 从 self.models_root 读取（由 __init__ 注入）
            # 子类应通过 super().__init__(models_root=config.models_root) 注入
            models_root=self.models_root,
            task_id=request.task_id or "",
            progress_reporter=progress_reporter,
            cancel_event=cancel_event,
            inputs=request.inputs,
            loras=[lora.model_dump() for lora in request.loras],
            capability=self.capability,
            extra={
                "workflow_name": workflow_name,
                "request_workflow": request.workflow,
            },
        )

        # 5. 执行工作流
        start_time = time.time()
        try:
            result = await executor.execute(request.inputs, ctx)
        except WorkflowExecutionError as e:
            await stream.put(make_error_frame(
                f"工作流执行失败（节点 {e.node_id!r} / {e.node_type!r}）：{e.cause}",
                error_type="workflow_execution_failed",
            ))
            return
        except WorkflowExecutorError as e:
            await stream.put(make_error_frame(
                f"工作流执行器错误：{e}",
                error_type="workflow_executor_error",
            ))
            return
        except asyncio.CancelledError:
            await stream.put(make_error_frame(
                "工作流执行被取消",
                error_type="workflow_cancelled",
            ))
            raise
        except Exception as e:
            logger.exception("工作流执行异常")
            await stream.put(make_error_frame(
                f"工作流执行异常：{e}",
                error_type="workflow_infer_error",
            ))
            return
        finally:
            # 节点资源清理由 WorkflowExecutor.execute 内部 finally 块负责
            # 此处无需额外 cleanup 调用
            pass

        # 6. 推送最终结果帧
        elapsed = time.time() - start_time
        self.last_used_at = time.time()
        await stream.put(make_final_frame(
            result=result,
            timing={
                "total_seconds": elapsed,
                "workflow": workflow_name,
                "variant": request.variant,
            },
        ))

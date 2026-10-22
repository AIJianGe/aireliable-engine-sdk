"""ModelSamplingAuraFlow 节点：给模型设置 AuraFlow 风格的 shift 采样参数。

v0.2.5 ComfyUI 风格节点补充（C3）。

参考 ComfyUI 的 ModelSamplingAuraFlow 节点设计：
    - 输入 1 个 model 端口，输出 1 个 model 端口（同名）
    - 1 个核心参数 shift（AuraFlow 官方推荐 3.0）
    - 转换而非消费：返回原 model 对象，仅修改其 scheduler 的 shift 配置

这是"模型变换器"节点模式的典型代表（参考 ComfyUI 的 ModelSamplingXxx 系列）：
    - 输入输出同名 ``model``（不像 Loader 节点有独立输出名）
    - 不真正加载新模型，只对输入 model 做参数变换
    - 通常串联在 UNetLoader/ModelLoader 和 Sampler 之间

双保险注入方案：
    - 方案 A：直接修改 ``model.scheduler.config.shift`` 和调用 ``set_shift``（若存在）
    - 方案 B：写 ``ctx.extra["model_shift"]`` 供下游 Sampler 读取
      （Sampler 会在采样前再次应用 shift，与方案 A 形成双保险）

ctx.extra 集成：
    - 读取：``ctx.extra["loaded_model"]``（若 inputs.model 为 None 时回退，与 UNetLoader 一致）
    - 写入：``ctx.extra["model_shift"]``（供 Sampler 读取）

参考：
    - v0.2.5-comfyui-nodes-analysis.md §二（3）
    - PR/architecture.md §20.19.10（ComfyUI 风格节点补充）
    - PR/development-guide.md §28.17.3（ModelSamplingAuraFlow 节点规范）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class ModelSamplingAuraFlowNode(BaseNode):
    """AuraFlow 风格 shift 采样参数设置节点（模型变换器节点）。

    平台基础节点库提供的模型变换器节点，参考 ComfyUI 的 ModelSamplingAuraFlow：
        - 输入 1 个 model 端口（UNet/Transformer 实例或 diffusers Pipeline）
        - 输出 1 个 model 端口（同名，返回同一对象，转换而非消费）
        - 1 个核心参数 shift（AuraFlow 官方推荐 3.0）

    双保险注入方案：
        - 方案 A：直接修改 ``model.scheduler.config.shift`` 和调用 ``set_shift``
        - 方案 B：写 ``ctx.extra["model_shift"]`` 供 Sampler 读取

    参数（params）：
        shift (float)：AuraFlow shift 参数，默认 3.0，范围 [0, 100]。
            官方推荐 3.0，用于 Flux/AuraFlow 系列模型的 FlowMatchEulerDiscreteScheduler。

    输入端口：
        model (model)：待设置 shift 的模型（UNet/Transformer/Pipeline）。

    输出端口：
        model (model)：设置 shift 后的模型（同一对象，转换而非消费）。

    ctx.extra 集成：
        - inputs.model 为 None 时回退到 ``ctx.extra["loaded_model"]``
        - 写 ``ctx.extra["model_shift"]`` 供下游 Sampler 读取
    """

    node_type = "ModelSamplingAuraFlow"
    category = "sample"
    subcategory = "sampler"  # 与 Sampler 同分类
    title = "采样算法 (AuraFlow)"
    icon = "🎲"
    description = "给模型设置 AuraFlow 风格的 shift 采样参数（Flux/AuraFlow 推荐 shift=3.0）"

    input_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            required=True,
            description="待设置 shift 的模型（UNet/Transformer/Pipeline，由 UNetLoader 或 ModelLoader 输出）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            description="设置 shift 后的模型（同一对象，转换而非消费）",
        ),
    ]
    params_schema = {
        "shift": {
            "type": "number", "required": False, "default": 3.0,
            "min": 0, "max": 100,
            "label": "移位",
            "description": "AuraFlow shift 参数（官方推荐 3.0，用于 FlowMatchEulerDiscreteScheduler）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._shift: float = float(self.get_param("shift", 3.0))

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not isinstance(self._shift, (int, float)):
            errors.append(f"params.shift 必须为数字，当前值：{self._shift!r}")
            return errors
        if self._shift < 0 or self._shift > 100:
            errors.append(
                f"params.shift 必须在 [0, 100] 范围内，当前值：{self._shift}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：给模型设置 AuraFlow shift 采样参数。

        流程：
            1. 参数校验
            2. 读取 model（优先 inputs.model，回退 ctx.extra["loaded_model"]）
            3. 上报开始进度
            4. 方案 A：修改 model.scheduler.config.shift + set_shift
            5. 方案 B：写 ctx.extra["model_shift"]
            6. 上报完成进度
            7. 返回原 model（转换而非消费）

        Args:
            inputs: 节点输入字典，含 model。
            ctx: 工作流上下文（提供 extra 全局状态）。

        Returns:
            dict 包含 model 字段（同一对象，转换而非消费）。

        Raises:
            NodeValidationError: 参数校验失败或未找到 model。
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 读取 model（优先 inputs.model，回退 ctx.extra["loaded_model"]）
        model = inputs.get("model")
        if model is None:
            # 回退到 ctx.extra（与 UNetLoader 的 ctx.extra 集成一致）
            model = ctx.extra.get("loaded_model")
        if model is None:
            raise NodeValidationError(
                self.node_type,
                [
                    "未找到 model。请：\n"
                    "  1. 通过 inputs.model 端口连接 UNetLoader / ModelLoader 的输出；\n"
                    "  2. 或在工作流中先放置 UNetLoader / ModelLoader 节点"
                    "（会自动写入 ctx.extra['loaded_model']）。"
                ],
            )

        shift = float(self._shift)

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="model_sampling_aura_flow.start",
            message=f"开始设置 shift={shift}",
            shift=shift,
            has_scheduler=hasattr(model, "scheduler") and model.scheduler is not None,
        )

        # 方案 A：直接修改 model.scheduler 的 shift（若 model 是 pipeline 或有 scheduler 属性）
        # 用 try-except 包裹：shift 注入失败不影响采样，仅 warning 日志
        # 下游 Sampler 会从 ctx.extra["model_shift"] 读取作为双保险
        scheduler = getattr(model, "scheduler", None)
        if scheduler is not None:
            try:
                # 修改 config.shift
                config = getattr(scheduler, "config", None)
                if config is not None:
                    if hasattr(config, "shift"):
                        # diffusers 0.27+ 的 config 是 dict-like，可能需要 __setitem__
                        try:
                            config.shift = shift
                        except (AttributeError, TypeError):
                            # 不可变 config，尝试 dict 赋值
                            try:
                                config["shift"] = shift
                            except Exception:
                                logger.warning(
                                    "ModelSamplingAuraFlow: scheduler.config.shift 不可修改"
                                    "（下游 Sampler 会从 ctx.extra 读取作为 fallback）"
                                )
                    else:
                        logger.debug(
                            "ModelSamplingAuraFlow: scheduler.config 无 shift 属性"
                            "（可能不是 FlowMatchEulerDiscreteScheduler，跳过方案 A）"
                        )
                # diffusers 0.27+ 的 FlowMatchEulerDiscreteScheduler 支持 set_shift 方法
                if hasattr(scheduler, "set_shift"):
                    scheduler.set_shift(shift)
            except Exception as exc:
                logger.warning(
                    "ModelSamplingAuraFlow: 修改 scheduler shift 失败（忽略，"
                    "下游 Sampler 会从 ctx.extra 读取）：%s", exc
                )

        # 方案 B：写 ctx.extra["model_shift"] 供 Sampler 读取（双保险）
        ctx.extra["model_shift"] = shift
        logger.debug(
            "ModelSamplingAuraFlow: 已写入 ctx.extra['model_shift']=%s（供 Sampler 读取）",
            shift,
        )

        # 上报完成
        await ctx.report_progress(
            1.0,
            stage="model_sampling_aura_flow.done",
            message=f"shift 设置完成（shift={shift}）",
            shift=shift,
        )

        # 返回原 model（转换而非消费，输入输出同名）
        return {"model": model}

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：转换节点不缓存（返回 None）。

        模型变换器节点不缓存，因为：
            - 它只修改 model 的属性，不产生新对象
            - 缓存会导致后续 Sampler 拿到的是"未修改"的 model，造成混乱
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shift={self._shift})"

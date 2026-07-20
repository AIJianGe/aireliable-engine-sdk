"""Sampler 节点：通用采样器节点（v0.2）。

设计为**抽象基类节点**：实际采样逻辑由引擎子类实现。
平台基础 SamplerNode 提供：
    - 标准输入端口（model / embeds / negative_embeds / images / scheduler）
    - 标准输出端口（latent）
    - 标准参数（steps / guidance / size / seed / batch_size / eta / sampler_name）
    - execute 流程（参数校验 → 进度上报 → 调用 _sample_impl → 回调进度）

支持 callback 进度上报：每个 step 上报一次进度，符合 diffusers pipeline 标准。

参考：
    - PR/architecture.md §12.2 / §12.2.6（节点示例）
    - PR/development-guide.md §28.5（数据类型规范）
    - v0.2-tasks.md 任务 3.6
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...types import Latent
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class SamplerNode(BaseNode):
    """通用采样器节点（抽象基类）。

    引擎子类只需实现 `_sample_impl(inputs, params, ctx, progress_callback) -> dict`：
        - 返回 dict，必须包含 "latent" 字段（Latent 类型）
        - 通过 progress_callback(step, total, **kwargs) 上报每步进度

    参数（params）：
        steps (int)：采样步数，默认 30。
        guidance (float)：CFG guidance scale，默认 7.5。
        size (str)：输出尺寸（"1024x1024"），默认 "1024x1024"。
        seed (Optional[int])：随机种子，None 随机。
        batch_size (int)：batch 大小，默认 1。
        eta (float)：DDIM eta（控制随机性），默认 0.0。
        sampler_name (Optional[str])：采样器名称（如 "euler" / "dpm++"）。
        cfg_scheduler (Optional[str])：CFG 调度策略（linear / constant 等）。
        device (Optional[str])：覆盖 ctx.device。
        call_timeout (Optional[float])：采样超时秒数，None 不限制。
    """

    node_type = "Sampler"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            required=True,
            description="主模型对象（由 ModelLoader 输出）",
        ),
        PortSpec(
            name="embeds",
            type="embeds",
            required=True,
            description="正向 prompt 编码（由 CLIPTextEncode 输出）",
        ),
        PortSpec(
            name="negative_embeds",
            type="embeds",
            required=False,
            description="负向 prompt 编码（None 表示无负向）",
        ),
        PortSpec(
            name="images",
            type="images",
            required=False,
            description="输入图片（图生图 / IPAdapter / ControlNet 用）",
        ),
        PortSpec(
            name="scheduler",
            type="model",
            required=False,
            description="调度器（若未指定，子类用模型默认）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="latent",
            type="latent",
            description="采样后的 latent（Latent 类型）",
        )
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._steps: int = int(self.get_param("steps", 30))
        self._guidance: float = float(self.get_param("guidance", 7.5))
        self._size: str = self.get_param("size", "1024x1024")
        self._seed: Optional[int] = self.get_param("seed")
        self._batch_size: int = int(self.get_param("batch_size", 1))
        self._eta: float = float(self.get_param("eta", 0.0))
        self._sampler_name: Optional[str] = self.get_param("sampler_name")
        self._cfg_scheduler: Optional[str] = self.get_param("cfg_scheduler")
        self._device_override: Optional[str] = self.get_param("device")
        self._call_timeout: Optional[float] = self.get_param("call_timeout")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not isinstance(self._steps, int) or self._steps <= 0:
            errors.append(f"params.steps 必须为正整数，当前值：{self._steps}")
        if not isinstance(self._guidance, (int, float)) or self._guidance < 0:
            errors.append(f"params.guidance 必须为非负数，当前值：{self._guidance}")
        if not isinstance(self._size, str) or "x" not in self._size:
            errors.append(f"params.size 必须为 'WxH' 格式字符串，当前值：{self._size!r}")
        if self._seed is not None and (not isinstance(self._seed, int) or self._seed < 0):
            errors.append(f"params.seed 必须为非负整数或 None，当前值：{self._seed}")
        if not isinstance(self._batch_size, int) or self._batch_size <= 0:
            errors.append(f"params.batch_size 必须为正整数，当前值：{self._batch_size}")
        if not isinstance(self._eta, (int, float)) or self._eta < 0 or self._eta > 1:
            errors.append(f"params.eta 必须在 [0, 1] 范围内，当前值：{self._eta}")
        if self._call_timeout is not None and (
            not isinstance(self._call_timeout, (int, float)) or self._call_timeout <= 0
        ):
            errors.append(
                f"params.call_timeout 必须为正数或 None，当前值：{self._call_timeout}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：调用 diffusers pipeline 的 __call__ 方法。

        流程：
            1. 参数校验
            2. 提取输入（model / embeds / negative_embeds / images / scheduler）
            3. 解析 size 字符串为 (width, height)
            4. 上报开始进度
            5. 调用 _sample_impl（子类实现），传入 progress_callback
            6. 检查返回结果
            7. 上报完成进度
            8. 返回 {"latent": Latent}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        model = inputs.get("model")
        embeds = inputs.get("embeds")
        negative_embeds = inputs.get("negative_embeds")
        images = inputs.get("images")
        scheduler = inputs.get("scheduler")

        if model is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.model 必填（采样器需要主模型）"],
            )
        if embeds is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.embeds 必填（采样器需要正向 prompt 编码）"],
            )

        # 解析 size
        try:
            width, height = self._parse_size(self._size)
        except ValueError as exc:
            raise NodeValidationError(
                self.node_type,
                [f"params.size 解析失败：{exc}"],
            )

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="sampler.start",
            message=f"开始采样 {self._steps} 步",
            steps=self._steps,
            guidance=self._guidance,
            size=self._size,
            seed=self._seed,
            device=device,
            has_negative=negative_embeds is not None,
            has_images=images is not None,
        )

        # 构造 progress_callback（子类调用）
        async def progress_callback(step: int, total: int, **kwargs: Any) -> None:
            if total <= 0:
                return
            progress = step / total
            await ctx.report_progress(
                progress,
                stage="sampler.step",
                message=f"采样 {step}/{total}",
                step=step,
                total=total,
                **kwargs,
            )

        # 调用子类实现
        try:
            result = await self._sample_impl(
                inputs={
                    "model": model,
                    "embeds": embeds,
                    "negative_embeds": negative_embeds,
                    "images": images,
                    "scheduler": scheduler,
                },
                params={
                    "steps": self._steps,
                    "guidance": self._guidance,
                    "width": width,
                    "height": height,
                    "seed": self._seed,
                    "batch_size": self._batch_size,
                    "eta": self._eta,
                    "sampler_name": self._sampler_name,
                    "cfg_scheduler": self._cfg_scheduler,
                    "device": device,
                },
                ctx=ctx,
                progress_callback=progress_callback,
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _sample_impl 方法，"
                "请继承 SamplerNode 并实现该方法"
            )
        except Exception as exc:
            logger.exception("Sampler: 采样失败")
            raise RuntimeError(f"采样失败: {exc}") from exc

        if not isinstance(result, dict) or "latent" not in result:
            raise TypeError(
                f"_sample_impl 必须返回包含 'latent' 字段的 dict，"
                f"实际返回：{type(result).__name__}"
            )

        await ctx.report_progress(
            1.0,
            stage="sampler.done",
            message="采样完成",
        )
        return {"latent": result["latent"]}

    async def _sample_impl(
        self,
        inputs: dict,
        params: dict,
        ctx: WorkflowContext,
        progress_callback,
    ) -> dict:
        """实际采样逻辑（子类必须实现）。

        Args:
            inputs: 包含 model / embeds / negative_embeds / images / scheduler。
            params: 已解析的参数 dict（steps / guidance / width / height / seed / ...）。
            ctx: 工作流上下文。
            progress_callback: async 回调，签名 (step, total, **kwargs) -> None。
                子类应在每个采样步后调用以触发进度上报。

        Returns:
            dict 必须包含 "latent" 字段（Latent 类型）。

        Raises:
            NotImplementedError: 基类未实现此方法。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 _sample_impl 方法"
        )

    def _parse_size(self, size: str) -> tuple[int, int]:
        """解析 'WxH' 字符串为 (width, height)。

        支持格式：
            - "1024x1024"
            - "1024X1024"（大写也接受）
            - "1024*1024"
            - "1024×1024"（Unicode 乘号）

        Raises:
            ValueError: 格式不正确。
        """
        # 统一替换分隔符
        normalized = size.lower().replace("×", "x").replace("*", "x")
        parts = normalized.split("x")
        if len(parts) != 2:
            raise ValueError(f"size 格式不正确：{size!r}，应为 'WxH' 如 '1024x1024'")
        try:
            width = int(parts[0])
            height = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"size 宽高必须为整数：{size!r}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"size 宽高必须为正数：{size!r}")
        # 检查是否为 8 的倍数（diffusers 标准）
        if width % 8 != 0 or height % 8 != 0:
            logger.warning(
                "Sampler: size %dx%d 不是 8 的倍数，部分模型可能报错",
                width, height,
            )
        return width, height

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(steps={self._steps}, "
            f"guidance={self._guidance}, size={self._size!r}, "
            f"seed={self._seed})"
        )

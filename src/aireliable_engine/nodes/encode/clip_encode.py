"""CLIPTextEncode 节点：CLIP 文本编码节点（v0.2）。

设计为**抽象基类节点**：实际编码逻辑由引擎子类实现。
平台基础 CLIPTextEncodeNode 提供：
    - 标准输入端口（text / clip）
    - 标准输出端口（embeds / negative_embeds）
    - 标准参数（max_length / batch_size / truncate）
    - execute 流程（参数校验 → 进度上报 → 调用 _encode_impl → 返回）

支持 prompt + negative_prompt 两路编码，符合 OpenAI / diffusers 标准。

参考：
    - PR/architecture.md §12.2 / §12.2.6（节点示例）
    - PR/development-guide.md §28.5（数据类型规范）
    - v0.2-tasks.md 任务 3.5
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...types import Tensor
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class CLIPTextEncodeNode(BaseNode):
    """CLIP 文本编码节点（抽象基类）。

    引擎子类只需实现 `_encode_impl(text, negative_text, clip, ctx) -> dict`：
        - 返回 dict，必须包含 "embeds" 字段（Tensor 类型）
        - 可选包含 "negative_embeds" 字段（None 表示无负向编码）

    参数（params）：
        max_length (int)：最大 token 长度，默认 77（CLIP 标准）。
        batch_size (int)：batch 大小，默认 1。
        truncate (bool)：超长文本是否截断，默认 True。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
        pool_output (bool)：是否输出 pooled embeddings
            （用于 SDXL/Flux 等 pipeline），默认 False。
        return_pooled (bool)：同 pool_output（兼容命名），默认 False。
    """

    node_type = "CLIPTextEncode"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="text",
            type="text",
            required=True,
            description="正向文本 prompt",
        ),
        PortSpec(
            name="clip",
            type="model",
            required=True,
            description="CLIP 文本编码器模型（由 ModelLoader / CLIPTextEncodeLoader 输出）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="embeds",
            type="embeds",
            description="正向 prompt 编码后的 embedding（Tensor）",
        ),
        PortSpec(
            name="negative_embeds",
            type="embeds",
            required=False,
            description="负向 prompt 编码后的 embedding（Tensor），None 表示无负向编码",
        ),
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._max_length: int = int(self.get_param("max_length", 77))
        self._batch_size: int = int(self.get_param("batch_size", 1))
        # 不强制 bool() 转换：保留原值以便 validate 检查类型
        self._truncate = self.get_param("truncate", True)
        self._device_override: Optional[str] = self.get_param("device")
        self._pool_output = self.get_param("pool_output", False)
        self._return_pooled = self.get_param("return_pooled", False)

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not isinstance(self._max_length, int) or self._max_length <= 0:
            errors.append(f"params.max_length 必须为正整数，当前值：{self._max_length}")
        if not isinstance(self._batch_size, int) or self._batch_size <= 0:
            errors.append(f"params.batch_size 必须为正整数，当前值：{self._batch_size}")
        if not isinstance(self._truncate, bool):
            errors.append(f"params.truncate 必须为 bool，当前值：{self._truncate!r}")
        if not isinstance(self._pool_output, bool):
            errors.append(f"params.pool_output 必须为 bool，当前值：{self._pool_output!r}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：编码 prompt + negative_prompt 并返回 embeds / negative_embeds。

        流程：
            1. 参数校验
            2. 提取 inputs.text / inputs.clip
            3. 上报开始进度
            4. 调用 _encode_impl（子类实现）
            5. 检查返回结果，标准化为 {embeds, negative_embeds}
            6. 上报完成进度
            7. 返回

        注意：
            - inputs.negative_text 是可选的（None 表示无负向编码）
            - 子类 _encode_impl 接收 negative_text 参数，None 时只编码正向
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        text = inputs.get("text")
        clip = inputs.get("clip")
        negative_text = inputs.get("negative_text")  # 可选

        if not text:
            raise NodeValidationError(
                self.node_type,
                ["inputs.text 必填且必须为非空字符串"],
            )
        if clip is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.clip 必填（CLIP 文本编码器模型）"],
            )

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="clip_encode.start",
            message="开始 CLIP 文本编码",
            text_preview=str(text)[:100],
            has_negative=negative_text is not None,
            max_length=self._max_length,
            device=device,
        )

        # 调用子类实现
        try:
            result = await self._encode_impl(text, negative_text, clip, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _encode_impl 方法，"
                "请继承 CLIPTextEncodeNode 并实现该方法"
            )
        except Exception as exc:
            logger.exception("CLIPTextEncode: 编码失败")
            raise RuntimeError(f"CLIP 文本编码失败: {exc}") from exc

        if not isinstance(result, dict) or "embeds" not in result:
            raise TypeError(
                f"_encode_impl 必须返回包含 'embeds' 字段的 dict，"
                f"实际返回：{type(result).__name__}"
            )

        # 标准化输出
        standardized = {
            "embeds": result["embeds"],
            "negative_embeds": result.get("negative_embeds"),
        }

        await ctx.report_progress(
            1.0,
            stage="clip_encode.done",
            message="CLIP 文本编码完成",
        )
        return standardized

    async def _encode_impl(
        self,
        text: str,
        negative_text: Optional[str],
        clip: Any,
        ctx: WorkflowContext,
    ) -> dict:
        """实际编码逻辑（子类必须实现）。

        Args:
            text: 正向 prompt 文本。
            negative_text: 负向 prompt 文本（None 表示无负向编码）。
            clip: CLIP 文本编码器模型（由 ModelLoader / CLIPTextEncodeLoader 输出）。
            ctx: 工作流上下文。

        Returns:
            dict 必须包含 "embeds" 字段（Tensor 类型）。
            可选包含 "negative_embeds" 字段（None 表示无负向编码）。

        Raises:
            NotImplementedError: 基类未实现此方法。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 _encode_impl 方法"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_length={self._max_length}, "
            f"batch_size={self._batch_size}, truncate={self._truncate!r})"
        )

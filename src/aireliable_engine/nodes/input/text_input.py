"""TextInput 节点：文本输入节点（v0.2.5 简化版）。

v0.2.5 简化：
    - 删除 field / required / default / strip / allow_empty 5 个 advanced 参数
    - 仅保留 value 一个参数
    - 默认行为：允许空格、允许空字符串（strip=False, allow_empty=True 硬编码）
    - 用户在节点 body 内嵌的 textarea 中直接输入文本

特性：
    - params.value 为画布内直接编辑的文本（最高优先级）
    - API 调用时可通过 ctx.inputs["value"] 或 InferRequest.inputs["value"] 注入
    - 输出：text (str)

参考：
    - v0.2-analysis.md §5.2 / §11.2
    - PR/architecture.md §12.2 / §28.5
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class TextInputNode(BaseNode):
    """文本输入节点（v0.2.5 简化版）。

    参数（params）：
        value (str)：文本内容，默认空字符串。
            画布调试时用户在节点 body 内嵌 textarea 中直接输入。
            API 调用时可通过 ctx.inputs["value"] 或 InferRequest.inputs["value"] 注入。

    输出：
        text (str)：文本值。

    v0.2.5 简化：
        - 不做 strip（保留空格和首尾空白）
        - 允许空字符串（allow_empty=True 硬编码）
        - 删除 field/required/default/strip/allow_empty 参数
    """

    node_type = "TextInput"
    title = "文本输入"
    icon = "📝"
    description = "文本输入节点（节点内直接编辑）"
    category = "input"
    subcategory = "text"

    # v0.2.5 简化：仅保留 value 参数
    params_schema = {
        "value": {
            "type": "string",
            "required": False,
            "default": "",
            "label": "文本内容",
            "format": "inline-textarea",
            "description": "直接在节点内输入文本",
        },
    }

    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="text",
            type="text",
            description="文本内容",
        )
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # v0.2.5 简化：仅读取 value 参数
        self._value: str = self.get_param("value", "")

    def validate(self) -> list[str]:
        """参数校验（v0.2.5 简化：无强制校验，允许空字符串）。"""
        return []

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：返回文本值。

        v0.2.5 简化优先级：
            1. params.value                      （画布内直接编辑，最高优先级）
            2. ctx.inputs["value"]               （直接调用注入）
            3. ctx.extra["__infer_request__"].inputs["value"]  （InferRequest 注入）
            4. ""                                 （默认空字符串）

        返回：{"text": str}
        """
        text = self._collect_value(ctx)
        return {"text": text}

    def _collect_value(self, ctx: WorkflowContext) -> str:
        """收集文本值（v0.2.5 简化优先级）。

        v0.2.5 简化：
            - params.value 非空时优先（画布内编辑）
            - 不做 strip（保留空格）
            - 允许空字符串
            - value 为空字符串时回退到 ctx.inputs（API 注入）
        """
        # 优先级 0：params.value（画布内直接编辑）
        # v0.2.5 简化：仅当 value 非空字符串时使用（空字符串视为"未设置"，回退到 API 注入）
        # 注意：不做 strip，"   "（纯空格）视为非空，直接返回
        if self._value is not None and len(self._value) > 0:
            return str(self._value)

        # 优先级 1：ctx.inputs["value"]
        if ctx.inputs and "value" in ctx.inputs:
            raw = ctx.inputs["value"]
            return self._to_str(raw)

        # 优先级 2：InferRequest.inputs["value"]
        request_obj = ctx.extra.get("__infer_request__") if ctx.extra else None
        if request_obj is not None:
            request_inputs = getattr(request_obj, "inputs", None)
            if isinstance(request_inputs, dict) and "value" in request_inputs:
                return self._to_str(request_inputs["value"])

        # 默认：空字符串
        return ""

    @staticmethod
    def _to_str(raw: Any) -> str:
        """类型转换：非 str 自动转 str（v0.2.5 简化：不做 strip）。"""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, bool):
            return "true" if raw else "false"
        if isinstance(raw, (int, float)):
            return str(raw)
        if isinstance(raw, list):
            return " ".join(str(item) for item in raw)
        return str(raw)

    def __repr__(self) -> str:
        return f"TextInputNode(value={self._value!r})"

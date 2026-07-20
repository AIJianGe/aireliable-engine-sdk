"""TextInput 节点：文本输入节点。

从请求的 inputs 字典中读取指定字段的文本值。
常用于 prompt / negative_prompt / style / artist 等文本参数接入工作流。

特性：
    - 通过 params.field 指定读取字段名
    - 通过 params.required 控制字段是否必需（默认 True）
    - 通过 params.default 指定可选字段的默认值
    - 支持 str / int / float / bool 自动转 str
    - 支持从 InferRequest.inputs 或 ctx.inputs 读取（优先级）

参考：
    - v0.2-analysis.md §5.2 / §11.2
    - PR/architecture.md §12.2 / §28.5
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class TextInputNode(BaseNode):
    """文本输入节点。

    参数（params）：
        field (str)：必填，从请求 inputs 中读取的字段名。
            示例："prompt" / "negative_prompt" / "style"。
        required (bool)：字段是否必需，默认 True。
            required=True 且字段缺失时抛 NodeValidationError。
            required=False 且字段缺失时返回 default。
        default (Any)：可选字段的默认值，默认 None（输出空字符串）。
            仅当 required=False 时生效。
        strip (bool)：是否去除首尾空白，默认 True。
        allow_empty (bool)：是否允许空字符串，默认 True。
            allow_empty=False 且字段为空字符串时报错。

    输出：
        text (str)：读取到的文本值。

    示例：
        # 必填 prompt
        TextInputNode({"field": "prompt"})

        # 可选 negative_prompt，缺省空字符串
        TextInputNode({
            "field": "negative_prompt",
            "required": False,
            "default": "",
        })

        # 必填且不允许空
        TextInputNode({
            "field": "prompt",
            "required": True,
            "allow_empty": False,
        })
    """

    node_type = "TextInput"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="text",
            type="text",
            description="从请求 inputs 中读取的文本字段",
        )
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._field: str = self.get_param("field", "")
        # 不强制 bool() 转换：保留原值以便 validate 检查类型
        self._required = self.get_param("required", True)
        self._default: Any = self.get_param("default", None)
        self._strip = self.get_param("strip", True)
        self._allow_empty = self.get_param("allow_empty", True)

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._field or not isinstance(self._field, str):
            errors.append("params.field 必填且必须为非空字符串")
        if not isinstance(self._required, bool):
            errors.append(f"params.required 必须为 bool，当前值：{self._required!r}")
        if not isinstance(self._strip, bool):
            errors.append(f"params.strip 必须为 bool，当前值：{self._strip!r}")
        if not isinstance(self._allow_empty, bool):
            errors.append(f"params.allow_empty 必须为 bool，当前值：{self._allow_empty!r}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：从 ctx.inputs 读取指定字段并返回 text。

        流程：
            1. 校验参数（field 不能为空）
            2. 从 ctx 收集字段值（ctx.inputs 优先，InferRequest.inputs 备选）
            3. 处理 required / default 逻辑
            4. 类型转换（非 str 自动转 str）
            5. strip / allow_empty 处理
            6. 返回 {"text": str}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        raw_value = self._collect_field_value(ctx)
        text = self._process_value(raw_value)
        return {"text": text}

    def _collect_field_value(self, ctx: WorkflowContext) -> Any:
        """从 ctx 收集字段值。

        优先级：
            1. ctx.inputs[field]                  （直接调用）
            2. ctx.extra["__infer_request__"].inputs[field] （完整 InferRequest 传递）

        返回值：
            - 字段存在：原始值（str/int/float/bool）
            - 字段不存在：哨兵 _MISSING
        """
        # 优先级 1：ctx.inputs
        if ctx.inputs and self._field in ctx.inputs:
            return ctx.inputs[self._field]

        # 优先级 2：完整 InferRequest 传递
        request_obj = ctx.extra.get("__infer_request__") if ctx.extra else None
        if request_obj is not None:
            request_inputs = getattr(request_obj, "inputs", None)
            if isinstance(request_inputs, dict) and self._field in request_inputs:
                return request_inputs[self._field]

        return _MISSING

    def _process_value(self, raw_value: Any) -> str:
        """处理原始值：required / default / 类型转换 / strip / allow_empty。"""
        if raw_value is _MISSING:
            if self._required:
                raise NodeValidationError(
                    self.node_type,
                    [f"必填字段 '{self._field}' 缺失"],
                )
            # 可选字段缺失，用 default
            raw_value = self._default

        # None 特殊处理
        if raw_value is None:
            if self._required:
                raise NodeValidationError(
                    self.node_type,
                    [f"必填字段 '{self._field}' 值为 None"],
                )
            return ""

        # 类型转换：非 str 自动转 str
        if isinstance(raw_value, str):
            text = raw_value
        elif isinstance(raw_value, bool):
            # bool 必须在 int 之前判断（bool 是 int 子类）
            text = "true" if raw_value else "false"
        elif isinstance(raw_value, (int, float)):
            text = str(raw_value)
        elif isinstance(raw_value, list):
            # 列表自动 join（如多 prompt 合并）
            text = " ".join(str(item) for item in raw_value)
        else:
            # 其他类型转 str
            text = str(raw_value)

        # strip
        if self._strip:
            text = text.strip()

        # allow_empty
        if not self._allow_empty and not text:
            raise NodeValidationError(
                self.node_type,
                [f"字段 '{self._field}' 不允许为空字符串"],
            )

        return text

    def __repr__(self) -> str:
        return (
            f"TextInputNode(field={self._field!r}, required={self._required!r}, "
            f"default={self._default!r})"
        )


class _MissingSentinel:
    """字段缺失哨兵（区分 None 与缺失）。"""

    _instance: "_MissingSentinel"

    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


_MISSING = _MissingSentinel()

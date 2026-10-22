"""TextInput 节点单元测试（v0.2.5 简化版）。

v0.2.5 简化：
    - 删除 field / required / default / strip / allow_empty 5 个参数
    - 仅保留 value 一个参数
    - 默认允许空格、允许空字符串（不做 strip）

覆盖：
    - 节点基本属性
    - value 画布内直接编辑
    - ctx.inputs 注入
    - InferRequest 注入
    - 类型转换
    - 优先级
    - 空值处理

"""

from __future__ import annotations

import pytest

from aireliable_engine import InferRequest, make_default_context
from aireliable_engine.nodes.input.text_input import TextInputNode


# ===== 基本属性测试 =====


class TestTextInputNodeBasic:
    """TextInputNode 基本属性测试（v0.2.5 简化版）。"""

    def test_node_type_is_textinput(self):
        node = TextInputNode()
        assert node.node_type == "TextInput"

    def test_input_specs_is_empty_list(self):
        node = TextInputNode()
        assert node.input_specs == []

    def test_output_specs_has_text_port(self):
        node = TextInputNode()
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "text"
        assert port.type == "text"

    def test_default_value_is_empty_string(self):
        """v0.2.5 简化：默认 value 为空字符串。"""
        node = TextInputNode()
        assert node._value == ""

    def test_init_with_value(self):
        """v0.2.5 简化：通过 value 参数初始化。"""
        node = TextInputNode({"value": "hello world"})
        assert node._value == "hello world"

    def test_validate_returns_empty(self):
        """v0.2.5 简化：validate 始终返回空列表（无强制校验）。"""
        node = TextInputNode()
        assert node.validate() == []

    def test_validate_returns_empty_with_empty_value(self):
        """空 value 也通过校验（允许空字符串）。"""
        node = TextInputNode({"value": ""})
        assert node.validate() == []

    def test_repr_contains_value(self):
        """v0.2.5 简化：repr 包含 value。"""
        node = TextInputNode({"value": "test"})
        repr_str = repr(node)
        assert "TextInputNode" in repr_str
        assert "test" in repr_str

    def test_params_schema_only_has_value(self):
        """v0.2.5 简化：params_schema 仅包含 value 一个参数。"""
        schema = TextInputNode.params_schema
        assert list(schema.keys()) == ["value"]
        assert schema["value"]["format"] == "inline-textarea"


# ===== value 画布内编辑测试 =====


class TestTextInputNodeValue:
    """value 参数（画布内直接编辑）测试。"""

    @pytest.mark.asyncio
    async def test_value_returned_directly(self):
        """params.value 直接作为输出。"""
        node = TextInputNode({"value": "a beautiful cat"})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == "a beautiful cat"

    @pytest.mark.asyncio
    async def test_empty_value_returns_empty_string(self):
        """空 value 返回空字符串。"""
        node = TextInputNode({"value": ""})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_value_with_whitespace_preserved(self):
        """v0.2.5 简化：不做 strip，保留空格。"""
        node = TextInputNode({"value": "  hello world  "})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == "  hello world  "

    @pytest.mark.asyncio
    async def test_value_with_newlines_preserved(self):
        """v0.2.5 简化：保留换行符。"""
        node = TextInputNode({"value": "line1\nline2\nline3"})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == "line1\nline2\nline3"

    @pytest.mark.asyncio
    async def test_value_takes_priority_over_ctx_inputs(self):
        """params.value 优先级高于 ctx.inputs["value"]。"""
        node = TextInputNode({"value": "from params"})
        ctx = make_default_context(inputs={"value": "from ctx"})
        result = await node.execute({}, ctx)
        assert result["text"] == "from params"


# ===== ctx.inputs 注入测试 =====


class TestTextInputNodeCtxInputs:
    """ctx.inputs 注入测试（API 调用场景）。"""

    @pytest.mark.asyncio
    async def test_ctx_inputs_value_when_params_value_empty(self):
        """params.value 为空时，从 ctx.inputs["value"] 读取。"""
        node = TextInputNode({"value": ""})
        ctx = make_default_context(inputs={"value": "from ctx"})
        result = await node.execute({}, ctx)
        assert result["text"] == "from ctx"

    @pytest.mark.asyncio
    async def test_ctx_inputs_string_value(self):
        """ctx.inputs 注入字符串。"""
        node = TextInputNode()
        ctx = make_default_context(inputs={"value": "hello world"})
        result = await node.execute({}, ctx)
        assert result["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_ctx_inputs_int_converted_to_str(self):
        """ctx.inputs 注入 int，自动转 str。"""
        node = TextInputNode()
        ctx = make_default_context(inputs={"value": 30})
        result = await node.execute({}, ctx)
        assert result["text"] == "30"

    @pytest.mark.asyncio
    async def test_ctx_inputs_float_converted_to_str(self):
        """ctx.inputs 注入 float，自动转 str。"""
        node = TextInputNode()
        ctx = make_default_context(inputs={"value": 7.5})
        result = await node.execute({}, ctx)
        assert result["text"] == "7.5"

    @pytest.mark.asyncio
    async def test_ctx_inputs_bool_true_converted_to_str(self):
        """ctx.inputs 注入 bool True，转为 "true"。"""
        node = TextInputNode()
        ctx = make_default_context(inputs={"value": True})
        result = await node.execute({}, ctx)
        assert result["text"] == "true"

    @pytest.mark.asyncio
    async def test_ctx_inputs_bool_false_converted_to_str(self):
        """ctx.inputs 注入 bool False，转为 "false"。"""
        node = TextInputNode()
        ctx = make_default_context(inputs={"value": False})
        result = await node.execute({}, ctx)
        assert result["text"] == "false"

    @pytest.mark.asyncio
    async def test_ctx_inputs_list_joined_with_space(self):
        """ctx.inputs 注入 list，用空格 join。"""
        node = TextInputNode()
        ctx = make_default_context(inputs={"value": ["cat", "cute", "fluffy"]})
        result = await node.execute({}, ctx)
        assert result["text"] == "cat cute fluffy"


# ===== InferRequest 注入测试 =====


class TestTextInputNodeInferRequest:
    """InferRequest 注入测试。"""

    @pytest.mark.asyncio
    async def test_read_from_infer_request_inputs(self):
        """params.value 为空时，从 InferRequest.inputs["value"] 读取。"""
        node = TextInputNode({"value": ""})
        request = InferRequest(
            variant="flux",
            inputs={"value": "from request"},
        )
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert result["text"] == "from request"

    @pytest.mark.asyncio
    async def test_ctx_inputs_takes_priority_over_infer_request(self):
        """ctx.inputs["value"] 优先级高于 InferRequest.inputs["value"]。"""
        node = TextInputNode({"value": ""})
        request = InferRequest(
            variant="flux",
            inputs={"value": "from request"},
        )
        ctx = make_default_context(
            inputs={"value": "from ctx"},
            extra={"__infer_request__": request},
        )
        result = await node.execute({}, ctx)
        assert result["text"] == "from ctx"

    @pytest.mark.asyncio
    async def test_params_value_takes_priority_over_everything(self):
        """params.value 优先级最高（高于 ctx.inputs 和 InferRequest）。"""
        node = TextInputNode({"value": "from params"})
        request = InferRequest(
            variant="flux",
            inputs={"value": "from request"},
        )
        ctx = make_default_context(
            inputs={"value": "from ctx"},
            extra={"__infer_request__": request},
        )
        result = await node.execute({}, ctx)
        assert result["text"] == "from params"

    @pytest.mark.asyncio
    async def test_all_empty_returns_empty_string(self):
        """所有来源都为空时，返回空字符串（不报错）。"""
        node = TextInputNode({"value": ""})
        request = InferRequest(variant="flux", inputs={})
        ctx = make_default_context(
            inputs={},
            extra={"__infer_request__": request},
        )
        result = await node.execute({}, ctx)
        assert result["text"] == ""


# ===== 集成场景测试 =====


class TestTextInputNodeIntegration:
    """集成场景测试。"""

    @pytest.mark.asyncio
    async def test_canvas_debugging_scenario(self):
        """画布调试场景：用户在节点 textarea 中输入文本。"""
        node = TextInputNode({"value": "一只可爱的猫咪坐在椅子上"})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == "一只可爱的猫咪坐在椅子上"

    @pytest.mark.asyncio
    async def test_api_call_scenario(self):
        """API 调用场景：通过 ctx.inputs 注入文本。"""
        node = TextInputNode({"value": ""})
        ctx = make_default_context(inputs={"value": "a beautiful landscape"})
        result = await node.execute({}, ctx)
        assert result["text"] == "a beautiful landscape"

    @pytest.mark.asyncio
    async def test_infer_request_scenario(self):
        """完整 InferRequest 调用场景。"""
        node = TextInputNode({"value": ""})
        request = InferRequest(
            variant="flux",
            inputs={"value": "surreal painting of a floating island"},
        )
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert result["text"] == "surreal painting of a floating island"

    @pytest.mark.asyncio
    async def test_multiline_text_scenario(self):
        """多行文本场景（textarea 支持）。"""
        multiline = "prompt line 1\nprompt line 2\nprompt line 3"
        node = TextInputNode({"value": multiline})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == multiline

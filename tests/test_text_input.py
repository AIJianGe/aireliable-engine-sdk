"""TextInput 节点单元测试。

覆盖任务 3.2 验收标准：
    - 从请求 inputs.<field> 读取
    - 支持 required / optional
    - required 字段缺失时报错

参考：v0.2-analysis.md §5.2 / PR/architecture.md §12.2
"""

from __future__ import annotations

import pytest

from aireliable_engine import InferRequest, make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.input.text_input import TextInputNode


# ===== 基本属性测试 =====


class TestTextInputNodeBasic:
    """TextInputNode 基本属性测试。"""

    def test_node_type_is_textinput(self):
        node = TextInputNode({"field": "prompt"})
        assert node.node_type == "TextInput"

    def test_input_specs_is_empty_list(self):
        node = TextInputNode({"field": "prompt"})
        assert node.input_specs == []

    def test_output_specs_has_text_port(self):
        node = TextInputNode({"field": "prompt"})
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "text"
        assert port.type == "text"

    def test_default_required_is_true(self):
        node = TextInputNode({"field": "prompt"})
        assert node._required is True

    def test_default_strip_is_true(self):
        node = TextInputNode({"field": "prompt"})
        assert node._strip is True

    def test_default_allow_empty_is_true(self):
        node = TextInputNode({"field": "prompt"})
        assert node._allow_empty is True

    def test_default_default_is_none(self):
        node = TextInputNode({"field": "prompt"})
        assert node._default is None

    def test_repr_contains_field_and_required(self):
        node = TextInputNode({"field": "prompt", "required": False, "default": ""})
        repr_str = repr(node)
        assert "TextInputNode" in repr_str
        assert "prompt" in repr_str
        assert "False" in repr_str


# ===== 参数校验测试 =====


class TestTextInputNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_valid_params(self):
        node = TextInputNode({"field": "prompt"})
        assert node.validate() == []

    def test_validate_reports_missing_field(self):
        node = TextInputNode({})
        errors = node.validate()
        assert any("field" in e for e in errors)

    def test_validate_reports_empty_field(self):
        node = TextInputNode({"field": ""})
        errors = node.validate()
        assert any("field" in e for e in errors)

    def test_validate_reports_non_bool_required(self):
        node = TextInputNode({"field": "prompt", "required": "yes"})
        errors = node.validate()
        assert any("required" in e for e in errors)


# ===== required 字段测试（验收标准）=====


class TestTextInputNodeRequired:
    """required 字段测试（任务 3.2 验收标准）。"""

    @pytest.mark.asyncio
    async def test_required_field_present_returns_value(self):
        """required=True 且字段存在，返回值。"""
        node = TextInputNode({"field": "prompt", "required": True})
        ctx = make_default_context(inputs={"prompt": "a beautiful cat"})
        result = await node.execute({}, ctx)
        assert result["text"] == "a beautiful cat"

    @pytest.mark.asyncio
    async def test_required_field_missing_raises_error(self):
        """required=True 且字段缺失，抛 NodeValidationError。"""
        node = TextInputNode({"field": "prompt", "required": True})
        ctx = make_default_context(inputs={})
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "prompt" in str(exc_info.value)
        assert "缺失" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_required_field_none_raises_error(self):
        """required=True 且字段值=None，抛 NodeValidationError。"""
        node = TextInputNode({"field": "prompt", "required": True})
        ctx = make_default_context(inputs={"prompt": None})
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_optional_field_missing_returns_default(self):
        """required=False 且字段缺失，返回 default。"""
        node = TextInputNode({
            "field": "negative_prompt",
            "required": False,
            "default": "low quality",
        })
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == "low quality"

    @pytest.mark.asyncio
    async def test_optional_field_missing_default_none_returns_empty(self):
        """required=False 且字段缺失，default=None 返回空字符串。"""
        node = TextInputNode({
            "field": "negative_prompt",
            "required": False,
            "default": None,
        })
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)
        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_optional_field_present_returns_value(self):
        """required=False 但字段存在，返回值（不使用 default）。"""
        node = TextInputNode({
            "field": "negative_prompt",
            "required": False,
            "default": "fallback",
        })
        ctx = make_default_context(inputs={"negative_prompt": "blurry"})
        result = await node.execute({}, ctx)
        assert result["text"] == "blurry"


# ===== 类型转换测试 =====


class TestTextInputNodeTypeConversion:
    """类型转换测试。"""

    @pytest.mark.asyncio
    async def test_int_value_converted_to_str(self):
        node = TextInputNode({"field": "steps"})
        ctx = make_default_context(inputs={"steps": 30})
        result = await node.execute({}, ctx)
        assert result["text"] == "30"

    @pytest.mark.asyncio
    async def test_float_value_converted_to_str(self):
        node = TextInputNode({"field": "guidance"})
        ctx = make_default_context(inputs={"guidance": 7.5})
        result = await node.execute({}, ctx)
        assert result["text"] == "7.5"

    @pytest.mark.asyncio
    async def test_bool_true_converted_to_str(self):
        node = TextInputNode({"field": "enhance"})
        ctx = make_default_context(inputs={"enhance": True})
        result = await node.execute({}, ctx)
        assert result["text"] == "true"

    @pytest.mark.asyncio
    async def test_bool_false_converted_to_str(self):
        node = TextInputNode({"field": "enhance"})
        ctx = make_default_context(inputs={"enhance": False})
        result = await node.execute({}, ctx)
        assert result["text"] == "false"

    @pytest.mark.asyncio
    async def test_list_value_joined_with_space(self):
        node = TextInputNode({"field": "tags"})
        ctx = make_default_context(inputs={"tags": ["cat", "cute", "fluffy"]})
        result = await node.execute({}, ctx)
        assert result["text"] == "cat cute fluffy"

    @pytest.mark.asyncio
    async def test_str_value_passed_through(self):
        node = TextInputNode({"field": "prompt"})
        ctx = make_default_context(inputs={"prompt": "hello world"})
        result = await node.execute({}, ctx)
        assert result["text"] == "hello world"


# ===== strip / allow_empty 测试 =====


class TestTextInputNodeStripAndEmpty:
    """strip 和 allow_empty 测试。"""

    @pytest.mark.asyncio
    async def test_strip_removes_whitespace(self):
        node = TextInputNode({"field": "prompt", "strip": True})
        ctx = make_default_context(inputs={"prompt": "  hello world  "})
        result = await node.execute({}, ctx)
        assert result["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_strip_disabled_preserves_whitespace(self):
        node = TextInputNode({"field": "prompt", "strip": False})
        ctx = make_default_context(inputs={"prompt": "  hello  "})
        result = await node.execute({}, ctx)
        assert result["text"] == "  hello  "

    @pytest.mark.asyncio
    async def test_allow_empty_false_raises_on_empty(self):
        """allow_empty=False 且字段为空字符串，抛错。"""
        node = TextInputNode({
            "field": "prompt",
            "required": True,
            "allow_empty": False,
        })
        ctx = make_default_context(inputs={"prompt": ""})
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "不允许为空" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_allow_empty_false_raises_on_whitespace_only(self):
        """strip=True + allow_empty=False + 仅空白字符 → 抛错。"""
        node = TextInputNode({
            "field": "prompt",
            "required": True,
            "allow_empty": False,
            "strip": True,
        })
        ctx = make_default_context(inputs={"prompt": "   "})
        with pytest.raises(NodeValidationError):
            await node.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_allow_empty_true_accepts_empty(self):
        """allow_empty=True 允许空字符串通过。"""
        node = TextInputNode({
            "field": "prompt",
            "required": True,
            "allow_empty": True,
        })
        ctx = make_default_context(inputs={"prompt": ""})
        result = await node.execute({}, ctx)
        assert result["text"] == ""


# ===== InferRequest 传递测试 =====


class TestTextInputNodeInferRequest:
    """通过完整 InferRequest 传递的测试。"""

    @pytest.mark.asyncio
    async def test_read_from_infer_request_inputs(self):
        """从 InferRequest.inputs 读取字段。"""
        node = TextInputNode({"field": "prompt"})
        request = InferRequest(
            variant="flux",
            inputs={"prompt": "a cat sitting on a chair"},
        )
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert result["text"] == "a cat sitting on a chair"

    @pytest.mark.asyncio
    async def test_ctx_inputs_takes_priority_over_infer_request(self):
        """ctx.inputs 优先级高于 InferRequest.inputs。"""
        node = TextInputNode({"field": "prompt"})
        request = InferRequest(
            variant="flux",
            inputs={"prompt": "from request"},
        )
        ctx = make_default_context(
            inputs={"prompt": "from ctx"},
            extra={"__infer_request__": request},
        )
        result = await node.execute({}, ctx)
        assert result["text"] == "from ctx"

    @pytest.mark.asyncio
    async def test_infer_request_fallback_when_ctx_inputs_empty(self):
        """ctx.inputs 为空时回退到 InferRequest.inputs。"""
        node = TextInputNode({"field": "negative_prompt"})
        request = InferRequest(
            variant="flux",
            inputs={"prompt": "a cat", "negative_prompt": "blurry"},
        )
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert result["text"] == "blurry"

    @pytest.mark.asyncio
    async def test_infer_request_field_missing_raises_when_required(self):
        """InferRequest.inputs 中字段缺失且 required=True，抛错。"""
        node = TextInputNode({"field": "nonexistent_field", "required": True})
        request = InferRequest(variant="flux", inputs={"prompt": "a cat"})
        ctx = make_default_context(extra={"__infer_request__": request})
        with pytest.raises(NodeValidationError):
            await node.execute({}, ctx)


# ===== 错误处理测试 =====


class TestTextInputNodeErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_missing_field_param_raises_validation_error(self):
        """params 未指定 field，execute 抛 NodeValidationError。"""
        node = TextInputNode({})
        ctx = make_default_context(inputs={"prompt": "hello"})
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "field" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_field_param_raises_validation_error(self):
        """params.field 为空字符串，execute 抛 NodeValidationError。"""
        node = TextInputNode({"field": ""})
        ctx = make_default_context(inputs={"prompt": "hello"})
        with pytest.raises(NodeValidationError):
            await node.execute({}, ctx)


# ===== 集成场景测试 =====


class TestTextInputNodeIntegration:
    """集成场景测试。"""

    @pytest.mark.asyncio
    async def test_typical_prompt_workflow(self):
        """典型文生图 prompt + negative_prompt 工作流。"""
        prompt_node = TextInputNode({"field": "prompt", "required": True})
        negative_node = TextInputNode({
            "field": "negative_prompt",
            "required": False,
            "default": "low quality, blurry",
        })
        ctx = make_default_context(inputs={
            "prompt": "a cute cat",
            # negative_prompt 故意缺失，用默认值
        })
        prompt_result = await prompt_node.execute({}, ctx)
        negative_result = await negative_node.execute({}, ctx)
        assert prompt_result["text"] == "a cute cat"
        assert negative_result["text"] == "low quality, blurry"

    @pytest.mark.asyncio
    async def test_typical_full_request_workflow(self):
        """完整请求工作流：prompt / negative / style 都存在。"""
        request = InferRequest(
            variant="flux",
            inputs={
                "prompt": "a cute cat in style of van gogh",
                "negative_prompt": "blurry, low quality",
                "style": "post-impressionism",
            },
        )
        ctx = make_default_context(extra={"__infer_request__": request})

        prompt_node = TextInputNode({"field": "prompt"})
        negative_node = TextInputNode({
            "field": "negative_prompt",
            "required": False,
            "default": "",
        })
        style_node = TextInputNode({
            "field": "style",
            "required": False,
            "default": "realism",
        })

        assert (await prompt_node.execute({}, ctx))["text"] == "a cute cat in style of van gogh"
        assert (await negative_node.execute({}, ctx))["text"] == "blurry, low quality"
        assert (await style_node.execute({}, ctx))["text"] == "post-impressionism"

    @pytest.mark.asyncio
    async def test_numeric_field_as_text(self):
        """数字字段（如 steps）作为文本读取。"""
        node = TextInputNode({"field": "steps"})
        ctx = make_default_context(inputs={"steps": 50})
        result = await node.execute({}, ctx)
        assert result["text"] == "50"

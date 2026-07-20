"""CLIPTextEncode 节点单元测试。

覆盖任务 3.5 验收标准：
    - 支持 prompt 和 negative_prompt

参考：v0.2-tasks.md 任务 3.5
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.encode.clip_encode import CLIPTextEncodeNode


# ===== Mock 子类 =====


class _MockCLIPTextEncode(CLIPTextEncodeNode):
    """模拟 CLIP 文本编码器。"""

    node_type = "MockCLIPTextEncode"

    def __init__(self, params, return_value: dict = None, raise_exc: Exception = None):
        super().__init__(params)
        # 默认返回 mock Tensor
        if return_value is None:
            return_value = {
                "embeds": MagicMock(name="positive_embeds"),
                "negative_embeds": MagicMock(name="negative_embeds"),
            }
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.encode_called_with_args = None

    async def _encode_impl(
        self,
        text: str,
        negative_text: Any,
        clip: Any,
        ctx,
    ) -> dict:
        self.encode_called_with_args = (text, negative_text, clip)
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _OnlyPositiveEncoder(CLIPTextEncodeNode):
    """只编码正向 prompt（无负向）。"""

    node_type = "OnlyPositiveEncoder"

    async def _encode_impl(self, text, negative_text, clip, ctx) -> dict:
        return {"embeds": MagicMock(name="positive_embeds")}  # 无 negative_embeds


class _FailingEncoder(CLIPTextEncodeNode):
    node_type = "FailingEncoder"

    async def _encode_impl(self, text, negative_text, clip, ctx) -> dict:
        raise RuntimeError("CLIP 编码失败")


class _InvalidReturnEncoder(CLIPTextEncodeNode):
    """返回 dict 缺少 embeds 字段。"""

    node_type = "InvalidReturnEncoder"

    async def _encode_impl(self, text, negative_text, clip, ctx) -> dict:
        return {"wrong_field": MagicMock()}


class _NonDictEncoder(CLIPTextEncodeNode):
    node_type = "NonDictEncoder"

    async def _encode_impl(self, text, negative_text, clip, ctx) -> dict:
        return "not a dict"


class _BareEncoder(CLIPTextEncodeNode):
    """未重写 _encode_impl。"""

    node_type = "BareEncoder"


# ===== 基本属性测试 =====


class TestCLIPTextEncodeNodeBasic:
    """CLIPTextEncodeNode 基本属性测试。"""

    def test_node_type_is_cliptextencode(self):
        node = CLIPTextEncodeNode()
        assert node.node_type == "CLIPTextEncode"

    def test_input_specs_has_text_and_clip(self):
        node = CLIPTextEncodeNode()
        names = [p.name for p in node.input_specs]
        assert "text" in names
        assert "clip" in names

    def test_output_specs_has_embeds_and_negative_embeds(self):
        """输出端口：embeds / negative_embeds（验收标准）。"""
        node = CLIPTextEncodeNode()
        names = [p.name for p in node.output_specs]
        assert "embeds" in names
        assert "negative_embeds" in names
        # embeds 必填，negative_embeds 可选
        embeds_port = next(p for p in node.output_specs if p.name == "embeds")
        negative_port = next(p for p in node.output_specs if p.name == "negative_embeds")
        assert embeds_port.required is True
        assert negative_port.required is False

    def test_default_max_length_is_77(self):
        """默认 max_length=77（CLIP 标准）。"""
        node = CLIPTextEncodeNode()
        assert node._max_length == 77

    def test_default_batch_size_is_1(self):
        node = CLIPTextEncodeNode()
        assert node._batch_size == 1

    def test_default_truncate_is_true(self):
        node = CLIPTextEncodeNode()
        assert node._truncate is True

    def test_init_with_params(self):
        node = CLIPTextEncodeNode({
            "max_length": 128,
            "batch_size": 2,
            "truncate": False,
            "device": "cuda:0",
            "pool_output": True,
        })
        assert node._max_length == 128
        assert node._batch_size == 2
        assert node._truncate is False
        assert node._device_override == "cuda:0"
        assert node._pool_output is True

    def test_repr_contains_max_length_and_batch_size(self):
        node = CLIPTextEncodeNode({"max_length": 128})
        repr_str = repr(node)
        assert "CLIPTextEncodeNode" in repr_str
        assert "128" in repr_str


# ===== 参数校验测试 =====


class TestCLIPTextEncodeNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_default_params(self):
        node = CLIPTextEncodeNode()
        assert node.validate() == []

    def test_validate_reports_negative_max_length(self):
        node = CLIPTextEncodeNode({"max_length": -1})
        errors = node.validate()
        assert any("max_length" in e for e in errors)

    def test_validate_reports_zero_max_length(self):
        node = CLIPTextEncodeNode({"max_length": 0})
        errors = node.validate()
        assert any("max_length" in e for e in errors)

    def test_validate_reports_negative_batch_size(self):
        node = CLIPTextEncodeNode({"batch_size": -1})
        errors = node.validate()
        assert any("batch_size" in e for e in errors)

    def test_validate_reports_non_bool_truncate(self):
        node = CLIPTextEncodeNode({"truncate": "yes"})
        errors = node.validate()
        assert any("truncate" in e for e in errors)


# ===== prompt + negative_prompt 测试（验收标准）=====


class TestCLIPTextEncodeNodePromptAndNegative:
    """prompt + negative_prompt 编码测试（任务 3.5 验收标准）。"""

    @pytest.mark.asyncio
    async def test_prompt_and_negative_passed_to_encode_impl(self):
        """prompt 和 negative_prompt 都传递到 _encode_impl。"""
        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context()
        mock_clip = MagicMock(name="mock_clip_model")
        inputs = {
            "text": "a beautiful cat",
            "negative_text": "blurry, low quality",
            "clip": mock_clip,
        }
        await encoder.execute(inputs, ctx)

        # 验证 _encode_impl 被正确调用
        called_text, called_neg, called_clip = encoder.encode_called_with_args
        assert called_text == "a beautiful cat"
        assert called_neg == "blurry, low quality"
        assert called_clip is mock_clip

    @pytest.mark.asyncio
    async def test_prompt_only_no_negative(self):
        """只传 prompt，negative_text 为 None。"""
        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context()
        inputs = {
            "text": "a beautiful cat",
            "clip": MagicMock(),
        }
        await encoder.execute(inputs, ctx)

        called_text, called_neg, _ = encoder.encode_called_with_args
        assert called_text == "a beautiful cat"
        assert called_neg is None

    @pytest.mark.asyncio
    async def test_returns_embeds_and_negative_embeds(self):
        """返回 embeds 和 negative_embeds 两个字段。"""
        positive = MagicMock(name="positive")
        negative = MagicMock(name="negative")
        encoder = _MockCLIPTextEncode(
            {},
            return_value={"embeds": positive, "negative_embeds": negative},
        )
        ctx = make_default_context()
        inputs = {
            "text": "a cat",
            "negative_text": "blurry",
            "clip": MagicMock(),
        }
        result = await encoder.execute(inputs, ctx)
        assert result["embeds"] is positive
        assert result["negative_embeds"] is negative

    @pytest.mark.asyncio
    async def test_only_positive_no_negative_in_result(self):
        """子类只返回 embeds，negative_embeds 填 None。"""
        encoder = _OnlyPositiveEncoder({})
        ctx = make_default_context()
        inputs = {"text": "a cat", "clip": MagicMock()}
        result = await encoder.execute(inputs, ctx)
        assert "embeds" in result
        assert result["embeds"] is not None
        assert result["negative_embeds"] is None


# ===== 错误处理测试 =====


class TestCLIPTextEncodeNodeErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_missing_text_raises_validation_error(self):
        """缺少 text 输入抛 NodeValidationError。"""
        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context()
        inputs = {"clip": MagicMock()}  # 缺 text
        with pytest.raises(NodeValidationError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "text" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_text_raises_validation_error(self):
        """text 为空字符串抛 NodeValidationError。"""
        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context()
        inputs = {"text": "", "clip": MagicMock()}
        with pytest.raises(NodeValidationError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "text" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_clip_raises_validation_error(self):
        """缺少 clip 输入抛 NodeValidationError。"""
        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context()
        inputs = {"text": "a cat"}  # 缺 clip
        with pytest.raises(NodeValidationError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "clip" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bare_encoder_raises_not_implemented(self):
        """未重写 _encode_impl 的子类抛 NotImplementedError。"""
        encoder = _BareEncoder({})
        ctx = make_default_context()
        inputs = {"text": "a cat", "clip": MagicMock()}
        with pytest.raises(NotImplementedError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "_encode_impl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_encode_impl_failure_wrapped_as_runtime_error(self):
        """_encode_impl 抛异常时被包装为 RuntimeError。"""
        encoder = _FailingEncoder({})
        ctx = make_default_context()
        inputs = {"text": "a cat", "clip": MagicMock()}
        with pytest.raises(RuntimeError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "CLIP 文本编码失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_return_dict_raises_type_error(self):
        """返回 dict 缺少 embeds 字段时抛 TypeError。"""
        encoder = _InvalidReturnEncoder({})
        ctx = make_default_context()
        inputs = {"text": "a cat", "clip": MagicMock()}
        with pytest.raises(TypeError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "embeds" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_dict_return_raises_type_error(self):
        """返回非 dict 时抛 TypeError。"""
        encoder = _NonDictEncoder({})
        ctx = make_default_context()
        inputs = {"text": "a cat", "clip": MagicMock()}
        with pytest.raises(TypeError) as exc_info:
            await encoder.execute(inputs, ctx)
        assert "dict" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_params_raises_validation_error(self):
        """参数校验失败抛 NodeValidationError。"""
        encoder = _MockCLIPTextEncode({"max_length": -1})
        ctx = make_default_context()
        inputs = {"text": "a cat", "clip": MagicMock()}
        with pytest.raises(NodeValidationError):
            await encoder.execute(inputs, ctx)


# ===== 进度上报测试 =====


class TestCLIPTextEncodeNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self):
        """编码时上报开始和完成进度。"""
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        inputs = {"text": "a cat", "clip": MagicMock()}
        await encoder.execute(inputs, ctx)

        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0.0, "clip_encode.start")
        assert progress_calls[-1] == (1.0, "clip_encode.done")

    @pytest.mark.asyncio
    async def test_text_preview_in_start_progress(self):
        """开始进度帧包含 text_preview 字段。"""
        progress_kwargs: list[dict] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_kwargs.append(kwargs)

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        encoder = _MockCLIPTextEncode({})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        inputs = {"text": "a beautiful sunset over mountains", "clip": MagicMock()}
        await encoder.execute(inputs, ctx)

        assert "text_preview" in progress_kwargs[0]
        assert "sunset" in progress_kwargs[0]["text_preview"]
        assert "has_negative" in progress_kwargs[0]

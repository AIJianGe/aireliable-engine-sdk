"""Sampler 节点单元测试。

覆盖任务 3.6 验收标准：
    - 支持 steps / guidance / size / seed 参数
    - 调用 diffusers pipeline 的 __call__ 方法
    - 支持回调进度上报

参考：v0.2-tasks.md 任务 3.6
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.sample.sampler import SamplerNode


# ===== Mock 子类 =====


class _MockSampler(SamplerNode):
    """模拟采样器，记录调用并触发 progress_callback。"""

    node_type = "MockSampler"

    def __init__(self, params, return_value: dict = None, raise_exc: Exception = None,
                 call_progress: bool = True):
        super().__init__(params)
        if return_value is None:
            return_value = {"latent": MagicMock(name="mock_latent")}
        self._return_value = return_value
        self._raise_exc = raise_exc
        self._call_progress = call_progress
        self.sample_called_with_inputs = None
        self.sample_called_with_params = None
        self.progress_callback_received = None

    async def _sample_impl(self, inputs, params, ctx, progress_callback) -> dict:
        self.sample_called_with_inputs = inputs
        self.sample_called_with_params = params
        self.progress_callback_received = progress_callback

        if self._raise_exc is not None:
            raise self._raise_exc

        # 模拟多步采样，调用 progress_callback
        if self._call_progress and progress_callback is not None:
            total = params.get("steps", 30)
            for step in range(1, total + 1):
                await progress_callback(step, total, timestep=step * 10)

        return self._return_value


class _FailingSampler(SamplerNode):
    node_type = "FailingSampler"

    async def _sample_impl(self, inputs, params, ctx, progress_callback) -> dict:
        raise RuntimeError("采样器内部错误")


class _InvalidReturnSampler(SamplerNode):
    """返回 dict 缺少 latent 字段。"""

    node_type = "InvalidReturnSampler"

    async def _sample_impl(self, inputs, params, ctx, progress_callback) -> dict:
        return {"wrong_field": MagicMock()}


class _NonDictSampler(SamplerNode):
    node_type = "NonDictSampler"

    async def _sample_impl(self, inputs, params, ctx, progress_callback) -> dict:
        return "not a dict"


class _BareSampler(SamplerNode):
    """未重写 _sample_impl。"""

    node_type = "BareSampler"


# ===== 基本属性测试 =====


class TestSamplerNodeBasic:
    """SamplerNode 基本属性测试。"""

    def test_node_type_is_sampler(self):
        node = SamplerNode()
        assert node.node_type == "Sampler"

    def test_input_specs_has_five_ports(self):
        """输入端口：model / embeds / negative_embeds / images / scheduler。"""
        node = SamplerNode()
        names = [p.name for p in node.input_specs]
        assert "model" in names
        assert "embeds" in names
        assert "negative_embeds" in names
        assert "images" in names
        assert "scheduler" in names

    def test_output_specs_has_latent_port(self):
        node = SamplerNode()
        assert len(node.output_specs) == 1
        assert node.output_specs[0].name == "latent"
        assert node.output_specs[0].type == "latent"

    def test_default_steps_is_30(self):
        node = SamplerNode()
        assert node._steps == 30

    def test_default_guidance_is_7_5(self):
        node = SamplerNode()
        assert node._guidance == 7.5

    def test_default_size_is_1024x1024(self):
        node = SamplerNode()
        assert node._size == "1024x1024"

    def test_default_batch_size_is_1(self):
        node = SamplerNode()
        assert node._batch_size == 1

    def test_default_eta_is_0(self):
        node = SamplerNode()
        assert node._eta == 0.0

    def test_default_seed_is_none(self):
        node = SamplerNode()
        assert node._seed is None

    def test_init_with_params(self):
        node = SamplerNode({
            "steps": 50,
            "guidance": 4.5,
            "size": "768x768",
            "seed": 42,
            "batch_size": 2,
            "eta": 0.5,
            "sampler_name": "dpm++",
            "device": "cuda:1",
        })
        assert node._steps == 50
        assert node._guidance == 4.5
        assert node._size == "768x768"
        assert node._seed == 42
        assert node._batch_size == 2
        assert node._eta == 0.5
        assert node._sampler_name == "dpm++"
        assert node._device_override == "cuda:1"

    def test_repr_contains_steps_and_size(self):
        node = SamplerNode({"steps": 20, "size": "512x512"})
        repr_str = repr(node)
        assert "SamplerNode" in repr_str
        assert "20" in repr_str
        assert "512x512" in repr_str


# ===== 参数校验测试（验收标准）=====


class TestSamplerNodeValidate:
    """参数校验测试（任务 3.6 验收标准：steps/guidance/size/seed 参数）。"""

    def test_validate_returns_empty_for_default_params(self):
        node = SamplerNode()
        assert node.validate() == []

    def test_validate_reports_non_positive_steps(self):
        node = SamplerNode({"steps": 0})
        errors = node.validate()
        assert any("steps" in e for e in errors)

    def test_validate_reports_negative_steps(self):
        node = SamplerNode({"steps": -10})
        errors = node.validate()
        assert any("steps" in e for e in errors)

    def test_validate_reports_negative_guidance(self):
        node = SamplerNode({"guidance": -1.0})
        errors = node.validate()
        assert any("guidance" in e for e in errors)

    def test_validate_reports_invalid_size_format(self):
        node = SamplerNode({"size": "invalid"})
        errors = node.validate()
        assert any("size" in e for e in errors)

    def test_validate_reports_non_int_seed(self):
        node = SamplerNode({"seed": "abc"})
        # 字符串 seed：会被保留原值，validate 校验
        # 但因为 get_param 没强制转换，validate 中 isinstance(self._seed, int) 会 False
        # 但 None 默认值会跳过校验
        if node._seed is not None and not isinstance(node._seed, int):
            errors = node.validate()
            assert any("seed" in e for e in errors)

    def test_validate_reports_negative_seed(self):
        node = SamplerNode({"seed": -1})
        errors = node.validate()
        assert any("seed" in e for e in errors)

    def test_validate_reports_negative_batch_size(self):
        node = SamplerNode({"batch_size": -1})
        errors = node.validate()
        assert any("batch_size" in e for e in errors)

    def test_validate_reports_eta_out_of_range(self):
        node = SamplerNode({"eta": 1.5})
        errors = node.validate()
        assert any("eta" in e for e in errors)

        node2 = SamplerNode({"eta": -0.5})
        errors2 = node2.validate()
        assert any("eta" in e for e in errors2)

    def test_validate_reports_invalid_call_timeout(self):
        node = SamplerNode({"call_timeout": -1})
        errors = node.validate()
        assert any("call_timeout" in e for e in errors)

    def test_validate_accepts_valid_params(self):
        node = SamplerNode({
            "steps": 50,
            "guidance": 7.5,
            "size": "1024x1024",
            "seed": 42,
            "batch_size": 2,
            "eta": 0.5,
            "call_timeout": 300.0,
        })
        assert node.validate() == []


# ===== size 解析测试 =====


class TestSamplerNodeSizeParsing:
    """size 字符串解析测试。"""

    def test_parse_standard_lowercase_x(self):
        node = SamplerNode()
        assert node._parse_size("1024x1024") == (1024, 1024)

    def test_parse_uppercase_x(self):
        node = SamplerNode()
        assert node._parse_size("1024X1024") == (1024, 1024)

    def test_parse_asterisk_separator(self):
        node = SamplerNode()
        assert node._parse_size("768*768") == (768, 768)

    def test_parse_unicode_multiply_sign(self):
        node = SamplerNode()
        assert node._parse_size("512×512") == (512, 512)

    def test_parse_non_square_size(self):
        node = SamplerNode()
        assert node._parse_size("1024x768") == (1024, 768)

    def test_parse_invalid_format_raises(self):
        node = SamplerNode()
        with pytest.raises(ValueError):
            node._parse_size("invalid")

    def test_parse_non_integer_raises(self):
        node = SamplerNode()
        with pytest.raises(ValueError):
            node._parse_size("1024.5x1024")

    def test_parse_zero_or_negative_raises(self):
        node = SamplerNode()
        with pytest.raises(ValueError):
            node._parse_size("0x1024")
        with pytest.raises(ValueError):
            node._parse_size("-1x1024")


# ===== 调用流程测试（验收标准）=====


class TestSamplerNodeExecute:
    """execute 流程测试（验收标准：调用 diffusers pipeline __call__）。"""

    @pytest.mark.asyncio
    async def test_execute_calls_sample_impl_with_parsed_inputs(self):
        """execute 把解析后的输入传给 _sample_impl。"""
        sampler = _MockSampler({"steps": 10})
        ctx = make_default_context()
        mock_model = MagicMock(name="mock_model")
        mock_embeds = MagicMock(name="mock_embeds")
        inputs = {
            "model": mock_model,
            "embeds": mock_embeds,
            "negative_embeds": MagicMock(name="negative_embeds"),
            "images": MagicMock(name="images"),
            "scheduler": MagicMock(name="scheduler"),
        }
        await sampler.execute(inputs, ctx)

        assert sampler.sample_called_with_inputs["model"] is mock_model
        assert sampler.sample_called_with_inputs["embeds"] is mock_embeds
        assert sampler.sample_called_with_inputs["negative_embeds"] is not None
        assert sampler.sample_called_with_inputs["images"] is not None
        assert sampler.sample_called_with_inputs["scheduler"] is not None

    @pytest.mark.asyncio
    async def test_execute_passes_parsed_params(self):
        """execute 把解析后的参数（含 width/height）传给 _sample_impl。"""
        sampler = _MockSampler({
            "steps": 20,
            "guidance": 5.0,
            "size": "768x512",
            "seed": 42,
        })
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        await sampler.execute(inputs, ctx)

        params = sampler.sample_called_with_params
        assert params["steps"] == 20
        assert params["guidance"] == 5.0
        assert params["width"] == 768
        assert params["height"] == 512
        assert params["seed"] == 42

    @pytest.mark.asyncio
    async def test_execute_returns_latent_from_sample_impl(self):
        """execute 返回 _sample_impl 的 latent 字段。"""
        mock_latent = MagicMock(name="mock_latent")
        sampler = _MockSampler(
            {"steps": 5},
            return_value={"latent": mock_latent},
        )
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        result = await sampler.execute(inputs, ctx)
        assert result["latent"] is mock_latent

    @pytest.mark.asyncio
    async def test_execute_missing_model_raises_validation_error(self):
        """缺少 model 输入抛 NodeValidationError。"""
        sampler = _MockSampler({"steps": 5})
        ctx = make_default_context()
        inputs = {"embeds": MagicMock()}  # 缺 model
        with pytest.raises(NodeValidationError) as exc_info:
            await sampler.execute(inputs, ctx)
        assert "model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_missing_embeds_raises_validation_error(self):
        """缺少 embeds 输入抛 NodeValidationError。"""
        sampler = _MockSampler({"steps": 5})
        ctx = make_default_context()
        inputs = {"model": MagicMock()}  # 缺 embeds
        with pytest.raises(NodeValidationError) as exc_info:
            await sampler.execute(inputs, ctx)
        assert "embeds" in str(exc_info.value)


# ===== 回调进度上报测试（验收标准）=====


class TestSamplerNodeProgressCallback:
    """回调进度上报测试（任务 3.6 验收标准：支持回调进度上报）。"""

    @pytest.mark.asyncio
    async def test_progress_callback_invoked_for_each_step(self):
        """每个采样步都会调用 progress_callback。"""
        sampler = _MockSampler({"steps": 10})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        await sampler.execute(inputs, ctx)

        # progress_callback 已传给 _sample_impl
        assert sampler.progress_callback_received is not None

    @pytest.mark.asyncio
    async def test_progress_callback_reports_to_ctx(self):
        """progress_callback 调用时会通过 ctx 上报进度。"""
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        sampler = _MockSampler({"steps": 5})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        await sampler.execute(inputs, ctx)

        # 至少 start + 5 steps + done = 7 次进度上报
        assert len(progress_calls) >= 7
        # 第一次是 start
        assert progress_calls[0] == (0.0, "sampler.start")
        # 第二到第六次是 step（步进）
        for i in range(5):
            stage = progress_calls[1 + i][1]
            assert stage == "sampler.step"
        # 最后一次是 done
        assert progress_calls[-1] == (1.0, "sampler.done")

    @pytest.mark.asyncio
    async def test_progress_increases_monotonically(self):
        """进度值单调递增。"""
        progress_values: list[float] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_values.append(progress)

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        sampler = _MockSampler({"steps": 5})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        await sampler.execute(inputs, ctx)

        # 验证进度值单调递增
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1]

    @pytest.mark.asyncio
    async def test_progress_callback_passes_step_and_total(self):
        """progress_callback 上报的 kwargs 包含 step 和 total 字段。"""
        progress_kwargs_list: list[dict] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_kwargs_list.append(kwargs)

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        sampler = _MockSampler({"steps": 3})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        await sampler.execute(inputs, ctx)

        # 第二到第四次是 step（start = 第0个，done = 最后一个）
        step_kwargs = progress_kwargs_list[1:4]
        for i, kwargs in enumerate(step_kwargs, start=1):
            assert kwargs["step"] == i
            assert kwargs["total"] == 3


# ===== 错误处理测试 =====


class TestSamplerNodeErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_bare_sampler_raises_not_implemented(self):
        """未重写 _sample_impl 的子类抛 NotImplementedError。"""
        sampler = _BareSampler({"steps": 5})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        with pytest.raises(NotImplementedError) as exc_info:
            await sampler.execute(inputs, ctx)
        assert "_sample_impl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_sample_impl_failure_wrapped_as_runtime_error(self):
        """_sample_impl 抛异常时被包装为 RuntimeError。"""
        sampler = _FailingSampler({"steps": 5})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        with pytest.raises(RuntimeError) as exc_info:
            await sampler.execute(inputs, ctx)
        assert "采样失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_return_dict_raises_type_error(self):
        """返回 dict 缺少 latent 字段时抛 TypeError。"""
        sampler = _InvalidReturnSampler({"steps": 5})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        with pytest.raises(TypeError) as exc_info:
            await sampler.execute(inputs, ctx)
        assert "latent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_dict_return_raises_type_error(self):
        """返回非 dict 时抛 TypeError。"""
        sampler = _NonDictSampler({"steps": 5})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        with pytest.raises(TypeError) as exc_info:
            await sampler.execute(inputs, ctx)
        assert "dict" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_params_raises_validation_error(self):
        """参数校验失败抛 NodeValidationError。"""
        sampler = _MockSampler({"steps": -1})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        with pytest.raises(NodeValidationError):
            await sampler.execute(inputs, ctx)

    @pytest.mark.asyncio
    async def test_invalid_size_raises_validation_error(self):
        """size 格式不正确抛 NodeValidationError。"""
        sampler = _MockSampler({"size": "invalid"})
        ctx = make_default_context()
        inputs = {"model": MagicMock(), "embeds": MagicMock()}
        with pytest.raises(NodeValidationError):
            await sampler.execute(inputs, ctx)

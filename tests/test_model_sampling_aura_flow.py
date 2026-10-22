"""ModelSamplingAuraFlow 节点单元测试（v0.2.5 ComfyUI 风格节点补充）。

覆盖 C3-4 验收标准：
    - 节点定义（node_type / category / subcategory / title / icon）
    - 输入端口为 model（1 个，type="model"，required=True）
    - 输出端口为 model（1 个，type="model"，输入输出同名）
    - params_schema 含 shift（default=3.0，min=0，max=100）
    - cache_key 返回 None（转换节点不缓存）
    - execute 写 ctx.extra["model_shift"]（mock model）
    - execute 修改 model.scheduler.config.shift（mock model.scheduler）
    - execute 返回原 model（同一对象，不创建新实例）
    - inputs.model 为 None 时回退到 ctx.extra["loaded_model"]
    - inputs.model 和 ctx.extra 都无时抛 NodeValidationError

参考：v0.2.5-comfyui-nodes-tasks.md C3-4
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.sample.model_sampling_aura_flow import (
    ModelSamplingAuraFlowNode,
)


# ===== Mock model 工具 =====


def _make_mock_model(has_scheduler: bool = True, has_config_shift: bool = True):
    """构造一个 mock model 对象（含 scheduler 属性）。"""
    model = MagicMock(name="mock_model")
    if has_scheduler:
        scheduler = MagicMock(name="mock_scheduler")
        if has_config_shift:
            config = MagicMock(name="mock_config")
            config.shift = 0.0  # 初始值
            scheduler.config = config
        else:
            scheduler.config = None
        # set_shift 方法
        scheduler.set_shift = MagicMock()
        model.scheduler = scheduler
    else:
        # 没有 scheduler 属性
        del model.scheduler
    return model


# ===== 基本属性测试 =====


class TestModelSamplingAuraFlowNodeBasic:
    """ModelSamplingAuraFlowNode 基本属性测试。"""

    def test_node_type_is_model_sampling_aura_flow(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.node_type == "ModelSamplingAuraFlow"

    def test_category_is_sample(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.category == "sample"

    def test_subcategory_is_sampler(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.subcategory == "sampler"

    def test_title_is_chinese(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.title == "采样算法 (AuraFlow)"

    def test_icon_is_dice_emoji(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.icon == "🎲"

    def test_input_specs_has_single_model_port(self):
        """输入端口为 model（1 个，required=True）。"""
        node = ModelSamplingAuraFlowNode({})
        assert len(node.input_specs) == 1
        port = node.input_specs[0]
        assert port.name == "model"
        assert port.type == "model"
        assert port.required is True

    def test_output_specs_has_single_model_port(self):
        """输出端口为 model（1 个，输入输出同名）。"""
        node = ModelSamplingAuraFlowNode({})
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "model"
        assert port.type == "model"

    def test_input_output_port_name_same(self):
        """C3-4-4：输入输出端口同名（model 变换器节点标志）。"""
        node = ModelSamplingAuraFlowNode({})
        in_name = node.input_specs[0].name
        out_name = node.output_specs[0].name
        assert in_name == out_name == "model"

    def test_default_shift_is_3(self):
        """C3-4-5：shift 默认值 3.0（AuraFlow 官方推荐）。"""
        node = ModelSamplingAuraFlowNode({})
        assert node._shift == 3.0

    def test_repr_contains_class_name_and_shift(self):
        node = ModelSamplingAuraFlowNode({"shift": 5.0})
        repr_str = repr(node)
        assert "ModelSamplingAuraFlowNode" in repr_str
        assert "5.0" in repr_str


# ===== params_schema 测试 =====


class TestModelSamplingAuraFlowNodeParamsSchema:
    """params_schema 测试。"""

    def test_params_schema_contains_shift(self):
        node = ModelSamplingAuraFlowNode({})
        assert "shift" in node.params_schema

    def test_shift_default_is_3(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.params_schema["shift"].get("default") == 3.0

    def test_shift_min_is_0(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.params_schema["shift"].get("min") == 0

    def test_shift_max_is_100(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.params_schema["shift"].get("max") == 100

    def test_shift_type_is_number(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.params_schema["shift"].get("type") == "number"

    def test_shift_label_is_chinese(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.params_schema["shift"].get("label") == "移位"


# ===== 参数校验测试 =====


class TestModelSamplingAuraFlowNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_default_shift(self):
        node = ModelSamplingAuraFlowNode({})
        assert node.validate() == []

    def test_validate_accepts_custom_shift(self):
        node = ModelSamplingAuraFlowNode({"shift": 7.5})
        assert node.validate() == []

    def test_validate_accepts_zero_shift(self):
        node = ModelSamplingAuraFlowNode({"shift": 0})
        assert node.validate() == []

    def test_validate_accepts_max_shift(self):
        node = ModelSamplingAuraFlowNode({"shift": 100})
        assert node.validate() == []

    def test_validate_reports_negative_shift(self):
        node = ModelSamplingAuraFlowNode({"shift": -1.0})
        errors = node.validate()
        assert any("shift" in e for e in errors)

    def test_validate_reports_exceeding_max_shift(self):
        node = ModelSamplingAuraFlowNode({"shift": 101.0})
        errors = node.validate()
        assert any("shift" in e for e in errors)


# ===== cache_key 测试 =====


class TestModelSamplingAuraFlowNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_returns_none(self):
        """C3-4-6：cache_key 返回 None（转换节点不缓存）。"""
        node = ModelSamplingAuraFlowNode({})
        assert node.cache_key({}) is None

    def test_cache_key_returns_none_with_inputs(self):
        """即使有 inputs 也返回 None。"""
        node = ModelSamplingAuraFlowNode({"shift": 5.0})
        assert node.cache_key({"model": MagicMock()}) is None


# ===== execute 测试 =====


class TestModelSamplingAuraFlowNodeExecute:
    """execute 完整流程测试。"""

    @pytest.mark.asyncio
    async def test_execute_writes_ctx_extra_model_shift(self):
        """C3-4-7：execute 写 ctx.extra["model_shift"]。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        await node.execute({"model": model}, ctx)

        assert "model_shift" in ctx.extra
        assert ctx.extra["model_shift"] == 3.0

    @pytest.mark.asyncio
    async def test_execute_writes_custom_shift_to_ctx_extra(self):
        """自定义 shift 值写入 ctx.extra。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 7.5})
        ctx = make_default_context()
        await node.execute({"model": model}, ctx)

        assert ctx.extra["model_shift"] == 7.5

    @pytest.mark.asyncio
    async def test_execute_modifies_scheduler_config_shift(self):
        """C3-4-8：execute 修改 model.scheduler.config.shift。"""
        model = _make_mock_model(has_scheduler=True, has_config_shift=True)
        # 初始 shift 为 0.0
        assert model.scheduler.config.shift == 0.0

        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        await node.execute({"model": model}, ctx)

        # 验证 config.shift 被修改
        assert model.scheduler.config.shift == 3.0

    @pytest.mark.asyncio
    async def test_execute_calls_set_shift_if_available(self):
        """execute 调用 scheduler.set_shift（若存在）。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        await node.execute({"model": model}, ctx)

        # 验证 set_shift 被调用一次，参数为 3.0
        model.scheduler.set_shift.assert_called_once_with(3.0)

    @pytest.mark.asyncio
    async def test_execute_returns_same_model_instance(self):
        """C3-4-9：execute 返回原 model（同一对象，不创建新实例）。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        result = await node.execute({"model": model}, ctx)

        assert "model" in result
        assert result["model"] is model  # 同一对象

    @pytest.mark.asyncio
    async def test_execute_fallback_to_ctx_extra_loaded_model(self):
        """C3-4-10：inputs.model 为 None 时回退到 ctx.extra["loaded_model"]。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        ctx.extra["loaded_model"] = model  # 模拟 UNetLoader 写入
        result = await node.execute({"model": None}, ctx)

        assert result["model"] is model
        assert ctx.extra["model_shift"] == 3.0

    @pytest.mark.asyncio
    async def test_execute_raises_when_no_model(self):
        """C3-4-11：inputs.model 和 ctx.extra 都无时抛 NodeValidationError。"""
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        # 不设置 ctx.extra["loaded_model"]
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"model": None}, ctx)
        assert "model" in str(exc_info.value).lower() or "未找到" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_invalid_shift_raises(self):
        """参数校验失败抛 NodeValidationError。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": -1.0})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await node.execute({"model": model}, ctx)

    @pytest.mark.asyncio
    async def test_execute_with_no_scheduler_attribute(self):
        """model 无 scheduler 属性时仍能正常工作（仅写 ctx.extra）。"""
        model = _make_mock_model(has_scheduler=False)
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        result = await node.execute({"model": model}, ctx)

        # 仅写 ctx.extra，不报错
        assert ctx.extra["model_shift"] == 3.0
        assert result["model"] is model

    @pytest.mark.asyncio
    async def test_execute_with_config_no_shift_attr(self):
        """scheduler.config 无 shift 属性时仍能正常工作（仅写 ctx.extra + set_shift）。"""
        model = _make_mock_model(has_scheduler=True, has_config_shift=False)
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        result = await node.execute({"model": model}, ctx)

        # 仅写 ctx.extra，不报错
        assert ctx.extra["model_shift"] == 3.0
        assert result["model"] is model

    @pytest.mark.asyncio
    async def test_execute_scheduler_exception_does_not_propagate(self):
        """scheduler 修改异常被 try-except 捕获，不影响 execute 返回。"""
        model = _make_mock_model()
        # 让 config.shift 赋值抛异常
        model.scheduler.config = MagicMock()
        type(model.scheduler.config).shift = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("mock error"))
        )

        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context()
        # 不应抛异常
        result = await node.execute({"model": model}, ctx)
        # ctx.extra 仍被写入
        assert ctx.extra["model_shift"] == 3.0
        assert result["model"] is model


# ===== 进度上报测试 =====


class TestModelSamplingAuraFlowNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self):
        """execute 时上报开始和完成进度。"""
        model = _make_mock_model()
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await node.execute({"model": model}, ctx)

        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0.0, "model_sampling_aura_flow.start")
        assert progress_calls[-1] == (1.0, "model_sampling_aura_flow.done")


# ===== shift 注入语义测试（v0.2.5 验收核心）=====


class TestModelSamplingAuraFlowShiftInjection:
    """shift 注入语义测试。

    v0.2.5 §28.17.3 验收核心：双保险注入方案
        - 方案 A：修改 model.scheduler.config.shift + 调用 set_shift
        - 方案 B：写 ctx.extra["model_shift"] 供 Sampler 读取
    """

    @pytest.mark.asyncio
    async def test_shift_injection_both_ways(self):
        """完整 shift 注入：方案 A + 方案 B 同时生效。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 4.5})
        ctx = make_default_context()

        await node.execute({"model": model}, ctx)

        # 方案 A：scheduler.config.shift 被修改
        assert model.scheduler.config.shift == 4.5
        # 方案 A：scheduler.set_shift 被调用
        model.scheduler.set_shift.assert_called_once_with(4.5)
        # 方案 B：ctx.extra["model_shift"] 被写入
        assert ctx.extra["model_shift"] == 4.5

    @pytest.mark.asyncio
    async def test_shift_overrides_previous_value(self):
        """多次调用时 shift 值覆盖（最近一次生效）。"""
        model = _make_mock_model()
        ctx = make_default_context()

        # 第一次：shift=3.0
        node1 = ModelSamplingAuraFlowNode({"shift": 3.0})
        await node1.execute({"model": model}, ctx)
        assert ctx.extra["model_shift"] == 3.0
        assert model.scheduler.config.shift == 3.0

        # 第二次：shift=7.5（覆盖）
        node2 = ModelSamplingAuraFlowNode({"shift": 7.5})
        await node2.execute({"model": model}, ctx)
        assert ctx.extra["model_shift"] == 7.5
        assert model.scheduler.config.shift == 7.5
        # set_shift 累计调用 2 次
        assert model.scheduler.set_shift.call_count == 2

    @pytest.mark.asyncio
    async def test_default_shift_value_injected(self):
        """默认 shift=3.0 正确注入。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({})  # 使用默认 shift=3.0
        ctx = make_default_context()

        await node.execute({"model": model}, ctx)

        assert ctx.extra["model_shift"] == 3.0
        assert model.scheduler.config.shift == 3.0
        model.scheduler.set_shift.assert_called_once_with(3.0)

    @pytest.mark.asyncio
    async def test_zero_shift_injection(self):
        """shift=0 边界值能正常注入。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 0})
        ctx = make_default_context()

        await node.execute({"model": model}, ctx)

        assert ctx.extra["model_shift"] == 0
        assert model.scheduler.config.shift == 0
        model.scheduler.set_shift.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_max_shift_injection(self):
        """shift=100 上限值能正常注入。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 100})
        ctx = make_default_context()

        await node.execute({"model": model}, ctx)

        assert ctx.extra["model_shift"] == 100
        assert model.scheduler.config.shift == 100
        model.scheduler.set_shift.assert_called_once_with(100)

    @pytest.mark.asyncio
    async def test_set_shift_exception_does_not_break_execution(self):
        """scheduler.set_shift 抛异常时不影响 ctx.extra 写入。"""
        model = _make_mock_model()
        model.scheduler.set_shift = MagicMock(side_effect=RuntimeError("mock set_shift error"))

        node = ModelSamplingAuraFlowNode({"shift": 5.0})
        ctx = make_default_context()

        # 不应抛异常
        result = await node.execute({"model": model}, ctx)

        # ctx.extra 仍被写入（方案 B 是双保险）
        assert ctx.extra["model_shift"] == 5.0
        assert result["model"] is model

    @pytest.mark.asyncio
    async def test_pipeline_with_scheduler_attribute(self):
        """Pipeline 形式（含 scheduler 属性）的 shift 注入。"""
        # 模拟 diffusers Pipeline：scheduler 通过属性访问
        model = MagicMock()
        model.scheduler = MagicMock()
        model.scheduler.config = MagicMock()
        model.scheduler.config.shift = 0.0
        model.scheduler.set_shift = MagicMock()

        node = ModelSamplingAuraFlowNode({"shift": 3.5})
        ctx = make_default_context()

        result = await node.execute({"model": model}, ctx)

        assert model.scheduler.config.shift == 3.5
        model.scheduler.set_shift.assert_called_once_with(3.5)
        assert ctx.extra["model_shift"] == 3.5
        assert result["model"] is model

    @pytest.mark.asyncio
    async def test_pipeline_without_scheduler_attribute(self):
        """无 scheduler 属性的 pipeline（仅写 ctx.extra，方案 A 跳过）。"""
        model = MagicMock()
        del model.scheduler  # 无 scheduler 属性

        node = ModelSamplingAuraFlowNode({"shift": 2.0})
        ctx = make_default_context()

        # 不应抛异常
        result = await node.execute({"model": model}, ctx)

        # 方案 A 跳过，方案 B 生效
        assert ctx.extra["model_shift"] == 2.0
        assert result["model"] is model

    @pytest.mark.asyncio
    async def test_sampler_can_read_shift_from_ctx_extra(self):
        """端到端：Sampler 可从 ctx.extra 读取 shift 值。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 6.0})
        ctx = make_default_context()

        await node.execute({"model": model}, ctx)

        # 模拟 Sampler 读取
        sampler_shift = ctx.extra.get("model_shift")
        assert sampler_shift == 6.0

    @pytest.mark.asyncio
    async def test_idempotency_with_same_shift(self):
        """相同 shift 值多次调用是幂等的。"""
        model = _make_mock_model()
        ctx = make_default_context()
        node = ModelSamplingAuraFlowNode({"shift": 3.0})

        await node.execute({"model": model}, ctx)
        shift_after_first = ctx.extra["model_shift"]
        config_shift_after_first = model.scheduler.config.shift

        await node.execute({"model": model}, ctx)
        assert ctx.extra["model_shift"] == shift_after_first
        assert model.scheduler.config.shift == config_shift_after_first
        # set_shift 调用 2 次
        assert model.scheduler.set_shift.call_count == 2

    @pytest.mark.asyncio
    async def test_extra_dict_initialized_when_empty(self):
        """ctx.extra 为空 dict 时仍能正确写入。"""
        model = _make_mock_model()
        node = ModelSamplingAuraFlowNode({"shift": 1.5})
        ctx = make_default_context()

        # 确认 extra 初始为空
        assert "model_shift" not in ctx.extra

        await node.execute({"model": model}, ctx)

        # 写入后存在
        assert "model_shift" in ctx.extra
        assert ctx.extra["model_shift"] == 1.5


# ===== 与其他节点协同的端到端测试 =====


class TestModelSamplingAuraFlowIntegration:
    """与其他节点协同的端到端测试。

    模拟完整工作流场景：
        UNetLoader → ModelSamplingAuraFlow → Sampler
    """

    @pytest.mark.asyncio
    async def test_workflow_pipeline_shift_propagation(self):
        """工作流：UNetLoader 写入 loaded_model → ModelSamplingAuraFlow 设置 shift。"""
        model = _make_mock_model()
        ctx = make_default_context()

        # 模拟 UNetLoader 写入 loaded_model
        ctx.extra["loaded_model"] = model

        # ModelSamplingAuraFlow 通过 ctx.extra 找到 model
        node = ModelSamplingAuraFlowNode({"shift": 3.0})
        result = await node.execute({"model": None}, ctx)

        # 验证 shift 已注入
        assert result["model"] is model
        assert ctx.extra["model_shift"] == 3.0
        assert model.scheduler.config.shift == 3.0

    @pytest.mark.asyncio
    async def test_inputs_model_priority_over_ctx_extra(self):
        """inputs.model 优先于 ctx.extra["loaded_model"]。"""
        ctx_model = _make_mock_model()
        inputs_model = _make_mock_model()

        ctx = make_default_context()
        ctx.extra["loaded_model"] = ctx_model

        node = ModelSamplingAuraFlowNode({"shift": 2.0})
        # inputs.model 显式提供（与 ctx.extra 不同）
        result = await node.execute({"model": inputs_model}, ctx)

        # 应使用 inputs_model 而非 ctx_model
        assert result["model"] is inputs_model
        assert result["model"] is not ctx_model
        # 仅 inputs_model 被修改
        assert inputs_model.scheduler.config.shift == 2.0
        assert ctx_model.scheduler.config.shift == 0.0  # 未被修改

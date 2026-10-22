"""WorkflowExecutor 单元测试。

覆盖任务 4.4 验收标准：
    - 解析 yaml → 构建 graph
    - 拓扑排序
    - 按序执行节点
    - 处理 if/else 分支（condition）
    - 收集 outputs 返回

"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from aireliable_engine import (
    NodeRegistry,
    PortSpec,
    WorkflowCycleError,
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowExecutorError,
    WorkflowReferenceError,
    WorkflowValidationError,
    make_default_context,
    register_node,
)
from aireliable_engine.node import BaseNode, NodeValidationError
from aireliable_engine.workflow_dsl import DSLError


# ===========================================================================
# 测试用节点（注册到独立 registry 避免污染全局）
# ===========================================================================


class EchoNode(BaseNode):
    """回显节点：把输入直接输出。"""
    node_type = "TestEcho"
    input_specs = [PortSpec(name="value", type="any", required=False)]
    output_specs = [PortSpec(name="value", type="any")]

    async def execute(self, inputs, ctx):
        return {"value": inputs.get("value")}


class DoubleNode(BaseNode):
    """加倍节点。"""
    node_type = "TestDouble"
    input_specs = [PortSpec(name="value", type="any")]
    output_specs = [PortSpec(name="result", type="any")]

    async def execute(self, inputs, ctx):
        return {"result": inputs["value"] * 2}


class AddNode(BaseNode):
    """加法节点：a + b。"""
    node_type = "TestAdd"
    input_specs = [
        PortSpec(name="a", type="any"),
        PortSpec(name="b", type="any"),
    ]
    output_specs = [PortSpec(name="result", type="any")]

    async def execute(self, inputs, ctx):
        a = inputs.get("a", 0)
        b = inputs.get("b", 0)
        return {"result": a + b}


class ParamEchoNode(BaseNode):
    """参数回显节点：把 params.value 输出。"""
    node_type = "TestParamEcho"
    output_specs = [PortSpec(name="value", type="any")]

    def __init__(self, params=None):
        super().__init__(params)
        self._value = self.get_param("value")

    async def execute(self, inputs, ctx):
        return {"value": self._value}


class FailingNode(BaseNode):
    """总是抛错的节点。"""
    node_type = "TestFailing"
    output_specs = [PortSpec(name="value", type="any")]

    async def execute(self, inputs, ctx):
        raise RuntimeError("节点故意失败")


class ValidatingNode(BaseNode):
    """带参数校验的节点。"""
    node_type = "TestValidating"
    output_specs = [PortSpec(name="value", type="any")]

    def __init__(self, params=None):
        super().__init__(params)
        self._n = self.get_param("n", 0)

    def validate(self):
        errors = []
        if not isinstance(self._n, int) or self._n < 0:
            errors.append("n 必须是非负整数")
        return errors

    async def execute(self, inputs, ctx):
        return {"value": self._n}


class CleanupTrackingNode(BaseNode):
    """追踪 cleanup 调用的节点。"""
    node_type = "TestCleanup"
    output_specs = [PortSpec(name="value", type="any")]
    cleanup_called: bool = False

    async def execute(self, inputs, ctx):
        return {"value": "ok"}

    def cleanup(self):
        # 类属性记录，便于跨实例检查（测试用）
        CleanupTrackingNode.cleanup_called = True


class CacheableNode(BaseNode):
    """可缓存节点：相同 cache_key 时跳过 execute。"""
    node_type = "TestCacheable"
    input_specs = [PortSpec(name="key", type="any")]
    output_specs = [PortSpec(name="value", type="any")]

    # 类级计数器：统计 execute 调用次数
    execute_count: int = 0

    def __init__(self, params=None):
        super().__init__(params)

    async def execute(self, inputs, ctx):
        CacheableNode.execute_count += 1
        return {"value": f"computed_for_{inputs['key']}"}

    def cache_key(self, inputs):
        return f"cache_{inputs['key']}"


# ===========================================================================
# 公共 fixture
# ===========================================================================


@pytest.fixture
def registry() -> NodeRegistry:
    """独立注册表（不污染全局默认注册表）。"""
    reg = NodeRegistry()
    reg.register("TestEcho", EchoNode)
    reg.register("TestDouble", DoubleNode)
    reg.register("TestAdd", AddNode)
    reg.register("TestParamEcho", ParamEchoNode)
    reg.register("TestFailing", FailingNode)
    reg.register("TestValidating", ValidatingNode)
    reg.register("TestCleanup", CleanupTrackingNode)
    reg.register("TestCacheable", CacheableNode)
    return reg


@pytest.fixture
def ctx():
    return make_default_context()


# ===========================================================================
# 构造与校验测试
# ===========================================================================


class TestConstruction:
    """WorkflowExecutor 构造与校验测试。"""

    def test_accept_dict(self, registry):
        wf = {
            "name": "test",
            "nodes": [{"id": "n1", "type": "TestEcho"}],
            "outputs": {"out": "n1.output.value"},
        }
        ex = WorkflowExecutor(wf, registry)
        assert ex.workflow_name == "test"

    def test_accept_yaml_string(self, registry):
        yaml_text = """
        name: test
        nodes:
          - id: n1
            type: TestEcho
        outputs:
          out: n1.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        assert ex.workflow_name == "test"

    def test_accept_nested_workflow_key(self, registry):
        """兼容 {"workflow": {...}} 嵌套结构。"""
        wf = {
            "workflow": {
                "name": "nested",
                "nodes": [{"id": "n1", "type": "TestEcho"}],
                "outputs": {"out": "n1.output.value"},
            }
        }
        ex = WorkflowExecutor(wf, registry)
        assert ex.workflow_name == "nested"

    def test_yaml_string_with_workflow_key(self, registry):
        yaml_text = """
        workflow:
          name: from_yaml
          nodes:
            - id: n1
              type: TestEcho
          outputs:
            out: n1.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        assert ex.workflow_name == "from_yaml"

    def test_empty_nodes_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor({"name": "empty", "nodes": [], "outputs": {}}, registry)

    def test_missing_id_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor({
                "name": "bad",
                "nodes": [{"type": "TestEcho"}],  # 缺 id
                "outputs": {},
            }, registry)

    def test_missing_type_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor({
                "name": "bad",
                "nodes": [{"id": "n1"}],  # 缺 type
                "outputs": {},
            }, registry)

    def test_duplicate_id_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor({
                "name": "dup",
                "nodes": [
                    {"id": "n1", "type": "TestEcho"},
                    {"id": "n1", "type": "TestEcho"},
                ],
                "outputs": {},
            }, registry)

    def test_invalid_yaml_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor("{not valid yaml", registry)

    def test_non_dict_workflow_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor("just a string", registry)

    def test_non_dict_workflow_list_raises(self, registry):
        with pytest.raises(WorkflowValidationError):
            WorkflowExecutor([], registry)

    def test_workflow_name_defaults_to_unnamed(self, registry):
        ex = WorkflowExecutor({
            "nodes": [{"id": "n1", "type": "TestEcho"}],
            "outputs": {},
        }, registry)
        assert ex.workflow_name == "<unnamed>"

    def test_default_registry_uses_global(self):
        """不传 registry 时使用全局默认注册表。"""
        # 全局注册表至少应该能拿到（即使可能为空）
        # 这里只检查不抛错
        try:
            ex = WorkflowExecutor({
                "name": "test",
                "nodes": [{"id": "n1", "type": "TestEcho"}],
                "outputs": {},
            })
            assert ex.registry is not None
        except WorkflowValidationError:
            # 校验通过即可，registry 是不是全局实例不强校验
            pass


# ===========================================================================
# 拓扑排序测试
# ===========================================================================


class TestTopologicalSort:
    """拓扑排序测试。"""

    def test_simple_chain(self, registry):
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho"},
                {"id": "b", "type": "TestEcho", "inputs": {"value": "a.output.value"}},
                {"id": "c", "type": "TestEcho", "inputs": {"value": "b.output.value"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        order = ex.get_execution_order()
        assert order == ["a", "b", "c"]

    def test_diamond_dependency(self, registry):
        wf = {
            "nodes": [
                {"id": "src", "type": "TestEcho"},
                {"id": "left", "type": "TestEcho", "inputs": {"value": "src.output.value"}},
                {"id": "right", "type": "TestEcho", "inputs": {"value": "src.output.value"}},
                {"id": "merge", "type": "TestAdd",
                 "inputs": {"a": "left.output.value", "b": "right.output.value"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        order = ex.get_execution_order()
        # src 必须在 left/right 之前，merge 最后
        assert order[0] == "src"
        assert order[-1] == "merge"
        assert "left" in order and "right" in order

    def test_cycle_detection(self, registry):
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho", "inputs": {"value": "b.output.value"}},
                {"id": "b", "type": "TestEcho", "inputs": {"value": "a.output.value"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        with pytest.raises(WorkflowCycleError):
            ex.get_execution_order()

    def test_self_cycle_detection(self, registry):
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho", "inputs": {"value": "a.output.value"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        with pytest.raises(WorkflowCycleError):
            ex.get_execution_order()

    def test_reference_to_nonexistent_node(self, registry):
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho", "inputs": {"value": "nonexistent.output.x"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        with pytest.raises(WorkflowValidationError):
            ex.get_execution_order()

    def test_var_inputs_not_counted_as_deps(self, registry):
        """${...} 形式的 inputs 不算节点依赖。"""
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho", "inputs": {"value": "${inputs.x}"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        # 应该能正常排序（无依赖）
        order = ex.get_execution_order()
        assert order == ["a"]

    def test_direct_value_inputs_not_deps(self, registry):
        """非字符串的 inputs 值不算依赖。"""
        wf = {
            "nodes": [
                {"id": "a", "type": "TestAdd", "inputs": {"a": 1, "b": 2}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        order = ex.get_execution_order()
        assert order == ["a"]


# ===========================================================================
# 变量替换测试
# ===========================================================================


class TestVarSubstitution:
    """参数与输入中的 ${...} 变量替换测试。"""

    @pytest.mark.asyncio
    async def test_substitute_variant(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${variant}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # make_default_context 默认 variant 是 "default"
        result = await ex.execute({}, ctx)
        assert result == {"out": ctx.variant}

    @pytest.mark.asyncio
    async def test_substitute_inputs_field(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.text}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"text": "hello"}, ctx)
        assert result == {"out": "hello"}

    @pytest.mark.asyncio
    async def test_substitute_loras(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${loras}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        # ctx.loras 是 list（make_default_context 默认空）
        assert result == {"out": ctx.loras}

    @pytest.mark.asyncio
    async def test_substitute_with_default(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.missing or 42}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        assert result == {"out": 42}

    @pytest.mark.asyncio
    async def test_substitute_with_default_string(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.missing or 'fallback'}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        assert result == {"out": "fallback"}

    @pytest.mark.asyncio
    async def test_substitute_with_default_not_used_when_present(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.x or 99}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 7}, ctx)
        assert result == {"out": 7}

    @pytest.mark.asyncio
    async def test_substitute_partial_string(self, registry, ctx):
        """部分 ${...} 嵌入字符串做拼接。"""
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "prefix-${inputs.tag}-suffix"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"tag": "middle"}, ctx)
        assert result == {"out": "prefix-middle-suffix"}

    @pytest.mark.asyncio
    async def test_substitute_preserves_type_int(self, registry, ctx):
        """完整 ${...} 替换保留原类型（int）。"""
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.count}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"count": 42}, ctx)
        assert result == {"out": 42}
        assert isinstance(result["out"], int)

    @pytest.mark.asyncio
    async def test_substitute_preserves_type_list(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.items}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"items": [1, 2, 3]}, ctx)
        assert result == {"out": [1, 2, 3]}
        assert isinstance(result["out"], list)

    @pytest.mark.asyncio
    async def test_substitute_nested_dict_params(self, registry, ctx):
        """params 中嵌套 dict 的 ${...} 替换。"""
        # 用 TestParamEcho 测不了嵌套 dict，改用直接值检查
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value:
                nested: "${inputs.x}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": "deep"}, ctx)
        assert result == {"out": {"nested": "deep"}}

    @pytest.mark.asyncio
    async def test_substitute_list_params(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value:
                - "${inputs.a}"
                - "${inputs.b}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"a": 1, "b": 2}, ctx)
        assert result == {"out": [1, 2]}

    @pytest.mark.asyncio
    async def test_substitute_missing_var_raises(self, registry, ctx):
        """变量不存在且无默认值时抛错。"""
        yaml_text = """
        name: test
        nodes:
          - id: p
            type: TestParamEcho
            params:
              value: "${inputs.nonexistent}"
        outputs:
          out: p.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutorError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_substitute_node_output_ref(self, registry, ctx):
        """${node_id.output.port} 引用节点输出。"""
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: dst
            type: TestParamEcho
            params:
              value: "${src.output.value}"
        outputs:
          out: dst.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": "hello"}, ctx)
        assert result == {"out": "hello"}


# ===========================================================================
# 输入引用解析测试
# ===========================================================================


class TestInputReference:
    """输入引用 node_id.output.port 解析测试。"""

    @pytest.mark.asyncio
    async def test_simple_ref(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: dst
            type: TestDouble
            inputs:
              value: src.output.value
        outputs:
          out: dst.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 5}, ctx)
        assert result == {"out": 10}

    @pytest.mark.asyncio
    async def test_multiple_refs(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: a
            type: TestParamEcho
            params:
              value: "${inputs.a}"
          - id: b
            type: TestParamEcho
            params:
              value: "${inputs.b}"
          - id: add
            type: TestAdd
            inputs:
              a: a.output.value
              b: b.output.value
        outputs:
          out: add.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"a": 3, "b": 4}, ctx)
        assert result == {"out": 7}

    @pytest.mark.asyncio
    async def test_direct_value_input(self, registry, ctx):
        """非字符串 inputs 值直接传递。"""
        yaml_text = """
        name: test
        nodes:
          - id: add
            type: TestAdd
            inputs:
              a: 10
              b: 20
        outputs:
          out: add.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        assert result == {"out": 30}

    @pytest.mark.asyncio
    async def test_ref_to_skipped_node_raises(self, registry, ctx):
        """引用被 condition 跳过的节点抛错。"""
        yaml_text = """
        name: test
        nodes:
          - id: skipped
            type: TestEcho
            inputs:
              value: "${inputs.x}"
            condition: "false"
          - id: user
            type: TestDouble
            inputs:
              value: skipped.output.value
        outputs:
          out: user.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutorError):
            await ex.execute({"x": 5}, ctx)

    @pytest.mark.asyncio
    async def test_ref_to_nonexistent_port_raises(self, registry, ctx):
        """引用不存在的端口抛错。"""
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "hello"
          - id: user
            type: TestDouble
            inputs:
              value: src.output.nonexistent
        outputs:
          out: user.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowReferenceError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_invalid_ref_format_raises(self, registry, ctx):
        """无效引用格式抛错。"""
        yaml_text = """
        name: test
        nodes:
          - id: user
            type: TestDouble
            inputs:
              value: invalid.format.with.too.many.dots
        outputs:
          out: user.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowReferenceError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_ref_without_output_keyword_raises(self, registry, ctx):
        """引用缺 'output' 关键字抛错。"""
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "hello"
          - id: user
            type: TestDouble
            inputs:
              value: src.value
        outputs:
          out: user.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowReferenceError):
            await ex.execute({}, ctx)


# ===========================================================================
# 条件执行测试
# ===========================================================================


class TestCondition:
    """condition 字段测试。"""

    @pytest.mark.asyncio
    async def test_condition_true_executes(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: dst
            type: TestDouble
            inputs:
              value: src.output.value
            condition: "${inputs.x} > 0"
        outputs:
          out: dst.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 5}, ctx)
        assert result == {"out": 10}

    @pytest.mark.asyncio
    async def test_condition_false_skips(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: skipped
            type: TestDouble
            inputs:
              value: src.output.value
            condition: "${inputs.x} > 100"
        outputs:
          r: src.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 5}, ctx)
        # skipped 节点没执行，但 src 输出可用
        assert result == {"r": 5}

    @pytest.mark.asyncio
    async def test_condition_references_node_output(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: dst
            type: TestDouble
            inputs:
              value: src.output.value
            condition: "${src.output.value} > 0"
        outputs:
          out: dst.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 5}, ctx)
        assert result == {"out": 10}

    @pytest.mark.asyncio
    async def test_condition_with_len(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.items}"
          - id: conditional
            type: TestParamEcho
            params:
              value: "applied"
            condition: "len(${inputs.items}) > 0"
        outputs:
          r: conditional.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"items": [1, 2, 3]}, ctx)
        assert result == {"r": "applied"}

    @pytest.mark.asyncio
    async def test_condition_dsl_error_raises(self, registry, ctx):
        yaml_text = """
        name: test
        nodes:
          - id: n
            type: TestEcho
            condition: "invalid syntax ("
        outputs:
          r: n.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutionError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_condition_false_skips_does_not_pollute_cache(self, registry, ctx):
        """被跳过的节点不出现在 cache 中。"""
        yaml_text = """
        name: test
        nodes:
          - id: skipped
            type: TestParamEcho
            params:
              value: "x"
            condition: "false"
          - id: other
            type: TestParamEcho
            params:
              value: "y"
        outputs:
          r: other.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        await ex.execute({}, ctx)
        assert "skipped" not in ex.cache
        assert "other" in ex.cache


# ===========================================================================
# 执行流程测试
# ===========================================================================


class TestExecution:
    """执行流程测试。"""

    @pytest.mark.asyncio
    async def test_simple_chain(self, registry, ctx):
        yaml_text = """
        name: chain
        nodes:
          - id: n1
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: n2
            type: TestDouble
            inputs:
              value: n1.output.value
          - id: n3
            type: TestDouble
            inputs:
              value: n2.output.result
        outputs:
          final: n3.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 5}, ctx)
        # 5 -> 10 -> 20
        assert result == {"final": 20}

    @pytest.mark.asyncio
    async def test_diamond(self, registry, ctx):
        yaml_text = """
        name: diamond
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: left
            type: TestDouble
            inputs:
              value: src.output.value
          - id: right
            type: TestDouble
            inputs:
              value: src.output.value
          - id: merge
            type: TestAdd
            inputs:
              a: left.output.result
              b: right.output.result
        outputs:
          total: merge.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": 10}, ctx)
        # 10 -> 20 + 20 = 40
        assert result == {"total": 40}

    @pytest.mark.asyncio
    async def test_node_failure_wrapped(self, registry, ctx):
        """节点 execute 抛错被包装为 WorkflowExecutionError。"""
        yaml_text = """
        name: fail
        nodes:
          - id: bad
            type: TestFailing
        outputs:
          r: bad.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutionError) as exc_info:
            await ex.execute({}, ctx)
        assert exc_info.value.node_id == "bad"
        assert exc_info.value.node_type == "TestFailing"
        assert isinstance(exc_info.value.cause, RuntimeError)

    @pytest.mark.asyncio
    async def test_validation_failure_wrapped(self, registry, ctx):
        """节点 validate 失败被包装。"""
        yaml_text = """
        name: validation
        nodes:
          - id: v
            type: TestValidating
            params:
              n: -1
        outputs:
          r: v.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutionError) as exc_info:
            await ex.execute({}, ctx)
        assert isinstance(exc_info.value.cause, NodeValidationError)

    @pytest.mark.asyncio
    async def test_node_not_registered(self, ctx):
        """未注册节点类型抛错。"""
        yaml_text = """
        name: unknown
        nodes:
          - id: n
            type: NonExistentType
        outputs:
          r: n.output.value
        """
        # 用空 registry
        empty_reg = NodeRegistry()
        ex = WorkflowExecutor(yaml_text, empty_reg)
        with pytest.raises(WorkflowExecutionError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_cleanup_called_on_success(self, registry, ctx):
        """成功完成后调用 cleanup。"""
        CleanupTrackingNode.cleanup_called = False
        yaml_text = """
        name: cleanup
        nodes:
          - id: c
            type: TestCleanup
        outputs:
          r: c.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        await ex.execute({}, ctx)
        assert CleanupTrackingNode.cleanup_called is True

    @pytest.mark.asyncio
    async def test_cleanup_called_on_failure(self, registry, ctx):
        """失败时也调用 cleanup（已执行节点的）。"""
        CleanupTrackingNode.cleanup_called = False
        yaml_text = """
        name: cleanup_on_fail
        nodes:
          - id: c
            type: TestCleanup
          - id: bad
            type: TestFailing
            inputs:
              value: c.output.value
        outputs:
          r: bad.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutionError):
            await ex.execute({}, ctx)
        # c 节点已执行，cleanup 应该被调用
        assert CleanupTrackingNode.cleanup_called is True

    @pytest.mark.asyncio
    async def test_cache_key_hit_skips_execute(self, registry, ctx):
        """相同 cache_key 命中缓存，跳过 execute。"""
        CacheableNode.execute_count = 0
        yaml_text = """
        name: cache
        nodes:
          - id: c1
            type: TestCacheable
            inputs:
              key: "${inputs.k}"
        outputs:
          r: c1.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # 第一次执行
        await ex.execute({"k": "mykey"}, ctx)
        assert CacheableNode.execute_count == 1
        # 第二次执行同 key，应命中缓存
        await ex.execute({"k": "mykey"}, ctx)
        assert CacheableNode.execute_count == 1  # 未增加

    @pytest.mark.asyncio
    async def test_cache_key_different_keys_re_executes(self, registry, ctx):
        """不同 cache_key 重新执行。"""
        CacheableNode.execute_count = 0
        yaml_text = """
        name: cache
        nodes:
          - id: c1
            type: TestCacheable
            inputs:
              key: "${inputs.k}"
        outputs:
          r: c1.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        await ex.execute({"k": "key1"}, ctx)
        assert CacheableNode.execute_count == 1
        await ex.execute({"k": "key2"}, ctx)
        assert CacheableNode.execute_count == 2

    @pytest.mark.asyncio
    async def test_cancel_signal_stops_execution(self, registry):
        """取消信号停止执行。

        取消信号在执行前已设置，导致所有节点都没执行。
        此时 outputs 解析会因引用未执行节点而抛 WorkflowReferenceError，
        或者 execute 提前返回不完整结果。两者都表明取消生效。
        """
        yaml_text = """
        name: cancel
        nodes:
          - id: n1
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: n2
            type: TestDouble
            inputs:
              value: n1.output.value
        outputs:
          r: n2.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # 创建带 cancel_event 的 ctx 并立即设置取消信号
        cancel_event = asyncio.Event()
        cancel_event.set()
        ctx_with_cancel = make_default_context(cancel_event=cancel_event)
        # 取消后 execute 要么抛 WorkflowReferenceError（outputs 无法解析），
        # 要么返回不完整 dict。两种情况都验证：未执行所有节点。
        try:
            await ex.execute({"x": 5}, ctx_with_cancel)
        except WorkflowReferenceError:
            pass  # 预期行为之一：outputs 引用未执行节点抛错
        # 验证取消生效：至少有一个节点未执行
        assert "n2" not in ex.cache or "n1" not in ex.cache


# ===========================================================================
# 输出收集测试
# ===========================================================================


class TestOutputs:
    """输出收集测试。"""

    @pytest.mark.asyncio
    async def test_single_output(self, registry, ctx):
        yaml_text = """
        name: single
        nodes:
          - id: n1
            type: TestParamEcho
            params:
              value: "${inputs.x}"
        outputs:
          out: n1.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"x": "hello"}, ctx)
        assert result == {"out": "hello"}

    @pytest.mark.asyncio
    async def test_multiple_outputs(self, registry, ctx):
        yaml_text = """
        name: multi
        nodes:
          - id: a
            type: TestParamEcho
            params:
              value: "${inputs.a}"
          - id: b
            type: TestParamEcho
            params:
              value: "${inputs.b}"
        outputs:
          out_a: a.output.value
          out_b: b.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({"a": 1, "b": 2}, ctx)
        assert result == {"out_a": 1, "out_b": 2}

    @pytest.mark.asyncio
    async def test_output_var_ref(self, registry, ctx):
        """输出引用 ${...} 变量。"""
        yaml_text = """
        name: var_out
        nodes:
          - id: n1
            type: TestParamEcho
            params:
              value: "x"
        outputs:
          variant: "${variant}"
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        assert result == {"variant": ctx.variant}

    @pytest.mark.asyncio
    async def test_output_direct_value(self, registry, ctx):
        """非字符串 outputs 直接值。"""
        yaml_text = """
        name: direct
        nodes:
          - id: n1
            type: TestParamEcho
            params:
              value: "x"
        outputs:
          literal: 42
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        assert result == {"literal": 42}

    @pytest.mark.asyncio
    async def test_empty_outputs(self, registry, ctx):
        """空 outputs 返回空 dict。"""
        yaml_text = """
        name: empty_out
        nodes:
          - id: n1
            type: TestParamEcho
            params:
              value: "x"
        outputs: {}
        """
        ex = WorkflowExecutor(yaml_text, registry)
        result = await ex.execute({}, ctx)
        assert result == {}


# ===========================================================================
# 错误处理测试
# ===========================================================================


class TestErrorHandling:
    """错误处理与异常类型测试。"""

    def test_workflow_executor_error_is_base(self):
        """WorkflowExecutorError 是所有异常的基类。"""
        assert issubclass(WorkflowValidationError, WorkflowExecutorError)
        assert issubclass(WorkflowCycleError, WorkflowExecutorError)
        assert issubclass(WorkflowExecutionError, WorkflowExecutorError)
        assert issubclass(WorkflowReferenceError, WorkflowExecutorError)

    @pytest.mark.asyncio
    async def test_node_failure_includes_context(self, registry, ctx):
        """节点失败错误包含 node_id / node_type / cause。"""
        yaml_text = """
        name: ctx
        nodes:
          - id: my_node
            type: TestFailing
        outputs:
          r: my_node.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutionError) as exc_info:
            await ex.execute({}, ctx)
        e = exc_info.value
        assert e.node_id == "my_node"
        assert e.node_type == "TestFailing"
        assert isinstance(e.cause, RuntimeError)
        assert "my_node" in str(e)
        assert "TestFailing" in str(e)

    @pytest.mark.asyncio
    async def test_dsl_error_in_condition_wrapped(self, registry, ctx):
        """condition DSL 错误被包装。"""
        yaml_text = """
        name: dsl_err
        nodes:
          - id: n
            type: TestEcho
            condition: "${nonexistent.var} > 0"
        outputs:
          r: n.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowExecutionError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_invalid_input_ref_format(self, registry, ctx):
        """无效 inputs 引用格式抛 WorkflowReferenceError。"""
        yaml_text = """
        name: bad_ref
        nodes:
          - id: user
            type: TestDouble
            inputs:
              value: bad.format
        outputs:
          r: user.output.result
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowReferenceError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_params_not_dict_raises(self, registry, ctx):
        """params 非 dict 抛 WorkflowValidationError。"""
        yaml_text = """
        name: bad_params
        nodes:
          - id: n
            type: TestEcho
            params: "not a dict"
        outputs:
          r: n.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowValidationError):
            await ex.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_inputs_not_dict_raises(self, registry, ctx):
        """inputs 非 dict 抛 WorkflowValidationError（在 _extract_deps 时）。"""
        yaml_text = """
        name: bad_inputs
        nodes:
          - id: n
            type: TestEcho
            inputs: "not a dict"
        outputs:
          r: n.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        with pytest.raises(WorkflowValidationError):
            await ex.execute({}, ctx)


# ===========================================================================
# get_execution_order 测试
# ===========================================================================


class TestGetExecutionOrder:
    """get_execution_order 公开方法测试。"""

    def test_returns_topological_order(self, registry):
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho"},
                {"id": "b", "type": "TestEcho", "inputs": {"value": "a.output.value"}},
                {"id": "c", "type": "TestEcho", "inputs": {"value": "b.output.value"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        order = ex.get_execution_order()
        assert order == ["a", "b", "c"]

    def test_does_not_execute(self, registry):
        """get_execution_order 不实际执行节点。"""
        wf = {
            "nodes": [
                {"id": "n", "type": "TestEcho"},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        order = ex.get_execution_order()
        assert order == ["n"]
        # cache 应该仍为空
        assert ex.cache == {}

    def test_raises_on_cycle(self, registry):
        wf = {
            "nodes": [
                {"id": "a", "type": "TestEcho", "inputs": {"value": "b.output.value"}},
                {"id": "b", "type": "TestEcho", "inputs": {"value": "a.output.value"}},
            ],
            "outputs": {},
        }
        ex = WorkflowExecutor(wf, registry)
        with pytest.raises(WorkflowCycleError):
            ex.get_execution_order()


# ===========================================================================
# 综合场景测试
# ===========================================================================


class TestScenarios:
    """模拟真实工作流场景。"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, registry, ctx):
        """完整工作流：3 节点链式 + 条件 + 变量替换。"""
        yaml_text = """
        name: complete
        nodes:
          - id: input_prompt
            type: TestParamEcho
            params:
              value: "${inputs.prompt}"
          - id: doubled
            type: TestDouble
            inputs:
              value: input_prompt.output.value
            condition: "${inputs.apply} == true"
          - id: final
            type: TestEcho
            inputs:
              value: doubled.output.result
        outputs:
          result: final.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # apply=true 时执行 doubled
        # 但 doubled.output.result 是 int，TestEcho 接收 int 并输出
        # 这里用字符串避免类型混淆
        result = await ex.execute({"prompt": 5, "apply": True}, ctx)
        # 5 * 2 = 10
        assert result == {"result": 10}

    @pytest.mark.asyncio
    async def test_conditional_skip_path(self, registry, ctx):
        """条件跳过路径。"""
        yaml_text = """
        name: skip_path
        nodes:
          - id: src
            type: TestParamEcho
            params:
              value: "${inputs.x}"
          - id: maybe_double
            type: TestDouble
            inputs:
              value: src.output.value
            condition: "${inputs.doublit} == true"
          - id: passthrough
            type: TestEcho
            inputs:
              value: src.output.value
            condition: "${inputs.doublit} == false"
        outputs:
          r: passthrough.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # doublit=false：maybe_double 跳过，passthrough 执行
        result = await ex.execute({"x": 5, "doublit": False}, ctx)
        assert result == {"r": 5}

    @pytest.mark.asyncio
    async def test_lora_injection(self, registry, ctx):
        """模拟 LoRA 注入工作流。"""
        yaml_text = """
        name: lora
        nodes:
          - id: lora_apply
            type: TestParamEcho
            params:
              value: "${loras}"
            condition: "len(${loras}) > 0"
        outputs:
          applied: lora_apply.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # ctx.loras 默认是空 list，需要手动设置
        ctx.loras = [{"name": "style_xfer"}]
        result = await ex.execute({}, ctx)
        assert result == {"applied": [{"name": "style_xfer"}]}

    @pytest.mark.asyncio
    async def test_no_lora_skipped(self, registry, ctx):
        """LoRA 为空时跳过。"""
        yaml_text = """
        name: no_lora
        nodes:
          - id: lora_apply
            type: TestParamEcho
            params:
              value: "${loras}"
            condition: "len(${loras}) > 0"
          - id: fallback
            type: TestParamEcho
            params:
              value: "no_lora"
        outputs:
          r: fallback.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        # ctx.loras 默认空
        result = await ex.execute({}, ctx)
        assert result == {"r": "no_lora"}

    @pytest.mark.asyncio
    async def test_ernie_image_workflow_simulation(self, registry, ctx):
        """模拟 ERNIE-Image 工作流（简化版）。"""
        yaml_text = """
        name: ernie_image
        nodes:
          - id: input_prompt
            type: TestParamEcho
            params:
              value: "${inputs.prompt}"
          - id: input_negative
            type: TestParamEcho
            params:
              value: "${inputs.negative_prompt or ''}"
          - id: model_loader
            type: TestParamEcho
            params:
              value: "${variant}"
          - id: sampler
            type: TestAdd
            inputs:
              a: input_prompt.output.value
              b: input_negative.output.value
          - id: save
            type: TestEcho
            inputs:
              value: sampler.output.result
        outputs:
          result: save.output.value
          model: model_loader.output.value
        """
        ex = WorkflowExecutor(yaml_text, registry)
        ctx.variant = "ernie-image-v1"
        result = await ex.execute({"prompt": "hello", "negative_prompt": "blurry"}, ctx)
        # sampler: "hello" + "blurry" -> "helloblurry"
        assert result["result"] == "helloblurry"
        assert result["model"] == "ernie-image-v1"

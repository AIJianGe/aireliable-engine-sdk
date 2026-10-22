"""工作流执行器（v0.2 新增）。

设计参考：
    - v0.2-analysis.md §12（工作流引擎架构）
    - PR/architecture.md §12.5（WorkflowExecutor）
    - PR/development-guide.md §29（工作流 yaml 规范）

核心特性：
    1. **yaml/dict 双输入**：接受 yaml 字符串或已解析的 dict
    2. **拓扑排序**：Kahn 算法 + 循环依赖检测
    3. **条件执行**：节点级 ``condition`` 字段（DSL 见 workflow_dsl）
    4. **输入引用解析**：``node_id.output.port`` / ``${variant}`` /
       ``${inputs.field}`` / ``${loras}`` / ``${inputs.field or default}``
    5. **参数变量替换**：``params`` 中的字符串值支持 ``${...}`` 替换
    6. **节点缓存**：``cache_key`` 非空时复用结果（同输入跳过 execute）
    7. **取消信号**：每节点执行前检查 ``ctx.is_cancelled()``
    8. **错误聚合**：节点执行失败时携带上下文信息（node_id / type / 原异常）
    9. **资源清理**：节点 ``cleanup`` 在异常时调用

使用示例::

    import asyncio
    from aireliable_engine import (
        WorkflowExecutor, NodeRegistry, register_node, make_default_context,
    )
    from aireliable_engine.nodes.input.text_input import TextInputNode
    from aireliable_engine.nodes.postprocess.image_save import ImageSaveNode

    register_node(TextInputNode)
    register_node(ImageSaveNode)

    yaml_text = '''
    workflow:
      name: my_workflow
      nodes:
        - id: input_prompt
          type: TextInput
          params: { field: prompt }
      outputs:
        result: input_prompt.output.text
    '''

    async def main():
        executor = WorkflowExecutor(yaml_text, NodeRegistry())
        ctx = make_default_context()
        result = await executor.execute({"prompt": "hello"}, ctx)
        print(result)  # {"result": "hello"}

    asyncio.run(main())
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Optional, Union

import yaml

from .node import BaseNode, NodeValidationError
from .node_registry import NodeNotFoundError, NodeRegistry
from .workflow_context import WorkflowContext
from .workflow_dsl import DSLError, evaluate_condition

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ===========================================================================
# 异常类
# ===========================================================================

class WorkflowExecutorError(Exception):
    """工作流执行器错误基类。"""


class WorkflowValidationError(WorkflowExecutorError):
    """工作流结构校验错误（yaml 格式 / 必填字段缺失 / 节点 id 重复等）。"""


class WorkflowCycleError(WorkflowExecutorError):
    """工作流循环依赖错误（拓扑排序检测到环）。"""

    def __init__(self, cycle_nodes: list[str]):
        self.cycle_nodes = cycle_nodes
        super().__init__(
            f"工作流检测到循环依赖：{' -> '.join(cycle_nodes)}"
        )


class WorkflowExecutionError(WorkflowExecutorError):
    """节点执行错误（含上下文信息）。"""

    def __init__(
        self,
        node_id: str,
        node_type: str,
        cause: BaseException,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.cause = cause
        super().__init__(
            f"节点 {node_id!r} (type={node_type!r}) 执行失败："
            f"{type(cause).__name__}: {cause}"
        )


class WorkflowReferenceError(WorkflowExecutorError):
    """输入引用错误（格式错误 / 节点未执行 / 端口不存在）。"""


# ===========================================================================
# 内部工具：引用解析
# ===========================================================================

# 完整 ${...} 匹配
_FULL_VAR_RE = re.compile(r"^\$\{([^}]+)\}$")

# 部分替换 ${...}（用于字符串拼接）
_PARTIAL_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# 输入引用：node_id.output.port
_INPUT_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.output\.[A-Za-z_][A-Za-z0-9_]*$")

# 默认值语法：var_path or default_literal
# 注意：默认值可以是数字、字符串、true/false/null
_OR_DEFAULT_RE = re.compile(r"^\s*(\S(?:.*?\S)?)\s+or\s+(\S.*?)\s*$", re.DOTALL)


def _parse_or_default(var_expr: str) -> tuple[str, Optional[str]]:
    """解析 ``${inputs.field or default}`` 语法。

    Args:
        var_expr: ${...} 内部的表达式（不含外层 ${}）

    Returns:
        (var_path, default_str) 元组。default_str 为 None 表示无默认值。
    """
    m = _OR_DEFAULT_RE.match(var_expr)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return var_expr.strip(), None


def _parse_literal(text: str) -> Any:
    """把默认值字符串解析为 Python 字面量。

    支持整数 / 浮点 / true/false/null / 带引号字符串 / 裸字符串。
    """
    text = text.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null" or text == "none":
        return None
    # 带引号字符串
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    # 整数
    try:
        return int(text)
    except ValueError:
        pass
    # 浮点
    try:
        return float(text)
    except ValueError:
        pass
    # 裸字符串
    return text


def _split_var_path(var_path: str) -> list[str]:
    """把 ``a.b.c`` 切分为 ``["a", "b", "c"]``。"""
    return [p.strip() for p in var_path.split(".") if p.strip()]


# ===========================================================================
# WorkflowExecutor
# ===========================================================================


class WorkflowExecutor:
    """工作流执行器。

    流程：
        1. 解析 yaml → 构建 graph
        2. 拓扑排序确定执行顺序（含循环检测）
        3. 按序执行节点，缓存中间结果
        4. 处理 ``condition`` 条件跳过
        5. 收集 outputs 返回

    Args:
        workflow: 工作流定义，可以是 yaml 字符串或已解析的 dict。
            dict 格式：``{"name": ..., "nodes": [...], "outputs": {...}}``
            或 ``{"workflow": {...}}`` 嵌套结构（兼容 development-guide.md §29.1）
        registry: 节点注册表。若为 None，使用全局默认注册表。

    Raises:
        WorkflowValidationError: 工作流结构错误
    """

    def __init__(
        self,
        workflow: Union[str, dict],
        registry: Optional[NodeRegistry] = None,
    ):
        # 1. 解析 workflow
        if isinstance(workflow, str):
            try:
                parsed = yaml.safe_load(workflow)
            except yaml.YAMLError as e:
                raise WorkflowValidationError(f"yaml 解析失败：{e}") from e
        else:
            parsed = workflow

        if not isinstance(parsed, dict):
            raise WorkflowValidationError(
                f"workflow 必须是 dict 或 yaml 字符串，实际 {type(parsed).__name__}"
            )

        # 兼容 {"workflow": {...}} 嵌套结构
        if "workflow" in parsed and isinstance(parsed["workflow"], dict):
            wf = parsed["workflow"]
        else:
            wf = parsed

        self.workflow: dict[str, Any] = wf
        # 默认使用全局注册表（与 register_node 装饰器一致）
        if registry is not None:
            self.registry = registry
        else:
            self.registry = NodeRegistry._get_default()

        # 节点缓存：node_id -> { port: value }
        self.cache: dict[str, dict[str, Any]] = {}
        # cache_key 缓存：cache_key_str -> outputs dict
        self._result_cache: dict[str, dict[str, Any]] = {}

        # 解析节点列表
        self.nodes: list[dict[str, Any]] = list(wf.get("nodes", []))
        self.outputs_spec: dict[str, str] = dict(wf.get("outputs", {}))
        self.workflow_name: str = wf.get("name", "<unnamed>")

        # 校验
        self._validate()

    # ===========================================================================
    # 校验
    # ===========================================================================

    def _validate(self) -> None:
        """校验 workflow 结构。"""
        if not self.nodes:
            raise WorkflowValidationError(
                f"工作流 {self.workflow_name!r} 没有节点"
            )

        # 节点 id 唯一性 + 必填字段
        seen_ids: set[str] = set()
        for node_def in self.nodes:
            if not isinstance(node_def, dict):
                raise WorkflowValidationError(
                    f"节点定义必须是 dict，实际 {type(node_def).__name__}"
                )
            node_id = node_def.get("id")
            if not node_id or not isinstance(node_id, str):
                raise WorkflowValidationError(
                    f"节点缺少必填字段 id（或类型不是 str）：{node_def!r}"
                )
            if node_id in seen_ids:
                raise WorkflowValidationError(
                    f"节点 id 重复：{node_id!r}"
                )
            seen_ids.add(node_id)

            node_type = node_def.get("type")
            if not node_type or not isinstance(node_type, str):
                raise WorkflowValidationError(
                    f"节点 {node_id!r} 缺少必填字段 type（或类型不是 str）"
                )

        # outputs 引用校验：
        # - str 类型：必须是 node_id.output.port 引用或 ${...} 变量注入
        # - 非 str 类型：作为直接字面值传递（如 int / list / dict）
        #   （与 _resolve_outputs 的处理逻辑保持一致）
        # 此处不做严格校验，留给 _resolve_outputs 在执行时处理。

    # ===========================================================================
    # graph 构建 + 拓扑排序
    # ===========================================================================

    def _node_id_set(self) -> set[str]:
        return {n["id"] for n in self.nodes}

    def _extract_deps(self, node_def: dict[str, Any]) -> list[str]:
        """从 node_def['inputs'] 提取依赖的 node_id 列表。

        输入引用格式：``node_id.output.port``
        """
        deps: list[str] = []
        inputs_spec = node_def.get("inputs", {}) or {}
        if not isinstance(inputs_spec, dict):
            raise WorkflowValidationError(
                f"节点 {node_def['id']!r} inputs 必须是 dict，"
                f"实际 {type(inputs_spec).__name__}"
            )
        for port_name, ref in inputs_spec.items():
            if not isinstance(ref, str):
                continue  # 非字符串引用（直接值）不算依赖
            # 跳过 ${...} 形式（这是变量注入，不是节点引用）
            if ref.startswith("${"):
                continue
            # 解析 node_id.output.port
            parts = ref.split(".")
            if len(parts) == 3 and parts[1] == "output":
                deps.append(parts[0])
            # 其他格式视为非法（在 _resolve_inputs 时会报错）
        return deps

    def _build_graph(self) -> dict[str, list[str]]:
        """构建邻接表：node_id -> [依赖的 node_id]。"""
        graph: dict[str, list[str]] = {}
        for node_def in self.nodes:
            graph[node_def["id"]] = self._extract_deps(node_def)
        return graph

    def _topological_sort(self, graph: dict[str, list[str]]) -> list[str]:
        """Kahn 算法拓扑排序，含循环检测。

        Returns:
            拓扑顺序的 node_id 列表

        Raises:
            WorkflowCycleError: 检测到循环依赖
        """
        # 计算入度
        in_degree: dict[str, int] = {nid: 0 for nid in graph}
        # 反向邻接表：node_id -> 依赖此节点的 node_id 列表
        dependents: dict[str, list[str]] = {nid: [] for nid in graph}

        for nid, deps in graph.items():
            for dep in deps:
                if dep not in graph:
                    raise WorkflowValidationError(
                        f"节点 {nid!r} 引用了不存在的节点 {dep!r}"
                    )
                in_degree[nid] += 1
                dependents[dep].append(nid)

        # 入度为 0 的节点入队
        queue: list[str] = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for dependent in dependents[nid]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(graph):
            # 存在循环：找出剩余节点
            remaining = [nid for nid in graph if nid not in order]
            raise WorkflowCycleError(remaining)

        return order

    # ===========================================================================
    # 变量替换 / 引用解析
    # ===========================================================================

    def _eval_var_ref(
        self,
        var_path: str,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> Any:
        """评估单个 ``${...}`` 变量引用。

        支持：
            - ``${variant}`` → ctx.variant
            - ``${loras}`` → ctx.loras
            - ``${inputs.field}`` → inputs[field]
            - ``${inputs.field or default}`` → inputs.get(field, default)
            - ``${node_id.output.port}`` → cache[node_id][port]
        """
        var_path_clean, default_str = _parse_or_default(var_path)
        parts = _split_var_path(var_path_clean)
        if not parts:
            raise WorkflowReferenceError(
                f"变量引用 ${{{var_path}}} 路径为空"
            )

        try:
            value = self._resolve_var_path(parts, inputs, ctx)
        except (KeyError, WorkflowReferenceError):
            if default_str is not None:
                return _parse_literal(default_str)
            raise

        if value is None and default_str is not None:
            return _parse_literal(default_str)

        return value

    def _resolve_var_path(
        self,
        parts: list[str],
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> Any:
        """按 path 段递归访问变量。

        parts[0] 是根变量名：
            - "variant" → ctx.variant
            - "loras" → ctx.loras
            - "inputs" → inputs dict
            - 其他 → 视为 node_id（从 cache 取）
        """
        root = parts[0]
        if root == "variant":
            current: Any = ctx.variant
        elif root == "loras":
            current = ctx.loras
        elif root == "inputs":
            current = inputs
        elif root in self.cache:
            # 节点输出：cache[node_id] = {port: value}
            # DSL 引用格式：${node_id.output.port}
            # parts = ["node_id", "output", "port"]
            current = self.cache[root]
        else:
            raise WorkflowReferenceError(
                f"变量 {root!r} 未定义（不在 variant/loras/inputs/节点输出 中）"
            )

        # 后续 path 段
        for seg in parts[1:]:
            current = self._access(current, seg)
        return current

    @staticmethod
    def _access(current: Any, seg: str) -> Any:
        """按 path 段访问 dict / list / 对象属性。

        对于节点输出 cache[root] = {port: value}，DSL 路径
        ``${node_id.output.port}`` 的第 2 段是 "output"，
        但 cache 结构是 {port: value}，没有 "output" 中间层。
        所以这里特殊处理：若 seg == "output" 且 current 是 dict 但
        "output" 不在 dict 中，则跳过 "output" 段直接访问下一段。
        """
        if current is None:
            raise WorkflowReferenceError(
                f"路径段 {seg!r} 无法在 None 上访问"
            )
        if isinstance(current, dict):
            # 特殊：节点输出 ${node_id.output.port}
            # cache[node_id] = {port: value}，但 DSL 写 "output" 段
            # 我们让 cache[node_id] = {"output": {port: value}} 吗？
            # 不，保持 cache[node_id] = {port: value}，但 DSL 路径
            # 段 "output" 跳过：若 "output" 不在 dict 但 dict 非空，
            # 视为整体 dict 就是 "output" 内容
            if seg in current:
                return current[seg]
            if seg == "output":
                # 跳过 "output" 段
                return current
            raise WorkflowReferenceError(
                f"key {seg!r} 不存在"
            )
        if isinstance(current, (list, tuple)):
            try:
                idx = int(seg)
            except ValueError:
                raise WorkflowReferenceError(
                    f"索引 {seg!r} 不是整数"
                )
            if idx < 0 or idx >= len(current):
                raise WorkflowReferenceError(
                    f"索引 {idx} 越界（长度 {len(current)}）"
                )
            return current[idx]
        # 对象属性
        if seg.startswith("_"):
            raise WorkflowReferenceError(
                f"禁止访问私有属性 {seg!r}"
            )
        if not hasattr(current, seg):
            raise WorkflowReferenceError(
                f"属性 {seg!r} 不存在（类型 {type(current).__name__}）"
            )
        return getattr(current, seg)

    def _substitute_vars(
        self,
        text: str,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> Any:
        """替换字符串中的 ``${...}`` 变量。

        - 整个字符串是 ``${...}``：返回原值（保留类型）
        - 部分包含 ``${...}``：字符串拼接（值转 str）
        """
        # 完整匹配 ${...}
        full_m = _FULL_VAR_RE.match(text)
        if full_m:
            return self._eval_var_ref(full_m.group(1), inputs, ctx)

        # 部分替换
        def replace(m: re.Match) -> str:
            value = self._eval_var_ref(m.group(1), inputs, ctx)
            return str(value)

        return _PARTIAL_VAR_RE.sub(replace, text)

    def _resolve_param_value(
        self,
        value: Any,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> Any:
        """递归解析 param 值中的 ``${...}``。"""
        if isinstance(value, str):
            return self._substitute_vars(value, inputs, ctx)
        if isinstance(value, dict):
            return {
                k: self._resolve_param_value(v, inputs, ctx)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [
                self._resolve_param_value(v, inputs, ctx)
                for v in value
            ]
        return value

    def _resolve_inputs(
        self,
        node_def: dict[str, Any],
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> dict[str, Any]:
        """解析节点的 inputs 引用。

        支持两种引用：
            - 节点引用：``node_id.output.port`` → cache[node_id][port]
            - 变量注入：``${...}`` → 变量替换
            - 直接值：非字符串值原样传递
        """
        result: dict[str, Any] = {}
        inputs_spec = node_def.get("inputs", {}) or {}
        for port_name, ref in inputs_spec.items():
            if isinstance(ref, str):
                if ref.startswith("${"):
                    # 变量注入
                    result[port_name] = self._substitute_vars(ref, inputs, ctx)
                else:
                    # 节点引用：node_id.output.port
                    result[port_name] = self._resolve_node_ref(ref)
            else:
                # 直接值
                result[port_name] = ref
        return result

    def _resolve_node_ref(self, ref: str) -> Any:
        """解析 ``node_id.output.port`` 形式的引用。"""
        parts = ref.split(".")
        if len(parts) != 3 or parts[1] != "output":
            raise WorkflowReferenceError(
                f"无效的输入引用 {ref!r}（期望格式 node_id.output.port）"
            )
        node_id, _, port = parts
        if node_id not in self.cache:
            raise WorkflowReferenceError(
                f"输入引用 {ref!r} 的节点 {node_id!r} 未执行"
                "（可能被 condition 跳过）"
            )
        outputs = self.cache[node_id]
        if port not in outputs:
            raise WorkflowReferenceError(
                f"输入引用 {ref!r} 的端口 {port!r} 不存在"
                f"（可用端口：{list(outputs.keys())}）"
            )
        return outputs[port]

    def _resolve_params(
        self,
        node_def: dict[str, Any],
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> dict[str, Any]:
        """解析节点的 params（含 ``${...}`` 替换）。"""
        params = node_def.get("params", {}) or {}
        if not isinstance(params, dict):
            raise WorkflowValidationError(
                f"节点 {node_def['id']!r} params 必须是 dict，"
                f"实际 {type(params).__name__}"
            )
        return self._resolve_param_value(params, inputs, ctx)

    def _resolve_outputs(
        self,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> dict[str, Any]:
        """解析 workflow outputs。"""
        result: dict[str, Any] = {}
        for out_name, ref in self.outputs_spec.items():
            if not isinstance(ref, str):
                result[out_name] = ref
                continue
            if ref.startswith("${"):
                result[out_name] = self._substitute_vars(ref, inputs, ctx)
            else:
                result[out_name] = self._resolve_node_ref(ref)
        return result

    def _build_dsl_context(
        self,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> dict[str, Any]:
        """构建 DSL 求值上下文。

        DSL 引用格式：``${node_id.output.port}`` →
            context["node_id"]["output"]["port"]
        所以 cache 需要包装为 {"output": {port: value}} 结构。
        """
        dsl_ctx: dict[str, Any] = {
            "inputs": inputs,
            "variant": ctx.variant,
            "loras": ctx.loras,
        }
        # 包装节点输出
        for node_id, outputs in self.cache.items():
            dsl_ctx[node_id] = {"output": outputs}
        return dsl_ctx

    # ===========================================================================
    # 执行入口
    # ===========================================================================

    async def execute(
        self,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> dict[str, Any]:
        """执行工作流。

        Args:
            inputs: 请求输入字段（如 prompt / size / num_inference_steps）
            ctx: 工作流上下文（含 variant / progress_reporter / cancel_event）

        Returns:
            工作流输出字典，key = outputs_spec 的 key

        Raises:
            WorkflowCycleError: 循环依赖
            WorkflowExecutionError: 节点执行失败
            WorkflowReferenceError: 输入引用错误
            DSLError: DSL 求值错误
            NodeNotFoundError: 节点类型未注册
        """
        # v0.2.5+ T6.1：用 log_context 包裹整个执行，让所有节点日志自动带 task_id
        # contextvars 保证 asyncio 任务间隔离，不影响其他并发推理
        from .log_context import log_context

        with log_context(task_id=ctx.task_id):
            return await self._execute_inner(inputs, ctx)

    async def _execute_inner(
        self,
        inputs: dict[str, Any],
        ctx: WorkflowContext,
    ) -> dict[str, Any]:
        """execute 的实际实现（已包裹在 log_context 中）。"""
        # 1. 构建 graph + 拓扑排序
        graph = self._build_graph()
        order = self._topological_sort(graph)

        logger.info(
            "工作流 %r 开始执行，拓扑顺序：%s",
            self.workflow_name, order,
        )

        # 2. 预校验所有节点的 validate()
        # （仅校验参数结构，不实际执行）
        # 此处跳过：节点 params 中的 ${} 在执行时才解析，
        # 校验阶段无 inputs/ctx 可用，会失败。
        # 实际校验放在每个节点 execute 前。

        # 3. 按序执行
        executed_nodes: list[tuple[str, BaseNode]] = []
        try:
            for node_id in order:
                # 取消信号检查
                if ctx.is_cancelled():
                    logger.info("工作流 %r 被取消", self.workflow_name)
                    break

                node_def = self._find_node_def(node_id)
                node_type = node_def["type"]

                # 构建 DSL 上下文（每次循环都重建，让 condition 能引用最新 cache）
                dsl_context = self._build_dsl_context(inputs, ctx)

                # 条件检查
                if "condition" in node_def:
                    cond_expr = node_def["condition"]
                    try:
                        should_run = evaluate_condition(cond_expr, dsl_context)
                    except DSLError as e:
                        raise WorkflowExecutionError(
                            node_id, node_type, e
                        ) from e
                    if not should_run:
                        logger.info(
                            "节点 %r (type=%r) 被条件 %r 跳过",
                            node_id, node_type, cond_expr,
                        )
                        continue

                # 解析 inputs / params
                node_inputs = self._resolve_inputs(node_def, inputs, ctx)
                node_params = self._resolve_params(node_def, inputs, ctx)

                # 实例化节点
                try:
                    node_cls = self.registry.get(node_type)
                except NodeNotFoundError as e:
                    raise WorkflowExecutionError(
                        node_id, node_type, e
                    ) from e

                node = node_cls(node_params)

                # 参数校验
                errors = node.validate()
                if errors:
                    raise WorkflowExecutionError(
                        node_id, node_type,
                        NodeValidationError(node_type, errors),
                    )

                # 缓存检查
                cache_key = node.cache_key(node_inputs)
                if cache_key is not None and cache_key in self._result_cache:
                    logger.info(
                        "节点 %r (type=%r) 命中缓存（cache_key=%r）",
                        node_id, node_type, cache_key,
                    )
                    outputs = self._result_cache[cache_key]
                else:
                    # 执行
                    logger.debug(
                        "执行节点 %r (type=%r)，inputs=%r",
                        node_id, node_type, list(node_inputs.keys()),
                    )
                    try:
                        outputs = await node.execute(node_inputs, ctx)
                    except Exception as e:
                        raise WorkflowExecutionError(
                            node_id, node_type, e
                        ) from e

                    # 缓存结果
                    if cache_key is not None:
                        self._result_cache[cache_key] = outputs

                # 记录到 cache
                self.cache[node_id] = outputs if outputs is not None else {}
                executed_nodes.append((node_id, node))

                logger.debug(
                    "节点 %r 执行完成，输出端口：%s",
                    node_id, list(self.cache[node_id].keys()),
                )

            # 4. 收集 outputs
            result = self._resolve_outputs(inputs, ctx)
            logger.info("工作流 %r 执行完成，输出键：%s",
                        self.workflow_name, list(result.keys()))
            return result

        finally:
            # 资源清理（无论成功失败都调用）
            for node_id, node in reversed(executed_nodes):
                try:
                    node.cleanup()
                except Exception as e:
                    logger.warning(
                        "节点 %r cleanup 失败：%s: %s",
                        node_id, type(e).__name__, e,
                    )

    # ===========================================================================
    # 辅助方法
    # ===========================================================================

    def _find_node_def(self, node_id: str) -> dict[str, Any]:
        """按 node_id 查找节点定义。"""
        for n in self.nodes:
            if n["id"] == node_id:
                return n
        raise WorkflowReferenceError(f"节点 {node_id!r} 不存在")

    def get_execution_order(self) -> list[str]:
        """获取拓扑排序后的执行顺序（不实际执行）。

        用于调试 / 工具可视化。

        Raises:
            WorkflowCycleError: 循环依赖
        """
        graph = self._build_graph()
        return self._topological_sort(graph)


__all__ = [
    # 异常
    "WorkflowExecutorError",
    "WorkflowValidationError",
    "WorkflowCycleError",
    "WorkflowExecutionError",
    "WorkflowReferenceError",
    # 执行器
    "WorkflowExecutor",
]

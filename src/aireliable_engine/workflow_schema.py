"""工作流 yaml schema 定义（v0.2 阶段 4 任务 4.5）。

使用 Pydantic 模型对工作流 yaml 进行严格校验，提供清晰错误信息。

设计参考：
    - v0.2-tasks.md 任务 4.5

校验规则：
    1. 顶层结构：name / description / nodes / outputs / branches
    2. 节点字段：id（唯一）/ type / pos / params / inputs / condition
    3. 输出定义：key=输出名，value=引用字符串或字面值
    4. 分支定义：name / condition / nodes / outputs（v0.2 仅 schema 校验，执行逻辑待后续阶段）
    5. 嵌套 {'workflow': {...}} 兼容

使用：
    from aireliable_engine.workflow_schema import (
        WorkflowSchema, NodeDef, BranchDef,
        validate_workflow_dict, validate_workflow_yaml,
        WorkflowSchemaError,
    )

    schema = validate_workflow_yaml(yaml_text)
    schema = validate_workflow_dict({"workflow": {...}})

    # 转换为普通 dict（供 WorkflowExecutor 使用）
    workflow_dict = schema.to_dict()
"""

from __future__ import annotations

import re
from typing import Any, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ===========================================================================
# 常量
# ===========================================================================

# 节点 id 模式：字母开头，仅含字母数字下划线（避免特殊字符与 DSL 冲突）
_NODE_ID_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 分支名模式：同节点 id
_BRANCH_NAME_PATTERN = _NODE_ID_PATTERN


# ===========================================================================
# 异常
# ===========================================================================

class WorkflowSchemaError(ValueError):
    """工作流 schema 校验错误。

    Attributes:
        field_path: 错误字段路径（如 "nodes[0].id"）。空字符串表示顶层错误。
    """

    def __init__(self, message: str, *, field_path: str = ""):
        self.field_path = field_path
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.field_path:
            return f"[{self.field_path}] {msg}"
        return msg


# ===========================================================================
# 节点定义
# ===========================================================================

class NodeDef(BaseModel):
    """节点定义。

    Attributes:
        id: 节点 id（工作流内唯一）。必须以字母或下划线开头，仅含字母数字下划线。
        type: 节点类型字符串（注册表中的 key）。
        pos: 画布坐标 [x, y]（可选，仅 UI 持久化用）。
        params: 节点参数字典（key=参数名，value=任意 JSON 兼容值）。
        inputs: 输入连接字典（key=端口名，value=引用字符串或字面值）。
                - ``node_id.output.port`` → 引用上游节点输出
                - ``${variant}`` / ``${inputs.field}`` / ``${loras}`` → 变量注入
                - 非 str 类型 → 字面值传递
        condition: 条件执行表达式（可选，DSL 见 workflow_dsl）。

    Note:
        schema 层不校验节点类型是否注册（由 WorkflowExecutor 在执行时校验），
        也不校验 inputs 引用是否有效（由 WorkflowExecutor 在 graph 构建时校验）。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    pos: Optional[list[int]] = None
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("节点 id 不能为空")
        if not _NODE_ID_PATTERN.match(v):
            raise ValueError(
                f"节点 id {v!r} 格式无效（必须以字母或下划线开头，"
                f"仅含字母数字下划线）"
            )
        return v

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("节点 type 不能为空")
        return v

    @field_validator("pos", mode="before")
    @classmethod
    def _validate_pos(cls, v: Any) -> Any:
        """pos 校验（在类型转换前，避免 Pydantic 把 str 自动转 int）。"""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError(f"节点 pos 必须是 list，实际 {type(v).__name__}")
        if len(v) != 2:
            raise ValueError(
                f"节点 pos 必须是 [x, y] 长度 2 列表，实际长度 {len(v)}"
            )
        for i, item in enumerate(v):
            # 拒绝 bool（虽然 bool 是 int 子类）和 str（避免 Pydantic 自动转换）
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(
                    f"节点 pos[{i}] 必须是 int，实际 {type(item).__name__}"
                )
        return v

    @field_validator("params")
    @classmethod
    def _validate_params(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError(
                f"params 必须是 dict，实际 {type(v).__name__}"
            )
        return v

    @field_validator("inputs")
    @classmethod
    def _validate_inputs(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError(
                f"inputs 必须是 dict，实际 {type(v).__name__}"
            )
        # key 必须是非空字符串
        for key in v.keys():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"inputs 的 key 必须是非空字符串，实际 {key!r}"
                )
        return v

    @field_validator("condition")
    @classmethod
    def _validate_condition(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(
                f"condition 必须是 str，实际 {type(v).__name__}"
            )
        if not v.strip():
            raise ValueError("condition 不能为空字符串（仅空白）")
        return v


# ===========================================================================
# 分支定义（v0.2 仅 schema 校验，执行逻辑待后续阶段）
# ===========================================================================

class BranchDef(BaseModel):
    """if/else 分支定义（v0.2 引入，可选）。

    当前阶段仅做 schema 校验，分支执行逻辑暂未实现。

    Attributes:
        name: 分支名（工作流内唯一）。必须以字母或下划线开头。
        condition: 分支条件表达式（DSL）。
        nodes: 分支内的节点列表（可选）。
        outputs: 分支输出（覆盖 workflow outputs，可选）。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    nodes: list[NodeDef] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("分支 name 不能为空")
        if not _BRANCH_NAME_PATTERN.match(v):
            raise ValueError(
                f"分支 name {v!r} 格式无效（必须以字母或下划线开头，"
                f"仅含字母数字下划线）"
            )
        return v

    @field_validator("condition")
    @classmethod
    def _validate_condition(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("分支 condition 不能为空")
        if not v.strip():
            raise ValueError("分支 condition 不能为空字符串（仅空白）")
        return v


# ===========================================================================
# 工作流 schema 顶层
# ===========================================================================

class WorkflowSchema(BaseModel):
    """工作流 yaml schema 顶层模型。

    Attributes:
        name: 工作流名（唯一标识）。非空字符串。
        description: 描述（可选，默认空字符串）。
        nodes: 节点列表（必填，至少 1 个）。节点 id 在工作流内唯一。
        outputs: 工作流输出（key=输出名，value=引用字符串或字面值）。
            默认空 dict（允许无输出工作流）。
        branches: if/else 分支列表（可选，默认空列表）。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = ""
    nodes: list[NodeDef] = Field(..., min_length=1)
    outputs: dict[str, Any] = Field(default_factory=dict)
    branches: list[BranchDef] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("工作流 name 不能为空")
        return v

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError(
                f"description 必须是 str，实际 {type(v).__name__}"
            )
        return v

    @model_validator(mode="after")
    def _validate_unique_node_ids(self) -> "WorkflowSchema":
        """校验节点 id 唯一性。"""
        seen: set[str] = set()
        for i, node in enumerate(self.nodes):
            if node.id in seen:
                raise ValueError(
                    f"节点 id 重复：{node.id!r}（第 {i + 1} 个节点）"
                )
            seen.add(node.id)
        return self

    @model_validator(mode="after")
    def _validate_unique_branch_names(self) -> "WorkflowSchema":
        """校验分支名唯一性。"""
        seen: set[str] = set()
        for i, branch in enumerate(self.branches):
            if branch.name in seen:
                raise ValueError(
                    f"分支 name 重复：{branch.name!r}（第 {i + 1} 个分支）"
                )
            seen.add(branch.name)
        return self

    def to_dict(self) -> dict[str, Any]:
        """转换为普通 dict（供 WorkflowExecutor 使用）。

        Returns:
            工作流 dict（不含 'workflow' 包装层）。
        """
        return self.model_dump(exclude_defaults=False)

    def to_yaml_dict(self) -> dict[str, Any]:
        """转换为 yaml 兼容 dict（包含 workflow 顶层 key）。

        Returns:
            ``{"workflow": {...}}`` 格式的 dict。
        """
        return {"workflow": self.to_dict()}

    def to_yaml(self) -> str:
        """序列化为 yaml 字符串。

        Returns:
            yaml 字符串（不含 'workflow' 包装层）。
        """
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)


# ===========================================================================
# 校验入口
# ===========================================================================

def _unwrap_workflow_dict(data: Any) -> dict[str, Any]:
    """剥离 {'workflow': {...}} 包装层。

    兼容两种输入：
        - 直接工作流 dict：``{"name": ..., "nodes": [...], ...}``
        - 包装格式：``{"workflow": {...}}``

    Raises:
        WorkflowSchemaError: 输入不是 dict，或包装层格式无效。
    """
    if data is None:
        raise WorkflowSchemaError("工作流数据为 None")
    if not isinstance(data, dict):
        raise WorkflowSchemaError(
            f"工作流必须是 dict，实际 {type(data).__name__}",
        )
    if "workflow" in data:
        # 必须是 {"workflow": {...}} 单字段包装
        if len(data) > 1:
            extra_keys = [k for k in data.keys() if k != "workflow"]
            raise WorkflowSchemaError(
                f"工作流 dict 同时含 'workflow' 包装层和其他字段：{extra_keys}",
                field_path="",
            )
        inner = data["workflow"]
        if not isinstance(inner, dict):
            raise WorkflowSchemaError(
                f"workflow 字段必须是 dict，实际 {type(inner).__name__}",
                field_path="workflow",
            )
        return inner
    return data


def _format_validation_error(e: ValidationError) -> WorkflowSchemaError:
    """将 Pydantic ValidationError 转换为 WorkflowSchemaError（含清晰路径）。"""
    errors = e.errors()
    if not errors:
        return WorkflowSchemaError(str(e))

    # 聚合所有错误（Pydantic 默认只显示第一个，但我们要全列出）
    messages: list[str] = []
    first_path = ""
    for err in errors:
        loc = err.get("loc", ())
        path = ".".join(str(x) for x in loc if x != "")
        msg = err.get("msg", "")
        typ = err.get("type", "")
        if not first_path:
            first_path = path
        # 简化 Pydantic 错误类型显示
        suffix = f"（类型: {typ}）" if typ and typ != "value_error" else ""
        prefix = f"[{path}] " if path else ""
        messages.append(f"{prefix}{msg}{suffix}")

    message = "\n".join(messages) if len(messages) == 1 else (
        f"工作流校验失败（{len(messages)} 处错误）：\n"
        + "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(messages))
    )
    return WorkflowSchemaError(message, field_path=first_path)


def validate_workflow_dict(data: dict[str, Any]) -> WorkflowSchema:
    """校验工作流 dict。

    Args:
        data: 工作流 dict。支持两种格式：
            - 直接格式：``{"name": ..., "nodes": [...], ...}``
            - 包装格式：``{"workflow": {...}}``

    Returns:
        WorkflowSchema 实例。

    Raises:
        WorkflowSchemaError: 校验失败。

    Example:
        >>> schema = validate_workflow_dict({
        ...     "name": "test",
        ...     "nodes": [{"id": "n1", "type": "Echo"}],
        ...     "outputs": {"r": "n1.output.value"},
        ... })
        >>> schema.name
        'test'
    """
    inner = _unwrap_workflow_dict(data)
    try:
        return WorkflowSchema.model_validate(inner)
    except ValidationError as e:
        raise _format_validation_error(e) from e


def validate_workflow_yaml(yaml_text: str) -> WorkflowSchema:
    """校验工作流 yaml 字符串。

    Args:
        yaml_text: yaml 字符串。支持两种格式：
            - 直接 yaml：``name: test\\nnodes: [...]``
            - 包装 yaml：``workflow:\\n  name: test``

    Returns:
        WorkflowSchema 实例。

    Raises:
        WorkflowSchemaError: yaml 解析失败或 schema 校验失败。

    Example:
        >>> schema = validate_workflow_yaml('''
        ... name: test
        ... nodes:
        ...   - id: n1
        ...     type: Echo
        ... outputs:
        ...   r: n1.output.value
        ... ''')
    """
    if not isinstance(yaml_text, str):
        raise WorkflowSchemaError(
            f"yaml_text 必须是 str，实际 {type(yaml_text).__name__}"
        )
    if not yaml_text.strip():
        raise WorkflowSchemaError("yaml_text 为空字符串")

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise WorkflowSchemaError(f"yaml 解析失败：{e}") from e

    return validate_workflow_dict(data)

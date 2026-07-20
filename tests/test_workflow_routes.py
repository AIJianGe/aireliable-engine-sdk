"""工作流 API 单元测试（v0.2 阶段 12 任务 12.3）。

覆盖：
    - WorkflowRouter 的 6 个端点（CRUD + reload + validate）
    - NodeRouter 的 5 个端点（list / get / definitions / packages）
    - 路径穿越防护
    - readonly 只读模式
    - 原子写入
    - engine 缓存清理
    - 工具函数（_safe_workflow_name / _infer_category / _node_to_definition）

参考：
    - v0.2-tasks.md 任务 12.3
    - sdk/.../workflow_routes.py
    - sdk/.../node_routes.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from aireliable_engine.node import BaseNode, PortSpec
from aireliable_engine.node_registry import NodeRegistry
from aireliable_engine.node_routes import (
    NodeRouter,
    _infer_category,
    _node_to_definition,
    create_node_router,
)
from aireliable_engine.workflow_routes import (
    WorkflowRouter,
    _safe_workflow_name,
    create_workflow_router,
)


# ===========================================================================
# 测试用工具
# ===========================================================================

VALID_WORKFLOW_YAML = """\
name: test_workflow
description: 测试工作流
nodes:
  - id: input1
    type: TextInput
    params:
      field: prompt
    inputs: {}
outputs:
  text: ${input1.output.text}
"""

INVALID_YAML_SYNTAX = """\
name: bad
nodes:
  - id: x
   type: bad indent
"""

INVALID_SCHEMA_YAML = """\
description: 缺少 name 和 nodes 字段
"""

UNKNOWN_NODE_YAML = """\
name: unknown_node_workflow
nodes:
  - id: n1
    type: NonExistentNodeType
    inputs: {}
outputs: {}
"""


def _make_registry_with_test_nodes() -> NodeRegistry:
    """创建一个含测试节点的注册表。"""
    registry = NodeRegistry()

    class _TextInput(BaseNode):
        node_type = "TextInput"
        title = "Text Input"
        input_specs: list[PortSpec] = []
        output_specs = [PortSpec(name="text", type="text")]

        async def execute(self, inputs: dict, ctx: Any) -> dict:
            return {"text": "hello"}

    class _ImagesInput(BaseNode):
        node_type = "ImagesInput"
        title = "Images Input"
        input_specs: list[PortSpec] = []
        output_specs = [PortSpec(name="images", type="images", variadic=True)]

        async def execute(self, inputs: dict, ctx: Any) -> dict:
            return {"images": []}

    class _ModelLoader(BaseNode):
        node_type = "ModelLoader"
        title = "Model Loader"
        input_specs: list[PortSpec] = []
        output_specs = [PortSpec(name="model", type="model")]

        async def execute(self, inputs: dict, ctx: Any) -> dict:
            return {"model": None}

    registry.register("TextInput", _TextInput)
    registry.register("ImagesInput", _ImagesInput)
    registry.register("ModelLoader", _ModelLoader)
    return registry


def _make_app_with_workflow_router(
    workflows_dir: Optional[Path] = None,
    node_registry: Optional[NodeRegistry] = None,
    readonly: bool = False,
    engine: Any = None,
    prefix: str = "",
) -> tuple[FastAPI, WorkflowRouter]:
    """构造一个含 WorkflowRouter 的 FastAPI app。"""
    app = FastAPI()
    router = WorkflowRouter(
        engine=engine,
        workflows_dir=workflows_dir,
        node_registry=node_registry,
        prefix=prefix,
        readonly=readonly,
    )
    app.include_router(router.router)
    return app, router


def _make_app_with_node_router(
    registry: Optional[NodeRegistry] = None,
    node_packages_dir: Optional[Path] = None,
    prefix: str = "",
) -> tuple[FastAPI, NodeRouter]:
    """构造一个含 NodeRouter 的 FastAPI app。"""
    app = FastAPI()
    router = NodeRouter(
        registry=registry,
        node_packages_dir=node_packages_dir,
        prefix=prefix,
    )
    app.include_router(router.router)
    return app, router


# ===========================================================================
# _safe_workflow_name 工具函数测试
# ===========================================================================

class TestSafeWorkflowName:
    """工作流名校验。"""

    def test_valid_name(self) -> None:
        assert _safe_workflow_name("default") == "default"
        assert _safe_workflow_name("workflow_v1") == "workflow_v1"
        assert _safe_workflow_name("ernie-image") == "ernie-image"

    def test_empty_name(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _safe_workflow_name("")
        assert exc_info.value.status_code == 400
        assert "不能为空" in exc_info.value.detail["error"]["message"]

    def test_none_name(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            _safe_workflow_name(None)  # type: ignore[arg-type]

    def test_path_traversal_slash(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _safe_workflow_name("etc/passwd")
        assert exc_info.value.status_code == 400
        assert "非法字符" in exc_info.value.detail["error"]["message"]

    def test_path_traversal_backslash(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            _safe_workflow_name("..\\secret")

    def test_path_traversal_double_dot(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            _safe_workflow_name("../etc/passwd")

    def test_leading_dot(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            _safe_workflow_name(".hidden")

    def test_special_chars(self) -> None:
        from fastapi import HTTPException
        for name in ['a"b', "a:b", "a*b", "a?b", "a|b", "a<b", "a>b"]:
            with pytest.raises(HTTPException):
                _safe_workflow_name(name)

    def test_null_byte(self) -> None:
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            _safe_workflow_name("evil\x00name")


# ===========================================================================
# WorkflowRouter - list_workflows
# ===========================================================================

class TestListWorkflows:
    """GET /workflows 端点。"""

    def test_empty_dir(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflows"] == []

    def test_with_files(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")
        (workflows_dir / "variant2.yaml").write_text("name: variant2\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n", encoding="utf-8")

        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows")
        assert resp.status_code == 200
        data = resp.json()
        names = [w["name"] for w in data["workflows"]]
        assert names == ["default", "variant2"]
        # 每条记录都有 size / mtime
        for w in data["workflows"]:
            assert "size" in w
            assert "mtime" in w
            assert w["size"] > 0
            assert w["mtime"] > 0

    def test_no_dir_configured(self) -> None:
        app, _ = _make_app_with_workflow_router(workflows_dir=None)
        client = TestClient(app)

        resp = client.get("/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflows"] == []

    def test_dir_not_exist(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent"
        app, _ = _make_app_with_workflow_router(workflows_dir=nonexistent)
        client = TestClient(app)

        resp = client.get("/workflows")
        assert resp.status_code == 200
        assert resp.json()["workflows"] == []

    def test_with_engine_current(self, tmp_path: Path) -> None:
        """有 engine 时附加 current 字段。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        # 用 mock engine（通过 __init__ 接收 workflows_dir，避免闭包作用域问题）
        class _MockEngine:
            def __init__(self, wf_dir: Path) -> None:
                self.current_workflow = "default"
                self.workflows_dir = wf_dir
                self.node_registry = None

            def _read_workflow_file(self, name: str) -> str:
                return (self.workflows_dir / f"{name}.yaml").read_text(encoding="utf-8")

            def reload_workflow(self, name: Optional[str] = None) -> None:
                pass

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            engine=_MockEngine(workflows_dir),
        )
        client = TestClient(app)

        resp = client.get("/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current"] == "default"

    def test_with_prefix(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir, prefix="/engines/ernie_image"
        )
        client = TestClient(app)

        resp = client.get("/engines/ernie_image/workflows")
        assert resp.status_code == 200


# ===========================================================================
# WorkflowRouter - get_workflow
# ===========================================================================

class TestGetWorkflow:
    """GET /workflows/{name} 端点。"""

    def test_get_success(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows/default")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/yaml")
        assert "test_workflow" in resp.text
        # Cache-Control: no-cache
        assert resp.headers["cache-control"] == "no-cache"

    def test_not_found(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows/nonexistent")
        assert resp.status_code == 404
        data = resp.json()
        assert data["detail"]["error"]["type"] == "workflow_not_found"

    def test_path_traversal_blocked_by_name_check(self) -> None:
        """路径穿越名称校验（_safe_workflow_name 层级）。

        HTTP 层 Starlette 会规范化 URL 路径（%2F → /），无法直接通过 URL 触发，
        但 _safe_workflow_name 函数会拒绝 `..` / `/` / `\\` 等名称。
        该函数级测试见 TestSafeWorkflowName。
        """
        # 此处仅验证函数调用：直接调用 _safe_workflow_name 验证抛 HTTPException
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _safe_workflow_name("../secret")
        assert exc_info.value.status_code == 400

    def test_no_dir_configured(self) -> None:
        app, _ = _make_app_with_workflow_router(workflows_dir=None)
        client = TestClient(app)

        resp = client.get("/workflows/default")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"]["type"] == "no_workflows_dir"

    def test_via_engine_caching(self, tmp_path: Path) -> None:
        """engine 路径优先于 workflows_dir，使用 engine 的缓存机制。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        class _MockEngine:
            def __init__(self, wf_dir: Path) -> None:
                self.current_workflow = "default"
                self.workflows_dir = wf_dir
                self.node_registry = None
                self._read_count = 0

            def _read_workflow_file(self, name: str) -> str:
                self._read_count += 1
                return (self.workflows_dir / f"{name}.yaml").read_text(encoding="utf-8")

            def reload_workflow(self, name: Optional[str] = None) -> None:
                pass

        engine = _MockEngine(workflows_dir)
        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            engine=engine,
        )
        client = TestClient(app)

        # 第一次 GET
        resp1 = client.get("/workflows/default")
        assert resp1.status_code == 200

        # 第二次 GET（应命中 engine 缓存）
        resp2 = client.get("/workflows/default")
        assert resp2.status_code == 200
        # 缓存机制由 WorkflowEngine._read_workflow_file 实现
        # _MockEngine 没有实现 mtime 缓存，所以会读两次（这里只验证响应一致）


# ===========================================================================
# WorkflowRouter - save_workflow
# ===========================================================================

class TestSaveWorkflow:
    """PUT /workflows/{name} 端点。"""

    def test_create_new(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.put(
            "/workflows/new_workflow",
            content=VALID_WORKFLOW_YAML,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "new_workflow"
        assert data["size"] > 0
        assert data["valid"] is True
        assert data["errors"] == []

        # 文件确实被写入
        assert (workflows_dir / "new_workflow.yaml").exists()
        assert "test_workflow" in (workflows_dir / "new_workflow.yaml").read_text(encoding="utf-8")

    def test_update_existing(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text("name: old\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n", encoding="utf-8")

        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        new_yaml = VALID_WORKFLOW_YAML
        resp = client.put(
            "/workflows/default",
            content=new_yaml,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 200
        # 文件被覆盖
        content = (workflows_dir / "default.yaml").read_text(encoding="utf-8")
        assert "test_workflow" in content

    def test_invalid_yaml_still_saved(self, tmp_path: Path) -> None:
        """校验失败但仍保存（便于用户修复）。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.put(
            "/workflows/bad_yaml",
            content=INVALID_YAML_SYNTAX,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0
        # 仍保存
        assert (workflows_dir / "bad_yaml.yaml").exists()

    def test_invalid_schema_returned_as_errors(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.put(
            "/workflows/invalid_schema",
            content=INVALID_SCHEMA_YAML,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_unknown_node_type_returned_as_errors(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        client = TestClient(app)

        resp = client.put(
            "/workflows/unknown_node",
            content=UNKNOWN_NODE_YAML,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        # errors 包含 "未在节点注册表中注册"
        assert any("NonExistentNodeType" in e for e in data["errors"])

    def test_clears_engine_cache(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        cleared: list[Optional[str]] = []

        class _MockEngine:
            def __init__(self, wf_dir: Path) -> None:
                self.current_workflow = "default"
                self.workflows_dir = wf_dir
                self.node_registry = None

            def _read_workflow_file(self, name: str) -> str:
                return (self.workflows_dir / f"{name}.yaml").read_text(encoding="utf-8")

            def reload_workflow(self, name: Optional[str] = None) -> None:
                cleared.append(name)

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            engine=_MockEngine(workflows_dir),
        )
        client = TestClient(app)

        resp = client.put(
            "/workflows/default",
            content=VALID_WORKFLOW_YAML,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 200
        # engine.reload_workflow 被调用，参数为工作流名
        assert cleared == ["default"]

    def test_atomic_write(self, tmp_path: Path) -> None:
        """保存后不应残留 .tmp 文件。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        client.put(
            "/workflows/atomic_test",
            content=VALID_WORKFLOW_YAML,
            headers={"content-type": "application/yaml"},
        )
        # .yaml 存在
        assert (workflows_dir / "atomic_test.yaml").exists()
        # .yaml.tmp 不应残留
        assert not (workflows_dir / "atomic_test.yaml.tmp").exists()

    def test_path_traversal_blocked_by_name_check(self) -> None:
        """路径穿越名称校验（PUT 端点）。

        Starlette 会规范化 URL 路径，无法通过 URL 触发；
        但 _safe_workflow_name 会拒绝 `..` 等名称。详见 TestSafeWorkflowName。
        """
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _safe_workflow_name("../evil")
        assert exc_info.value.status_code == 400

    def test_no_dir_configured(self) -> None:
        app, _ = _make_app_with_workflow_router(workflows_dir=None)
        client = TestClient(app)

        resp = client.put(
            "/workflows/default",
            content=VALID_WORKFLOW_YAML,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"]["type"] == "no_workflows_dir"


# ===========================================================================
# WorkflowRouter - delete_workflow
# ===========================================================================

class TestDeleteWorkflow:
    """DELETE /workflows/{name} 端点。"""

    def test_delete_success(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "to_delete.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.delete("/workflows/to_delete")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "to_delete"
        assert data["deleted"] is True
        assert not (workflows_dir / "to_delete.yaml").exists()

    def test_delete_not_found(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.delete("/workflows/nonexistent")
        assert resp.status_code == 404

    def test_delete_clears_engine_cache(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        cleared: list[Optional[str]] = []

        class _MockEngine:
            def __init__(self, wf_dir: Path) -> None:
                self.current_workflow = "default"
                self.workflows_dir = wf_dir
                self.node_registry = None

            def _read_workflow_file(self, name: str) -> str:
                return (self.workflows_dir / f"{name}.yaml").read_text(encoding="utf-8")

            def reload_workflow(self, name: Optional[str] = None) -> None:
                cleared.append(name)

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            engine=_MockEngine(workflows_dir),
        )
        client = TestClient(app)

        resp = client.delete("/workflows/default")
        assert resp.status_code == 200
        assert cleared == ["default"]

    def test_path_traversal_blocked_by_name_check(self) -> None:
        """路径穿越名称校验（DELETE 端点）。"""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _safe_workflow_name("../secret")
        assert exc_info.value.status_code == 400


# ===========================================================================
# WorkflowRouter - reload_workflow
# ===========================================================================

class TestReloadWorkflow:
    """POST /workflows/{name}/reload 端点。"""

    def test_with_engine(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        reloaded: list[Optional[str]] = []

        class _MockEngine:
            def __init__(self, wf_dir: Path) -> None:
                self.current_workflow = "default"
                self.workflows_dir = wf_dir
                self.node_registry = None

            def _read_workflow_file(self, name: str) -> str:
                return (self.workflows_dir / f"{name}.yaml").read_text(encoding="utf-8")

            def reload_workflow(self, name: Optional[str] = None) -> None:
                reloaded.append(name)

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            engine=_MockEngine(workflows_dir),
        )
        client = TestClient(app)

        resp = client.post("/workflows/default/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "default"
        assert data["reloaded"] is True
        assert reloaded == ["default"]

    def test_without_engine(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.post("/workflows/default/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reloaded"] is True
        assert "note" in data  # 无 engine 缓存说明


# ===========================================================================
# WorkflowRouter - validate_workflow
# ===========================================================================

class TestValidateWorkflow:
    """GET /workflows/{name}/validate 端点。"""

    def test_valid_workflow(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        client = TestClient(app)

        resp = client.get("/workflows/default/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "default"
        assert data["valid"] is True
        assert data["errors"] == []

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "bad.yaml").write_text(INVALID_YAML_SYNTAX, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows/bad/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_invalid_schema(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "bad_schema.yaml").write_text(INVALID_SCHEMA_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows/bad_schema/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_unknown_node_type(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "unknown.yaml").write_text(UNKNOWN_NODE_YAML, encoding="utf-8")

        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        client = TestClient(app)

        resp = client.get("/workflows/unknown/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert any("NonExistentNodeType" in e for e in data["errors"])

    def test_not_found(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows/nonexistent/validate")
        assert resp.status_code == 404

    def test_no_registry_skips_node_check(self, tmp_path: Path) -> None:
        """无 node_registry 时跳过节点类型校验。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "unknown.yaml").write_text(UNKNOWN_NODE_YAML, encoding="utf-8")

        # 不传 node_registry
        app, _ = _make_app_with_workflow_router(workflows_dir=workflows_dir)
        client = TestClient(app)

        resp = client.get("/workflows/unknown/validate")
        assert resp.status_code == 200
        data = resp.json()
        # 没有 registry 校验，schema 通过即视为 valid
        assert data["valid"] is True
        assert data["errors"] == []


# ===========================================================================
# WorkflowRouter - readonly 模式
# ===========================================================================

class TestReadonlyMode:
    """readonly 只读模式。"""

    def test_blocks_save(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            readonly=True,
        )
        client = TestClient(app)

        resp = client.put(
            "/workflows/default",
            content=VALID_WORKFLOW_YAML,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["error"]["type"] == "readonly"
        # 文件不应被写入
        assert not (workflows_dir / "default.yaml").exists()

    def test_blocks_delete(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            readonly=True,
        )
        client = TestClient(app)

        resp = client.delete("/workflows/default")
        assert resp.status_code == 403
        # 文件仍存在
        assert (workflows_dir / "default.yaml").exists()

    def test_allows_get(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            readonly=True,
        )
        client = TestClient(app)

        resp = client.get("/workflows/default")
        assert resp.status_code == 200

    def test_allows_list(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            readonly=True,
        )
        client = TestClient(app)

        resp = client.get("/workflows")
        assert resp.status_code == 200

    def test_allows_validate(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        app, _ = _make_app_with_workflow_router(
            workflows_dir=workflows_dir,
            readonly=True,
        )
        client = TestClient(app)

        resp = client.get("/workflows/default/validate")
        assert resp.status_code == 200


# ===========================================================================
# create_workflow_router 工厂函数
# ===========================================================================

class TestCreateWorkflowRouter:
    """create_workflow_router 工厂函数。"""

    def test_returns_workflow_router_instance(self, tmp_path: Path) -> None:
        router = create_workflow_router(workflows_dir=tmp_path)
        assert isinstance(router, WorkflowRouter)

    def test_with_prefix(self) -> None:
        router = create_workflow_router(prefix="/engines/ernie_image")
        assert router.prefix == "/engines/ernie_image"

    def test_with_readonly(self) -> None:
        router = create_workflow_router(readonly=True)
        assert router.readonly is True

    def test_strips_trailing_slash(self) -> None:
        router = create_workflow_router(prefix="/engines/ernie_image/")
        assert router.prefix == "/engines/ernie_image"


# ===========================================================================
# NodeRouter - list_nodes
# ===========================================================================

class TestListNodes:
    """GET /nodes 端点。"""

    def test_empty_registry(self) -> None:
        registry = NodeRegistry()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["count"] == 0

    def test_with_registered_nodes(self) -> None:
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        types = [n["type"] for n in data["nodes"]]
        assert "TextInput" in types
        assert "ImagesInput" in types
        assert "ModelLoader" in types

        # 每条记录都有 title / category / module / class
        for n in data["nodes"]:
            assert "title" in n
            assert "category" in n
            assert "module" in n
            assert "class" in n

    def test_no_registry_configured(self) -> None:
        app, _ = _make_app_with_node_router(registry=None)
        client = TestClient(app)

        resp = client.get("/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert "note" in data


# ===========================================================================
# NodeRouter - get_node
# ===========================================================================

class TestGetNode:
    """GET /nodes/{node_type} 端点。"""

    def test_found(self) -> None:
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes/TextInput")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "TextInput"
        assert data["title"] == "Text Input"
        assert data["category"] == "custom"  # 测试节点模块不在标准分类路径下
        assert "inputs" in data
        assert "outputs" in data
        assert "params" in data

    def test_not_found(self) -> None:
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes/NonExistent")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"]["type"] == "node_not_found"

    def test_no_registry(self) -> None:
        app, _ = _make_app_with_node_router(registry=None)
        client = TestClient(app)

        resp = client.get("/nodes/TextInput")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"]["type"] == "no_registry"


# ===========================================================================
# NodeRouter - get_all_definitions
# ===========================================================================

class TestGetAllDefinitions:
    """GET /nodes/definitions 端点。"""

    def test_returns_definitions(self) -> None:
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes/definitions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0"
        assert data["count"] == 3
        assert "TextInput" in data["nodes"]
        assert "ImagesInput" in data["nodes"]
        assert "ModelLoader" in data["nodes"]

    def test_definitions_contain_required_fields(self) -> None:
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes/definitions")
        data = resp.json()
        for node_def in data["nodes"].values():
            assert "type" in node_def
            assert "title" in node_def
            assert "category" in node_def
            assert "color" in node_def
            assert "icon" in node_def
            assert "description" in node_def
            assert "inputs" in node_def
            assert "outputs" in node_def
            assert "params" in node_def

    def test_categories_collected(self) -> None:
        registry = _make_registry_with_test_nodes()
        app, _ = _make_app_with_node_router(registry=registry)
        client = TestClient(app)

        resp = client.get("/nodes/definitions")
        data = resp.json()
        # 测试节点都分类为 custom
        assert "custom" in data["categories"]
        assert "title" in data["categories"]["custom"]
        assert "color" in data["categories"]["custom"]

    def test_no_registry(self) -> None:
        app, _ = _make_app_with_node_router(registry=None)
        client = TestClient(app)

        resp = client.get("/nodes/definitions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == {}
        assert data["categories"] == {}


# ===========================================================================
# NodeRouter - list_packages
# ===========================================================================

class TestListPackages:
    """GET /nodes/packages 端点。"""

    def test_returns_list(self, monkeypatch: MonkeyPatch) -> None:
        """mock entry_points 返回空列表。"""
        # mock importlib.metadata.entry_points 返回空
        from importlib.metadata import EntryPoint
        from importlib.metadata import distributions as _distributions

        def _mock_entry_points(group=None):
            return []

        def _mock_distributions():
            return []

        # NodeRouter.list_packages 内部 from importlib.metadata import entry_points
        # 所以 monkeypatch importlib.metadata.entry_points
        import importlib.metadata as ilm
        monkeypatch.setattr(ilm, "entry_points", _mock_entry_points)
        monkeypatch.setattr("aireliable_engine.node_routes.distributions", _mock_distributions)

        app, _ = _make_app_with_node_router()
        client = TestClient(app)

        resp = client.get("/nodes/packages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["packages"] == []
        assert data["count"] == 0

    def test_with_mock_entries(self, monkeypatch: MonkeyPatch) -> None:
        """模拟已安装一个节点包。"""
        from importlib.metadata import EntryPoint

        # 构造 mock entry point
        ep = EntryPoint(
            name="MyNode",
            value="my_pkg.my_module:MyNode",
            group="aireliable.nodes",
        )

        def _mock_entry_points(group=None):
            if group == "aireliable.nodes":
                return [ep]
            return []

        def _mock_distributions():
            """模拟一个 distribution。"""

            class _MockDist:
                class _MockMetadata:
                    def __getitem__(self, key):
                        if key == "Name":
                            return "my-pkg"
                        if key == "Version":
                            return "1.0.0"
                        return None

                metadata = _MockMetadata()
                version = "1.0.0"

                def read_text(self, filename):
                    if filename == "top_level.txt":
                        return "my_pkg\n"
                    return None

            return [_MockDist()]

        import importlib.metadata as ilm
        monkeypatch.setattr(ilm, "entry_points", _mock_entry_points)
        monkeypatch.setattr("aireliable_engine.node_routes.distributions", _mock_distributions)

        app, _ = _make_app_with_node_router()
        client = TestClient(app)

        resp = client.get("/nodes/packages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        pkg = data["packages"][0]
        assert pkg["name"] == "my-pkg"
        assert pkg["version"] == "1.0.0"
        assert "MyNode" in pkg["nodes"]


# ===========================================================================
# NodeRouter - get_package_definitions
# ===========================================================================

class TestGetPackageDefinitions:
    """GET /nodes/packages/{package_name}/definitions 端点。"""

    def test_from_node_packages_dir(self, tmp_path: Path) -> None:
        """从 node_packages_dir 读取前端定义。"""
        node_packages_dir = tmp_path / "node_packages"
        pkg_dir = node_packages_dir / "my_pkg"
        frontend_dir = pkg_dir / "frontend"
        frontend_dir.mkdir(parents=True)
        definitions = {
            "version": "1.0",
            "nodes": {
                "MyNode": {"type": "MyNode", "title": "My Node"},
            },
        }
        (frontend_dir / "node_definitions.json").write_text(
            json.dumps(definitions), encoding="utf-8"
        )

        app, _ = _make_app_with_node_router(node_packages_dir=node_packages_dir)
        client = TestClient(app)

        resp = client.get("/nodes/packages/my_pkg/definitions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0"
        assert "MyNode" in data["nodes"]

    def test_not_found(self, tmp_path: Path) -> None:
        node_packages_dir = tmp_path / "node_packages"
        node_packages_dir.mkdir()
        app, _ = _make_app_with_node_router(node_packages_dir=node_packages_dir)
        client = TestClient(app)

        resp = client.get("/nodes/packages/nonexistent/definitions")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"]["type"] == "package_not_found"

    def test_invalid_json(self, tmp_path: Path) -> None:
        node_packages_dir = tmp_path / "node_packages"
        pkg_dir = node_packages_dir / "bad_pkg" / "frontend"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "node_definitions.json").write_text(
            "not a valid json {{{",
            encoding="utf-8",
        )

        app, _ = _make_app_with_node_router(node_packages_dir=node_packages_dir)
        client = TestClient(app)

        resp = client.get("/nodes/packages/bad_pkg/definitions")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"]["type"] == "json_decode_error"


# ===========================================================================
# create_node_router 工厂函数
# ===========================================================================

class TestCreateNodeRouter:
    """create_node_router 工厂函数。"""

    def test_returns_node_router_instance(self) -> None:
        router = create_node_router()
        assert isinstance(router, NodeRouter)

    def test_with_prefix(self) -> None:
        router = create_node_router(prefix="/engines/ernie_image")
        assert router.prefix == "/engines/ernie_image"

    def test_strips_trailing_slash(self) -> None:
        router = create_node_router(prefix="/engines/ernie_image/")
        assert router.prefix == "/engines/ernie_image"


# ===========================================================================
# 工具函数测试
# ===========================================================================

class TestInferCategory:
    """_infer_category 函数。"""

    def test_input_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.input.text_input"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "input"

    def test_loader_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.loader.model_loader"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "loader"

    def test_encode_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.encode.clip_encode"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "encode"

    def test_sampler_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.sampler.sampler"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "sampler"

    def test_decode_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.decode.vae_decode"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "decode"

    def test_lora_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.lora.lora_loader"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "lora"

    def test_output_category(self) -> None:
        class _N(BaseNode):
            __module__ = "aireliable_engine.nodes.output.image_save"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "output"

    def test_custom_category_for_unknown_module(self) -> None:
        class _N(BaseNode):
            __module__ = "some_other_package.my_node"
            node_type = "Test"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        assert _infer_category(_N) == "custom"


class TestNodeToDefinition:
    """_node_to_definition 函数。"""

    def test_converts_node_class(self) -> None:
        class _MyNode(BaseNode):
            node_type = "MyTestNode"
            title = "My Test Node"
            input_specs = [PortSpec(name="text", type="text", description="input text")]
            output_specs = [PortSpec(name="result", type="text", description="output result")]

            async def execute(self, inputs, ctx):
                return {"result": "ok"}

        definition = _node_to_definition(_MyNode)
        assert definition["type"] == "MyTestNode"
        assert definition["title"] == "My Test Node"
        assert definition["category"] == "custom"
        assert definition["color"] == "#7F8C8D"  # custom 颜色
        assert definition["icon"] == "•"
        assert len(definition["inputs"]) == 1
        assert definition["inputs"][0]["name"] == "text"
        assert definition["inputs"][0]["type"] == "text"
        assert definition["inputs"][0]["required"] is True
        assert definition["inputs"][0]["variadic"] is False
        assert len(definition["outputs"]) == 1
        assert definition["outputs"][0]["name"] == "result"

    def test_uses_default_title_when_not_set(self) -> None:
        class _NoTitleNode(BaseNode):
            node_type = "NoTitle"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        definition = _node_to_definition(_NoTitleNode)
        # 没有 title 类属性时，回退到 node_type
        assert definition["title"] == "NoTitle"

    def test_uses_params_schema_when_present(self) -> None:
        class _WithSchemaNode(BaseNode):
            node_type = "WithSchema"
            input_specs = []
            output_specs = []
            params_schema = {
                "field": {"type": "str", "default": "prompt"},
            }

            async def execute(self, inputs, ctx):
                return {}

        definition = _node_to_definition(_WithSchemaNode)
        assert definition["params"] == {
            "field": {"type": "str", "default": "prompt"},
        }

    def test_empty_params_when_no_schema(self) -> None:
        class _NoSchemaNode(BaseNode):
            node_type = "NoSchema"
            input_specs = []
            output_specs = []

            async def execute(self, inputs, ctx):
                return {}

        definition = _node_to_definition(_NoSchemaNode)
        assert definition["params"] == {}


# ===========================================================================
# 集成：WorkflowRouter + NodeRouter 挂载到同一 app
# ===========================================================================

class TestIntegrationBothRouters:
    """同时挂载 WorkflowRouter + NodeRouter 到同一 app。"""

    def test_both_routers_work_together(self, tmp_path: Path) -> None:
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(VALID_WORKFLOW_YAML, encoding="utf-8")

        registry = _make_registry_with_test_nodes()

        app = FastAPI()
        wf_router = WorkflowRouter(
            workflows_dir=workflows_dir,
            node_registry=registry,
            prefix="/engines/ernie_image",
        )
        node_router = NodeRouter(
            registry=registry,
            prefix="/engines/ernie_image",
        )
        app.include_router(wf_router.router)
        app.include_router(node_router.router)
        client = TestClient(app)

        # 列工作流
        resp = client.get("/engines/ernie_image/workflows")
        assert resp.status_code == 200
        assert resp.json()["workflows"][0]["name"] == "default"

        # 列节点
        resp = client.get("/engines/ernie_image/nodes")
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

        # 获取节点定义
        resp = client.get("/engines/ernie_image/nodes/definitions")
        assert resp.status_code == 200
        assert "TextInput" in resp.json()["nodes"]

        # 校验工作流
        resp = client.get("/engines/ernie_image/workflows/default/validate")
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

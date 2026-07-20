"""远程引擎路由 + 鉴权 + 只读模式集成测试（v0.2 阶段 13 任务 13.1/13.2）。

覆盖：
    - RemoteEngineConfig（SDK 侧）：配置数据类（auth_token / workflow_sync / readonly_workflows / allowed_paths）
    - create_engine_app：构造完整远程引擎 app
    - mount_engine_routes：挂载 8 端点 + workflow + node
    - AuthMiddleware：Bearer token 鉴权
    - readonly_workflows / workflow_sync 只读模式

注意：
    - SDK 侧的 RemoteEngineConfig **不含** url 字段（url 是客户端概念）
    - core 侧的 RemoteEngineClient 测试在 core/tests/ 目录中（独立模块）
    - 此文件专注 SDK 侧的远程路由 + auth middleware + readonly 模式
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aireliable_engine import (
    AuthMiddleware,
    BaseEngine,
    CapabilitiesResponse,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraLoadResponse,
    LoraUnloadRequest,
    LoraUnloadResponse,
    RemoteEngineConfig,
    StatusResponse,
    UnloadResponse,
    create_engine_app,
    make_final_frame,
    make_frame,
    mount_engine_routes,
)
from aireliable_engine.node import BaseNode, PortSpec
from aireliable_engine.node_registry import NodeRegistry


# ===========================================================================
# Mock 引擎
# ===========================================================================

def _make_mock_engine() -> BaseEngine:
    """构造一个最小可用的 BaseEngine 子类实例。"""

    class _MockEngine(BaseEngine):
        def __init__(self):
            super().__init__(
                engine_id="mock_remote",
                engine_name="Mock Remote Engine",
                capability="text-to-image",
                engine_type="image",
            )
            self.workflows_dir = None
            self.node_registry = None

        async def _capabilities_impl(self) -> CapabilitiesResponse:
            return CapabilitiesResponse(
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                capability=self.capability,
                type=self.engine_type,
            )

        async def _load_impl(self, variant: str) -> LoadResponse:
            return LoadResponse(variant=variant, state="ready", vram_used="0GB")

        async def _unload_impl(self) -> UnloadResponse:
            return UnloadResponse(state="idle")

        async def _status_impl(self) -> StatusResponse:
            return StatusResponse(state=self.state_machine.state.value)

        async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
            return LoraLoadResponse(name=name, weight=weight, loaded_loras=[name])

        async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
            return LoraUnloadResponse(name=name, loaded_loras=[])

        async def _infer_impl(self, request: Any, stream: Any) -> None:
            await stream.put(make_frame(0.5, "denoising"))
            await stream.put(
                make_final_frame(
                    result={"image_urls": ["http://example.com/a.png"]},
                    timing={"total": 1.0},
                )
            )

    return _MockEngine()


def _make_registry() -> NodeRegistry:
    """创建含一个测试节点的注册表。"""
    registry = NodeRegistry()

    class _TextInput(BaseNode):
        node_type = "TextInput"
        title = "Text Input"
        input_specs: list[PortSpec] = []
        output_specs = [PortSpec(name="text", type="text")]

        async def execute(self, inputs: dict, ctx: Any) -> dict:
            return {"text": "hello"}

    registry.register("TextInput", _TextInput)
    return registry


# ===========================================================================
# RemoteEngineConfig 测试（SDK 侧字段）
# ===========================================================================

class TestRemoteEngineConfig:
    """RemoteEngineConfig 数据类（SDK 侧）。

    SDK 侧的 RemoteEngineConfig 仅含：
        auth_token / workflow_sync / readonly_workflows / allowed_paths

    不含 url / timeout / infer_timeout / headers / writable / from_dict
    （那些是 core 侧 RemoteEngineClient 的字段）
    """

    def test_default_config(self) -> None:
        cfg = RemoteEngineConfig()
        assert cfg.auth_token is None
        assert cfg.workflow_sync is True
        assert cfg.readonly_workflows is False
        assert cfg.allowed_paths == ("/health", "/capabilities")

    def test_auth_token(self) -> None:
        cfg = RemoteEngineConfig(auth_token="Bearer secret-xxx")
        assert cfg.auth_token == "Bearer secret-xxx"

    def test_workflow_sync_false(self) -> None:
        cfg = RemoteEngineConfig(workflow_sync=False)
        assert cfg.workflow_sync is False

    def test_readonly_workflows_true(self) -> None:
        cfg = RemoteEngineConfig(readonly_workflows=True)
        assert cfg.readonly_workflows is True

    def test_custom_allowed_paths(self) -> None:
        cfg = RemoteEngineConfig(allowed_paths=("/health", "/metrics"))
        assert "/metrics" in cfg.allowed_paths

    def test_verify_token_no_auth_configured(self) -> None:
        """auth_token=None 时永远返回 True（无鉴权）。"""
        cfg = RemoteEngineConfig()
        assert cfg.verify_token(None) is True
        assert cfg.verify_token("anything") is True
        assert cfg.verify_token("") is True

    def test_verify_token_correct(self) -> None:
        cfg = RemoteEngineConfig(auth_token="Bearer secret-xxx")
        assert cfg.verify_token("Bearer secret-xxx") is True

    def test_verify_token_wrong(self) -> None:
        cfg = RemoteEngineConfig(auth_token="Bearer secret-xxx")
        assert cfg.verify_token("Bearer wrong") is False
        assert cfg.verify_token(None) is False
        assert cfg.verify_token("") is False

    def test_verify_token_case_sensitive(self) -> None:
        """token 大小写敏感（secrets.compare_digest 行为）。"""
        cfg = RemoteEngineConfig(auth_token="Bearer Secret")
        assert cfg.verify_token("Bearer Secret") is True
        assert cfg.verify_token("Bearer secret") is False

    def test_verify_token_returns_bool(self) -> None:
        """verify_token 始终返回 bool 类型。"""
        cfg = RemoteEngineConfig(auth_token="Bearer secret")
        assert isinstance(cfg.verify_token("Bearer secret"), bool)
        assert isinstance(cfg.verify_token(None), bool)
        assert isinstance(cfg.verify_token(""), bool)


# ===========================================================================
# create_engine_app 测试
# ===========================================================================

class TestCreateEngineApp:
    """create_engine_app 工厂函数。"""

    def test_creates_app_with_all_routes(self, tmp_path: Path) -> None:
        engine = _make_mock_engine()
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        registry = _make_registry()

        app = create_engine_app(
            engine=engine,
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        client = TestClient(app)

        # /health
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["engine_id"] == "mock_remote"

        # /capabilities
        resp = client.get("/capabilities")
        assert resp.status_code == 200
        assert resp.json()["engine_id"] == "mock_remote"

        # /status
        resp = client.get("/status")
        assert resp.status_code == 200

        # /workflows
        resp = client.get("/workflows")
        assert resp.status_code == 200

        # /nodes
        resp = client.get("/nodes")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

        # /nodes/definitions
        resp = client.get("/nodes/definitions")
        assert resp.status_code == 200

    def test_no_auth_allows_all(self) -> None:
        """无 auth_token 配置时所有端点可访问。"""
        engine = _make_mock_engine()
        app = create_engine_app(engine=engine)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200

    def test_auth_blocks_unauthenticated(self) -> None:
        """有 auth_token 但未携带 → 401。"""
        engine = _make_mock_engine()
        config = RemoteEngineConfig(auth_token="Bearer secret-xxx")
        app = create_engine_app(engine=engine, config=config)
        client = TestClient(app)

        # 未带 Authorization → 401（除白名单外）
        resp = client.get("/status")
        assert resp.status_code == 401

        # 白名单端点 /health /capabilities 可访问
        resp = client.get("/health")
        assert resp.status_code == 200

        resp = client.get("/capabilities")
        assert resp.status_code == 200

    def test_auth_allows_correct_token(self) -> None:
        """有 auth_token 且携带正确 token → 通过。"""
        engine = _make_mock_engine()
        config = RemoteEngineConfig(auth_token="Bearer secret-xxx")
        app = create_engine_app(engine=engine, config=config)
        client = TestClient(app)

        # 带正确 token
        resp = client.get(
            "/status",
            headers={"Authorization": "Bearer secret-xxx"},
        )
        assert resp.status_code == 200

    def test_auth_rejects_wrong_token(self) -> None:
        """错误 token → 401。"""
        engine = _make_mock_engine()
        config = RemoteEngineConfig(auth_token="Bearer secret-xxx")
        app = create_engine_app(engine=engine, config=config)
        client = TestClient(app)

        resp = client.get(
            "/status",
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_readonly_workflows_blocks_save(self, tmp_path: Path) -> None:
        """readonly_workflows=True 时 PUT/DELETE 返回 403。"""
        engine = _make_mock_engine()
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(
            "name: test\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n",
            encoding="utf-8",
        )

        config = RemoteEngineConfig(readonly_workflows=True)
        app = create_engine_app(
            engine=engine,
            config=config,
            workflows_dir=workflows_dir,
        )
        client = TestClient(app)

        # PUT → 403
        resp = client.put(
            "/workflows/default",
            content="name: test\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n",
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 403
        # detail 是结构化错误 dict，handler 透传 error 字段
        assert resp.json()["error"]["type"] == "readonly"

        # DELETE → 403
        resp = client.delete("/workflows/default")
        assert resp.status_code == 403

        # GET → 200（只读不影响查询）
        resp = client.get("/workflows/default")
        assert resp.status_code == 200

    def test_workflow_sync_false_blocks_writes(self, tmp_path: Path) -> None:
        """workflow_sync=False 等同 readonly_workflows=True。"""
        engine = _make_mock_engine()
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()

        config = RemoteEngineConfig(workflow_sync=False)
        app = create_engine_app(
            engine=engine,
            config=config,
            workflows_dir=workflows_dir,
        )
        client = TestClient(app)

        resp = client.put(
            "/workflows/default",
            content="name: test\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n",
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 403


# ===========================================================================
# mount_engine_routes 测试
# ===========================================================================

class TestMountEngineRoutes:
    """mount_engine_routes 函数。"""

    def test_returns_routers(self, tmp_path: Path) -> None:
        engine = _make_mock_engine()
        app = FastAPI()
        workflow_router, node_router = mount_engine_routes(
            app=app,
            engine=engine,
            workflows_dir=tmp_path,
            node_registry=_make_registry(),
        )
        assert workflow_router is not None
        assert node_router is not None
        assert hasattr(workflow_router, "router")
        assert hasattr(node_router, "router")

    def test_mounts_eight_endpoints(self, tmp_path: Path) -> None:
        engine = _make_mock_engine()
        app = FastAPI()
        mount_engine_routes(app=app, engine=engine)

        client = TestClient(app)

        # /health /capabilities /status（GET）
        for path in ["/health", "/capabilities", "/status"]:
            resp = client.get(path)
            assert resp.status_code == 200, f"GET {path} failed: {resp.status_code}"

        # /load /unload /lora/load /lora/unload（POST）
        resp = client.post("/load", json={"variant": "base"})
        assert resp.status_code == 200

        resp = client.post("/unload")
        assert resp.status_code == 200

        resp = client.post("/lora/load", json={"name": "lora1", "weight": 0.5})
        assert resp.status_code == 200

        resp = client.post("/lora/unload", json={"name": "lora1"})
        assert resp.status_code == 200

    def test_mounts_with_prefix(self, tmp_path: Path) -> None:
        engine = _make_mock_engine()
        app = FastAPI()
        mount_engine_routes(
            app=app,
            engine=engine,
            workflows_dir=tmp_path,
            prefix="/engines/mock",
        )
        client = TestClient(app)

        # 8 端点不带 prefix
        resp = client.get("/health")
        assert resp.status_code == 200

        # workflows/nodes 带 prefix
        resp = client.get("/engines/mock/workflows")
        assert resp.status_code == 200

        resp = client.get("/engines/mock/nodes")
        assert resp.status_code == 200

    def test_infer_endpoint_streams_sse(self, tmp_path: Path) -> None:
        engine = _make_mock_engine()
        app = FastAPI()
        mount_engine_routes(app=app, engine=engine)

        client = TestClient(app)

        # /infer 返回 SSE 流
        resp = client.post("/infer", json={"variant": "base", "inputs": {}})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # 应包含 data: 帧
        assert "data:" in resp.text

    def test_mount_static_output(self, tmp_path: Path) -> None:
        engine = _make_mock_engine()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "image").mkdir()
        (output_dir / "image" / "test.png").write_bytes(b"fake png")

        app = FastAPI()
        mount_engine_routes(
            app=app,
            engine=engine,
            mount_output=output_dir,
        )
        client = TestClient(app)

        resp = client.get("/output/image/test.png")
        assert resp.status_code == 200


# ===========================================================================
# AuthMiddleware 测试
# ===========================================================================

class TestAuthMiddleware:
    """AuthMiddleware 鉴权中间件。"""

    def test_no_auth_passes_through(self) -> None:
        """无 auth_token 配置时全部放行。"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        # auth_token=None → 无鉴权
        app.add_middleware(AuthMiddleware, config=RemoteEngineConfig())
        client = TestClient(app)

        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_auth_blocks_non_whitelist(self) -> None:
        """非白名单路径需要 Authorization。"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        config = RemoteEngineConfig(auth_token="Bearer secret")
        app.add_middleware(AuthMiddleware, config=config)
        client = TestClient(app)

        # 无 token → 401
        resp = client.get("/test")
        assert resp.status_code == 401

        # 错误 token → 401
        resp = client.get("/test", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        # 正确 token → 200
        resp = client.get("/test", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200

    def test_whitelist_paths_pass(self) -> None:
        """白名单路径无需 auth。"""
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"ok": True}

        @app.get("/capabilities")
        async def capabilities():
            return {"ok": True}

        config = RemoteEngineConfig(auth_token="Bearer secret")
        app.add_middleware(AuthMiddleware, config=config)
        client = TestClient(app)

        # 白名单
        resp = client.get("/health")
        assert resp.status_code == 200

        resp = client.get("/capabilities")
        assert resp.status_code == 200

    def test_custom_whitelist(self) -> None:
        """自定义白名单。"""
        app = FastAPI()

        @app.get("/public")
        async def public():
            return {"ok": True}

        @app.get("/private")
        async def private():
            return {"ok": True}

        config = RemoteEngineConfig(
            auth_token="Bearer secret",
            allowed_paths=("/public",),
        )
        app.add_middleware(AuthMiddleware, config=config)
        client = TestClient(app)

        # 白名单
        resp = client.get("/public")
        assert resp.status_code == 200

        # 非白名单
        resp = client.get("/private")
        assert resp.status_code == 401

    def test_unauthorized_response_format(self) -> None:
        """401 响应格式。"""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        config = RemoteEngineConfig(auth_token="Bearer secret")
        app.add_middleware(AuthMiddleware, config=config)
        client = TestClient(app)

        resp = client.get("/test")
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["type"] == "unauthorized"
        # WWW-Authenticate header
        assert resp.headers.get("www-authenticate") == "Bearer"

    def test_whitelist_with_prefix(self) -> None:
        """白名单路径以 endswith 匹配，支持 prefix 路径。"""
        app = FastAPI()

        @app.get("/engines/mock/health")
        async def health():
            return {"ok": True}

        # 默认白名单 ["/health", "/capabilities"]，通过 endswith 匹配
        config = RemoteEngineConfig(auth_token="Bearer secret")
        app.add_middleware(AuthMiddleware, config=config)
        client = TestClient(app)

        resp = client.get("/engines/mock/health")
        assert resp.status_code == 200


# ===========================================================================
# 集成：auth + workflows + nodes
# ===========================================================================

class TestAuthWithWorkflows:
    """鉴权 + 工作流 + 节点端点协同。"""

    def test_workflow_endpoints_require_auth(self, tmp_path: Path) -> None:
        """工作流端点需要 auth。"""
        engine = _make_mock_engine()
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text(
            "name: test\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n",
            encoding="utf-8",
        )

        config = RemoteEngineConfig(auth_token="Bearer xxx")
        app = create_engine_app(
            engine=engine,
            config=config,
            workflows_dir=workflows_dir,
            node_registry=_make_registry(),
        )
        client = TestClient(app)

        # 无 token → 401
        resp = client.get("/workflows")
        assert resp.status_code == 401

        resp = client.get("/nodes")
        assert resp.status_code == 401

        # 带 token → 200
        headers = {"Authorization": "Bearer xxx"}
        resp = client.get("/workflows", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/nodes", headers=headers)
        assert resp.status_code == 200

    def test_auth_with_save_workflow(self, tmp_path: Path) -> None:
        """保存工作流需要 auth。"""
        engine = _make_mock_engine()
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()

        config = RemoteEngineConfig(auth_token="Bearer xxx")
        app = create_engine_app(
            engine=engine,
            config=config,
            workflows_dir=workflows_dir,
        )
        client = TestClient(app)

        yaml_text = "name: test\nnodes:\n  - id: n1\n    type: TextInput\n    inputs: {}\n"

        # 无 token → 401
        resp = client.put(
            "/workflows/default",
            content=yaml_text,
            headers={"content-type": "application/yaml"},
        )
        assert resp.status_code == 401

        # 带 token → 200
        resp = client.put(
            "/workflows/default",
            content=yaml_text,
            headers={
                "content-type": "application/yaml",
                "Authorization": "Bearer xxx",
            },
        )
        assert resp.status_code == 200

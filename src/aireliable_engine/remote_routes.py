"""远程引擎路由挂载（v0.2 阶段 13 任务 13.1/13.2）。

提供：
    1. ``mount_engine_routes(app, engine, ...)``：一次性把 BaseEngine 的 8 个标准端点
       + WorkflowRouter + NodeRouter 挂载到 FastAPI app
    2. ``RemoteEngineConfig``：远程引擎配置容器（auth_token / workflow_sync / readonly_workflows）
    3. ``AuthMiddleware``：Bearer Token 鉴权中间件
    4. ``create_engine_app(engine, ...)``：构造一个完整的远程引擎 FastAPI app

设计要点：
    - 远程引擎与本地引擎使用相同的端点协议（兼容）
    - 远程引擎额外暴露 paradigm: "workflow" 字段（在 /capabilities 返回）
    - 远程引擎支持 auth_token 鉴权（任务 13.2）
    - 远程引擎支持 readonly_workflows 模式（任务 13.2）
    - 远程引擎支持 workflow_sync 开关（控制 PUT/DELETE 是否可用）

参考：
    - v0.2-analysis.md §13 远程引擎范式 B
    - v0.2-tasks.md 任务 13.1 / 13.2
    - PR/architecture.md §15 远程引擎范式 B
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from .engine_base import BaseEngine
from .node_registry import NodeRegistry
from .node_routes import NodeRouter, create_node_router
from .schemas import (
    HealthResponse,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraUnloadRequest,
    UnloadResponse,
)
from .workflow_engine import WorkflowEngine
from .workflow_routes import WorkflowRouter, create_workflow_router

logger = logging.getLogger(__name__)


# ===========================================================================
# 远程引擎配置（任务 13.2）
# ===========================================================================

@dataclass
class RemoteEngineConfig:
    """远程引擎配置。

    用于在远程部署引擎时控制：
        1. 鉴权（auth_token）
        2. 工作流同步权限（workflow_sync）
        3. 只读模式（readonly_workflows）

    配置示例（来自 config/engines/<engine_id>.yaml）：

        deployment:
          mode: remote
          remote_config:
            url: https://gpu-server.com/ernie_image
            auth_token: "Bearer xxx"
            workflow_sync: true
            readonly_workflows: false

    Attributes:
        auth_token: 鉴权 token（如 "Bearer xxx"）。为 None 时无鉴权。
            客户端必须通过 Authorization header 传递相同值。
        workflow_sync: 是否允许客户端 PUT/DELETE 工作流。
            False 时 PUT/DELETE 返回 403。默认 True。
        readonly_workflows: 工作流只读模式（更强约束）。
            True 时 WorkflowRouter.readonly=True，所有写操作返回 403。默认 False。
        allowed_paths: 鉴权白名单路径（如 ["/health", "/capabilities"]）。
            这些路径无需 auth_token。默认 ["health", "capabilities"]。
    """

    auth_token: Optional[str] = None
    workflow_sync: bool = True
    readonly_workflows: bool = False
    allowed_paths: tuple[str, ...] = ("/health", "/capabilities")

    def verify_token(self, authorization: Optional[str]) -> bool:
        """校验 Authorization header。

        Args:
            authorization: HTTP Authorization header 值。

        Returns:
            True 表示通过，False 表示拒绝。
        """
        if self.auth_token is None:
            return True
        if not authorization:
            return False
        # 用 secrets.compare_digest 防止时序攻击
        try:
            return secrets.compare_digest(authorization, self.auth_token)
        except Exception:
            return False


# ===========================================================================
# 鉴权中间件（任务 13.2）
# ===========================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer Token 鉴权中间件。

    用法：
        app.add_middleware(AuthMiddleware, config=remote_config)

    工作原理：
        1. 白名单路径（如 /health /capabilities）直接放行
        2. 其他路径校验 Authorization header
        3. 校验失败返回 401 Unauthorized
        4. 校验通过继续处理

    Note:
        - token 比对用 secrets.compare_digest 防止时序攻击
        - 白名单路径默认为 /health 和 /capabilities（用于健康检查和能力声明）
    """

    def __init__(self, app: Any, config: Optional[RemoteEngineConfig] = None) -> None:
        super().__init__(app)
        self.config = config or RemoteEngineConfig()

    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: Callable[[StarletteRequest], Awaitable[StarletteResponse]],
    ) -> StarletteResponse:
        """中间件处理函数。"""
        # 无 auth_token 配置 → 无鉴权
        if self.config.auth_token is None:
            return await call_next(request)

        # 白名单路径放行
        path = request.url.path
        # 去掉 prefix 后比较
        for allowed in self.config.allowed_paths:
            if path == allowed or path.endswith(allowed):
                return await call_next(request)

        # 校验 Authorization
        authorization = request.headers.get("Authorization")
        if not self.config.verify_token(authorization):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "type": "unauthorized",
                        "message": "Authorization header 缺失或无效",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


# ===========================================================================
# 路由挂载
# ===========================================================================

def mount_engine_routes(
    app: FastAPI,
    engine: BaseEngine,
    *,
    workflows_dir: Optional[Path] = None,
    node_registry: Optional[NodeRegistry] = None,
    node_packages_dir: Optional[Path] = None,
    prefix: str = "",
    readonly: bool = False,
    mount_output: Optional[Path] = None,
) -> tuple[WorkflowRouter, NodeRouter]:
    """把引擎的全部路由挂载到 FastAPI app。

    挂载的端点：
        - 8 个标准端点：/health /capabilities /load /unload /infer /status /lora/load /lora/unload
        - 工作流端点：/workflows/*（WorkflowRouter 提供）
        - 节点端点：/nodes/*（NodeRouter 提供）
        - 输出静态文件（可选）：/output/*

    Args:
        app: FastAPI 应用实例。
        engine: BaseEngine 实例（提供 8 端点 + 工作流加载）。
        workflows_dir: 工作流目录。为 None 时从 engine.workflows_dir 取。
        node_registry: 节点注册表。为 None 时从 engine.node_registry 取。
        node_packages_dir: 节点包目录（用于查询第三方节点包前端定义）。
        prefix: 路由前缀（如 "/engines/ernie_image"）。
        readonly: 工作流只读模式（PUT/DELETE 返回 403）。
        mount_output: 输出目录路径。若提供，挂载到 /output。

    Returns:
        (workflow_router, node_router) 元组。
    """
    _register_standard_endpoints(app, engine)

    # 挂载工作流路由
    workflow_router = create_workflow_router(
        engine=engine if isinstance(engine, WorkflowEngine) else None,
        workflows_dir=workflows_dir or getattr(engine, "workflows_dir", None),
        node_registry=node_registry or getattr(engine, "node_registry", None),
        prefix=prefix,
        readonly=readonly,
    )
    app.include_router(workflow_router.router)

    # 挂载节点路由
    node_router = create_node_router(
        registry=node_registry or getattr(engine, "node_registry", None),
        node_packages_dir=node_packages_dir,
        prefix=prefix,
    )
    app.include_router(node_router.router)

    # 挂载静态文件（输出目录）
    if mount_output is not None:
        mount_output = Path(mount_output)
        mount_output.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/output",
            StaticFiles(directory=str(mount_output)),
            name="output",
        )
        logger.info("静态输出目录已挂载到 /output（%s）", mount_output)

    logger.info(
        "引擎 %s 路由已挂载（prefix=%r, readonly=%s）",
        engine.engine_id, prefix, readonly,
    )
    return workflow_router, node_router


def _register_standard_endpoints(app: FastAPI, engine: BaseEngine) -> None:
    """注册 8 个标准 HTTP 端点。

    与 engines/<engine>/server.py 中的 _register_endpoints 等价，
    但以注入方式而非全局 get_engine() 方式工作。

    v0.2.5+ T6.2：额外注册 /log_level 端点（GET 查询 + POST 动态调整）。
    """
    from .schemas import ErrorResponse
    # v0.2.5+ T6.2：注册日志级别动态调整端点
    from .logging_config import register_log_level_endpoint
    register_log_level_endpoint(app)

    @app.get("/health", response_model=HealthResponse, tags=["engine"])
    async def health() -> HealthResponse:
        """健康检查。"""
        return await engine.health()

    @app.get("/capabilities", tags=["engine"])
    async def capabilities() -> dict:
        """能力声明。

        远程引擎通过此端点告知本地：paradigm（integrated / workflow / custom）、
        可用工作流、可用节点等。
        """
        return (await engine.capabilities()).model_dump()

    @app.post("/load", tags=["engine"])
    async def load(request: LoadRequest) -> dict:
        """加载 variant 到显存（v0.2.5+ 幂等）。

        幂等规则（详见 PR/architecture.md §3.12）：
            - 已加载同 variant → 直接返回 success（不重复加载）
            - 已加载不同 variant → 先卸载旧模型再加载新模型
            - 未加载 → 正常加载
        """
        try:
            # 幂等检查：查询当前状态
            try:
                status = await engine.status()
                if status.loaded_variant == request.variant and status.state == "ready":
                    logger.info(
                        "幂等命中：variant=%s 已加载，跳过重复加载",
                        request.variant,
                    )
                    return LoadResponse(
                        variant=request.variant,
                        state="ready",
                        vram_used=status.vram_used,
                        load_time=0.0,
                    ).model_dump()
            except Exception as e:
                logger.debug("幂等检查查询状态失败，继续正常加载: %s", e)

            response = await engine.load(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/unload", tags=["engine"])
    async def unload() -> dict:
        """卸载模型（v0.2.5+ 幂等）。

        幂等规则（详见 PR/architecture.md §3.12）：
            - 已卸载（无模型加载）→ 直接返回 success
            - 已加载 → 正常卸载
        """
        try:
            # 幂等检查：查询当前状态
            try:
                status = await engine.status()
                if status.loaded_variant is None or status.state == "idle":
                    logger.info("幂等命中：无模型加载，跳过卸载")
                    return UnloadResponse(
                        state="idle",
                        vram_freed="0GB",
                    ).model_dump()
            except Exception as e:
                logger.debug("幂等检查查询状态失败，继续正常卸载: %s", e)

            response = await engine.unload()
            return response.model_dump()
        except Exception as e:
            logger.exception("卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/infer", tags=["engine"])
    async def infer(request: Request):
        """推理（SSE 流式返回进度）。"""
        # 解析请求体（任意 JSON 结构，由引擎自定义）
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"请求体 JSON 解析失败：{e}",
            )

        # 把 dict 转为 InferRequest（兼容）
        from .schemas import InferRequest
        try:
            infer_req = InferRequest.model_validate(body)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"请求参数校验失败：{e}",
            )

        sse_stream = engine.infer(infer_req)

        return StreamingResponse(
            sse_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/status", tags=["engine"])
    async def get_status() -> dict:
        """引擎状态查询。"""
        return (await engine.status()).model_dump()

    @app.post("/lora/load", tags=["engine"])
    async def lora_load(request: LoraLoadRequest) -> dict:
        """加载 LoRA。"""
        try:
            response = await engine.load_lora(request)
            return response.model_dump()
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception("LoRA 加载失败")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/lora/unload", tags=["engine"])
    async def lora_unload(request: LoraUnloadRequest) -> dict:
        """卸载 LoRA。"""
        try:
            response = await engine.unload_lora(request)
            return response.model_dump()
        except Exception as e:
            logger.exception("LoRA 卸载失败")
            raise HTTPException(status_code=500, detail=str(e))

    # 统一错误处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # 如果 detail 是 dict 且含 "error" 键（结构化错误），直接透传
        # WorkflowRouter / NodeRouter 抛出的 detail 形如：
        #   {"error": {"type": "readonly", "message": "..."}}
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail,
            )
        # 否则包装成 ErrorResponse（保持与原行为兼容）
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error={
                "type": exc.__class__.__name__,
                "message": str(exc.detail),
                "code": exc.status_code,
            }).model_dump(),
        )


# ===========================================================================
# 远程引擎 app 工厂
# ===========================================================================

def create_engine_app(
    engine: BaseEngine,
    *,
    config: Optional[RemoteEngineConfig] = None,
    workflows_dir: Optional[Path] = None,
    node_registry: Optional[NodeRegistry] = None,
    node_packages_dir: Optional[Path] = None,
    prefix: str = "",
    mount_output: Optional[Path] = None,
    title: Optional[str] = None,
    version: str = "0.2.0",
) -> FastAPI:
    """构造一个完整的远程引擎 FastAPI app。

    Args:
        engine: BaseEngine 实例。
        config: 远程引擎配置（auth_token / readonly_workflows / workflow_sync）。
            为 None 时使用默认配置（无鉴权 + 可读写）。
        workflows_dir: 工作流目录。
        node_registry: 节点注册表。
        node_packages_dir: 节点包目录。
        prefix: 路由前缀。
        mount_output: 输出目录挂载路径。
        title: app 标题（默认从 engine.engine_name 取）。
        version: app 版本。

    Returns:
        FastAPI app 实例，可直接用 uvicorn 启动。

    Example:
        engine = MyWorkflowEngine(...)
        config = RemoteEngineConfig(
            auth_token="Bearer secret-xxx",
            readonly_workflows=False,
        )
        app = create_engine_app(engine, config=config, workflows_dir=Path("workflows"))
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    config = config or RemoteEngineConfig()

    app = FastAPI(
        title=title or f"{engine.engine_name} Remote Engine",
        description=(
            f"AIReliable 远程引擎（engine_id={engine.engine_id}, "
            f"capability={engine.capability}, type={engine.engine_type}）"
        ),
        version=version,
    )

    # CORS（允许跨域，远程引擎部署在云端时需要）
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 鉴权中间件（任务 13.2）
    if config.auth_token is not None:
        app.add_middleware(AuthMiddleware, config=config)
        logger.info("远程引擎 %s 已启用鉴权（token=%s...）",
                    engine.engine_id, config.auth_token[:16] if config.auth_token else "")

    # readonly_workflows 控制 workflow_sync
    readonly = config.readonly_workflows or (not config.workflow_sync)

    # 挂载全部路由
    mount_engine_routes(
        app=app,
        engine=engine,
        workflows_dir=workflows_dir,
        node_registry=node_registry,
        node_packages_dir=node_packages_dir,
        prefix=prefix,
        readonly=readonly,
        mount_output=mount_output,
    )

    logger.info(
        "远程引擎 app 创建完成（engine_id=%s, auth=%s, readonly_workflows=%s）",
        engine.engine_id,
        bool(config.auth_token),
        readonly,
    )
    return app


__all__ = [
    "RemoteEngineConfig",
    "AuthMiddleware",
    "mount_engine_routes",
    "create_engine_app",
]

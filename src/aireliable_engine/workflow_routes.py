"""工作流管理 FastAPI 路由（v0.2 阶段 12 任务 12.1）。

提供工作流 CRUD 端点，供画布编辑器与外部工具调用。

端点：
    GET    /workflows                     → 列出所有工作流
    GET    /workflows/{name}              → 获取工作流 yaml
    PUT    /workflows/{name}              → 保存（创建/更新）工作流
    DELETE /workflows/{name}              → 删除工作流
    POST   /workflows/{name}/reload       → 强制重新加载工作流（清缓存）
    GET    /workflows/{name}/validate     → 校验工作流合法性

权限：
    - 默认所有端点无鉴权（本地部署）
    - 远程部署时通过 reverse proxy 加 auth middleware

数据格式：
    - GET 返回 yaml 文本（Content-Type: application/yaml）
    - GET /workflows 返回 JSON 列表
    - PUT 接收 yaml 文本 body
    - 所有错误返回 JSON：{"error": {"type": "...", "message": "..."}}

参考文档：
    - v0.2-analysis.md §13 工作流管理 API
    - v0.2-tasks.md 任务 12.1
    - PR/architecture.md §12.8 工作流可视化
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .node_registry import NodeRegistry
from .workflow_engine import WorkflowEngine, WorkflowEngineError, WorkflowNotFoundError
from .workflow_executor import WorkflowExecutionError
from .workflow_schema import WorkflowSchemaError, validate_workflow_dict
from .node import BaseNode

logger = logging.getLogger(__name__)


# ===========================================================================
# 工具函数
# ===========================================================================

def _yaml_to_dict(yaml_text: str) -> dict:
    """把 yaml 文本解析为 dict。

    Args:
        yaml_text: yaml 文本。

    Returns:
        解析后的 dict。

    Raises:
        WorkflowSchemaError: yaml 解析失败。
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise WorkflowSchemaError(f"PyYAML 未安装：{e}") from e

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise WorkflowSchemaError(f"yaml 解析失败：{e}") from e

    if not isinstance(data, dict):
        raise WorkflowSchemaError(
            f"工作流根节点必须是 dict，实际类型：{type(data).__name__}"
        )
    return data


def _dict_to_yaml(data: dict) -> str:
    """把 dict 序列化为 yaml 文本。"""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise WorkflowSchemaError(f"PyYAML 未安装：{e}") from e

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _safe_workflow_name(name: str) -> str:
    """校验工作流名合法性，防止路径穿越攻击。

    Args:
        name: 工作流名（不含 .yaml 扩展名）。

    Returns:
        清理后的工作流名。

    Raises:
        HTTPException: 名称非法或包含路径穿越字符。
    """
    if not name or not isinstance(name, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"type": "invalid_name", "message": "工作流名不能为空"}},
        )
    # 禁止路径穿越
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "type": "invalid_name",
                    "message": f"工作流名包含非法字符：{name!r}",
                }
            },
        )
    # 禁止特殊字符
    invalid_chars = set('<>:"|?*\x00')
    if any(c in invalid_chars for c in name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "type": "invalid_name",
                    "message": f"工作流名包含特殊字符：{name!r}",
                }
            },
        )
    return name


def _make_error_response(
    exc: Exception,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> JSONResponse:
    """构造标准错误响应。"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
        },
    )


# ===========================================================================
# WorkflowRouter 类
# ===========================================================================

class WorkflowRouter:
    """工作流管理路由封装。

    负责把 WorkflowEngine 的工作流能力暴露为 REST API。
    支持注入到任意 FastAPI app（本地引擎或远程引擎）。

    用法：
        engine = MyWorkflowEngine(...)
        router = WorkflowRouter(engine=engine, prefix="/engines/{engine_id}")
        app.include_router(router.router)
    """

    def __init__(
        self,
        engine: Optional[WorkflowEngine] = None,
        workflows_dir: Optional[Path] = None,
        node_registry: Optional[NodeRegistry] = None,
        prefix: str = "",
        readonly: bool = False,
    ) -> None:
        """初始化路由。

        Args:
            engine: WorkflowEngine 实例（提供工作流加载/缓存能力）。
                若提供，则 GET/POST reload 调用 engine 的方法。
                若为 None，则直接操作 workflows_dir 文件系统。
            workflows_dir: 工作流目录。若 engine 提供，从 engine.workflows_dir 取。
                两者都为 None 时所有路径相关操作返回 500。
            node_registry: 节点注册表（用于校验节点类型）。
                若未提供，校验端点跳过节点类型检查。
            prefix: 路由前缀（如 "/engines/ernie_image"）。
            readonly: 只读模式（仅允许 GET，禁止 PUT/DELETE）。
        """
        self.engine = engine
        self.workflows_dir = (
            workflows_dir
            or (engine.workflows_dir if engine else None)
        )
        if self.workflows_dir is not None:
            self.workflows_dir = Path(self.workflows_dir)
        self.node_registry = node_registry or (
            engine.node_registry if engine else None
        )
        self.prefix = prefix.rstrip("/")
        self.readonly = readonly

        # 创建 FastAPI router
        self.router = APIRouter(prefix=self.prefix, tags=["workflows"])
        self._register_routes()

    # ----------------------------------------------------------------------
    # 路由注册
    # ----------------------------------------------------------------------

    def _register_routes(self) -> None:
        """注册所有 CRUD 端点。"""

        # 1. GET /workflows：列出所有工作流
        self.router.add_api_route(
            "/workflows",
            self.list_workflows,
            methods=["GET"],
            response_model=None,
            summary="列出所有工作流",
        )

        # 2. GET /workflows/{name}：获取工作流 yaml
        self.router.add_api_route(
            "/workflows/{name}",
            self.get_workflow,
            methods=["GET"],
            response_model=None,
            summary="获取工作流 yaml",
        )

        # 3. PUT /workflows/{name}：保存工作流
        self.router.add_api_route(
            "/workflows/{name}",
            self.save_workflow,
            methods=["PUT"],
            response_model=None,
            summary="保存工作流",
        )

        # 4. DELETE /workflows/{name}：删除工作流
        self.router.add_api_route(
            "/workflows/{name}",
            self.delete_workflow,
            methods=["DELETE"],
            response_model=None,
            summary="删除工作流",
        )

        # 5. POST /workflows/{name}/reload：强制重新加载
        self.router.add_api_route(
            "/workflows/{name}/reload",
            self.reload_workflow,
            methods=["POST"],
            response_model=None,
            summary="强制重新加载工作流（清缓存）",
        )

        # 6. GET /workflows/{name}/validate：校验工作流
        self.router.add_api_route(
            "/workflows/{name}/validate",
            self.validate_workflow,
            methods=["GET"],
            response_model=None,
            summary="校验工作流合法性",
        )

    # ----------------------------------------------------------------------
    # 路由处理函数
    # ----------------------------------------------------------------------

    async def list_workflows(self) -> dict:
        """GET /workflows：列出所有工作流。

        Returns:
            {"workflows": [{"name": str, "size": int, "mtime": float}, ...]}
        """
        if self.workflows_dir is None or not self.workflows_dir.exists():
            return {"workflows": []}

        workflows = []
        for p in sorted(self.workflows_dir.glob("*.yaml")):
            try:
                stat = p.stat()
                workflows.append({
                    "name": p.stem,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            except OSError as e:
                logger.warning("获取工作流 %s 状态失败：%s", p, e)
                continue

        # 若有 engine，附加当前工作流
        current = None
        if self.engine is not None:
            current = getattr(self.engine, "current_workflow", None)

        return {"workflows": workflows, "current": current}

    async def get_workflow(self, name: str) -> Response:
        """GET /workflows/{name}：获取工作流 yaml。

        Returns:
            PlainTextResponse（Content-Type: application/yaml）
        """
        name = _safe_workflow_name(name)

        # 优先用 engine 的缓存（含热重载）
        if self.engine is not None:
            try:
                text = self.engine._read_workflow_file(name)
                return PlainTextResponse(
                    text,
                    media_type="application/yaml",
                    headers={"Cache-Control": "no-cache"},
                )
            except WorkflowNotFoundError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": {
                            "type": "workflow_not_found",
                            "message": str(e),
                            "available": e.available if hasattr(e, "available") else [],
                        }
                    },
                )
            except WorkflowEngineError as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": {"type": "workflow_load_error", "message": str(e)}},
                )

        # 无 engine，直接读文件
        if self.workflows_dir is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "no_workflows_dir", "message": "workflows_dir 未配置"}},
            )

        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"type": "workflow_not_found", "message": f"工作流 {name!r} 不存在"}},
            )

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "io_error", "message": str(e)}},
            )

        return PlainTextResponse(
            text,
            media_type="application/yaml",
            headers={"Cache-Control": "no-cache"},
        )

    async def save_workflow(self, name: str, request: Request) -> dict:
        """PUT /workflows/{name}：保存工作流。

        请求体：yaml 文本（Content-Type: application/yaml 或 text/plain）。

        Returns:
            {"name": str, "size": int, "valid": bool, "errors": list[str]}
        """
        if self.readonly:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": {"type": "readonly", "message": "工作流目录为只读模式"}},
            )

        name = _safe_workflow_name(name)

        if self.workflows_dir is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "no_workflows_dir", "message": "workflows_dir 未配置"}},
            )

        # 确保目录存在
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

        # 读取请求体
        try:
            body = await request.body()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"type": "body_read_error", "message": str(e)}},
            )

        try:
            yaml_text = body.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"type": "encoding_error", "message": str(e)}},
            )

        # 校验 yaml 合法性（不强制中断保存，只返回警告）
        valid = True
        errors: list[str] = []
        try:
            yaml_dict = _yaml_to_dict(yaml_text)
            validate_workflow_dict(yaml_dict)
            # 节点类型校验
            if self.node_registry is not None:
                errors = self._validate_node_types(yaml_dict)
                if errors:
                    valid = False
        except WorkflowSchemaError as e:
            valid = False
            errors.append(str(e))
        except Exception as e:
            valid = False
            errors.append(f"校验异常：{e}")

        # 写入文件（即使校验失败也保存，便于用户修复）
        path = self.workflows_dir / f"{name}.yaml"
        try:
            # 原子写入：先写 .tmp 再重命名
            tmp_path = path.with_suffix(".yaml.tmp")
            tmp_path.write_text(yaml_text, encoding="utf-8")
            tmp_path.replace(path)
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "io_error", "message": str(e)}},
            )

        # 清除 engine 缓存
        if self.engine is not None:
            self.engine.reload_workflow(name)

        size = path.stat().st_size
        logger.info(
            "工作流 %r 已保存（%d 字节，valid=%s）",
            name, size, valid,
        )

        return {
            "name": name,
            "size": size,
            "valid": valid,
            "errors": errors,
        }

    async def delete_workflow(self, name: str) -> dict:
        """DELETE /workflows/{name}：删除工作流。

        Returns:
            {"name": str, "deleted": True}
        """
        if self.readonly:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": {"type": "readonly", "message": "工作流目录为只读模式"}},
            )

        name = _safe_workflow_name(name)

        if self.workflows_dir is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "no_workflows_dir", "message": "workflows_dir 未配置"}},
            )

        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"type": "workflow_not_found", "message": f"工作流 {name!r} 不存在"}},
            )

        try:
            path.unlink()
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "io_error", "message": str(e)}},
            )

        # 清除 engine 缓存
        if self.engine is not None:
            self.engine.reload_workflow(name)

        logger.info("工作流 %r 已删除", name)
        return {"name": name, "deleted": True}

    async def reload_workflow(self, name: str) -> dict:
        """POST /workflows/{name}/reload：强制重新加载工作流（清缓存）。

        Returns:
            {"name": str, "reloaded": True}
        """
        name = _safe_workflow_name(name)

        if self.engine is None:
            # 无 engine 时无缓存，直接返回成功
            return {"name": name, "reloaded": True, "note": "无 engine 缓存"}

        try:
            self.engine.reload_workflow(name)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "reload_error", "message": str(e)}},
            )

        return {"name": name, "reloaded": True}

    async def validate_workflow(self, name: str) -> dict:
        """GET /workflows/{name}/validate：校验工作流合法性。

        校验项：
            1. yaml 语法合法
            2. 符合 WorkflowSchema
            3. 节点类型已注册（若有 node_registry）

        Returns:
            {"name": str, "valid": bool, "errors": list[str]}
        """
        name = _safe_workflow_name(name)

        # 获取 yaml 文本
        try:
            if self.engine is not None:
                yaml_text = self.engine._read_workflow_file(name)
            elif self.workflows_dir is not None:
                path = self.workflows_dir / f"{name}.yaml"
                if not path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={"error": {"type": "workflow_not_found", "message": f"工作流 {name!r} 不存在"}},
                    )
                yaml_text = path.read_text(encoding="utf-8")
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": {"type": "no_workflows_dir", "message": "workflows_dir 未配置"}},
                )
        except WorkflowNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"type": "workflow_not_found", "message": str(e)}},
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "read_error", "message": str(e)}},
            )

        # 校验
        valid = True
        errors: list[str] = []
        try:
            yaml_dict = _yaml_to_dict(yaml_text)
            validate_workflow_dict(yaml_dict)
            if self.node_registry is not None:
                errors = self._validate_node_types(yaml_dict)
                if errors:
                    valid = False
        except WorkflowSchemaError as e:
            valid = False
            errors.append(str(e))
        except Exception as e:
            valid = False
            errors.append(f"校验异常：{e}")

        return {
            "name": name,
            "valid": valid,
            "errors": errors,
        }

    # ----------------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------------

    def _validate_node_types(self, yaml_dict: dict) -> list[str]:
        """校验工作流 yaml 中引用的所有节点类型已注册。

        Args:
            yaml_dict: 工作流 yaml dict。

        Returns:
            错误信息列表（空列表表示通过）。
        """
        errors: list[str] = []
        if self.node_registry is None:
            return errors

        nodes = yaml_dict.get("nodes", [])
        if not isinstance(nodes, list):
            return errors

        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"nodes[{i}] 必须为 dict，实际类型：{type(node).__name__}")
                continue
            node_type = node.get("type")
            if not node_type:
                errors.append(f"nodes[{i}].type 必填")
                continue
            if not self.node_registry.contains(node_type):
                errors.append(
                    f"nodes[{i}].type {node_type!r} 未在节点注册表中注册"
                )

        return errors


# ===========================================================================
# 工厂函数
# ===========================================================================

def create_workflow_router(
    engine: Optional[WorkflowEngine] = None,
    workflows_dir: Optional[Path] = None,
    node_registry: Optional[NodeRegistry] = None,
    prefix: str = "",
    readonly: bool = False,
) -> WorkflowRouter:
    """创建工作流管理路由。

    Args:
        engine: WorkflowEngine 实例。
        workflows_dir: 工作流目录。
        node_registry: 节点注册表。
        prefix: 路由前缀。
        readonly: 只读模式。

    Returns:
        WorkflowRouter 实例。
    """
    return WorkflowRouter(
        engine=engine,
        workflows_dir=workflows_dir,
        node_registry=node_registry,
        prefix=prefix,
        readonly=readonly,
    )


__all__ = [
    "WorkflowRouter",
    "create_workflow_router",
]

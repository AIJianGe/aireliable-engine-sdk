"""节点定义 FastAPI 路由（v0.2 阶段 12 任务 12.2）。

提供节点查询端点，供画布编辑器加载节点库与节点包前端定义。

端点：
    GET /nodes                          → 列出所有已注册节点
    GET /nodes/{node_type}              → 获取单个节点详情
    GET /nodes/definitions              → 所有节点的前端定义（画布用）
    GET /nodes/packages                 → 列出已安装的第三方节点包
    GET /nodes/packages/{package_name}/definitions → 节点包的前端定义

v0.2.5 P0-H3 增强：
    GET /nodes?include_errors=true      → 返回加载失败的节点列表（前端红色标记）
    GET /nodes/definitions?include_errors=true → 同上

数据格式：
    - GET /nodes 返回 JSON 列表
    - 节点定义镜像前端格式（与 src-tauri/ui/workflow_editor/node_definitions.js 一致）

参考文档：
    - v0.2-analysis.md §14.7 节点包生态
    - v0.2-tasks.md 任务 12.2
    - PR/development-guide.md §32 节点包发布规范
"""

from __future__ import annotations

import json
import logging
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from .node import BaseNode, PortSpec
from .node_registry import NodeRegistry

logger = logging.getLogger(__name__)


# ===========================================================================
# 节点定义镜像生成
# ===========================================================================

def _port_to_dict(spec: PortSpec) -> dict:
    """把 PortSpec 转换为前端可用的 dict。"""
    return {
        "name": spec.name,
        "type": spec.type,
        "required": spec.required,
        "default": spec.default,
        "description": spec.description,
        "variadic": spec.variadic,
    }


def _node_to_definition(node_cls: type[BaseNode]) -> dict:
    """把节点类转换为前端定义 dict。

    v0.2.5 P0-H2：params 字段透传 params_schema 中的 format 字段
    （multiline / enum-from-dir / file-paths / info / enum），
    供前端 _renderParamWidget 按 type+format 分发渲染控件。

    v0.2.5 P-CategoryFix：透传 category / subcategory 字段。
        - category 优先从类属性读取（节点显式声明），否则用 _infer_category 推断
        - subcategory 优先从类属性读取，否则为空字符串（前端归到「其他」子分类）

    Args:
        node_cls: 节点类。

    Returns:
        节点定义 dict（与 NODE_DEFINITIONS 格式一致）。
    """
    # 从类属性读取规格
    # v0.2.5 P0-O9：title 为空字符串时回退到 node_type（BaseNode 默认 title=""）
    title = getattr(node_cls, "title", "") or node_cls.node_type
    # v0.2.5 P-CategoryFix：优先读取类属性 category / subcategory
    category = getattr(node_cls, "category", None) or _infer_category(node_cls)
    subcategory = getattr(node_cls, "subcategory", None) or ""
    return {
        "type": node_cls.node_type,
        "title": title,
        "category": category,
        "subcategory": subcategory,
        "color": _infer_color(node_cls, category),
        "icon": getattr(node_cls, "icon", "•") or "•",
        "description": (node_cls.__doc__ or "").split("\n\n")[0].strip()
            if node_cls.__doc__ else "",
        "inputs": [_port_to_dict(s) for s in node_cls.input_specs],
        "outputs": [_port_to_dict(s) for s in node_cls.output_specs],
        # v0.2.5 P0-H2：_infer_params 直接透传 params_schema（含 format 字段）
        "params": _infer_params(node_cls),
    }


def _infer_category(node_cls: type[BaseNode]) -> str:
    """从节点类模块路径推断分类（fallback，节点未显式声明 category 时使用）。

    v0.2.5 P-CategoryFix：本函数仅作为 fallback。节点类应显式声明 ``category`` 类属性。
    引擎自定义节点（模块路径非 ``aireliable_engine.nodes.*``）必须显式声明 category，
    否则会被归到 ``custom``。

    匹配规则（按 SDK 标准目录结构）：
        - ``.nodes.input.`` / ``.input.``      → input
        - ``.nodes.loader.`` / ``.loader.``     → loader
        - ``.nodes.encode.`` / ``.encode.``     → encode
        - ``.nodes.sampler.`` / ``.sampler.`` / ``.nodes.sample.`` / ``.sample.`` → sample
        - ``.nodes.decode.`` / ``.decode.``     → decode
        - ``.nodes.lora.`` / ``.lora.``         → lora
        - ``.nodes.latent.`` / ``.latent.``     → latent
        - ``.nodes.output.`` / ``.output.``     → output
        - ``.nodes.postprocess.`` / ``.postprocess.`` → output（v0.2.5 P-CategoryFix 新增）
        - 其他                                  → custom

    Args:
        node_cls: 节点类。

    Returns:
        分类字符串（input / loader / encode / sample / decode / lora / latent / output / custom）。
    """
    module = node_cls.__module__
    if ".nodes.input" in module or ".input." in module:
        return "input"
    if ".nodes.loader" in module or ".loader." in module:
        return "loader"
    if ".nodes.encode" in module or ".encode." in module:
        return "encode"
    # v0.2.5：统一使用 sample（与 SamplerNode.category = "sample" 一致）
    if ".nodes.sampler" in module or ".sampler." in module:
        return "sample"
    if ".nodes.sample" in module or ".sample." in module:
        return "sample"
    if ".nodes.decode" in module or ".decode." in module:
        return "decode"
    if ".nodes.lora" in module or ".lora." in module:
        return "lora"
    if ".nodes.latent" in module or ".latent." in module:
        return "latent"
    # v0.2.5 P-CategoryFix：postprocess 归到 output（ImageSaveNode 在 postprocess 目录）
    if ".nodes.output" in module or ".output." in module:
        return "output"
    if ".nodes.postprocess" in module or ".postprocess." in module:
        return "output"
    return "custom"


def _infer_color(node_cls: type[BaseNode], category: Optional[str] = None) -> str:
    """根据分类推断颜色。

    Args:
        node_cls: 节点类。
        category: 已知的分类（避免重复推断）。若为 None 则调用 _infer_category。
    """
    if category is None:
        category = _infer_category(node_cls)
    colors = {
        "input": "#4A90E2",
        "loader": "#9B59B6",
        "encode": "#E67E22",
        "sample": "#E74C3C",
        "sampler": "#E74C3C",  # 兼容（部分老节点可能用 sampler）
        "decode": "#16A085",
        "lora": "#F39C12",
        "latent": "#3498DB",
        "output": "#27AE60",
        "custom": "#7F8C8D",
    }
    return colors.get(category, "#7F8C8D")


def _infer_params(node_cls: type[BaseNode]) -> dict:
    """从类属性 params_schema 推断参数定义。

    v0.2.5 P0-H1：params_schema 各项支持以下字段（与前端 widget 渲染对齐）：
        - type: string / integer / number / boolean / enum / array / info
        - format: multiline / enum / enum-from-dir / file-paths / info
            （前端按 type + format 组合分发到不同 HTML 控件）
        - label: 中文显示名
        - description: tooltip 描述
        - required: 是否必填
        - default: 默认值
        - options: enum 类型的可选值列表
        - min / max: 数值范围

    v0.2.5 P1-1：params_schema 各项新增 ``advanced`` 字段：
        - advanced: bool         是否为高级参数（默认 False，不写此键等同 False）。
                                 True 时前端属性面板将其折叠到「高级选项 ▾」区域，
                                 降低新用户认知负担。API 调用不受影响。
        典型用法：核心参数（steps / guidance / value）不标记，调优参数
                  （device / call_timeout / loras）标记 ``"advanced": True``。

    Returns:
        params_schema dict（含 format / advanced 字段），前端直接消费。
    """
    # 节点可在类属性中声明 params_schema
    params_schema = getattr(node_cls, "params_schema", None)
    if isinstance(params_schema, dict):
        return params_schema
    return {}


# ===========================================================================
# NodeRouter 类
# ===========================================================================

class NodeRouter:
    """节点定义查询路由。

    用法：
        registry = NodeRegistry()
        registry.discover_plugins()
        router = NodeRouter(registry=registry, prefix="/engines/{engine_id}")
        app.include_router(router.router)
    """

    def __init__(
        self,
        registry: Optional[NodeRegistry] = None,
        node_packages_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> None:
        """初始化路由。

        Args:
            registry: 节点注册表。若为 None，使用全局默认注册表。
            node_packages_dir: 节点包目录（用于查找 frontend/node_definitions.json）。
                若为 None，从已安装包的 site-packages 查找。
            prefix: 路由前缀。
        """
        self.registry = registry
        self.node_packages_dir = node_packages_dir
        self.prefix = prefix.rstrip("/")
        self.router = APIRouter(prefix=self.prefix, tags=["nodes"])
        self._register_routes()

    # ----------------------------------------------------------------------
    # 路由注册
    # ----------------------------------------------------------------------

    def _register_routes(self) -> None:
        self.router.add_api_route(
            "/nodes",
            self.list_nodes,
            methods=["GET"],
            response_model=None,
            summary="列出所有已注册节点",
        )
        self.router.add_api_route(
            "/nodes/definitions",
            self.get_all_definitions,
            methods=["GET"],
            response_model=None,
            summary="所有节点的前端定义（画布用）",
        )
        self.router.add_api_route(
            "/nodes/packages",
            self.list_packages,
            methods=["GET"],
            response_model=None,
            summary="列出已安装的第三方节点包",
        )
        self.router.add_api_route(
            "/nodes/packages/{package_name}/definitions",
            self.get_package_definitions,
            methods=["GET"],
            response_model=None,
            summary="节点包的前端定义",
        )
        self.router.add_api_route(
            "/nodes/{node_type}",
            self.get_node,
            methods=["GET"],
            response_model=None,
            summary="获取单个节点详情",
        )

    # ----------------------------------------------------------------------
    # 路由处理函数
    # ----------------------------------------------------------------------

    async def list_nodes(
        self,
        include_errors: bool = Query(
            False, description="包含加载失败的节点（前端红色标记 + 错误 tooltip）"
        ),
    ) -> dict:
        """GET /nodes：列出所有已注册节点。

        v0.2.5 P0-H3：支持 include_errors=true 查询参数，
        返回加载失败的节点列表（供前端显示红色标记 + 错误 tooltip）。

        Args:
            include_errors: 是否包含加载失败的节点。默认 False。

        Returns:
            {"nodes": [{"type", "title", "category", "module"}, ...], "count": int}
            include_errors=True 时额外返回 {"load_errors": {...}}
        """
        if self.registry is None:
            return {"nodes": [], "count": 0, "note": "registry 未配置"}

        nodes = []
        for node_type, node_cls in self.registry.list().items():
            # v0.2.5 P-CategoryFix：优先读取类属性 category，与 _node_to_definition 一致
            category = getattr(node_cls, "category", None) or _infer_category(node_cls)
            nodes.append({
                "type": node_type,
                "title": getattr(node_cls, "title", node_type),
                "category": category,
                "module": node_cls.__module__,
                "class": node_cls.__name__,
            })

        result: dict[str, Any] = {"nodes": nodes, "count": len(nodes)}
        # v0.2.5 P0-H3：可选返回加载失败的节点
        if include_errors:
            result["load_errors"] = self.registry.list_errors()
            result["error_count"] = len(result["load_errors"])
        return result

    async def get_node(self, node_type: str) -> dict:
        """GET /nodes/{node_type}：获取单个节点详情。

        Returns:
            节点完整定义 dict。
        """
        if self.registry is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": {"type": "no_registry", "message": "registry 未配置"}},
            )

        try:
            node_cls = self.registry.get(node_type)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": {"type": "node_not_found", "message": f"节点 {node_type!r} 未注册"}},
            )

        return _node_to_definition(node_cls)

    async def get_all_definitions(
        self,
        include_errors: bool = Query(
            False, description="包含加载失败的节点（前端红色标记 + 错误 tooltip）"
        ),
    ) -> dict:
        """GET /nodes/definitions：所有节点的前端定义。

        v0.2.5 P0-H3：支持 include_errors=true 查询参数，
        返回加载失败的节点列表（供前端显示红色标记 + 错误 tooltip）。

        Args:
            include_errors: 是否包含加载失败的节点。默认 False。

        Returns:
            {"version": "1.0", "nodes": {...}, "categories": {...}}
            include_errors=True 时额外返回 {"load_errors": {...}}
        """
        if self.registry is None:
            return {"version": "1.0", "nodes": {}, "categories": {}}

        nodes_by_type = {}
        for node_type, node_cls in self.registry.list().items():
            nodes_by_type[node_type] = _node_to_definition(node_cls)

        # 收集分类
        categories = {}
        for node_def in nodes_by_type.values():
            cat = node_def["category"]
            if cat not in categories:
                categories[cat] = {
                    "title": _category_title(cat),
                    "color": node_def["color"],
                }

        result: dict[str, Any] = {
            "version": "1.0",
            "nodes": nodes_by_type,
            "categories": categories,
            "count": len(nodes_by_type),
        }
        # v0.2.5 P0-H3：可选返回加载失败的节点
        if include_errors:
            result["load_errors"] = self.registry.list_errors()
            result["error_count"] = len(result["load_errors"])
        return result

    async def list_packages(self) -> dict:
        """GET /nodes/packages：列出已安装的第三方节点包。

        通过扫描 importlib.metadata.distributions() 找出所有声明了
        "aireliable.nodes" entry points 的包。

        Returns:
            {"packages": [{"name", "version", "nodes": [entry_name, ...]}, ...]}
        """
        packages = []
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="aireliable.nodes")
        except Exception as e:
            logger.warning("扫描 entry points 失败：%s", e)
            return {"packages": []}

        # 按 package 分组
        pkg_groups: dict[str, list[str]] = {}
        for ep in eps:
            # 通过 ep.value 找到所属 distribution
            module_path = ep.value.split(":")[0]
            pkg_name = _find_distribution_for_module(module_path)
            if pkg_name is None:
                pkg_name = "unknown"
            pkg_groups.setdefault(pkg_name, []).append(ep.name)

        # 获取每个包的版本
        dist_versions = {d.metadata["Name"]: d.version for d in distributions()}

        for pkg_name, entry_names in pkg_groups.items():
            packages.append({
                "name": pkg_name,
                "version": dist_versions.get(pkg_name, "unknown"),
                "nodes": entry_names,
            })

        return {"packages": packages, "count": len(packages)}

    async def get_package_definitions(self, package_name: str) -> dict:
        """GET /nodes/packages/{package_name}/definitions：节点包的前端定义。

        从已安装的节点包中读取 frontend/node_definitions.json。

        Returns:
            节点包前端定义 dict。
        """
        # 查找包的安装位置
        for dist in distributions():
            if dist.metadata["Name"] != package_name:
                continue
            # 找到包根目录
            files = dist.files or []
            for file in files:
                # 查找 frontend/node_definitions.json
                if str(file).endswith("frontend/node_definitions.json"):
                    # 构造绝对路径
                    full_path = dist.locate_file(file)
                    if Path(full_path).exists():
                        try:
                            data = json.loads(Path(full_path).read_text(encoding="utf-8"))
                            return data
                        except json.JSONDecodeError as e:
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": {
                                    "type": "json_decode_error",
                                    "message": f"节点包 {package_name!r} 的 node_definitions.json 解析失败：{e}",
                                }},
                            )

        # 在 node_packages_dir 查找（备选）
        if self.node_packages_dir is not None:
            pkg_dir = self.node_packages_dir / package_name
            definitions_path = pkg_dir / "frontend" / "node_definitions.json"
            if definitions_path.exists():
                try:
                    return json.loads(definitions_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail={"error": {
                            "type": "json_decode_error",
                            "message": f"节点包 {package_name!r} 的 node_definitions.json 解析失败：{e}",
                        }},
                    )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "type": "package_not_found",
                    "message": f"节点包 {package_name!r} 未安装或未提供前端定义",
                }
            },
        )


# ===========================================================================
# 工具函数
# ===========================================================================

def _category_title(category: str) -> str:
    """分类标题（中文）。"""
    titles = {
        "input": "输入",
        "loader": "加载器",
        "encode": "编码器",
        "sampler": "采样器",
        "decode": "解码器",
        "lora": "LoRA",
        "output": "输出",
        "custom": "自定义",
    }
    return titles.get(category, category)


def _find_distribution_for_module(module_path: str) -> Optional[str]:
    """根据模块路径查找所属的 distribution 名。

    Args:
        module_path: 如 "ipadapter_plus_nodes.ipadapter_apply"

    Returns:
        distribution 名（如 "aireliable-node-ipadapter-plus"），找不到返回 None。
    """
    top_package = module_path.split(".")[0]
    for dist in distributions():
        # 检查 top-level packages
        top_level = dist.read_text("top_level.txt")
        if top_level and top_package in top_level.split():
            return dist.metadata["Name"]
        # 检查 requires
        if dist.metadata["Name"] and top_package.replace("_", "-") in (
            dist.metadata["Name"].lower()
        ):
            return dist.metadata["Name"]
    return None


# ===========================================================================
# 工厂函数
# ===========================================================================

def create_node_router(
    registry: Optional[NodeRegistry] = None,
    node_packages_dir: Optional[Path] = None,
    prefix: str = "",
) -> NodeRouter:
    """创建节点定义查询路由。"""
    return NodeRouter(
        registry=registry,
        node_packages_dir=node_packages_dir,
        prefix=prefix,
    )


__all__ = [
    "NodeRouter",
    "create_node_router",
]

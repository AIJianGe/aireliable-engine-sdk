"""ImagesInput 节点：多图输入核心节点（v0.2）。

解决 ComfyUI 硬伤：一个工作流不能同时支持 1/N 张图输入。
本节点输出端口 images 为 variadic（Images 类型），长度 0~N，
下游节点（IPAdapter / ControlNet / 图生图）按数组消费。

数据来源（按优先级）：
    1. ctx.extra["__infer_request__"].images （完整 InferRequest 对象传递）
    2. ctx.inputs["images"]                  （直接 list[ImageInput]）
    3. ctx.inputs["images"]                  （list[dict]，自动转 ImageInput）
    4. params.image_paths                    （画布调试用本地图片路径列表）

支持三种图片来源：
    - url：HTTP/HTTPS 下载（异步 httpx，固定 30s 超时）
    - b64_json：base64 解码
    - path：本地路径加载（PIL）

v0.2.5 简化：删除 role_filter / max_count / download_timeout / max_image_size /
accept_roles 5 个 advanced 参数，节点仅保留 image_paths 一个画布调试参数。
API 调用时通过 ctx.inputs 注入图片列表，无需 advanced 参数。

"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

from ...node import BaseNode, PortSpec
from ...node_registry import register_node
from ...schemas import ImageInput
from ...types import Image, Images
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

# v0.2.5 简化：URL 下载超时硬编码（原 download_timeout 参数已删除）
_DOWNLOAD_TIMEOUT: float = 30.0


@register_node
class ImagesInputNode(BaseNode):
    """多图输入节点（v0.2 核心，v0.2.5 简化）。

    特性：
        - 输出端口 images 为 variadic（Images 类型），长度 0~N
        - 单图输入时数组长度 = 1
        - 无图输入时数组长度 = 0（下游节点按需跳过）
        - 工作流 yaml 不需要为不同图片数量复制多份

    参数（params）：
        image_paths (list[str])：画布调试用本地图片路径列表。
            format: "file-paths"
            优先级：ctx.inputs["images"] > params.image_paths
            画布调试时用户可在节点上或属性面板选择本地图片，
            无需走 API 注入流程。

    v0.2.5 简化说明：
        原有的 role_filter / max_count / download_timeout / max_image_size /
        accept_roles 5 个 advanced 参数已删除。节点职责聚焦于"图片输入"。
        如需 role 过滤或数量限制，由调用方在 API 请求中预先处理。
    """

    node_type = "ImagesInput"
    # v0.2.5 前端元数据（供 NodeRouter 暴露给画布 UI）
    title = "图片输入"
    icon = "🖼️"
    description = "多图输入节点（支持 URL/base64/本地路径 + 画布调试）"
    # v0.2.5 P-CategoryFix：显式声明分类（与前端 NODE_CATEGORIES 对齐）
    category = "input"
    subcategory = "images"
    # v0.2.5 简化：仅保留 image_paths 一个参数（画布调试用）
    params_schema = {
        "image_paths": {
            "type": "array", "required": False, "default": [], "label": "本地图片路径",
            "format": "file-paths",
            "description": "v0.2.5 画布调试用本地图片路径列表（优先级低于 ctx.inputs，留空则从 API 注入）",
        },
    }
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="images",
            type="images",
            variadic=True,
            description="输入图片数组，长度 0~N",
        )
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # v0.2.5 P0-G2：image_paths 字段（画布调试用）
        self._image_paths: list[str] = self.get_param("image_paths", []) or []

    def validate(self) -> list[str]:
        """参数校验。"""
        # v0.2.5 简化：仅 image_paths 一个参数，无需复杂校验
        return []

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载所有图片并返回 Images。

        流程：
            1. 从 ctx 收集 ImageInput 列表（API 注入优先，回退到 params.image_paths）
            2. 逐张加载（URL/b64/path）
            3. 上报进度
            4. 返回 {"images": Images(items=[...])}
        """
        image_inputs = self._collect_image_inputs(ctx)

        await ctx.report_progress(
            0.0,
            stage="images_input.start",
            message=f"准备加载 {len(image_inputs)} 张图片",
        )

        loaded_images: list[Image] = []
        for idx, image_input in enumerate(image_inputs):
            ctx.raise_if_cancelled()
            try:
                image = await self._load_single_image(image_input, ctx)
                if image is not None:
                    loaded_images.append(image)
                    logger.debug(
                        "ImagesInput: 已加载第 %d/%d 张图 (source=%s)",
                        idx + 1,
                        len(image_inputs),
                        self._describe_source(image_input),
                    )
            except Exception as exc:
                logger.warning(
                    "ImagesInput: 第 %d 张图加载失败 (%s): %s",
                    idx + 1,
                    self._describe_source(image_input),
                    exc,
                )
                await ctx.report_info(
                    f"第 {idx + 1} 张图加载失败: {exc}",
                    stage="images_input.warn",
                    index=idx,
                    source=self._describe_source(image_input),
                )

            progress = (idx + 1) / max(len(image_inputs), 1)
            await ctx.report_progress(
                progress,
                stage="images_input.loading",
                message=f"已加载 {idx + 1}/{len(image_inputs)}",
            )

        images = Images(items=loaded_images)
        await ctx.report_progress(
            1.0,
            stage="images_input.done",
            message=f"加载完成，共 {len(images)} 张图",
        )
        return {"images": images}

    def _collect_image_inputs(self, ctx: WorkflowContext) -> list[ImageInput]:
        """从 ctx 收集 ImageInput 列表。

        v0.2.5 P0-G3 双轨读取优先级：
            1. ctx.extra["__infer_request__"].images （完整 InferRequest 传递）
            2. ctx.inputs["images"]                  （直接 list[ImageInput] 或 list[dict]）
            3. params.image_paths                    （画布调试用本地路径列表）

        v0.2.5 简化：删除 role_filter / accept_roles 过滤逻辑，
        调用方在 API 请求中预先过滤。
        """
        # 优先级 1：完整 InferRequest 传递
        request_obj = ctx.extra.get("__infer_request__") if ctx.extra else None
        if request_obj is not None and hasattr(request_obj, "images"):
            images_field = getattr(request_obj, "images", None)
            if images_field:
                return list(images_field)

        # 优先级 2：ctx.inputs["images"]
        raw_images = ctx.inputs.get("images") if ctx.inputs else None
        if raw_images:
            if isinstance(raw_images, Images):
                # 已经是 Images 包装类（来自上游节点输出）
                return []
            if not isinstance(raw_images, list):
                raw_images = [raw_images]

            result: list[ImageInput] = []
            for item in raw_images:
                if isinstance(item, ImageInput):
                    result.append(item)
                elif isinstance(item, dict):
                    result.append(ImageInput(**item))
                elif isinstance(item, str):
                    # 纯字符串：按 URL 处理
                    result.append(ImageInput(url=item))
                else:
                    logger.warning("ImagesInput: 跳过无法识别的图片输入 %r", type(item))
            if result:
                return result

        # v0.2.5 P0-G3：优先级 3 — params.image_paths（画布调试用本地路径列表）
        if self._image_paths:
            result = []
            for path in self._image_paths:
                if isinstance(path, str) and path:
                    result.append(ImageInput(path=path))
            if result:
                logger.info(
                    "ImagesInput: 使用 params.image_paths 加载 %d 张本地图片",
                    len(result),
                )
            return result

        return []

    async def _load_single_image(
        self, image_input: ImageInput, ctx: WorkflowContext
    ) -> Optional[Image]:
        """加载单张图片：URL 下载 / b64 解码 / 本地路径。

        优先级：path > b64_json > url（path 是最便宜的，无 IO）。
        """
        if image_input.path:
            return await self._load_from_path(image_input.path, image_input.role)
        if image_input.b64_json:
            return self._load_from_b64(image_input.b64_json, image_input.role)
        if image_input.url:
            return await self._load_from_url(image_input.url, image_input.role, ctx)
        logger.warning("ImagesInput: ImageInput 三字段全为空，跳过")
        return None

    async def _load_from_path(
        self, path: str, role: Optional[str] = None
    ) -> Optional[Image]:
        """从本地路径或 dataURL 加载图片。

        v0.2.5 浏览器兼容：path 可以是 dataURL（data:image/...;base64,...），
        此时提取 base64 部分用 _load_from_b64 加载。
        """
        # dataURL 检测（浏览器环境选的图）
        if path.startswith("data:"):
            # 提取 base64 部分：data:image/png;base64,<data>
            header, _, b64_data = path.partition(",")
            if not b64_data:
                raise ValueError("dataURL 格式无效：缺少 base64 数据")
            return self._load_from_b64(b64_data, role)

        from pathlib import Path

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"图片路径不存在: {path}")
        if not path_obj.is_file():
            raise ValueError(f"路径不是文件: {path}")
        # PIL 延迟导入，避免无 PIL 环境下 import 报错
        from PIL import Image as PILImage

        with PILImage.open(path_obj) as pil_img:
            pil_img.load()
        metadata = {"role": role} if role else {}
        return Image(pil=pil_img.copy(), path=path_obj, metadata=metadata)

    def _load_from_b64(
        self, b64_json: str, role: Optional[str] = None
    ) -> Optional[Image]:
        """从 base64 字符串加载图片。"""
        try:
            image_bytes = base64.b64decode(b64_json)
        except Exception as exc:
            raise ValueError(f"base64 解码失败: {exc}") from exc
        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(image_bytes))
        pil_img.load()
        metadata = {"role": role} if role else {}
        return Image(pil=pil_img, metadata=metadata)

    async def _load_from_url(
        self,
        url: str,
        role: Optional[str],
        ctx: WorkflowContext,
    ) -> Optional[Image]:
        """从 URL 下载图片。"""
        # httpx 延迟导入，避免无网络环境或单元测试环境报错
        import httpx

        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            image_bytes = response.content

        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(image_bytes))
        pil_img.load()
        metadata = {"role": role, "source_url": url} if role else {"source_url": url}
        return Image(pil=pil_img, metadata=metadata)

    @staticmethod
    def _describe_source(image_input: ImageInput) -> str:
        """生成 ImageInput 的可读描述（用于日志）。"""
        if image_input.path:
            return f"path={image_input.path}"
        if image_input.b64_json:
            return f"b64_json(len={len(image_input.b64_json)})"
        if image_input.url:
            return f"url={image_input.url}"
        return "empty"

    def __repr__(self) -> str:
        return f"ImagesInputNode(image_paths_count={len(self._image_paths)})"

"""ImagesInput 节点：多图输入核心节点（v0.2）。

解决 ComfyUI 硬伤：一个工作流不能同时支持 1/N 张图输入。
本节点输出端口 images 为 variadic（Images 类型），长度 0~N，
下游节点（IPAdapter / ControlNet / 图生图）按数组消费。

数据来源（按优先级）：
    1. ctx.extra["__infer_request__"].images （完整 InferRequest 对象传递）
    2. ctx.inputs["images"]                  （直接 list[ImageInput]）
    3. ctx.inputs["images"]                  （list[dict]，自动转 ImageInput）

支持三种图片来源：
    - url：HTTP/HTTPS 下载（异步 httpx）
    - b64_json：base64 解码
    - path：本地路径加载（PIL）

参考：
    - v0.2-analysis.md §5.2.3 多图输入请求体
    - PR/architecture.md §12.2.4 多图输入节点
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

from ...node import BaseNode, PortSpec
from ...schemas import ImageInput
from ...types import Image, Images
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class ImagesInputNode(BaseNode):
    """多图输入节点（v0.2 核心）。

    特性：
        - 输出端口 images 为 variadic（Images 类型），长度 0~N
        - 单图输入时数组长度 = 1
        - 无图输入时数组长度 = 0（下游节点按需跳过）
        - 工作流 yaml 不需要为不同图片数量复制多份

    参数（params）：
        role_filter (Optional[str])：按 role 字段过滤，None=全部。
            示例：role_filter="reference" 只取参考图。
        max_count (Optional[int])：限制最大张数，None=不限制。
        download_timeout (float)：URL 下载超时秒数，默认 30.0。
        max_image_size (Optional[int])：单张图片最大像素（宽 * 高），
            超过则按比例缩小，None=不限制。
        accept_roles (Optional[list[str]])：白名单 role 列表，
            与 role_filter 互斥（同时设置时 role_filter 优先）。
    """

    node_type = "ImagesInput"
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
        self._role_filter: Optional[str] = self.get_param("role_filter")
        self._max_count: Optional[int] = self.get_param("max_count")
        self._download_timeout: float = float(self.get_param("download_timeout", 30.0))
        self._max_image_size: Optional[int] = self.get_param("max_image_size")
        self._accept_roles: Optional[list[str]] = self.get_param("accept_roles")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        max_count = self._max_count
        if max_count is not None and (not isinstance(max_count, int) or max_count < 0):
            errors.append(f"max_count 必须为非负整数，当前值：{max_count}")
        if self._download_timeout <= 0:
            errors.append(f"download_timeout 必须为正数，当前值：{self._download_timeout}")
        if self._max_image_size is not None and (
            not isinstance(self._max_image_size, int) or self._max_image_size <= 0
        ):
            errors.append(f"max_image_size 必须为正整数，当前值：{self._max_image_size}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载所有图片并返回 Images。

        流程：
            1. 从 ctx 获取 ImageInput 列表
            2. 按 role 过滤
            3. 限制最大数量
            4. 逐张加载（URL/b64/path）
            5. 上报进度
            6. 返回 {"images": Images(items=[...])}
        """
        image_inputs = self._collect_image_inputs(ctx)
        image_inputs = self._filter_by_role(image_inputs)
        image_inputs = self._apply_max_count(image_inputs)

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

        优先级：
            1. ctx.extra["__infer_request__"].images （完整 InferRequest 传递）
            2. ctx.inputs["images"]                  （直接 list[ImageInput] 或 list[dict]）
        """
        # 优先级 1：完整 InferRequest 传递
        request_obj = ctx.extra.get("__infer_request__")
        if request_obj is not None and hasattr(request_obj, "images"):
            images_field = getattr(request_obj, "images", None)
            if images_field:
                return list(images_field)

        # 优先级 2：ctx.inputs["images"]
        raw_images = ctx.inputs.get("images") if ctx.inputs else None
        if not raw_images:
            return []
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
        return result

    def _filter_by_role(self, image_inputs: list[ImageInput]) -> list[ImageInput]:
        """按 role 字段过滤。"""
        # role_filter 优先
        if self._role_filter is not None:
            return [img for img in image_inputs if img.role == self._role_filter]
        # 白名单次之
        if self._accept_roles is not None:
            return [img for img in image_inputs if img.role in self._accept_roles]
        return image_inputs

    def _apply_max_count(self, image_inputs: list[ImageInput]) -> list[ImageInput]:
        """限制最大张数。"""
        if self._max_count is None or self._max_count <= 0:
            return image_inputs
        if len(image_inputs) <= self._max_count:
            return image_inputs
        logger.info(
            "ImagesInput: 图片数量 %d 超过 max_count %d，截断",
            len(image_inputs),
            self._max_count,
        )
        return image_inputs[: self._max_count]

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
        """从本地路径加载图片。"""
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
            pil_img = self._resize_if_needed(pil_img)
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
        pil_img = self._resize_if_needed(pil_img)
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

        async with httpx.AsyncClient(timeout=self._download_timeout) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            image_bytes = response.content

        from PIL import Image as PILImage

        pil_img = PILImage.open(io.BytesIO(image_bytes))
        pil_img.load()
        pil_img = self._resize_if_needed(pil_img)
        metadata = {"role": role, "source_url": url} if role else {"source_url": url}
        return Image(pil=pil_img, metadata=metadata)

    def _resize_if_needed(self, pil_img: Any) -> Any:
        """按 max_image_size 限制图片尺寸（等比缩小）。"""
        if self._max_image_size is None:
            return pil_img
        width, height = pil_img.size
        current_size = width * height
        if current_size <= self._max_image_size:
            return pil_img
        # 等比缩小
        scale = (self._max_image_size / current_size) ** 0.5
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        logger.info(
            "ImagesInput: 图片尺寸 %dx%d 超限，缩小至 %dx%d",
            width,
            height,
            new_width,
            new_height,
        )
        return pil_img.resize((new_width, new_height))

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
        return (
            f"ImagesInputNode(role_filter={self._role_filter!r}, "
            f"max_count={self._max_count!r})"
        )

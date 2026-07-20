"""ImageSave 节点：图片保存节点（v0.2）。

将 Images 包装类保存到 data/files/ 目录，返回图片 URL 列表。
是工作流的最后一个节点，位于 VAEDecode → ImageSave 的末尾。

支持：
    - 多种图片格式：PNG / JPEG / WebP
    - 元数据注入：PNG info（parameters / workflow / 自定义字段）
    - 文件名模板：{task_id} / {index} / {timestamp} / {ext}
    - URL 生成：可配置 base_url 前缀
    - 同步保存 metadata.json（可选）

工作流示例：
    - id: save
      type: ImageSave
      inputs:
        images: vae_decode.output.images

    outputs:
      image_urls: save.output.image_urls

参考：
    - PR/architecture.md §12.2 / §12.4（工作流示例）
    - PR/development-guide.md §28.5（数据类型规范）
    - v0.2-tasks.md 任务 3.8
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

from ...node import BaseNode, NodeValidationError, PortSpec
from ...types import Image, Images
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


# ===========================================================================
# 支持的图片格式映射表
# ===========================================================================

# PIL 支持的格式（save 方法 format 参数）
_FORMAT_MAP: dict[str, str] = {
    "PNG": "PNG",
    "JPEG": "JPEG",
    "JPG": "JPEG",
    "WEBP": "WEBP",
    "WEBPLOSSLESS": "WEBP",
}

# 格式对应的文件扩展名
_EXTENSION_MAP: dict[str, str] = {
    "PNG": "png",
    "JPEG": "jpg",
    "WEBP": "webp",
}

# 支持 quality 参数的格式
_QUALITY_FORMATS = {"JPEG", "WEBP"}

# 支持 lossless 参数的格式
_LOSSLESS_FORMATS = {"WEBP"}

# 支持 optimize 参数的格式
_OPTIMIZE_FORMATS = {"PNG", "JPEG", "WEBP"}


class ImageSaveNode(BaseNode):
    """图片保存节点。

    将 Images 包装类中的所有图片保存到 data/files/ 目录，
    返回每张图片的 URL 列表。

    节点特性：
        - **非抽象节点**：直接使用，引擎子类无需重写
        - **可配置输出路径**：output_dir 参数控制保存位置
        - **元数据注入**：通过 metadata 参数注入 PNG info
        - **URL 生成**：基于 base_url 生成可访问 URL

    参数（params）：
        format (str)：图片格式 "PNG" / "JPEG" / "WEBP"（默认 "PNG"）。
            不区分大小写。
        quality (Optional[int])：JPEG / WEBP 质量（1-95）。
            None=使用 PIL 默认值。PNG 忽略此参数。
        output_dir (Optional[str])：输出目录（绝对路径或相对 output_root）。
            None=output_root/files/（默认）。
        filename_template (str)：文件名模板，支持占位符：
            - {task_id}：任务 ID（默认 "default"）
            - {index}：图片序号（从 0 开始）
            - {timestamp}：当前时间戳（int）
            - {ext}：文件扩展名（不含点）
            默认："{task_id}_{index:03d}.{ext}"
        base_url (Optional[str])：URL 前缀。
            如 "http://localhost:8000/files/"。
            None=返回相对路径（"/files/xxx.png"）。
        metadata (Optional[dict])：额外元数据注入（PNG info / EXIF）。
            仅对 PNG 格式生效，写入 PngInfo。
        include_workflow_info (bool)：是否将 workflow 信息写入 PNG info。
            需要传入的 Image.metadata 包含 "workflow" 字段。默认 False。
        save_metadata_json (bool)：是否同时保存 metadata.json 文件。
            默认 False。
        optimize (bool)：是否优化图片大小。默认 False。
            PNG：使用 optimize=True
            JPEG：使用 optimize=True
            WEBP：使用 method=6
        lossless (bool)：WEBP 无损压缩。默认 False。
    """

    node_type = "ImageSave"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="images",
            type="images",
            required=True,
            variadic=True,
            description="要保存的图片数组（Images 包装类）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="image_urls",
            type="any",
            description="保存后的图片 URL 列表（list[str]）",
        )
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # 格式归一化：JPG → JPEG，统一为大写
        raw_format = str(self.get_param("format", "PNG")).upper()
        self._format: str = _FORMAT_MAP.get(raw_format, raw_format)
        self._quality: Optional[int] = self.get_param("quality")
        self._output_dir: Optional[str] = self.get_param("output_dir")
        self._filename_template: str = self.get_param(
            "filename_template", "{task_id}_{index:03d}.{ext}"
        )
        self._base_url: Optional[str] = self.get_param("base_url")
        self._metadata: Optional[dict] = self.get_param("metadata")
        # 不强制 bool() 转换，保留原值供 validate() 检查类型
        self._include_workflow_info: Any = self.get_param(
            "include_workflow_info", False
        )
        self._save_metadata_json: Any = self.get_param("save_metadata_json", False)
        self._optimize: Any = self.get_param("optimize", False)
        self._lossless: Any = self.get_param("lossless", False)

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if self._format not in _FORMAT_MAP:
            errors.append(
                f"params.format 必须为 {sorted(_FORMAT_MAP.keys())} 之一，"
                f"当前值：{self._format!r}"
            )
        if self._quality is not None and (
            not isinstance(self._quality, int) or self._quality < 1 or self._quality > 100
        ):
            errors.append(
                f"params.quality 必须为 1-100 整数或 None，当前值：{self._quality}"
            )
        pil_format = _FORMAT_MAP.get(self._format)
        if pil_format and self._quality is not None and pil_format not in _QUALITY_FORMATS:
            errors.append(
                f"params.quality 仅对 {sorted(_QUALITY_FORMATS)} 格式生效，"
                f"当前格式：{pil_format}"
            )
        if not isinstance(self._filename_template, str) or not self._filename_template:
            errors.append(
                f"params.filename_template 必须为非空字符串，"
                f"当前值：{self._filename_template!r}"
            )
        else:
            # 仅当 template 是字符串时才校验占位符
            template_errors = self._validate_filename_template(self._filename_template)
            errors.extend(template_errors)
        # bool 类型校验
        if not isinstance(self._include_workflow_info, bool):
            errors.append(
                f"params.include_workflow_info 必须为 bool 类型，"
                f"当前值：{self._include_workflow_info!r}"
            )
        if not isinstance(self._save_metadata_json, bool):
            errors.append(
                f"params.save_metadata_json 必须为 bool 类型，"
                f"当前值：{self._save_metadata_json!r}"
            )
        if not isinstance(self._optimize, bool):
            errors.append(
                f"params.optimize 必须为 bool 类型，当前值：{self._optimize!r}"
            )
        if not isinstance(self._lossless, bool):
            errors.append(
                f"params.lossless 必须为 bool 类型，当前值：{self._lossless!r}"
            )
        if self._lossless and pil_format and pil_format not in _LOSSLESS_FORMATS:
            errors.append(
                f"params.lossless 仅对 {sorted(_LOSSLESS_FORMATS)} 格式生效，"
                f"当前格式：{pil_format}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：保存所有图片到磁盘并返回 URL 列表。

        流程：
            1. 参数校验
            2. 提取 images 输入
            3. 解析输出目录（output_root 优先）
            4. 创建目录
            5. 逐张保存图片，注入 PNG info
            6. 上报进度
            7. 生成 URL 列表
            8. 可选保存 metadata.json
            9. 返回 {"image_urls": ["...", ...]}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        images_value = inputs.get("images")
        if images_value is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.images 必填（ImageSave 需要至少一张图片）"],
            )

        # 标准化为 Images
        images = self._normalize_images_input(images_value)
        if len(images) == 0:
            raise NodeValidationError(
                self.node_type,
                ["inputs.images 不能为空（ImageSave 需要至少一张图片）"],
            )

        # 解析输出目录
        output_dir = self._resolve_output_dir(ctx)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 任务标识
        task_id = ctx.task_id or "default"
        timestamp = int(time.time())
        ext = _EXTENSION_MAP[_FORMAT_MAP[self._format]]

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="image_save.start",
            message=f"开始保存 {len(images)} 张图片",
            image_count=len(images),
            format=self._format,
            output_dir=str(output_dir),
        )

        # 逐张保存
        image_urls: list[str] = []
        pil_format = _FORMAT_MAP[self._format]
        for index, image in enumerate(images):
            ctx.raise_if_cancelled()

            filename = self._render_filename(
                template=self._filename_template,
                task_id=task_id,
                index=index,
                timestamp=timestamp,
                ext=ext,
            )
            file_path = output_dir / filename

            # 构造 PNG info（仅 PNG 格式）
            png_info = self._build_png_info(image, pil_format)

            # 构造保存参数
            save_kwargs = self._build_save_kwargs(pil_format, png_info)

            try:
                pil_image = image.pil
                # 模式转换：JPEG 不支持 RGBA / P
                if pil_format == "JPEG" and pil_image.mode in {"RGBA", "P", "LA"}:
                    pil_image = pil_image.convert("RGB")
                elif pil_format == "WEBP" and pil_image.mode == "P":
                    pil_image = pil_image.convert("RGBA" if "transparency" in image.pil.info else "RGB")

                pil_image.save(file_path, format=pil_format, **save_kwargs)
                logger.info("ImageSave: 已保存图片 %s", file_path)
            except Exception as exc:
                logger.exception("ImageSave: 保存图片 %s 失败", file_path)
                raise RuntimeError(f"保存图片 {filename} 失败: {exc}") from exc

            # 生成 URL
            url = self._build_url(file_path, filename)
            image_urls.append(url)

            # 可选：保存 metadata.json
            if isinstance(self._save_metadata_json, bool) and self._save_metadata_json:
                self._save_metadata_file(file_path, image, task_id, index, timestamp)

            # 上报进度
            progress = (index + 1) / len(images)
            await ctx.report_progress(
                progress,
                stage="image_save.progress",
                message=f"已保存 {index + 1}/{len(images)}",
                index=index,
                total=len(images),
                filename=filename,
            )

        await ctx.report_progress(
            1.0,
            stage="image_save.done",
            message=f"保存完成，共 {len(image_urls)} 张图片",
            image_count=len(image_urls),
        )
        return {"image_urls": image_urls}

    # ===== 内部辅助方法 =====

    def _normalize_images_input(self, value: Any) -> Images:
        """将输入标准化为 Images 包装类。"""
        if isinstance(value, Images):
            return value
        if isinstance(value, Image):
            return Images.from_single(value)
        if isinstance(value, list):
            for i, item in enumerate(value):
                if not isinstance(item, Image):
                    raise NodeValidationError(
                        self.node_type,
                        [
                            f"inputs.images list[{i}] 不是 Image 类型，"
                            f"实际类型：{type(item).__name__}"
                        ],
                    )
            return Images.from_list(value)
        raise NodeValidationError(
            self.node_type,
            [
                f"inputs.images 必须为 Images / list[Image] / Image，"
                f"实际类型：{type(value).__name__}"
            ],
        )

    def _resolve_output_dir(self, ctx: WorkflowContext) -> Path:
        """解析输出目录。

        优先级：
            1. output_dir 参数（绝对路径直接用，相对路径相对 output_root）
            2. output_root / "files"（默认）
            3. ctx.output_root 为 None 时使用 cwd / "data" / "files"
        """
        if self._output_dir:
            output_dir = Path(self._output_dir)
            if not output_dir.is_absolute() and ctx.output_root is not None:
                output_dir = ctx.output_root / output_dir
            return output_dir

        if ctx.output_root is not None:
            return ctx.output_root / "files"

        # 兜底：当前工作目录
        return Path.cwd() / "data" / "files"

    def _validate_filename_template(self, template: str) -> list[str]:
        """校验文件名模板占位符。"""
        errors: list[str] = []
        try:
            # 试渲染一次，捕获未知占位符
            template.format(
                task_id="test",
                index=0,
                timestamp=0,
                ext="png",
            )
        except KeyError as exc:
            errors.append(
                f"params.filename_template 包含未知占位符 {exc}，"
                f"仅支持 {{task_id}} / {{index}} / {{timestamp}} / {{ext}}"
            )
        except (ValueError, IndexError) as exc:
            errors.append(
                f"params.filename_template 格式错误：{exc}"
            )
        return errors

    def _render_filename(
        self,
        template: str,
        task_id: str,
        index: int,
        timestamp: int,
        ext: str,
    ) -> str:
        """渲染文件名模板。"""
        return template.format(
            task_id=task_id,
            index=index,
            timestamp=timestamp,
            ext=ext,
        )

    def _build_png_info(self, image: Image, pil_format: str) -> Optional[PngInfo]:
        """构造 PNG info（仅 PNG 格式）。

        合并来源：
            1. image.metadata（Image 自带的元数据，"workflow" 键除外）
            2. self._metadata（节点参数指定的额外元数据）
            3. workflow 信息（若 include_workflow_info=True 才注入）
        """
        if pil_format != "PNG":
            return None

        png_info = PngInfo()
        has_content = False
        include_workflow = (
            isinstance(self._include_workflow_info, bool) and self._include_workflow_info
        )

        # 1. image.metadata（"workflow" 键仅在 include_workflow_info=True 时注入）
        if image.metadata:
            for key, value in image.metadata.items():
                if key in {"exif", "original_url"}:
                    continue
                if key == "workflow" and not include_workflow:
                    continue
                png_info.add_text(str(key), self._stringify_metadata_value(value))
                has_content = True

        # 2. self._metadata（节点参数）
        if isinstance(self._metadata, dict):
            for key, value in self._metadata.items():
                png_info.add_text(str(key), self._stringify_metadata_value(value))
                has_content = True

        return png_info if has_content else None

    def _stringify_metadata_value(self, value: Any) -> str:
        """将元数据值转换为字符串（PngInfo 要求 str）。"""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False, default=str)

    def _build_save_kwargs(
        self,
        pil_format: str,
        png_info: Optional[PngInfo],
    ) -> dict:
        """构造 PIL Image.save 的额外参数。"""
        kwargs: dict[str, Any] = {}

        if self._quality is not None and pil_format in _QUALITY_FORMATS:
            kwargs["quality"] = self._quality

        if (
            isinstance(self._optimize, bool)
            and self._optimize
            and pil_format in _OPTIMIZE_FORMATS
        ):
            kwargs["optimize"] = True

        if (
            isinstance(self._lossless, bool)
            and self._lossless
            and pil_format in _LOSSLESS_FORMATS
        ):
            kwargs["lossless"] = True

        if png_info is not None:
            kwargs["pnginfo"] = png_info

        return kwargs

    def _build_url(self, file_path: Path, filename: str) -> str:
        """生成图片 URL。"""
        if self._base_url is not None:
            # 用户指定 base_url，拼接 filename
            base = self._base_url
            if not base.endswith("/"):
                base = base + "/"
            return base + filename

        # 默认：返回相对路径 /files/filename
        return f"/files/{filename}"

    def _save_metadata_file(
        self,
        image_path: Path,
        image: Image,
        task_id: str,
        index: int,
        timestamp: int,
    ) -> None:
        """保存 metadata.json 文件（与图片同目录）。"""
        metadata_path = image_path.with_suffix(".json")
        metadata = {
            "task_id": task_id,
            "index": index,
            "timestamp": timestamp,
            "filename": image_path.name,
            "image_metadata": image.metadata or {},
            "extra_metadata": self._metadata or {},
        }
        try:
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("ImageSave: 已保存元数据 %s", metadata_path)
        except Exception:
            logger.warning("ImageSave: 保存元数据失败 %s", metadata_path, exc_info=True)

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：ImageSave 不缓存（每次都需要保存新文件）。"""
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(format={self._format!r}, "
            f"quality={self._quality!r}, "
            f"optimize={self._optimize!r}, "
            f"lossless={self._lossless!r})"
        )

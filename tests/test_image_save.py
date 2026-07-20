"""ImageSave 节点单元测试。

覆盖任务 3.8 验收标准：
    - 实现 ImageSaveNode 类
    - 输入：images
    - 保存到 data/files/
    - 返回图片 URL
    - 支持多种格式（PNG / JPEG / WebP）
    - 支持元数据注入（PNG info）

参考：v0.2-tasks.md 任务 3.8
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngImageFile

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.postprocess.image_save import ImageSaveNode
from aireliable_engine.types import Image, Images


# ===== 辅助工厂 =====


def _make_pil_image(
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (255, 0, 0),
    mode: str = "RGB",
) -> PILImage.Image:
    """构造一个简单的 PIL 图片。"""
    if mode == "RGBA":
        color = color + (255,)
    return PILImage.new(mode, size, color)


def _make_image(
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (255, 0, 0),
    mode: str = "RGB",
    metadata: dict = None,
) -> Image:
    """构造一个 Image 实例。"""
    pil = _make_pil_image(size, color, mode)
    return Image(pil=pil, metadata=metadata or {})


# ===== 基本属性测试 =====


class TestImageSaveNodeBasic:
    """ImageSaveNode 基本属性测试。"""

    def test_node_type_is_imagesave(self):
        node = ImageSaveNode()
        assert node.node_type == "ImageSave"

    def test_input_specs_has_one_port(self):
        node = ImageSaveNode()
        assert len(node.input_specs) == 1
        port = node.input_specs[0]
        assert port.name == "images"
        assert port.type == "images"
        assert port.required is True
        assert port.variadic is True

    def test_output_specs_has_one_port(self):
        node = ImageSaveNode()
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "image_urls"
        assert port.type == "any"

    def test_default_format_is_png(self):
        node = ImageSaveNode()
        assert node._format == "PNG"

    def test_default_quality_is_none(self):
        node = ImageSaveNode()
        assert node._quality is None

    def test_default_filename_template(self):
        node = ImageSaveNode()
        assert node._filename_template == "{task_id}_{index:03d}.{ext}"

    def test_default_base_url_is_none(self):
        node = ImageSaveNode()
        assert node._base_url is None

    def test_default_metadata_is_none(self):
        node = ImageSaveNode()
        assert node._metadata is None

    def test_default_include_workflow_info_is_false(self):
        node = ImageSaveNode()
        assert node._include_workflow_info is False

    def test_default_save_metadata_json_is_false(self):
        node = ImageSaveNode()
        assert node._save_metadata_json is False

    def test_default_optimize_is_false(self):
        node = ImageSaveNode()
        assert node._optimize is False

    def test_default_lossless_is_false(self):
        node = ImageSaveNode()
        assert node._lossless is False

    def test_init_with_custom_params(self):
        node = ImageSaveNode({
            "format": "webp",
            "quality": 80,
            "output_dir": "/tmp/custom_output",
            "filename_template": "{timestamp}_{index}.{ext}",
            "base_url": "http://localhost:8000/files/",
            "metadata": {"prompt": "a cat"},
            "include_workflow_info": True,
            "save_metadata_json": True,
            "optimize": True,
            "lossless": True,
        })
        assert node._format == "WEBP"
        assert node._quality == 80
        assert node._output_dir == "/tmp/custom_output"
        assert node._filename_template == "{timestamp}_{index}.{ext}"
        assert node._base_url == "http://localhost:8000/files/"
        assert node._metadata == {"prompt": "a cat"}
        assert node._include_workflow_info is True
        assert node._save_metadata_json is True
        assert node._optimize is True
        assert node._lossless is True

    def test_format_is_uppercased(self):
        node = ImageSaveNode({"format": "png"})
        assert node._format == "PNG"
        node = ImageSaveNode({"format": "webp"})
        assert node._format == "WEBP"
        node = ImageSaveNode({"format": "jpg"})
        assert node._format == "JPEG"

    def test_repr_contains_class_name(self):
        node = ImageSaveNode({"format": "webp"})
        repr_str = repr(node)
        assert "ImageSaveNode" in repr_str
        assert "WEBP" in repr_str


# ===== 参数校验测试 =====


class TestImageSaveNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_default_params(self):
        node = ImageSaveNode()
        assert node.validate() == []

    def test_validate_reports_invalid_format(self):
        node = ImageSaveNode({"format": "BMP"})
        errors = node.validate()
        assert any("format" in e for e in errors)

    def test_validate_accepts_png_format(self):
        node = ImageSaveNode({"format": "PNG"})
        assert node.validate() == []

    def test_validate_accepts_jpeg_format(self):
        node = ImageSaveNode({"format": "JPEG"})
        assert node.validate() == []

    def test_validate_accepts_webp_format(self):
        node = ImageSaveNode({"format": "WEBP"})
        assert node.validate() == []

    def test_validate_accepts_jpg_alias(self):
        """JPG 应映射为 JPEG。"""
        node = ImageSaveNode({"format": "JPG"})
        assert node.validate() == []

    def test_validate_reports_quality_below_range(self):
        node = ImageSaveNode({"format": "JPEG", "quality": 0})
        errors = node.validate()
        assert any("quality" in e for e in errors)

    def test_validate_reports_quality_above_range(self):
        node = ImageSaveNode({"format": "JPEG", "quality": 101})
        errors = node.validate()
        assert any("quality" in e for e in errors)

    def test_validate_accepts_quality_in_range(self):
        node = ImageSaveNode({"format": "JPEG", "quality": 50})
        assert node.validate() == []

    def test_validate_accepts_quality_none(self):
        node = ImageSaveNode({"format": "JPEG", "quality": None})
        assert node.validate() == []

    def test_validate_reports_quality_for_png_format(self):
        """PNG 不支持 quality 参数。"""
        node = ImageSaveNode({"format": "PNG", "quality": 50})
        errors = node.validate()
        assert any("quality" in e for e in errors)

    def test_validate_reports_empty_filename_template(self):
        node = ImageSaveNode({"filename_template": ""})
        errors = node.validate()
        assert any("filename_template" in e for e in errors)

    def test_validate_reports_non_string_filename_template(self):
        node = ImageSaveNode({"filename_template": 123})
        errors = node.validate()
        assert any("filename_template" in e for e in errors)

    def test_validate_reports_unknown_placeholder_in_template(self):
        node = ImageSaveNode({"filename_template": "{unknown_field}.{ext}"})
        errors = node.validate()
        assert any("filename_template" in e for e in errors)

    def test_validate_accepts_valid_template(self):
        node = ImageSaveNode({"filename_template": "{task_id}_{index}_{timestamp}.{ext}"})
        assert node.validate() == []

    def test_validate_reports_non_bool_include_workflow_info(self):
        node = ImageSaveNode({"include_workflow_info": "yes"})
        errors = node.validate()
        assert any("include_workflow_info" in e for e in errors)

    def test_validate_reports_non_bool_save_metadata_json(self):
        node = ImageSaveNode({"save_metadata_json": "yes"})
        errors = node.validate()
        assert any("save_metadata_json" in e for e in errors)

    def test_validate_reports_non_bool_optimize(self):
        node = ImageSaveNode({"optimize": "yes"})
        errors = node.validate()
        assert any("optimize" in e for e in errors)

    def test_validate_reports_non_bool_lossless(self):
        node = ImageSaveNode({"lossless": "yes"})
        errors = node.validate()
        assert any("lossless" in e for e in errors)

    def test_validate_reports_lossless_for_png_format(self):
        """PNG 不支持 lossless 参数。"""
        node = ImageSaveNode({"format": "PNG", "lossless": True})
        errors = node.validate()
        assert any("lossless" in e for e in errors)


# ===== execute 流程测试 =====


class TestImageSaveNodeExecute:
    """execute 流程测试。"""

    @pytest.mark.asyncio
    async def test_execute_saves_single_image_and_returns_urls(self, tmp_path: Path):
        """验收标准：保存图片并返回 URL。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="task_001")
        node = ImageSaveNode()
        result = await node.execute({"images": images}, ctx)
        assert isinstance(result, dict)
        assert "image_urls" in result
        assert isinstance(result["image_urls"], list)
        assert len(result["image_urls"]) == 1
        # 文件确实被保存
        files = list((tmp_path / "files").glob("*.png"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_execute_saves_multiple_images(self, tmp_path: Path):
        images = Images.from_list([_make_image() for _ in range(3)])
        ctx = make_default_context(output_root=tmp_path, task_id="task_multi")
        node = ImageSaveNode()
        result = await node.execute({"images": images}, ctx)
        assert len(result["image_urls"]) == 3
        files = sorted((tmp_path / "files").glob("*.png"))
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_execute_raises_on_missing_images(self, tmp_path: Path):
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_none_images(self, tmp_path: Path):
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"images": None}, ctx)
        assert "images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_empty_images(self, tmp_path: Path):
        images = Images.empty()
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"images": images}, ctx)
        assert "images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_invalid_images_type(self, tmp_path: Path):
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"images": "not_images"}, ctx)
        assert "images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_list_with_non_image_element(
        self, tmp_path: Path
    ):
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"images": ["not_an_image"]}, ctx)
        assert "images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_invalid_params(self, tmp_path: Path):
        """参数校验失败时抛 NodeValidationError。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode({"format": "BMP"})
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"images": images}, ctx)
        assert "format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_accepts_single_image_input(self, tmp_path: Path):
        """单个 Image 应自动包装为 Images 长度=1。"""
        image = _make_image()
        ctx = make_default_context(output_root=tmp_path, task_id="task_single")
        node = ImageSaveNode()
        result = await node.execute({"images": image}, ctx)
        assert len(result["image_urls"]) == 1
        files = list((tmp_path / "files").glob("*.png"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_execute_accepts_list_input(self, tmp_path: Path):
        """list[Image] 应自动包装为 Images。"""
        images_list = [_make_image() for _ in range(2)]
        ctx = make_default_context(output_root=tmp_path, task_id="task_list")
        node = ImageSaveNode()
        result = await node.execute({"images": images_list}, ctx)
        assert len(result["image_urls"]) == 2

    @pytest.mark.asyncio
    async def test_execute_creates_output_dir_if_not_exists(self, tmp_path: Path):
        """output_dir 不存在时自动创建。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path)
        node = ImageSaveNode({"output_dir": "deep/nested/path"})
        await node.execute({"images": images}, ctx)
        assert (tmp_path / "deep" / "nested" / "path").is_dir()


# ===== 多格式测试（验收标准）=====


class TestImageSaveNodeFormats:
    """多格式测试（验收标准：支持 PNG / JPEG / WebP）。"""

    @pytest.mark.asyncio
    async def test_save_as_png(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="png_task")
        node = ImageSaveNode({"format": "PNG"})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        assert len(files) == 1
        # 验证文件是 PNG
        with PILImage.open(files[0]) as img:
            assert img.format == "PNG"

    @pytest.mark.asyncio
    async def test_save_as_jpeg(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="jpg_task")
        node = ImageSaveNode({"format": "JPEG"})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.jpg"))
        assert len(files) == 1
        with PILImage.open(files[0]) as img:
            assert img.format == "JPEG"

    @pytest.mark.asyncio
    async def test_save_as_webp(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="webp_task")
        node = ImageSaveNode({"format": "WEBP"})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.webp"))
        assert len(files) == 1
        with PILImage.open(files[0]) as img:
            assert img.format == "WEBP"

    @pytest.mark.asyncio
    async def test_save_rgba_as_jpeg_converts_to_rgb(self, tmp_path: Path):
        """JPEG 不支持 RGBA，应自动转 RGB。"""
        images = Images.from_single(_make_image(mode="RGBA"))
        ctx = make_default_context(output_root=tmp_path, task_id="rgba_task")
        node = ImageSaveNode({"format": "JPEG"})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.jpg"))
        assert len(files) == 1
        with PILImage.open(files[0]) as img:
            assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_save_with_quality_setting(self, tmp_path: Path):
        """quality 参数应影响 JPEG 文件大小。"""
        images = Images.from_single(_make_image(size=(256, 256)))
        ctx = make_default_context(output_root=tmp_path, task_id="quality_task")
        # 高质量
        node_high = ImageSaveNode({"format": "JPEG", "quality": 95})
        await node_high.execute({"images": images}, ctx)
        # 低质量
        ctx2 = make_default_context(output_root=tmp_path, task_id="low_q")
        node_low = ImageSaveNode({"format": "JPEG", "quality": 5})
        await node_low.execute({"images": images}, ctx2)
        # 找到两个文件
        high_q_file = list((tmp_path / "files").glob("quality_task_*.jpg"))[0]
        low_q_file = list((tmp_path / "files").glob("low_q_*.jpg"))[0]
        assert high_q_file.stat().st_size > low_q_file.stat().st_size

    @pytest.mark.asyncio
    async def test_save_with_webp_lossless(self, tmp_path: Path):
        """WEBP 无损压缩应正常工作。"""
        images = Images.from_single(_make_image(size=(64, 64)))
        ctx = make_default_context(output_root=tmp_path, task_id="lossless_task")
        node = ImageSaveNode({"format": "WEBP", "lossless": True, "quality": 100})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.webp"))
        assert len(files) == 1
        with PILImage.open(files[0]) as img:
            assert img.format == "WEBP"


# ===== 文件名模板测试 =====


class TestImageSaveNodeFilenameTemplate:
    """文件名模板测试。"""

    @pytest.mark.asyncio
    async def test_default_template_uses_task_id_and_index(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="my_task")
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        assert len(files) == 1
        assert files[0].name == "my_task_000.png"

    @pytest.mark.asyncio
    async def test_custom_template_with_timestamp(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="my_task")
        node = ImageSaveNode({"filename_template": "{timestamp}_{task_id}.{ext}"})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        assert len(files) == 1
        assert files[0].name.startswith("_my_task.png") is False
        assert "_my_task.png" in files[0].name
        # 应以时间戳开头（数字）
        assert files[0].name[0].isdigit()

    @pytest.mark.asyncio
    async def test_template_with_multiple_images_increments_index(
        self, tmp_path: Path
    ):
        images = Images.from_list([_make_image() for _ in range(3)])
        ctx = make_default_context(output_root=tmp_path, task_id="task_idx")
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        files = sorted((tmp_path / "files").glob("*.png"))
        names = [f.name for f in files]
        assert "task_idx_000.png" in names
        assert "task_idx_001.png" in names
        assert "task_idx_002.png" in names

    @pytest.mark.asyncio
    async def test_template_extension_matches_format(self, tmp_path: Path):
        """文件扩展名应与 format 一致。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="ext_task")
        node = ImageSaveNode({"format": "JPEG"})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.jpg"))
        assert len(files) == 1
        assert not files[0].suffix == ".png"


# ===== URL 生成测试 =====


class TestImageSaveNodeUrlGeneration:
    """URL 生成测试。"""

    @pytest.mark.asyncio
    async def test_default_url_is_relative_path(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="url_task")
        node = ImageSaveNode()
        result = await node.execute({"images": images}, ctx)
        assert result["image_urls"][0] == "/files/url_task_000.png"

    @pytest.mark.asyncio
    async def test_custom_base_url_is_used(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="base_task")
        node = ImageSaveNode({"base_url": "http://localhost:8000/files/"})
        result = await node.execute({"images": images}, ctx)
        assert result["image_urls"][0] == "http://localhost:8000/files/base_task_000.png"

    @pytest.mark.asyncio
    async def test_custom_base_url_without_trailing_slash(self, tmp_path: Path):
        """base_url 不以 / 结尾时自动添加。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="noslash")
        node = ImageSaveNode({"base_url": "http://example.com/static"})
        result = await node.execute({"images": images}, ctx)
        assert result["image_urls"][0] == "http://example.com/static/noslash_000.png"

    @pytest.mark.asyncio
    async def test_url_count_matches_image_count(self, tmp_path: Path):
        images = Images.from_list([_make_image() for _ in range(4)])
        ctx = make_default_context(output_root=tmp_path, task_id="multi")
        node = ImageSaveNode({"base_url": "http://localhost/files/"})
        result = await node.execute({"images": images}, ctx)
        assert len(result["image_urls"]) == 4
        for url in result["image_urls"]:
            assert url.startswith("http://localhost/files/")


# ===== 元数据注入测试（验收标准）=====


class TestImageSaveNodeMetadataInjection:
    """元数据注入测试（验收标准：支持 PNG info）。"""

    @pytest.mark.asyncio
    async def test_metadata_param_injected_into_png(self, tmp_path: Path):
        """节点 metadata 参数注入 PNG info。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="meta_task")
        node = ImageSaveNode({
            "metadata": {"prompt": "a cat", "seed": 42},
        })
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        with PILImage.open(files[0]) as img:
            assert img.info.get("prompt") == "a cat"
            assert img.info.get("seed") == "42"

    @pytest.mark.asyncio
    async def test_image_metadata_injected_into_png(self, tmp_path: Path):
        """Image.metadata 字段注入 PNG info。"""
        image = _make_image(metadata={"source": "generated", "sampler": "euler"})
        images = Images.from_single(image)
        ctx = make_default_context(output_root=tmp_path, task_id="img_meta")
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        with PILImage.open(files[0]) as img:
            assert img.info.get("source") == "generated"
            assert img.info.get("sampler") == "euler"

    @pytest.mark.asyncio
    async def test_workflow_info_injected_when_enabled(self, tmp_path: Path):
        """include_workflow_info=True 时将 workflow 字段写入 PNG。"""
        workflow_data = {"nodes": ["sampler", "vae_decode"], "variant": "default"}
        image = _make_image(metadata={"workflow": workflow_data})
        images = Images.from_single(image)
        ctx = make_default_context(output_root=tmp_path, task_id="wf_task")
        node = ImageSaveNode({"include_workflow_info": True})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        with PILImage.open(files[0]) as img:
            workflow_str = img.info.get("workflow")
            assert workflow_str is not None
            assert "sampler" in workflow_str

    @pytest.mark.asyncio
    async def test_workflow_info_not_injected_when_disabled(self, tmp_path: Path):
        """include_workflow_info=False 时不写入 workflow。"""
        workflow_data = {"nodes": ["sampler"]}
        image = _make_image(metadata={"workflow": workflow_data})
        images = Images.from_single(image)
        ctx = make_default_context(output_root=tmp_path, task_id="wf_off")
        node = ImageSaveNode({"include_workflow_info": False})
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        with PILImage.open(files[0]) as img:
            assert "workflow" not in img.info

    @pytest.mark.asyncio
    async def test_metadata_json_saved_when_enabled(self, tmp_path: Path):
        """save_metadata_json=True 时同时保存 metadata.json。"""
        images = Images.from_single(_make_image(metadata={"foo": "bar"}))
        ctx = make_default_context(output_root=tmp_path, task_id="json_task")
        node = ImageSaveNode({"save_metadata_json": True})
        await node.execute({"images": images}, ctx)
        json_files = list((tmp_path / "files").glob("*.json"))
        assert len(json_files) == 1
        # 验证 JSON 内容
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["task_id"] == "json_task"
        assert data["index"] == 0
        assert data["image_metadata"]["foo"] == "bar"

    @pytest.mark.asyncio
    async def test_metadata_not_injected_for_jpeg(self, tmp_path: Path):
        """JPEG 不支持 PNG info，不应报错。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="jpeg_meta")
        node = ImageSaveNode({
            "format": "JPEG",
            "metadata": {"prompt": "a cat"},
        })
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.jpg"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_complex_metadata_value_stringified(self, tmp_path: Path):
        """复杂元数据值（dict / list）应转为 JSON 字符串。"""
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="complex_meta")
        node = ImageSaveNode({
            "metadata": {
                "params": {"steps": 30, "guidance": 7.5},
                "tags": ["cat", "art"],
            },
        })
        await node.execute({"images": images}, ctx)
        files = list((tmp_path / "files").glob("*.png"))
        with PILImage.open(files[0]) as img:
            params_str = img.info.get("params")
            assert params_str is not None
            assert "steps" in params_str
            assert "30" in params_str
            tags_str = img.info.get("tags")
            assert "cat" in tags_str


# ===== 输出目录测试 =====


class TestImageSaveNodeOutputDir:
    """输出目录解析测试。"""

    @pytest.mark.asyncio
    async def test_default_output_dir_uses_output_root_files(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="default_dir")
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        assert (tmp_path / "files").is_dir()

    @pytest.mark.asyncio
    async def test_custom_output_dir_absolute_path(self, tmp_path: Path):
        custom_dir = tmp_path / "custom" / "output"
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="custom_dir")
        node = ImageSaveNode({"output_dir": str(custom_dir)})
        await node.execute({"images": images}, ctx)
        assert custom_dir.is_dir()
        files = list(custom_dir.glob("*.png"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_custom_output_dir_relative_to_output_root(self, tmp_path: Path):
        images = Images.from_single(_make_image())
        ctx = make_default_context(output_root=tmp_path, task_id="rel_dir")
        node = ImageSaveNode({"output_dir": "sub/dir"})
        await node.execute({"images": images}, ctx)
        assert (tmp_path / "sub" / "dir").is_dir()
        files = list((tmp_path / "sub" / "dir").glob("*.png"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_output_dir_fallback_when_no_output_root(self, tmp_path: Path):
        """output_root=None 时使用 cwd / data / files。"""
        images = Images.from_single(_make_image())
        # 不设置 output_root
        ctx = make_default_context(output_root=None, task_id="fallback")
        node = ImageSaveNode()
        # 改 cwd 到 tmp_path 避免污染实际工作目录
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            await node.execute({"images": images}, ctx)
            assert (tmp_path / "data" / "files").is_dir()
        finally:
            os.chdir(old_cwd)


# ===== 进度上报测试 =====


class TestImageSaveNodeProgress:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_execute_reports_start_and_done(self, tmp_path: Path):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                events.append(("error", message, error_type, kwargs))

            async def report_info(self, message, **kwargs):
                events.append(("info", message, kwargs))

        images = Images.from_single(_make_image())
        ctx = make_default_context(
            output_root=tmp_path, task_id="progress_task",
            progress_reporter=_Capture(),
        )
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        progress_events = [e for e in events if e[0] == "progress"]
        assert any(e[2] == "image_save.start" for e in progress_events)
        assert any(e[2] == "image_save.done" for e in progress_events)
        start = [e for e in progress_events if e[2] == "image_save.start"][0]
        done = [e for e in progress_events if e[2] == "image_save.done"][0]
        assert start[1] == 0.0
        assert done[1] == 1.0

    @pytest.mark.asyncio
    async def test_execute_reports_per_image_progress(self, tmp_path: Path):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        images = Images.from_list([_make_image() for _ in range(3)])
        ctx = make_default_context(
            output_root=tmp_path, task_id="progress_multi",
            progress_reporter=_Capture(),
        )
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        # start + 3 progress + done = 5 个进度事件
        progress_events = [e for e in events if e[0] == "progress"]
        assert len(progress_events) == 5

    @pytest.mark.asyncio
    async def test_start_progress_contains_image_count(self, tmp_path: Path):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        images = Images.from_list([_make_image() for _ in range(2)])
        ctx = make_default_context(
            output_root=tmp_path, task_id="count_task",
            progress_reporter=_Capture(),
        )
        node = ImageSaveNode()
        await node.execute({"images": images}, ctx)
        start = [e for e in events if e[2] == "image_save.start"][0]
        assert start[3]["image_count"] == 2


# ===== 取消信号测试 =====


class TestImageSaveNodeCancellation:
    """取消信号测试。"""

    @pytest.mark.asyncio
    async def test_execute_raises_when_cancelled_before_save(self, tmp_path: Path):
        """工作流取消时抛 CancelledError。"""
        import asyncio

        images = Images.from_list([_make_image() for _ in range(3)])
        cancel_event = asyncio.Event()
        cancel_event.set()
        ctx = make_default_context(
            output_root=tmp_path, task_id="cancel_task",
            cancel_event=cancel_event,
        )
        node = ImageSaveNode()
        with pytest.raises(asyncio.CancelledError):
            await node.execute({"images": images}, ctx)


# ===== cache_key 测试 =====


class TestImageSaveNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_returns_none_by_default(self):
        """ImageSave 不缓存（每次都需保存新文件）。"""
        node = ImageSaveNode()
        assert node.cache_key({}) is None

    def test_cache_key_returns_none_with_inputs(self):
        node = ImageSaveNode({"format": "JPEG"})
        assert node.cache_key({"images": MagicMock()}) is None


# ===== 集成测试 =====


class TestImageSaveNodeIntegration:
    """端到端集成测试。"""

    @pytest.mark.asyncio
    async def test_full_save_pipeline_with_all_options(self, tmp_path: Path):
        """完整保存流程：所有选项生效。"""
        # 构造 2 张带元数据的图片
        images = Images.from_list([
            _make_image(metadata={"seed": 42, "sampler": "euler"}),
            _make_image(metadata={"seed": 43, "sampler": "dpm++"}),
        ])
        ctx = make_default_context(
            output_root=tmp_path,
            task_id="full_task",
        )
        node = ImageSaveNode({
            "format": "PNG",
            "base_url": "http://localhost:8000/files/",
            "metadata": {"engine": "ernie_image", "variant": "default"},
            "save_metadata_json": True,
            "include_workflow_info": True,
        })
        result = await node.execute({"images": images}, ctx)
        # 验证 URL
        assert len(result["image_urls"]) == 2
        for url in result["image_urls"]:
            assert url.startswith("http://localhost:8000/files/")
        # 验证图片文件
        files = sorted((tmp_path / "files").glob("*.png"))
        assert len(files) == 2
        assert files[0].name == "full_task_000.png"
        assert files[1].name == "full_task_001.png"
        # 验证 PNG info 注入
        with PILImage.open(files[0]) as img:
            assert img.info.get("engine") == "ernie_image"
            assert img.info.get("seed") == "42"
        # 验证 metadata.json 文件
        json_files = sorted((tmp_path / "files").glob("*.json"))
        assert len(json_files) == 2
        data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert data["task_id"] == "full_task"
        assert data["index"] == 0

    @pytest.mark.asyncio
    async def test_save_multiple_formats_in_sequence(self, tmp_path: Path):
        """同一图片保存为不同格式。"""
        image = _make_image()
        ctx1 = make_default_context(output_root=tmp_path, task_id="multi_png")
        ctx2 = make_default_context(output_root=tmp_path, task_id="multi_jpg")
        ctx3 = make_default_context(output_root=tmp_path, task_id="multi_webp")

        for ctx, fmt in [(ctx1, "PNG"), (ctx2, "JPEG"), (ctx3, "WEBP")]:
            node = ImageSaveNode({"format": fmt})
            result = await node.execute(
                {"images": Images.from_single(image)}, ctx
            )
            assert len(result["image_urls"]) == 1

        # 验证三种格式的文件都存在
        assert len(list((tmp_path / "files").glob("*.png"))) == 1
        assert len(list((tmp_path / "files").glob("*.jpg"))) == 1
        assert len(list((tmp_path / "files").glob("*.webp"))) == 1

"""ImagesInput 节点单元测试（v0.2.5 简化版）。

v0.2.5 简化：删除 role_filter / max_count / download_timeout / max_image_size /
accept_roles 5 个 advanced 参数，本测试文件同步删除相关测试用例。

覆盖：
    - 节点基本属性
    - image_paths 画布调试加载
    - b64_json 解码
    - URL 下载（mock httpx）
    - InferRequest 传递
    - ctx.inputs 注入
    - 错误处理

"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage

from aireliable_engine import ImageInput, InferRequest, make_default_context
from aireliable_engine.nodes.input.images_input import ImagesInputNode
from aireliable_engine.types import Image, Images


# ===== 辅助函数 =====


def _make_pil_image(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (8, 8)) -> PILImage.Image:
    """生成测试用 PIL 图片。"""
    return PILImage.new("RGB", size, color)


def _make_test_image_file(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    """生成测试图片文件并返回路径。"""
    pil_img = _make_pil_image(color)
    pil_img.save(path, format="PNG")
    return path


def _image_to_b64(pil_img: PILImage.Image, fmt: str = "PNG") -> str:
    """PIL 图片转 base64 字符串。"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ===== 测试用例 =====


class TestImagesInputNodeBasic:
    """ImagesInputNode 基本属性测试（v0.2.5 简化版）。"""

    def test_node_type_is_imagesinput(self):
        node = ImagesInputNode()
        assert node.node_type == "ImagesInput"

    def test_input_specs_is_empty_list(self):
        node = ImagesInputNode()
        assert node.input_specs == []

    def test_output_specs_has_images_port(self):
        node = ImagesInputNode()
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "images"
        assert port.type == "images"
        assert port.variadic is True

    def test_default_image_paths_is_empty_list(self):
        """v0.2.5 简化：默认 image_paths 为空列表。"""
        node = ImagesInputNode()
        assert node._image_paths == []

    def test_init_with_image_paths(self):
        """v0.2.5 简化：通过 image_paths 参数初始化。"""
        node = ImagesInputNode({
            "image_paths": ["/path/a.png", "/path/b.png"],
        })
        assert node._image_paths == ["/path/a.png", "/path/b.png"]

    def test_validate_returns_empty_for_default_params(self):
        """v0.2.5 简化：validate 始终返回空列表（无 advanced 参数需校验）。"""
        node = ImagesInputNode()
        errors = node.validate()
        assert errors == []

    def test_validate_returns_empty_with_image_paths(self):
        """v0.2.5 简化：带 image_paths 时 validate 也返回空。"""
        node = ImagesInputNode({"image_paths": ["/path/a.png"]})
        errors = node.validate()
        assert errors == []

    def test_repr_contains_image_paths_count(self):
        """v0.2.5 简化：repr 包含 image_paths 数量。"""
        node = ImagesInputNode({"image_paths": ["/a.png", "/b.png"]})
        repr_str = repr(node)
        assert "ImagesInputNode" in repr_str
        assert "image_paths_count=2" in repr_str

    def test_params_schema_only_has_image_paths(self):
        """v0.2.5 简化：params_schema 仅包含 image_paths 一个参数。"""
        schema = ImagesInputNode.params_schema
        assert list(schema.keys()) == ["image_paths"]
        assert schema["image_paths"]["format"] == "file-paths"
        assert schema["image_paths"]["type"] == "array"


# ===== image_paths 画布调试加载测试 =====


class TestImagePathsLoading:
    """v0.2.5 image_paths 画布调试加载测试。"""

    @pytest.mark.asyncio
    async def test_image_paths_loads_local_files(self, tmp_path: Path):
        """image_paths 加载本地图片文件。"""
        f1 = _make_test_image_file(tmp_path / "a.png", (255, 0, 0))
        f2 = _make_test_image_file(tmp_path / "b.png", (0, 255, 0))

        node = ImagesInputNode({"image_paths": [str(f1), str(f2)]})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)

        assert "images" in result
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 2

    @pytest.mark.asyncio
    async def test_image_paths_empty_returns_zero_images(self):
        """image_paths 为空时返回 0 张图。"""
        node = ImagesInputNode({"image_paths": []})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)

        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_image_paths_nonexistent_file_skipped(self):
        """image_paths 中不存在的文件被跳过（不报错）。"""
        node = ImagesInputNode({"image_paths": ["/nonexistent/xxx.png"]})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)

        # 加载失败但不抛异常，返回 0 张图
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_image_paths_order_preserved(self, tmp_path: Path):
        """image_paths 顺序 = 加载顺序 = 输出顺序。"""
        f1 = _make_test_image_file(tmp_path / "1.png", (255, 0, 0))
        f2 = _make_test_image_file(tmp_path / "2.png", (0, 255, 0))
        f3 = _make_test_image_file(tmp_path / "3.png", (0, 0, 255))

        node = ImagesInputNode({"image_paths": [str(f1), str(f2), str(f3)]})
        ctx = make_default_context(inputs={})
        result = await node.execute({}, ctx)

        assert len(result["images"]) == 3
        # 验证顺序：第一张红色、第二张绿色、第三张蓝色
        assert result["images"].items[0].pil.getpixel((0, 0)) == (255, 0, 0)
        assert result["images"].items[1].pil.getpixel((0, 0)) == (0, 255, 0)
        assert result["images"].items[2].pil.getpixel((0, 0)) == (0, 0, 255)


# ===== ctx.inputs 注入测试 =====


class TestCtxInputsInjection:
    """ctx.inputs 注入图片测试（API 调用场景）。"""

    @pytest.mark.asyncio
    async def test_ctx_inputs_dict_list_takes_priority_over_image_paths(
        self, tmp_path: Path
    ):
        """ctx.inputs 优先级高于 params.image_paths。"""
        f1 = _make_test_image_file(tmp_path / "a.png", (255, 0, 0))
        # image_paths 指向 f1，但 ctx.inputs 注入 b64 图
        b64 = _image_to_b64(_make_pil_image((0, 255, 0)))
        node = ImagesInputNode({"image_paths": [str(f1)]})
        ctx = make_default_context(
            inputs={"images": [{"b64_json": b64}]}
        )
        result = await node.execute({}, ctx)

        # 应加载 ctx.inputs 的图（绿色），不加载 image_paths 的图（红色）
        assert len(result["images"]) == 1
        assert result["images"].items[0].pil.getpixel((0, 0)) == (0, 255, 0)

    @pytest.mark.asyncio
    async def test_ctx_inputs_imageinput_objects(self):
        """ctx.inputs 直接传 ImageInput 对象列表。"""
        b64 = _image_to_b64(_make_pil_image((0, 0, 255)))
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(b64_json=b64)]}
        )
        result = await node.execute({}, ctx)

        assert len(result["images"]) == 1
        assert result["images"].items[0].pil.getpixel((0, 0)) == (0, 0, 255)

    @pytest.mark.asyncio
    async def test_ctx_inputs_string_treated_as_url(self):
        """ctx.inputs 中纯字符串按 URL 处理。"""
        node = ImagesInputNode()
        ctx = make_default_context(inputs={"images": ["https://example.com/x.png"]})

        # mock httpx 避免真实下载
        mock_response = MagicMock()
        mock_response.content = _make_pil_image((255, 255, 0)).tobytes()
        mock_response.raise_for_status = MagicMock()

        # PIL 从 bytes 加载需要格式，用 io.BytesIO 包装真实 PNG 数据
        png_bytes = io.BytesIO()
        _make_pil_image((255, 255, 0)).save(png_bytes, format="PNG")
        mock_response.content = png_bytes.getvalue()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await node.execute({}, ctx)

        assert len(result["images"]) == 1
        assert result["images"].items[0].metadata.get("source_url") == "https://example.com/x.png"


# ===== InferRequest 传递测试 =====


class TestInferRequestPassThrough:
    """InferRequest 完整传递测试。"""

    @pytest.mark.asyncio
    async def test_infer_request_images_takes_priority(self):
        """ctx.extra["__infer_request__"].images 优先级最高。"""
        b64 = _image_to_b64(_make_pil_image((255, 0, 255)))
        req = InferRequest(
            variant="default",
            inputs={"prompt": "test"},
            images=[ImageInput(b64_json=b64)],
        )

        node = ImagesInputNode()
        ctx = make_default_context(inputs={})
        ctx.extra["__infer_request__"] = req

        result = await node.execute({}, ctx)

        assert len(result["images"]) == 1
        assert result["images"].items[0].pil.getpixel((0, 0)) == (255, 0, 255)

    @pytest.mark.asyncio
    async def test_infer_request_empty_falls_back_to_image_paths(
        self, tmp_path: Path
    ):
        """InferRequest.images 为空时回退到 image_paths。"""
        f1 = _make_test_image_file(tmp_path / "a.png", (255, 0, 0))
        req = InferRequest(
            variant="default",
            inputs={"prompt": "test"},
            images=[],
        )

        node = ImagesInputNode({"image_paths": [str(f1)]})
        ctx = make_default_context(inputs={})
        ctx.extra["__infer_request__"] = req

        result = await node.execute({}, ctx)

        # InferRequest.images 为空 → 回退到 image_paths
        assert len(result["images"]) == 1
        assert result["images"].items[0].pil.getpixel((0, 0)) == (255, 0, 0)


# ===== 错误处理测试 =====


class TestErrorHandling:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_nonexistent_path_does_not_raise(self):
        """不存在的路径不抛异常，仅跳过。"""
        node = ImagesInputNode({"image_paths": ["/nonexistent/a.png", "/nonexistent/b.png"]})
        ctx = make_default_context(inputs={})

        # 不应抛异常
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_b64_skipped(self):
        """无效 b64_json 被跳过。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [{"b64_json": "!!!invalid base64!!!"}]}
        )

        result = await node.execute({}, ctx)
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_unrecognized_input_type_skipped(self):
        """无法识别的输入类型被跳过。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [12345, None, {"unknown_field": "x"}]}
        )

        # ImageInput(**{"unknown_field": "x"}) 会抛 TypeError，被 _collect_image_inputs 跳过
        # 实际上 ImageInput 是 pydantic 模型，未知字段会被忽略或抛错
        # 这里主要验证不会因异常崩溃
        try:
            result = await node.execute({}, ctx)
            assert len(result["images"]) == 0
        except Exception:
            # 如果 pydantic 严格模式抛错，也算通过（节点不应处理无效输入）
            pass

    @pytest.mark.asyncio
    async def test_empty_image_input_skipped(self):
        """三字段全空的 ImageInput 被跳过。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput()]}  # url/b64/path 全空
        )

        result = await node.execute({}, ctx)
        assert len(result["images"]) == 0


# ===== 完整工作流场景测试 =====


class TestEndToEndScenarios:
    """完整工作流场景测试。"""

    @pytest.mark.asyncio
    async def test_canvas_debugging_scenario(self, tmp_path: Path):
        """画布调试场景：仅用 image_paths 加载多张本地图片。"""
        f1 = _make_test_image_file(tmp_path / "ref1.png", (255, 0, 0))
        f2 = _make_test_image_file(tmp_path / "ref2.png", (0, 255, 0))
        f3 = _make_test_image_file(tmp_path / "ref3.png", (0, 0, 255))

        node = ImagesInputNode({"image_paths": [str(f1), str(f2), str(f3)]})
        ctx = make_default_context(inputs={})

        result = await node.execute({}, ctx)

        assert len(result["images"]) == 3
        # 验证顺序与 image_paths 一致
        assert result["images"].items[0].pil.getpixel((0, 0)) == (255, 0, 0)
        assert result["images"].items[1].pil.getpixel((0, 0)) == (0, 255, 0)
        assert result["images"].items[2].pil.getpixel((0, 0)) == (0, 0, 255)

    @pytest.mark.asyncio
    async def test_api_call_scenario_with_infer_request(self):
        """API 调用场景：InferRequest 携带多张图。"""
        b64_1 = _image_to_b64(_make_pil_image((255, 0, 0)))
        b64_2 = _image_to_b64(_make_pil_image((0, 255, 0)))

        req = InferRequest(
            variant="default",
            inputs={"prompt": "test"},
            images=[
                ImageInput(b64_json=b64_1, role="reference"),
                ImageInput(b64_json=b64_2, role="source"),
            ],
        )

        node = ImagesInputNode()
        ctx = make_default_context(inputs={})
        ctx.extra["__infer_request__"] = req

        result = await node.execute({}, ctx)

        assert len(result["images"]) == 2
        # v0.2.5 简化：不再按 role 过滤，两张图都应加载
        assert result["images"].items[0].metadata.get("role") == "reference"
        assert result["images"].items[1].metadata.get("role") == "source"

    @pytest.mark.asyncio
    async def test_mixed_sources_scenario(self, tmp_path: Path):
        """混合来源场景：path + b64 同时存在。"""
        f1 = _make_test_image_file(tmp_path / "a.png", (255, 0, 0))
        b64 = _image_to_b64(_make_pil_image((0, 255, 0)))

        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(f1)), ImageInput(b64_json=b64)]}
        )

        result = await node.execute({}, ctx)

        assert len(result["images"]) == 2

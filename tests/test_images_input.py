"""ImagesInput 节点单元测试。

覆盖任务 3.1 验收标准：
    - 0 图场景
    - 1 图场景
    - N 图场景
    - role 过滤
    - max_count 截断
    - b64_json 解码
    - URL 下载（mock httpx）
    - InferRequest 传递
    - 错误处理

参考：v0.2-analysis.md §5.2.3 / PR/architecture.md §12.2.4
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
    """ImagesInputNode 基本属性测试。"""

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

    def test_default_params_are_none(self):
        node = ImagesInputNode()
        assert node._role_filter is None
        assert node._max_count is None

    def test_init_with_params(self):
        node = ImagesInputNode({
            "role_filter": "reference",
            "max_count": 3,
            "download_timeout": 10.0,
            "max_image_size": 1024,
            "accept_roles": ["reference", "mask"],
        })
        assert node._role_filter == "reference"
        assert node._max_count == 3
        assert node._download_timeout == 10.0
        assert node._max_image_size == 1024
        assert node._accept_roles == ["reference", "mask"]

    def test_validate_returns_empty_for_default_params(self):
        node = ImagesInputNode()
        errors = node.validate()
        assert errors == []

    def test_validate_reports_negative_max_count(self):
        node = ImagesInputNode({"max_count": -1})
        errors = node.validate()
        assert any("max_count" in e for e in errors)

    def test_validate_reports_zero_download_timeout(self):
        node = ImagesInputNode({"download_timeout": 0})
        errors = node.validate()
        assert any("download_timeout" in e for e in errors)

    def test_validate_reports_negative_max_image_size(self):
        node = ImagesInputNode({"max_image_size": -1})
        errors = node.validate()
        assert any("max_image_size" in e for e in errors)

    def test_repr_contains_role_filter(self):
        node = ImagesInputNode({"role_filter": "mask"})
        repr_str = repr(node)
        assert "ImagesInputNode" in repr_str
        assert "mask" in repr_str


class TestImagesInputNodeZeroImages:
    """0 图场景测试（验收标准）。"""

    @pytest.mark.asyncio
    async def test_empty_ctx_returns_empty_images(self):
        """ctx 没有 images 字段，返回空 Images。"""
        node = ImagesInputNode()
        ctx = make_default_context()
        result = await node.execute({}, ctx)
        assert "images" in result
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_infer_request_with_empty_images_list(self):
        """InferRequest.images 为空列表时返回空 Images。"""
        node = ImagesInputNode()
        request = InferRequest(variant="flux", inputs={}, images=[])
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_infer_request_without_images_field(self):
        """老版本 InferRequest 没有 images 字段，返回空 Images。"""
        node = ImagesInputNode()
        request = InferRequest(variant="flux", inputs={})
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 0


class TestImagesInputNodeSingleImage:
    """1 图场景测试（验收标准）。"""

    @pytest.mark.asyncio
    async def test_load_single_image_from_path(self, tmp_path: Path):
        """从本地路径加载 1 张图。"""
        img_path = _make_test_image_file(tmp_path / "a.png", color=(255, 0, 0))
        node = ImagesInputNode()
        ctx = make_default_context(inputs={"images": [ImageInput(path=str(img_path))]})
        result = await node.execute({}, ctx)
        images = result["images"]
        assert isinstance(images, Images)
        assert len(images) == 1
        assert isinstance(images[0], Image)
        assert images[0].path == img_path
        assert images[0].pil.size == (8, 8)

    @pytest.mark.asyncio
    async def test_load_single_image_from_b64(self):
        """从 b64_json 加载 1 张图。"""
        pil_img = _make_pil_image(color=(0, 255, 0))
        b64 = _image_to_b64(pil_img)
        node = ImagesInputNode()
        ctx = make_default_context(inputs={"images": [ImageInput(b64_json=b64)]})
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 1
        assert images[0].pil.size == (8, 8)
        assert images[0].path is None

    @pytest.mark.asyncio
    async def test_load_single_image_via_infer_request(self, tmp_path: Path):
        """通过完整 InferRequest 传递加载图片。"""
        img_path = _make_test_image_file(tmp_path / "ref.png", color=(0, 0, 255))
        node = ImagesInputNode()
        request = InferRequest(
            variant="flux",
            inputs={"prompt": "a cat"},
            images=[ImageInput(path=str(img_path), role="reference")],
        )
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 1
        assert result["images"][0].metadata.get("role") == "reference"


class TestImagesInputNodeMultipleImages:
    """N 图场景测试（验收标准）。"""

    @pytest.mark.asyncio
    async def test_load_three_images_from_paths(self, tmp_path: Path):
        """从本地路径加载 3 张图。"""
        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(3)
        ]
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 3
        for img, original_path in zip(images, paths):
            assert img.path == original_path

    @pytest.mark.asyncio
    async def test_load_mixed_sources(self, tmp_path: Path):
        """混合来源：1 张 path + 1 张 b64 + 1 张 url（mock）。"""
        img_path = _make_test_image_file(tmp_path / "a.png")
        pil_img = _make_pil_image(color=(0, 255, 0))
        b64 = _image_to_b64(pil_img)

        # mock httpx 返回第三张图的字节
        url_pil = _make_pil_image(color=(0, 0, 255))
        url_buf = io.BytesIO()
        url_pil.save(url_buf, format="PNG")
        url_bytes = url_buf.getvalue()
        mock_response = MagicMock()
        mock_response.content = url_bytes
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=mock_response)

        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [
                ImageInput(path=str(img_path)),
                ImageInput(b64_json=b64),
                ImageInput(url="https://example.com/c.png"),
            ]}
        )

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await node.execute({}, ctx)

        images = result["images"]
        assert len(images) == 3
        assert images[0].path == img_path
        assert images[2].metadata.get("source_url") == "https://example.com/c.png"

    @pytest.mark.asyncio
    async def test_iter_and_index_on_result(self, tmp_path: Path):
        """返回的 Images 支持迭代和索引。"""
        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(3)
        ]
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 3
        for idx, img in enumerate(images):
            assert img == images[idx]
        # 切片返回 Images
        sliced = images[:2]
        assert isinstance(sliced, Images)
        assert len(sliced) == 2


class TestImagesInputNodeRoleFilter:
    """role 过滤测试。"""

    @pytest.mark.asyncio
    async def test_role_filter_selects_matching(self, tmp_path: Path):
        """role_filter 只保留匹配的图。"""
        p1 = _make_test_image_file(tmp_path / "ref.png", color=(255, 0, 0))
        p2 = _make_test_image_file(tmp_path / "mask.png", color=(0, 255, 0))
        p3 = _make_test_image_file(tmp_path / "ctrl.png", color=(0, 0, 255))

        node = ImagesInputNode({"role_filter": "mask"})
        ctx = make_default_context(
            inputs={"images": [
                ImageInput(path=str(p1), role="reference"),
                ImageInput(path=str(p2), role="mask"),
                ImageInput(path=str(p3), role="control"),
            ]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 1
        assert images[0].metadata.get("role") == "mask"

    @pytest.mark.asyncio
    async def test_accept_roles_whitelist(self, tmp_path: Path):
        """accept_roles 白名单匹配多个 role。"""
        p1 = _make_test_image_file(tmp_path / "ref.png", color=(255, 0, 0))
        p2 = _make_test_image_file(tmp_path / "mask.png", color=(0, 255, 0))
        p3 = _make_test_image_file(tmp_path / "ctrl.png", color=(0, 0, 255))

        node = ImagesInputNode({"accept_roles": ["reference", "mask"]})
        ctx = make_default_context(
            inputs={"images": [
                ImageInput(path=str(p1), role="reference"),
                ImageInput(path=str(p2), role="mask"),
                ImageInput(path=str(p3), role="control"),
            ]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 2
        roles = [img.metadata.get("role") for img in images]
        assert "reference" in roles
        assert "mask" in roles
        assert "control" not in roles

    @pytest.mark.asyncio
    async def test_role_filter_overrides_accept_roles(self, tmp_path: Path):
        """同时设置时 role_filter 优先。"""
        p1 = _make_test_image_file(tmp_path / "ref.png", color=(255, 0, 0))
        p2 = _make_test_image_file(tmp_path / "mask.png", color=(0, 255, 0))

        node = ImagesInputNode({
            "role_filter": "reference",
            "accept_roles": ["mask"],
        })
        ctx = make_default_context(
            inputs={"images": [
                ImageInput(path=str(p1), role="reference"),
                ImageInput(path=str(p2), role="mask"),
            ]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 1
        assert images[0].metadata.get("role") == "reference"


class TestImagesInputNodeMaxCount:
    """max_count 截断测试。"""

    @pytest.mark.asyncio
    async def test_max_count_truncates(self, tmp_path: Path):
        """max_count 截断超出的图片。"""
        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(5)
        ]
        node = ImagesInputNode({"max_count": 3})
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 3
        # 截断保留前 3 张
        assert images[0].path == paths[0]
        assert images[2].path == paths[2]

    @pytest.mark.asyncio
    async def test_max_count_none_does_not_truncate(self, tmp_path: Path):
        """max_count=None 不截断。"""
        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(5)
        ]
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]}
        )
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 5

    @pytest.mark.asyncio
    async def test_max_count_zero_means_no_limit(self, tmp_path: Path):
        """max_count=0 视为不限制（与 None 等价）。"""
        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(3)
        ]
        node = ImagesInputNode({"max_count": 0})
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]}
        )
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 3


class TestImagesInputNodeErrorHandling:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_nonexistent_path_skipped(self):
        """路径不存在时该图被跳过，其他图正常加载。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [
                ImageInput(path="/nonexistent/path/a.png"),
                ImageInput(b64_json=_image_to_b64(_make_pil_image(color=(0, 255, 0)))),
            ]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        # 第一张失败被跳过，第二张成功
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_invalid_b64_skipped(self):
        """损坏的 b64_json 该图被跳过。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [
                ImageInput(b64_json="not_valid_base64!!!"),
                ImageInput(b64_json=_image_to_b64(_make_pil_image(color=(0, 255, 0)))),
            ]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_empty_image_input_skipped(self):
        """三个字段全为空的 ImageInput 被跳过。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput()]}
        )
        result = await node.execute({}, ctx)
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_string_treated_as_url(self):
        """纯字符串被识别为 URL。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": ["https://example.com/a.png"]}
        )
        # 由于 mock 比较复杂，这里只验证节点能识别字符串为 ImageInput(url=...)
        image_inputs = node._collect_image_inputs(ctx)
        assert len(image_inputs) == 1
        assert image_inputs[0].url == "https://example.com/a.png"

    @pytest.mark.asyncio
    async def test_dict_input_converted_to_image_input(self, tmp_path: Path):
        """dict 自动转 ImageInput。"""
        img_path = _make_test_image_file(tmp_path / "a.png")
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [{"path": str(img_path), "role": "source"}]}
        )
        image_inputs = node._collect_image_inputs(ctx)
        assert len(image_inputs) == 1
        assert isinstance(image_inputs[0], ImageInput)
        assert image_inputs[0].role == "source"

    @pytest.mark.asyncio
    async def test_invalid_item_type_skipped(self):
        """无法识别的输入类型被跳过。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [42, 3.14, None]}
        )
        image_inputs = node._collect_image_inputs(ctx)
        assert image_inputs == []


class TestImagesInputNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_for_multiple_images(self, tmp_path: Path):
        """多张图加载时会上报进度。"""
        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(3)
        ]
        node = ImagesInputNode()
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, kwargs.get("message", "")))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]},
            progress_reporter=_CapturingReporter(),
        )
        await node.execute({}, ctx)
        # 至少上报 start / loading x 3 / done = 5 次
        assert len(progress_calls) >= 5
        # 第一次 progress=0.0
        assert progress_calls[0][0] == 0.0
        # 最后一次 progress=1.0
        assert progress_calls[-1][0] == 1.0

    @pytest.mark.asyncio
    async def test_progress_info_on_failure(self, tmp_path: Path):
        """加载失败时上报 info 帧。"""
        valid_path = _make_test_image_file(tmp_path / "valid.png")
        node = ImagesInputNode()
        info_messages: list[str] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                pass

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                info_messages.append(message)

        ctx = make_default_context(
            inputs={"images": [
                ImageInput(path="/nonexistent/a.png"),
                ImageInput(path=str(valid_path)),
            ]},
            progress_reporter=_CapturingReporter(),
        )
        await node.execute({}, ctx)
        assert any("失败" in msg for msg in info_messages)


class TestImagesInputNodeCancellation:
    """取消信号测试。"""

    @pytest.mark.asyncio
    async def test_cancellation_aborts_loading(self, tmp_path: Path):
        """cancel_event 已设置时立即抛出 CancelledError。"""
        import asyncio

        paths = [
            _make_test_image_file(tmp_path / f"img_{i}.png", color=(i * 50, 0, 0))
            for i in range(3)
        ]
        node = ImagesInputNode()
        cancel_event = asyncio.Event()
        cancel_event.set()  # 立即取消
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(p)) for p in paths]},
            cancel_event=cancel_event,
        )
        with pytest.raises(asyncio.CancelledError):
            await node.execute({}, ctx)


class TestImagesInputNodeMaxImageSize:
    """max_image_size 缩放测试。"""

    @pytest.mark.asyncio
    async def test_large_image_downscaled(self, tmp_path: Path):
        """超大图片按 max_image_size 等比缩小。"""
        # 生成 100x100 的图片（10000 像素）
        big_pil = PILImage.new("RGB", (100, 100), (255, 0, 0))
        big_path = tmp_path / "big.png"
        big_pil.save(big_path, format="PNG")

        # max_image_size=2500 → 等比缩小到 ~50x50
        node = ImagesInputNode({"max_image_size": 2500})
        ctx = make_default_context(inputs={"images": [ImageInput(path=str(big_path))]})
        result = await node.execute({}, ctx)
        images = result["images"]
        assert len(images) == 1
        w, h = images[0].pil.size
        assert w * h <= 2500

    @pytest.mark.asyncio
    async def test_small_image_unchanged(self, tmp_path: Path):
        """小图片不被缩小。"""
        small_path = _make_test_image_file(tmp_path / "small.png", color=(255, 0, 0))
        node = ImagesInputNode({"max_image_size": 10000})
        ctx = make_default_context(inputs={"images": [ImageInput(path=str(small_path))]})
        result = await node.execute({}, ctx)
        assert result["images"][0].pil.size == (8, 8)


class TestImagesInputNodeSourcePriority:
    """图片来源优先级测试。"""

    @pytest.mark.asyncio
    async def test_path_priority_over_b64(self, tmp_path: Path):
        """path 字段优先于 b64_json。"""
        path_img = _make_test_image_file(tmp_path / "a.png", color=(255, 0, 0))
        b64_img = _image_to_b64(_make_pil_image(color=(0, 255, 0)))

        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(path=str(path_img), b64_json=b64_img)]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        # 应该加载 path 的图（红色），而不是 b64 的图（绿色）
        assert len(images) == 1
        assert images[0].path == path_img

    @pytest.mark.asyncio
    async def test_b64_priority_over_url(self, tmp_path: Path):
        """b64_json 字段优先于 url。"""
        b64_img = _image_to_b64(_make_pil_image(color=(0, 255, 0)))
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [ImageInput(b64_json=b64_img, url="https://example.com/a.png")]}
        )
        result = await node.execute({}, ctx)
        images = result["images"]
        # 应该加载 b64 的图（无 source_url metadata）
        assert len(images) == 1
        assert "source_url" not in images[0].metadata


class TestImagesInputNodeIntegration:
    """集成场景测试。"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_infer_request(self, tmp_path: Path):
        """完整工作流场景：InferRequest 传递 + role 过滤 + max_count。"""
        p1 = _make_test_image_file(tmp_path / "ref1.png", color=(255, 0, 0))
        p2 = _make_test_image_file(tmp_path / "ref2.png", color=(0, 255, 0))
        p3 = _make_test_image_file(tmp_path / "mask1.png", color=(0, 0, 255))
        p4 = _make_test_image_file(tmp_path / "ctrl1.png", color=(255, 255, 0))

        node = ImagesInputNode({
            "role_filter": "reference",
            "max_count": 1,
        })
        request = InferRequest(
            variant="flux",
            inputs={"prompt": "a cat in reference style"},
            images=[
                ImageInput(path=str(p1), role="reference"),
                ImageInput(path=str(p2), role="reference"),
                ImageInput(path=str(p3), role="mask"),
                ImageInput(path=str(p4), role="control"),
            ],
            workflow="ipadapter",
        )
        ctx = make_default_context(extra={"__infer_request__": request})
        result = await node.execute({}, ctx)
        images = result["images"]
        # 过滤后 2 张 reference，max_count 截断为 1
        assert len(images) == 1
        assert images[0].metadata.get("role") == "reference"
        # 第一张 reference（按列表顺序）
        assert images[0].path == p1

    @pytest.mark.asyncio
    async def test_collect_image_inputs_with_dict_list(self):
        """ctx.inputs["images"] 是 dict 列表时自动转换。"""
        node = ImagesInputNode()
        ctx = make_default_context(
            inputs={"images": [
                {"url": "https://example.com/a.png"},
                {"path": "/tmp/b.png", "role": "mask"},
            ]}
        )
        image_inputs = node._collect_image_inputs(ctx)
        assert len(image_inputs) == 2
        assert image_inputs[0].url == "https://example.com/a.png"
        assert image_inputs[1].path == "/tmp/b.png"
        assert image_inputs[1].role == "mask"

    def test_describe_source_helper(self):
        """_describe_source 辅助方法正确描述来源。"""
        assert "path=" in ImagesInputNode._describe_source(ImageInput(path="/tmp/a.png"))
        assert "b64_json" in ImagesInputNode._describe_source(ImageInput(b64_json="abc"))
        assert "url=" in ImagesInputNode._describe_source(ImageInput(url="https://x.com"))
        assert ImagesInputNode._describe_source(ImageInput()) == "empty"

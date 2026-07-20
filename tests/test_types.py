"""types.py 数据类型单元测试。

覆盖：
    - Image：构造 / 工厂方法 / 元数据 / 属性
    - Images：容器协议 / 工厂方法 / 切片 / 迭代
    - Audio：构造 / 属性
    - Video：构造 / 属性
    - Tensor / Latent：构造（torch 缺失时不强制）
    - PORT_TYPE_MAP / get_port_type_class / is_valid_port_type
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from aireliable_engine.types import (
    PORT_TYPE_MAP,
    Audio,
    Image,
    Images,
    Latent,
    Tensor,
    Video,
    get_port_type_class,
    is_valid_port_type,
)


# ===========================================================================
# 测试 fixtures
# ===========================================================================

@pytest.fixture
def sample_pil() -> PILImage.Image:
    """构造一张 32x32 RGB 测试图。"""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # 红色
    return PILImage.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_image(sample_pil: PILImage.Image) -> Image:
    """构造一个 Image 实例。"""
    return Image(pil=sample_pil, metadata={"source": "test"})


@pytest.fixture
def sample_array() -> np.ndarray:
    """构造一个 4x4 uint8 数组。"""
    return np.ones((4, 4, 3), dtype=np.uint8) * 128


# ===========================================================================
# Image 测试
# ===========================================================================

class TestImage:
    """Image 数据类测试。"""

    def test_construct_basic(self, sample_pil: PILImage.Image) -> None:
        """基础构造。"""
        img = Image(pil=sample_pil)
        assert img.pil is sample_pil
        assert img.path is None
        assert img.metadata == {}

    def test_construct_with_path_and_metadata(
        self, sample_pil: PILImage.Image, tmp_path
    ) -> None:
        """带 path 和 metadata 构造。"""
        from pathlib import Path

        path = Path("/tmp/test.png")
        img = Image(pil=sample_pil, path=path, metadata={"role": "source"})
        assert img.path == path
        assert img.metadata == {"role": "source"}

    def test_width_height_properties(self, sample_image: Image) -> None:
        """width / height 属性。"""
        assert sample_image.width == 32
        assert sample_image.height == 32

    def test_from_path(self, tmp_path, sample_pil: PILImage.Image) -> None:
        """from_path 工厂方法。"""
        # 保存一张测试图
        path = tmp_path / "test.png"
        sample_pil.save(path)

        img = Image.from_path(path, metadata={"source": "file"})
        assert img.path == path
        assert img.pil.size == (32, 32)
        assert img.metadata["source"] == "file"

    def test_from_path_with_mode_conversion(
        self, tmp_path, sample_pil: PILImage.Image
    ) -> None:
        """from_path 模式转换。"""
        path = tmp_path / "test.png"
        sample_pil.save(path)

        img = Image.from_path(path, mode="RGBA")
        assert img.pil.mode == "RGBA"

    def test_from_array(self, sample_array: np.ndarray) -> None:
        """from_array 工厂方法。"""
        img = Image.from_array(sample_array)
        assert img.pil.size == (4, 4)
        assert img.path is None

    def test_from_array_with_mode(self) -> None:
        """from_array 带模式。"""
        arr = np.zeros((8, 8), dtype=np.uint8)
        img = Image.from_array(arr, mode="L")
        assert img.pil.mode == "L"

    def test_with_metadata_immutable_update(self, sample_image: Image) -> None:
        """with_metadata 返回新实例（不可变更新）。"""
        new_img = sample_image.with_metadata(role="mask", extra="x")
        assert new_img is not sample_image
        assert new_img.pil is sample_image.pil  # pil 引用相同
        assert new_img.metadata == {"source": "test", "role": "mask", "extra": "x"}
        # 原实例 metadata 不变
        assert sample_image.metadata == {"source": "test"}

    def test_metadata_default_factory(self, sample_pil: PILImage.Image) -> None:
        """metadata 默认值为独立 dict（非共享可变默认值）。"""
        img1 = Image(pil=sample_pil)
        img2 = Image(pil=sample_pil)
        img1.metadata["key"] = "value"
        assert "key" not in img2.metadata


# ===========================================================================
# Images 测试
# ===========================================================================

class TestImages:
    """Images 包装类测试。"""

    def test_empty(self) -> None:
        """empty() 工厂方法。"""
        imgs = Images.empty()
        assert len(imgs) == 0
        assert bool(imgs) is False
        assert imgs.first() is None

    def test_from_single(self, sample_image: Image) -> None:
        """from_single() 工厂方法。"""
        imgs = Images.from_single(sample_image)
        assert len(imgs) == 1
        assert bool(imgs) is True
        assert imgs[0] is sample_image

    def test_from_list(self, sample_image: Image) -> None:
        """from_list() 工厂方法。"""
        imgs = Images.from_list([sample_image, sample_image, sample_image])
        assert len(imgs) == 3

    def test_from_list_copies(self, sample_image: Image) -> None:
        """from_list() 应复制 list（不共享引用）。"""
        original = [sample_image, sample_image]
        imgs = Images.from_list(original)
        original.append(sample_image)  # 修改原 list
        assert len(imgs) == 2  # Images 不受影响

    def test_int_index(self, sample_image: Image) -> None:
        """int 索引返回 Image。"""
        imgs = Images(items=[sample_image, sample_image, sample_image])
        result = imgs[1]
        assert isinstance(result, Image)
        assert result is sample_image

    def test_slice_index_returns_images(self, sample_image: Image) -> None:
        """slice 切片返回 Images（类型一致性）。"""
        imgs = Images(items=[sample_image, sample_image, sample_image])
        result = imgs[0:2]
        assert isinstance(result, Images)
        assert len(result) == 2

    def test_iteration(self, sample_image: Image) -> None:
        """迭代。"""
        imgs = Images(items=[sample_image, sample_image])
        items = [img for img in imgs]
        assert len(items) == 2
        assert all(img is sample_image for img in items)

    def test_append(self, sample_image: Image) -> None:
        """append 方法。"""
        imgs = Images.empty()
        imgs.append(sample_image)
        assert len(imgs) == 1
        imgs.append(sample_image)
        assert len(imgs) == 2

    def test_extend(self, sample_image: Image) -> None:
        """extend 方法。"""
        imgs1 = Images.from_single(sample_image)
        imgs2 = Images.from_single(sample_image)
        imgs1.extend(imgs2)
        assert len(imgs1) == 2

    def test_first(self, sample_image: Image) -> None:
        """first() 方法。"""
        imgs_empty = Images.empty()
        assert imgs_empty.first() is None

        imgs = Images.from_single(sample_image)
        assert imgs.first() is sample_image

    def test_paths_property(self, sample_image: Image, tmp_path) -> None:
        """paths 属性。"""
        from pathlib import Path

        path1 = Path("/tmp/a.png")
        path2 = None
        img1 = Image(pil=sample_image.pil, path=path1)
        img2 = Image(pil=sample_image.pil, path=path2)
        imgs = Images(items=[img1, img2])
        assert imgs.paths == [path1, None]

    def test_repr_empty(self) -> None:
        """空 Images 的 repr。"""
        imgs = Images.empty()
        assert "empty" in repr(imgs)

    def test_repr_with_items(self, sample_image: Image) -> None:
        """非空 Images 的 repr。"""
        imgs = Images(items=[sample_image])
        repr_str = repr(imgs)
        assert "n=1" in repr_str
        assert "32x32" in repr_str

    def test_repr_with_many_items_truncated(self, sample_image: Image) -> None:
        """多图 repr 截断（显示前 3 张）。"""
        imgs = Images(items=[sample_image] * 5)
        repr_str = repr(imgs)
        assert "n=5" in repr_str
        assert "..." in repr_str


# ===========================================================================
# Audio 测试
# ===========================================================================

class TestAudio:
    """Audio 数据类测试。"""

    def test_construct_mono(self) -> None:
        """单声道构造。"""
        samples = np.zeros((16000,), dtype=np.float32)
        audio = Audio(samples=samples, sample_rate=16000)
        assert audio.sample_rate == 16000
        assert audio.path is None
        assert audio.channels == 1
        assert audio.duration_seconds == 1.0

    def test_construct_stereo(self) -> None:
        """立体声构造。"""
        samples = np.zeros((2, 44100), dtype=np.float32)
        audio = Audio(samples=samples, sample_rate=44100)
        assert audio.channels == 2
        assert audio.duration_seconds == 1.0

    def test_empty_audio_duration(self) -> None:
        """空音频时长为 0。"""
        samples = np.zeros((0,), dtype=np.float32)
        audio = Audio(samples=samples, sample_rate=16000)
        assert audio.duration_seconds == 0.0

    def test_metadata_default_factory(self) -> None:
        """metadata 默认值独立性。"""
        samples = np.zeros((100,), dtype=np.float32)
        a1 = Audio(samples=samples, sample_rate=16000)
        a2 = Audio(samples=samples, sample_rate=16000)
        a1.metadata["k"] = "v"
        assert "k" not in a2.metadata


# ===========================================================================
# Video 测试
# ===========================================================================

class TestVideo:
    """Video 数据类测试。"""

    def test_construct_empty(self) -> None:
        """空视频构造。"""
        video = Video()
        assert video.frames is None
        assert video.path is None
        assert video.fps == 30
        assert video.frame_count == 0
        assert video.duration_seconds == 0.0

    def test_construct_with_frames(self, sample_image: Image) -> None:
        """带帧列表构造。"""
        frames = [sample_image] * 60
        video = Video(frames=frames, fps=30)
        assert video.frame_count == 60
        assert video.duration_seconds == 2.0

    def test_construct_with_path(self, tmp_path) -> None:
        """带路径构造（未解码帧）。"""
        from pathlib import Path

        path = Path("/tmp/video.mp4")
        video = Video(path=path, fps=24)
        assert video.frame_count == 0  # frames 为 None
        assert video.duration_seconds == 0.0


# ===========================================================================
# Tensor / Latent 测试（不依赖 torch 安装）
# ===========================================================================

class TestTensorLatent:
    """Tensor / Latent 数据类测试。

    不实际创建 torch.Tensor 实例（torch 在 SDK 核心依赖中不存在），
    只验证类定义可被 import + 类型注解正确。
    """

    def test_tensor_class_exists(self) -> None:
        """Tensor 类可被 import。"""
        assert Tensor is not None
        # 验证 dataclass 字段
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(Tensor)}
        assert field_names == {"data", "device", "metadata"}

    def test_latent_class_exists(self) -> None:
        """Latent 类可被 import。"""
        assert Latent is not None
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(Latent)}
        assert field_names == {"tensor", "vae_type", "metadata"}

    def test_tensor_construct_with_mock_data(self) -> None:
        """Tensor 用 mock 对象构造（不依赖 torch）。"""
        # 用一个带 device 属性的 mock 对象
        class MockTensor:
            @property
            def device(self) -> str:
                return "cuda:0"

        tensor = Tensor(data=MockTensor())  # type: ignore[arg-type]
        # __post_init__ 应从 mock.device 推断
        assert tensor.device == "cuda:0"

    def test_tensor_construct_with_explicit_device(self) -> None:
        """显式指定 device 时不查询 tensor.device。"""
        # 用一个会抛异常的 mock（模拟 torch 未安装）
        class BadMock:
            @property
            def device(self) -> str:
                raise RuntimeError("torch not installed")

        tensor = Tensor(data=BadMock(), device="cpu")  # type: ignore[arg-type]
        assert tensor.device == "cpu"  # 显式指定的优先

    def test_tensor_construct_unknown_device_when_no_torch(self) -> None:
        """无 torch 时 device 推断为 'unknown'。"""
        # mock 一个没有 device 属性的对象
        class NoDevice:
            pass

        tensor = Tensor(data=NoDevice())  # type: ignore[arg-type]
        assert tensor.device == "unknown"

    def test_latent_construct_basic(self) -> None:
        """Latent 基础构造（用 mock tensor）。"""
        latent = Latent(tensor=None, vae_type="ernie-image")  # type: ignore[arg-type]
        assert latent.vae_type == "ernie-image"
        assert latent.metadata == {}


# ===========================================================================
# 端口类型映射测试
# ===========================================================================

class TestPortTypeMap:
    """端口类型映射测试。"""

    def test_all_expected_types_in_map(self) -> None:
        """所有预期端口类型在映射表中。"""
        expected = {"image", "images", "audio", "audios", "video", "text", "embeds", "latent", "model", "any"}
        assert expected.issubset(PORT_TYPE_MAP.keys())

    def test_get_port_type_class_known(self) -> None:
        """已知类型返回对应 Python 类。"""
        assert get_port_type_class("image") is Image
        assert get_port_type_class("images") is Images
        assert get_port_type_class("audio") is Audio
        assert get_port_type_class("video") is Video
        assert get_port_type_class("embeds") is Tensor
        assert get_port_type_class("latent") is Latent
        assert get_port_type_class("text") is str

    def test_get_port_type_class_unknown_returns_any(self) -> None:
        """未知类型返回 Any。"""
        from typing import Any

        assert get_port_type_class("unknown_type") is Any

    def test_is_valid_port_type_valid(self) -> None:
        """合法端口类型。"""
        for t in ("image", "images", "audio", "audios", "video", "text", "embeds", "latent", "model", "any"):
            assert is_valid_port_type(t), f"{t} 应为合法类型"

    def test_is_valid_port_type_invalid(self) -> None:
        """非法端口类型。"""
        for t in ("", "foo", "Image", "Images", "int", None):
            assert not is_valid_port_type(t), f"{t!r} 应为非法类型"

"""节点数据类型（v0.2 新增）。

定义节点间传递的标准数据类型，配合 PortSpec.type 字段使用。

设计参考：
    - v0.2-analysis.md §5.4 / §11.2
    - PR/architecture.md §12.3
    - PR/development-guide.md §28.5

端口类型映射表（PR/development-guide.md §28.7）：
    | type 值    | Python 类型         | 用途                |
    |------------|---------------------|---------------------|
    | image      | Image               | 单张图片            |
    | images     | Images              | 多图数组（variadic）|
    | audio      | Audio               | 单段音频            |
    | audios     | list[Audio]         | 多段音频            |
    | video      | Video               | 单个视频            |
    | text       | str                 | 文本                |
    | embeds     | Tensor              | torch.Tensor + device |
    | latent     | Latent              | 潜空间表示          |
    | model      | Any                 | 模型对象            |
    | any        | Any                 | 任意类型            |

依赖说明：
    - PIL / numpy：SDK 核心依赖，可直接 import
    - torch：可选依赖（[image]/[video]/[voice]/[llm] extras）
        types.py 模块本身可在无 torch 环境下 import（用于节点规格查询）
        但 Tensor / Latent 实例化需要 torch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

# PIL 是 SDK 核心依赖（pillow）
from PIL import Image as PILImage

# numpy 是 SDK 核心依赖（pillow 间接依赖，但显式声明更清晰）
import numpy as np

# torch 是可选依赖（[image]/[video]/[voice]/[llm] extras）
# 使用 TYPE_CHECKING 让注解不求值，模块可在无 torch 环境下 import
if TYPE_CHECKING:
    import torch


# ===========================================================================
# Image：单张图片
# ===========================================================================

@dataclass
class Image:
    """单张图片数据类型（PortSpec.type="image"）。

    包装 PIL.Image，附加路径与元数据。

    Attributes:
        pil: PIL.Image.Image 对象。是图像数据的实际载体。
        path: 图片本地路径（如已下载到本地）。None 表示仅内存对象。
            下游节点（如 ImageSave）可直接复用路径避免重复保存。
        metadata: 图片元数据字典。可包含：
            - source: 来源标记（"user_upload" / "url_download" / "generated"）
            - role: 角色标记（"source" / "mask" / "reference" / "control"）
            - exif: EXIF 信息
            - original_url: 原始 URL（如来自下载）
            - width / height: 宽高（冗余字段，便于不加载 PIL 即可查询）

    设计原则：
        - Image 是不可变数据载体：pil 字段不修改，metadata 通过新实例传递
        - 路径优先：若 path 存在，下游节点应优先从路径加载（避免 PIL 对象序列化开销）
    """

    pil: "PILImage.Image"
    path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        """图片宽度（冗余字段，便于不查询 PIL）。"""
        return self.pil.width

    @property
    def height(self) -> int:
        """图片高度（冗余字段，便于不查询 PIL）。"""
        return self.pil.height

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        metadata: Optional[dict] = None,
        mode: Optional[str] = None,
    ) -> "Image":
        """从本地路径加载图片。

        Args:
            path: 图片文件路径。
            metadata: 附加元数据。
            mode: PIL 加载模式（如 "RGB" / "RGBA"）。None 表示不转换。

        Returns:
            Image 实例。
        """
        path = Path(path)
        pil = PILImage.open(path)
        if mode and pil.mode != mode:
            pil = pil.convert(mode)
        return cls(pil=pil, path=path, metadata=metadata or {})

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        metadata: Optional[dict] = None,
        mode: Optional[str] = None,
    ) -> "Image":
        """从 numpy 数组创建图片。

        Args:
            array: HxWxC 或 HxW 的 numpy 数组（uint8 / float32）。
            metadata: 附加元数据。
            mode: PIL 模式。None 时自动推断（RGB / L）。

        Returns:
            Image 实例（无 path）。
        """
        pil = PILImage.fromarray(array, mode=mode) if mode else PILImage.fromarray(array)
        return cls(pil=pil, metadata=metadata or {})

    def with_metadata(self, **kwargs: Any) -> "Image":
        """返回带附加元数据的新 Image（不可变更新）。

        Args:
            **kwargs: 要追加的元数据键值对。

        Returns:
            新的 Image 实例（pil 引用相同，metadata 合并）。
        """
        new_meta = {**self.metadata, **kwargs}
        return Image(pil=self.pil, path=self.path, metadata=new_meta)


# ===========================================================================
# Images：多图包装类（variadic 端口专用）
# ===========================================================================

@dataclass
class Images:
    """多图包装类（PortSpec.type="images", variadic=True）。

    设计考量：
        - 性能：包装类持 list 引用，不复制图像数据
        - 类型安全：明确类型，下游节点能区分单图/多图
        - 兼容性高：支持迭代 / 索引 / len()，类似 list 用法
        - 自动包装：from_single() 将单张 Image 包装为 Images([image])

    为何用 Images 包装类而非 list[Image]：
        | 维度       | list[Image]      | Images 包装类          |
        |------------|------------------|------------------------|
        | 类型安全   | 弱               | 强（PortSpec 校验）    |
        | 自动包装   | 不支持           | 支持 from_single()     |
        | 元数据传递 | 无               | 可携带（来源/EXIF）    |
        | 调试       | list 难定位      | 类有 __repr__          |

    使用示例：
        # 工厂方法
        images = Images.from_single(image)   # 长度=1
        images = Images.empty()               # 长度=0（下游节点跳过）

        # 类 list 操作
        for img in images:
            print(img.width)
        first = images[0]
        first_two = images[0:2]   # 切片返回 Images

    Attributes:
        items: 内部 Image 列表。

    Note:
        本类不保证 items 不可变（list 本身可变），但建议当作只读使用。
        需要修改时创建新 Images 实例。
    """

    items: list[Image] = field(default_factory=list)

    # ===== 容器协议 =====

    def __len__(self) -> int:
        """返回图片数量。"""
        return len(self.items)

    def __iter__(self) -> Iterator[Image]:
        """迭代图片。"""
        return iter(self.items)

    def __getitem__(self, index: Union[int, slice]) -> Union[Image, "Images"]:
        """索引访问。

        - int 索引：返回 Image
        - slice 切片：返回 Images（保持类型一致性）
        """
        if isinstance(index, int):
            return self.items[index]
        # slice 切片返回 Images 包装
        return Images(items=self.items[index])

    def __bool__(self) -> bool:
        """是否非空（len > 0）。"""
        return len(self.items) > 0

    def __repr__(self) -> str:
        if not self.items:
            return "Images(empty)"
        sizes = [f"{img.width}x{img.height}" for img in self.items[:3]]
        suffix = ", ..." if len(self.items) > 3 else ""
        return f"Images(n={len(self.items)}, sizes=[{', '.join(sizes)}{suffix}])"

    # ===== 工厂方法 =====

    @classmethod
    def from_single(cls, image: Image) -> "Images":
        """从单张图片创建 Images（长度=1）。

        用于将单图输出自动包装为多图数组，便于下游 variadic 端口消费。
        """
        return cls(items=[image])

    @classmethod
    def empty(cls) -> "Images":
        """创建空 Images（长度=0）。

        下游节点检测到空 Images 时应按需跳过（如 IPAdapter 无图时不应用）。
        """
        return cls(items=[])

    @classmethod
    def from_list(cls, images: list[Image]) -> "Images":
        """从 Image 列表创建 Images。"""
        return cls(items=list(images))

    # ===== 辅助方法 =====

    def append(self, image: Image) -> None:
        """追加一张图片（原地修改）。"""
        self.items.append(image)

    def extend(self, images: "Images") -> None:
        """追加另一组 Images（原地修改）。"""
        self.items.extend(images.items)

    def first(self) -> Optional[Image]:
        """返回第一张图片，空则返回 None。"""
        return self.items[0] if self.items else None

    @property
    def paths(self) -> list[Optional[Path]]:
        """所有图片路径列表（无路径的为 None）。"""
        return [img.path for img in self.items]


# ===========================================================================
# Audio：单段音频
# ===========================================================================

@dataclass
class Audio:
    """单段音频数据类型（PortSpec.type="audio"）。

    Attributes:
        samples: 音频采样数据。形状 (channels, samples) 或 (samples,)。
            单声道可为 (samples,)，多声道为 (channels, samples)。
            dtype 通常为 float32，范围 [-1.0, 1.0]。
        sample_rate: 采样率（Hz）。常用 16000 / 22050 / 44100 / 48000。
        path: 音频文件路径（如已落盘）。None 表示仅内存对象。
        metadata: 元数据（如 speaker / language / duration）。
    """

    samples: np.ndarray
    sample_rate: int
    path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """音频时长（秒）。"""
        if self.samples.size == 0:
            return 0.0
        return self.samples.shape[-1] / self.sample_rate

    @property
    def channels(self) -> int:
        """声道数。"""
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]


# ===========================================================================
# Video：单个视频
# ===========================================================================

@dataclass
class Video:
    """单个视频数据类型（PortSpec.type="video"）。

    Attributes:
        frames: 视频帧列表（每帧是一个 Image）。None 表示尚未解码帧
            （仅有 path，需要时再解码）。
        path: 视频文件路径。None 表示仅内存对象。
        fps: 帧率（每秒帧数）。默认 30。
        metadata: 元数据（如 codec / duration / resolution）。
    """

    frames: Optional[list[Image]] = None
    path: Optional[Path] = None
    fps: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        """帧数（若 frames 为 None 返回 0）。"""
        return len(self.frames) if self.frames else 0

    @property
    def duration_seconds(self) -> float:
        """视频时长（秒）。"""
        if not self.frames or self.fps <= 0:
            return 0.0
        return len(self.frames) / self.fps


# ===========================================================================
# Tensor：torch.Tensor 包装（embeds 端口类型）
# ===========================================================================

@dataclass
class Tensor:
    """torch.Tensor 包装数据类型（PortSpec.type="embeds"）。

    用于在节点间传递模型嵌入（embeddings），携带 device 元数据。

    Attributes:
        data: torch.Tensor 张量数据。
        device: 设备字符串（如 "cuda:0" / "cpu"）。
            冗余字段，便于不查询 tensor.device 即可知。
        metadata: 元数据（如 dtype / shape / model_id）。
    """

    data: "torch.Tensor"
    device: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """后置初始化：若 device 未指定，从 tensor 推断。"""
        if not self.device:
            try:
                self.device = str(self.data.device)
            except (AttributeError, RuntimeError):
                # data 不是真正的 torch.Tensor（torch 未安装）
                self.device = "unknown"


# ===========================================================================
# Latent：潜空间表示
# ===========================================================================

@dataclass
class Latent:
    """潜空间表示数据类型（PortSpec.type="latent"）。

    VAE 编码后的潜空间张量，作为 Sampler 的输入与 VAEDecode 的输入。

    Attributes:
        tensor: 潜空间张量。形状通常为 (batch, channels, h, w)。
        vae_type: VAE 类型标识（如 "sd15" / "sdxl" / "ernie-image"）。
            用于校验 VAEDecode 是否能解码此 latent。
        metadata: 元数据（如 scaling_factor / shift_factor）。
    """

    tensor: "torch.Tensor"
    vae_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# 类型注册表：PortSpec.type 字符串 → Python 类型映射
# ===========================================================================

# 端口类型字符串 → Python 类型的映射（用于运行时类型校验）
# 注意：torch 相关类型在无 torch 环境下为 None，校验时需特殊处理
PORT_TYPE_MAP: dict[str, Any] = {
    "image": Image,
    "images": Images,
    "audio": Audio,
    "audios": list,  # list[Audio]，运行时为 list
    "video": Video,
    "text": str,
    "embeds": Tensor,
    "latent": Latent,
    "model": Any,  # 模型对象，任意类型
    "any": Any,
}


def get_port_type_class(type_name: str) -> Any:
    """根据 PortSpec.type 字符串查询对应的 Python 类型。

    Args:
        type_name: 端口类型字符串（如 "image" / "images" / "embeds"）。

    Returns:
        对应的 Python 类型。未知类型返回 Any。
        torch 相关类型（embeds / latent）在 torch 未安装时
        仍返回 Tensor / Latent 类（定义本身不依赖 torch 运行时）。
    """
    return PORT_TYPE_MAP.get(type_name, Any)


def is_valid_port_type(type_name: str) -> bool:
    """检查端口类型字符串是否合法。

    Args:
        type_name: 端口类型字符串。

    Returns:
        True 表示合法（在 PORT_TYPE_MAP 中）。
    """
    return type_name in PORT_TYPE_MAP


__all__ = [
    # 数据类型
    "Image",
    "Images",
    "Audio",
    "Video",
    "Tensor",
    "Latent",
    # 类型映射
    "PORT_TYPE_MAP",
    "get_port_type_class",
    "is_valid_port_type",
]

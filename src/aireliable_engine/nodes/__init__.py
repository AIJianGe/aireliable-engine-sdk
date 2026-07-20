"""平台基础节点库（v0.2）。

按 v0.2-analysis.md §11.2 / PR/architecture.md §12.2 组织：

    input/           输入节点（数据接入）
        ImagesInput          - 多图输入（variadic，v0.2 核心）
        TextInput            - 文本输入（任务 3.2）
        AudioInput           - 音频输入（后续阶段）
        VideoInput           - 视频输入（后续阶段）

    loader/          加载节点（模型资源）
        ModelLoader          - 主模型加载（任务 3.3）
        VAELoader            - VAE 加载（任务 3.4）
        CLIPTextEncodeLoader - CLIP 文本编码器加载
        ...

    encode/          编码节点（特征提取）
        CLIPTextEncode       - CLIP 文本编码（任务 3.5）
        ...

    sample/          采样节点
        Sampler              - 通用采样器（任务 3.6）
        Scheduler            - 调度器

    decode/          解码节点
        VAEDecode            - VAE 解码（任务 3.7）
        VAEEncode            - VAE 编码（后续阶段）

    postprocess/    后处理节点
        ImageSave            - 图片保存（任务 3.8）
        VideoSave            - 视频保存（后续阶段）
        AudioSave            - 音频保存（后续阶段）

各节点通过 entry points 或显式 register_node 注册到 NodeRegistry。
导入子包即触发节点注册。
"""

# 子包导入会触发节点自动注册
from .input.images_input import ImagesInputNode
from .input.text_input import TextInputNode
from .loader.model_loader import ModelLoaderNode
from .loader.vae_loader import VAELoaderNode
from .encode.clip_encode import CLIPTextEncodeNode
from .sample.sampler import SamplerNode
from .decode.vae_decode import VAEDecodeNode
from .postprocess.image_save import ImageSaveNode

__all__ = [
    "ImagesInputNode",
    "TextInputNode",
    "ModelLoaderNode",
    "VAELoaderNode",
    "CLIPTextEncodeNode",
    "SamplerNode",
    "VAEDecodeNode",
    "ImageSaveNode",
]

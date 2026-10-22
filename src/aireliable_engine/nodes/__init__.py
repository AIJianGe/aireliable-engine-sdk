"""平台基础节点库（v0.2 + v0.2.5 通用节点改造）。

按 v0.2-analysis.md §11.2 / PR/architecture.md §12.2 组织：

    input/           输入节点（数据接入）
        ImagesInput          - 多图输入（variadic，v0.2 核心）
        TextInput            - 文本输入（任务 3.2）

    loader/          加载节点（模型资源）
        ModelLoader          - 主模型加载（v0.2.5 含 pipeline/components 默认实现）
        VAELoader            - VAE 加载（v0.2.5 含 AutoencoderKL 默认实现）
        UNetLoader           - UNet/Transformer 加载（v0.2.5 ComfyUI 风格组件分离）
        CLIPLoader           - CLIP 文本编码器加载（v0.2.5 ComfyUI 风格组件分离）

    encode/          编码节点（特征提取）
        CLIPTextEncode       - CLIP 文本编码（v0.2.5 含 CLIPTokenizer 默认实现）
        VAEEncode            - VAE 编码（v0.2.5 对标 ComfyUI VAEEncode，支持多图批量）

    sample/          采样节点
        Sampler              - 通用采样器（v0.2.5 含 pipeline/components 双架构默认实现）
        EmptyLatentImage     - 空 latent 生成（v0.2.5 新增，参考 ComfyUI）
        ModelSamplingAuraFlow - AuraFlow shift 采样参数设置（v0.2.5 ComfyUI 风格）

    decode/          解码节点
        VAEDecode            - VAE 解码（v0.2.5 含 VAE.decode 默认实现）

    lora/            LoRA 加载节点（v0.2.5 新增）
        LoRALoader           - 简单版 LoRA 加载器（model → model）
        LoRALoaderAdvanced   - 高级版 LoRA 加载器（model+clip → model+clip）

    postprocess/    后处理节点
        ImageSave            - 图片保存（任务 3.8）

v0.2.5 通用节点改造要点：
    - 所有平台基础节点都添加 @register_node 装饰器
    - 纯 SDK 安装时 NodeRegistry 自动注册全部 10 个平台节点
    - 引擎节点与平台节点通过 node_type 区分（引擎覆盖同名平台节点）

各节点通过 @register_node 装饰器自动注册到 NodeRegistry。
导入子包即触发节点注册。
"""

# 子包导入会触发节点自动注册
from .input.images_input import ImagesInputNode
from .input.text_input import TextInputNode
from .loader.model_loader import ModelLoaderNode
from .loader.vae_loader import VAELoaderNode
# v0.2.5 ComfyUI 风格节点补充（C1）：UNet 独立加载器
from .loader.unet_loader import UNetLoaderNode
# v0.2.5 ComfyUI 风格节点补充（C2）：CLIP 独立加载器
from .loader.clip_loader import CLIPLoaderNode
from .encode.clip_encode import CLIPTextEncodeNode
from .encode.vae_encode import VAEEncodeNode
from .sample.sampler import SamplerNode
from .sample.empty_latent import EmptyLatentImageNode
# v0.2.5 ComfyUI 风格节点补充（C3）：AuraFlow shift 采样参数设置
from .sample.model_sampling_aura_flow import ModelSamplingAuraFlowNode
from .decode.vae_decode import VAEDecodeNode
from .lora.lora_loader import LoRALoaderNode, LoRALoaderAdvancedNode
from .postprocess.image_save import ImageSaveNode

__all__ = [
    # 输入节点
    "ImagesInputNode",
    "TextInputNode",
    # 加载节点
    "ModelLoaderNode",
    "VAELoaderNode",
    "UNetLoaderNode",
    "CLIPLoaderNode",
    # 编码节点
    "CLIPTextEncodeNode",
    "VAEEncodeNode",
    # 采样节点
    "SamplerNode",
    "EmptyLatentImageNode",
    "ModelSamplingAuraFlowNode",
    # 解码节点
    "VAEDecodeNode",
    # LoRA 节点（v0.2.5 新增）
    "LoRALoaderNode",
    "LoRALoaderAdvancedNode",
    # 后处理节点
    "ImageSaveNode",
]

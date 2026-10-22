"""加载器节点子包。

包含模型加载相关的节点：
    ModelLoaderNode  - 主模型加载（一体化，含 pipeline/components 双模式）
    VAELoaderNode    - VAE 加载
    UNetLoaderNode   - UNet/Transformer 加载（v0.2.5 ComfyUI 风格组件分离）
    CLIPLoaderNode   - CLIP 文本编码器加载（v0.2.5 ComfyUI 风格组件分离）

导入子包即触发节点注册。
"""

from .model_loader import ModelLoaderNode
from .vae_loader import VAELoaderNode
# v0.2.5 ComfyUI 风格节点补充（C1）：UNet 独立加载器
from .unet_loader import UNetLoaderNode
# v0.2.5 ComfyUI 风格节点补充（C2）：CLIP 独立加载器
from .clip_loader import CLIPLoaderNode

__all__ = [
    "ModelLoaderNode",
    "VAELoaderNode",
    "UNetLoaderNode",
    "CLIPLoaderNode",
]

"""加载器节点子包。

包含模型加载相关的节点：
    ModelLoaderNode  - 主模型加载（抽象基类）
    VAELoaderNode    - VAE 加载（任务 3.4）
    CLIPTextEncodeLoaderNode / CLIPVisionLoaderNode / T5LoaderNode / LoRALoaderNode ...

导入子包即触发节点注册。
"""

from .model_loader import ModelLoaderNode
from .vae_loader import VAELoaderNode

__all__ = ["ModelLoaderNode", "VAELoaderNode"]

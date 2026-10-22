"""LoRA 节点子包（v0.2.5 新增）。

包含 LoRA 加载相关节点：
    LoRALoaderNode          - 简单版 LoRA 加载器（model → model）
    LoRALoaderAdvancedNode  - 高级版 LoRA 加载器（model+clip → model+clip）

导入子包即触发节点注册。
"""

from .lora_loader import LoRALoaderNode, LoRALoaderAdvancedNode

__all__ = ["LoRALoaderNode", "LoRALoaderAdvancedNode"]

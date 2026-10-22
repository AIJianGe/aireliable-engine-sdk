"""采样节点子包。

包含采样相关的节点：
    SamplerNode                   - 通用采样器（v0.2 任务 3.6 + v0.2.5 默认实现）
    EmptyLatentImageNode          - 空 latent 生成（v0.2.5 新增，参考 ComfyUI）
    ModelSamplingAuraFlowNode     - AuraFlow shift 采样参数设置（v0.2.5 ComfyUI 风格）

导入子包即触发节点注册。
"""

from .sampler import SamplerNode
from .empty_latent import EmptyLatentImageNode
# v0.2.5 ComfyUI 风格节点补充（C3）：AuraFlow shift 采样参数设置
from .model_sampling_aura_flow import ModelSamplingAuraFlowNode

__all__ = [
    "SamplerNode",
    "EmptyLatentImageNode",
    "ModelSamplingAuraFlowNode",
]

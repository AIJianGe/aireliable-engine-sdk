"""编码节点子包。

包含特征提取相关的节点：
    CLIPTextEncodeNode  - CLIP 文本编码（任务 3.5）
    T5EncodeNode / CLIPVisionEncodeNode ...

导入子包即触发节点注册。
"""

from .clip_encode import CLIPTextEncodeNode

__all__ = ["CLIPTextEncodeNode"]

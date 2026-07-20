"""后处理节点子包。

包含：
    - ImageSave：图片保存节点（任务 3.8）
    - VideoSave：视频保存节点（后续阶段）
    - AudioSave：音频保存节点（后续阶段）
    - MetadataInject：元数据注入节点（后续阶段）
"""

from .image_save import ImageSaveNode

__all__ = ["ImageSaveNode"]

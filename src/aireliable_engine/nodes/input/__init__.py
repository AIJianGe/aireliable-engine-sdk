"""输入节点子包。

导入子包即触发节点注册到 NodeRegistry 默认实例。
"""

from .images_input import ImagesInputNode
from .text_input import TextInputNode

__all__ = ["ImagesInputNode", "TextInputNode"]

"""输入节点子包。

按 v0.2-analysis.md §11.2 / PR/architecture.md §12.2 组织。
导入子包即触发节点注册到 NodeRegistry 默认实例。
"""

from .images_input import ImagesInputNode
from .text_input import TextInputNode

__all__ = ["ImagesInputNode", "TextInputNode"]

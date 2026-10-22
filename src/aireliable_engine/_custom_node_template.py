"""自定义节点模板（v0.2.5 P0-J4）。

本文件是用户自定义节点的开发模板，展示最小可用节点结构。

使用方式：
    1. 复制本文件到 ``engines/<engine_id>/custom_nodes/`` 目录下。
    2. 重命名为有意义的名称（如 ``my_style_transfer.py``）。
    3. 修改 ``MyCustomNode`` 类名为你的节点名。
    4. 修改 ``node_type`` 类属性（必须全局唯一）。
    5. 实现 ``execute`` 方法。

注意：
    - 以 ``_`` 开头的文件会被 ``discover_local_dir`` 跳过，
      因此本模板文件不会被引擎扫描到，仅作为参考。
    - 节点类的 ``node_type`` 属性必须全局唯一，
      不可与内置节点（如 ``TextInput`` / ``ErnieSampler`` 等）重名。
    - 自定义节点的依赖请声明到 ``manifest.yaml`` 的
      ``custom_node_dependencies`` 字段，**严禁运行时 pip install**。

params_schema 编写规范（v0.2.5 P0-H1）：
    每个字段支持以下属性：
        - type: string / integer / number / boolean / enum / array / info
        - format: multiline / enum / enum-from-dir / file-paths / info
            （前端按 type + format 组合分发到不同 HTML 控件）
        - label: 中文显示名
        - description: tooltip 描述
        - required: 是否必填
        - default: 默认值
        - options: enum 类型的可选值列表
        - min / max: 数值范围

参考：
    - v0.2.5-workflow-editor-nodes-analysis.md §4 节点定义规范
    - PR/architecture.md §20.13 自定义节点扩展机制
    - PR/development-guide.md §28.x 自定义节点开发规范
"""

from __future__ import annotations

from typing import Any

from .node import BaseNode, PortSpec
from .workflow_context import WorkflowContext


class MyCustomNode(BaseNode):
    """我的自定义节点（示例模板）。

    这是一个最小可用节点示例，展示 BaseNode 的基本结构。
    请根据实际需求修改输入/输出端口和 execute 实现。
    """

    # ===== 节点元数据 =====
    node_type = "MyCustomNode"  # 必须全局唯一
    title = "My Custom Node"
    icon = "🔧"
    description = "我的自定义节点（示例模板）"

    # ===== 输入/输出端口 =====
    # PortSpec(name, type, required=True, default=None, description=None, variadic=False)
    # type 可选值：text / images / audio / video / embeds / latent / model / any
    input_specs: list[PortSpec] = [
        PortSpec(name="text", type="text", required=True, description="输入文本"),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(name="result", type="text", description="输出文本"),
    ]

    # ===== 参数 schema（v0.2.5 P0-H1）=====
    # 前端根据 params_schema 自动渲染属性面板控件。
    # type + format 组合决定渲染的 HTML 控件类型：
    #   - string + multiline → textarea
    #   - string → text input
    #   - integer / number → number input
    #   - boolean → toggle
    #   - enum → combo（select）
    #   - info → 只读卡片（不在节点上渲染，由属性面板展示）
    #   - array + file-paths → 自定义文件选择 widget
    #   - enum-from-dir → 自定义目录枚举 widget
    params_schema = {
        "prefix": {
            "type": "string",
            "default": "",
            "label": "前缀",
            "description": "输出文本前添加的前缀",
        },
        "suffix": {
            "type": "string",
            "default": "",
            "label": "后缀",
            "description": "输出文本后追加的后缀",
        },
        "uppercase": {
            "type": "boolean",
            "default": False,
            "label": "大写转换",
            "description": "是否将输出文本转为大写",
        },
    }

    # ===== execute 实现 =====
    async def execute(self, inputs: dict[str, Any], ctx: WorkflowContext) -> dict[str, Any]:
        """执行节点逻辑。

        Args:
            inputs: 输入端口数据字典，key 为 input_specs 中的 name。
            ctx: 工作流执行上下文，提供 logger / device / extra 等。

        Returns:
            输出端口数据字典，key 为 output_specs 中的 name。
        """
        text = inputs.get("text", "")
        if not isinstance(text, str):
            text = str(text)

        prefix = self.get_param("prefix", "") or ""
        suffix = self.get_param("suffix", "") or ""
        uppercase = self.get_param("uppercase", False)

        result = f"{prefix}{text}{suffix}"
        if uppercase:
            result = result.upper()

        ctx.logger.info("MyCustomNode: input=%r, output=%r", text, result)

        return {"result": result}

"""节点抽象基类（v0.2 新增）。

定义工作流编排的基本单元：节点（Node）。

设计参考：
    - v0.2-analysis.md §4.1 / §4.2 / §11.2
    - PR/architecture.md §12.2
    - PR/development-guide.md §28.1

核心概念：
    - BaseNode：所有节点的抽象基类，子类实现 execute(inputs, ctx) -> outputs
    - PortSpec：端口规格，描述输入/输出端口的类型与元数据
    - 节点间通过 dict 传递数据，key = 端口名

节点生命周期：
    1. __init__(params)：从工作流 yaml 的 params 字段初始化节点参数
    2. validate()：参数校验，返回错误列表（空列表表示通过）
    3. execute(inputs, ctx)：执行节点逻辑，返回输出字典
    4. cache_key(inputs)：（可选）返回缓存键，相同键的节点结果会被复用
    5. cleanup()：节点卸载时清理资源（如释放显存）

使用示例：
    class MyNode(BaseNode):
        node_type = "MyNode"
        input_specs = [PortSpec(name="text", type="text")]
        output_specs = [PortSpec(name="result", type="text")]

        async def execute(self, inputs, ctx):
            return {"result": inputs["text"].upper()}

    register_node("MyNode", MyNode)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .workflow_context import WorkflowContext


# ===========================================================================
# PortSpec：端口规格
# ===========================================================================

@dataclass
class PortSpec:
    """端口规格（输入/输出端口元数据）。

    用于描述节点的输入/输出端口，供：
        - WorkflowExecutor 做类型校验与自动包装
        - 画布 UI 渲染端口位置与类型
        - 节点包前端定义镜像同步

    Attributes:
        name: 端口名（如 "images" / "latent" / "prompt_embeds"）。
            同一节点内端口名必须唯一。
        type: 端口类型字符串，详见 PR/development-guide.md §28.7：
            - "image"   → Image（单张图片）
            - "images"  → Images（多图数组，variadic）
            - "audio"   → Audio（单段音频）
            - "audios"  → list[Audio]（多段音频）
            - "video"   → Video（单个视频）
            - "text"    → str（文本）
            - "embeds"  → Tensor（torch.Tensor + device 元数据）
            - "latent"  → Latent（潜空间表示）
            - "model"   → Any（模型对象，diffusers pipeline / transformer）
            - "any"     → Any（任意类型）
        required: 是否必填（仅对输入端口有意义）。默认 True。
            False 时若上游未提供，节点应使用 default 或跳过该输入。
        default: 默认值。required=False 时使用。
        description: 端口描述（用于画布 UI tooltip 与文档生成）。
        variadic: 是否为变长数组端口。True 表示该端口接受数组
            （如 images: Images 接受 0~N 张图）。默认 False。

    节点间数据传递规则详见 PR/development-guide.md §28.6。
    """

    name: str
    type: str = "any"
    required: bool = True
    default: Any = None
    description: str = ""
    variadic: bool = False

    def __post_init__(self) -> None:
        """端口规格基础校验。"""
        if not self.name:
            raise ValueError("PortSpec.name 不能为空字符串")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"PortSpec[{self.name}].type 必须是非空字符串")
        # variadic 端口类型建议复数形式（images / audios）
        # 这里不强制，仅做约定
        if self.variadic and self.type in {"image", "audio", "video"}:
            # 单数类型 + variadic=True 通常是笔误，但不阻止（开发者可能自定义）
            pass


# ===========================================================================
# BaseNode：节点抽象基类
# ===========================================================================

class BaseNode(ABC):
    """工作流节点抽象基类。

    所有节点继承 BaseNode，实现 execute(inputs, ctx) -> outputs。
    节点间通过 dict 传递数据（key = 端口名）。

    子类必须设置类属性：
        - node_type: str         节点类型唯一标识（如 "ModelLoader"）
        - input_specs: list[PortSpec]    输入端口规格（可为空列表）
        - output_specs: list[PortSpec]   输出端口规格（可为空列表）

    v0.2.5 新增：子类应设置前端元数据类属性（供 NodeRouter 自动暴露给画布 UI）：
        - title: str             中文显示名（如「ERNIE-Image 主模型加载」）
        - icon: str              emoji 图标（如「📦」），默认「•」
        - description: str       简短描述（tooltip 用），若空则从 __doc__ 提取首段
        - params_schema: dict   参数 schema（前端属性面板渲染用），结构：
                                 {param_name: {type, label, required, default, options, ...}}

    v0.2.5 节点系统优化（P-CategoryFix）：子类可显式声明分类类属性：
        - category: str          一级分类（input / loader / encode / sampler /
                                 decode / lora / output / custom）。
                                 若未声明，NodeRouter 用 ``_infer_category`` 从模块路径推断。
                                 **引擎自定义节点（模块路径非 ``aireliable_engine.nodes.*``）
                                 必须显式声明 category**，否则会被归到 ``custom``。
        - subcategory: str       二级分类（如 ``text`` / ``images`` / ``model`` / ``vae``）。
                                 若未声明，前端按 ``generic`` 归到「其他」子分类。
                                 建议与前端 ``NODE_CATEGORIES[cat].subcategories`` 的 key 对齐。

    v0.2.5 节点系统优化（P1-1）：params_schema 各项支持 ``advanced`` 字段：
        - advanced: bool         是否为高级参数（默认 False）。
                                 True 时前端属性面板将其折叠到「高级选项 ▾」区域，
                                 降低新用户认知负担。API 调用不受影响（仍按 param_name 注入）。
        典型用法：核心参数（如 steps / guidance / prompt.value）不标记 advanced，
                  调优参数（如 device / call_timeout / loras）标记 advanced=true。

    子类必须实现：
        - async execute(inputs, ctx) -> dict

    子类可选实现：
        - validate() -> list[str]        参数校验，返回错误信息列表
        - cache_key(inputs) -> str|None  返回缓存键（None 表示不缓存）
        - cleanup() -> None              资源清理

    设计原则：
        - 节点应无状态（stateless）：相同 inputs + params 应产生相同 outputs
            有状态的节点（如 LoRA 加载后切换 model）应通过 cleanup() 复原
        - 节点应幂等：重复执行同一节点不应产生副作用
        - 节点应可缓存：实现 cache_key 可显著加速工作流（如 ModelLoader）
        - 节点不应直接调用 HTTP / 文件系统以外的副作用接口

    Args:
        params: 节点参数字典，来自工作流 yaml 的 params 字段。
            子类应在 __init__ 中调用 super().__init__(params) 并解析所需字段。

    Example:
        class TextInput(BaseNode):
            node_type = "TextInput"
            input_specs = []
            output_specs = [PortSpec(name="text", type="text")]

            # v0.2.5 前端元数据
            title = "文本输入"
            icon = "📝"
            description = "从工作流输入字段读取文本"
            params_schema = {
                "field": {"type": "string", "required": True, "label": "字段名"},
            }

            def __init__(self, params: dict):
                super().__init__(params)
                self.field = params.get("field", "prompt")
                self.required = params.get("required", True)

            def validate(self) -> list[str]:
                errors = []
                if not self.field:
                    errors.append("field 参数必填")
                return errors

            async def execute(self, inputs, ctx):
                text = ctx.inputs.get(self.field, "")
                if self.required and not text:
                    raise ValueError(f"输入字段 {self.field} 必填")
                return {"text": text}
    """

    # ===== 类属性（子类必须覆盖）=====
    # 注意：使用类属性而非实例属性，便于节点注册表通过类直接查询规格
    # 子类覆盖时直接赋新列表，不会触发可变默认值问题（每个子类有独立列表对象）
    node_type: str = ""
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = []

    # ===== v0.2.5 新增：前端元数据类属性（子类可覆盖）=====
    # NodeRouter._node_to_definition() 通过 getattr() 读取这些属性，
    # 用于把节点类序列化为前端 NODE_DEFINITIONS 兼容格式（画布 UI 渲染用）。
    # 子类不覆盖时使用以下默认值。
    title: str = ""           # 中文显示名（空时 NodeRouter 兜底用 node_type）
    icon: str = "•"            # emoji 图标
    description: str = ""      # 简短描述（空时 NodeRouter 从 __doc__ 提取首段）
    params_schema: dict = {}   # 参数 schema（前端属性面板渲染用）

    # ===== 实例属性 =====

    def __init__(self, params: Optional[dict] = None) -> None:
        """初始化节点。

        Args:
            params: 节点参数字典。子类应先调用 super().__init__(params)
                再解析自己的字段。None 等价于 {}。
        """
        self.params: dict = dict(params) if params else {}

    # ===== 抽象方法（子类必须实现）=====

    @abstractmethod
    async def execute(self, inputs: dict, ctx: "WorkflowContext") -> dict:
        """执行节点逻辑。

        Args:
            inputs: 上游节点输出字典，key = 输入端口名，value = 上游输出值。
                若某输入端口 required=False 且上游未连接，则该 key 不存在。
            ctx: 工作流上下文，提供：
                - engine_id / variant / device / models_root 等元数据
                - vram_manager：显存管理器（如 offload / prefetch）
                - progress_reporter：进度上报（SSE 流）
                - cancel_event：取消事件（节点应定期检查并主动退出）

        Returns:
            输出字典，key = 输出端口名，value = 输出值。
            供下游节点通过 inputs 引用。

        Raises:
            任何异常都会被 WorkflowExecutor 捕获并记录，工作流执行中止。
            节点应在异常中携带清晰的错误信息以便调试。
        """
        raise NotImplementedError

    # ===== 可选方法（子类按需覆盖）=====

    def validate(self) -> list[str]:
        """参数校验。

        Returns:
            错误信息列表。空列表表示校验通过。
            WorkflowExecutor 在执行前会调用所有节点的 validate()，
            若任一节点返回非空错误列表，工作流拒绝执行。
        """
        return []

    def cache_key(self, inputs: dict) -> Optional[str]:
        """计算缓存键。

        Args:
            inputs: 上游节点输出（同 execute 的 inputs）。

        Returns:
            缓存键字符串。若返回非 None，WorkflowExecutor 会以此键缓存
            节点输出；后续相同键的节点调用会直接复用缓存结果，跳过 execute。
            返回 None 表示不缓存（默认行为）。

        使用场景：
            - ModelLoader：cache_key = f"{variant}:{cache_dir}:{model_id}"
            - CLIPTextEncode：cache_key = hash(text + clip_id)
            - Sampler：通常不缓存（每次推理都需执行）
        """
        return None

    def cleanup(self) -> None:
        """资源清理。

        在以下场景被调用：
            - 工作流执行完毕后（释放节点持有的临时资源）
            - 节点异常退出时（尝试清理已分配资源）
            - 引擎 unload 时（释放所有节点资源）

        子类应在此释放：
            - torch.Tensor（显存）
            - 文件句柄
            - 临时文件

        默认空实现。子类覆盖时应保证幂等性（多次调用不报错）。
        """
        pass

    # ===== 辅助方法 =====

    def get_param(self, key: str, default: Any = None) -> Any:
        """从 params 中读取字段（便捷方法）。

        Args:
            key: 参数名。
            default: 默认值（参数不存在时返回）。

        Returns:
            参数值或默认值。
        """
        return self.params.get(key, default)

    def get_input_spec(self, name: str) -> Optional[PortSpec]:
        """按名称查找输入端口规格。

        Args:
            name: 端口名。

        Returns:
            匹配的 PortSpec，未找到返回 None。
        """
        for spec in self.input_specs:
            if spec.name == name:
                return spec
        return None

    def get_output_spec(self, name: str) -> Optional[PortSpec]:
        """按名称查找输出端口规格。

        Args:
            name: 端口名。

        Returns:
            匹配的 PortSpec，未找到返回 None。
        """
        for spec in self.output_specs:
            if spec.name == name:
                return spec
        return None

    def __repr__(self) -> str:
        """节点可读表示（用于日志与调试）。"""
        return f"<{self.__class__.__name__} node_type={self.node_type!r}>"


# ===========================================================================
# 节点异常
# ===========================================================================

class NodeValidationError(Exception):
    """节点参数校验失败异常。

    WorkflowExecutor 在执行前会调用所有节点的 validate()，
    若任一节点返回非空错误列表，抛出此异常中止工作流执行。
    """

    def __init__(self, node_type: str, errors: list[str]):
        self.node_type = node_type
        self.errors = errors
        message = f"节点 {node_type} 参数校验失败：\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class NodeExecutionError(Exception):
    """节点执行失败异常。

    WorkflowExecutor 捕获节点 execute 抛出的任意异常后，
    包装为 NodeExecutionError 以携带节点 id 与类型信息。
    """

    def __init__(self, node_id: str, node_type: str, cause: Exception):
        self.node_id = node_id
        self.node_type = node_type
        self.cause = cause
        message = f"节点 {node_id} ({node_type}) 执行失败：{cause}"
        super().__init__(message)

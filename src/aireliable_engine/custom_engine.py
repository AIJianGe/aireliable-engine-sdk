"""CustomEngine 全自定义引擎基类（v0.2.5 恢复）。

适用于固定流程、封装式调用的引擎（如 TTS / Chat）。
子类需实现全部 7 个 ``_xxx_impl`` 方法。

与 WorkflowEngine 的区别：
    - WorkflowEngine：工作流模式，推理由 WorkflowExecutor 执行 yaml 工作流
    - CustomEngine：全自定义模式，推理由子类完全控制

UI 挂载：
    - WorkflowEngine：挂载工作流编辑器静态文件
    - CustomEngine：挂载 Gradio UI（或其他自定义 UI）

使用示例：

    from aireliable_engine import CustomEngine

    class MyTTSEngine(CustomEngine):
        def __init__(self, config):
            super().__init__(
                engine_id="my_tts",
                engine_name="MyTTS",
                capability="text-to-speech",
                engine_type="voice",
            )
            self.output_root = config.output_root

        async def _capabilities_impl(self):
            ...

        async def _load_impl(self, variant: str):
            ...

        async def _unload_impl(self):
            ...

        async def _infer_impl(self, request, stream):
            ...

        async def _status_impl(self):
            ...

        async def _load_lora_impl(self, name, weight):
            ...

        async def _unload_lora_impl(self, name):
            ...
"""

from __future__ import annotations

import logging

from .engine_base import BaseEngine

logger = logging.getLogger(__name__)


class CustomEngine(BaseEngine):
    """全自定义引擎基类。

    v0.2.5 恢复的范式，适用于：
        - 固定流程引擎（TTS / ASR / Chat）
        - 封装外部二进制的引擎（如 llama.cpp）
        - 不需要用户拖拽节点编排的引擎

    子类必须实现全部 7 个 ``_xxx_impl`` 方法：
        - _capabilities_impl
        - _load_impl
        - _unload_impl
        - _infer_impl
        - _status_impl
        - _load_lora_impl
        - _unload_lora_impl

    与 WorkflowEngine 的对比：

    | 维度       | WorkflowEngine            | CustomEngine              |
    |------------|---------------------------|---------------------------|
    | 推理实现   | WorkflowExecutor 执行 yaml | 子类完全控制 _infer_impl  |
    | 灵活度     | 受节点系统约束             | 可直接调用任意 Python API |
    | UI         | 工作流编辑器（拖拽节点）   | Gradio UI（表单交互）     |
    | 适用场景   | 需用户编排的复杂流程       | 固定流程、封装式调用      |
    | manifest   | paradigm: workflow        | paradigm: integrated      |
    """

    # __init_subclass__ 在 BaseEngine 中已豁免 CustomEngine，
    # 此处无需重复定义

    pass

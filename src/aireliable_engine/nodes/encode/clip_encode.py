"""CLIPTextEncode 节点：CLIP 文本编码节点（v0.2 + v0.2.5 通用节点改造）。

v0.2.5 通用节点改造（U1-3）：
    - 不再是纯抽象基类，提供基于 transformers CLIPTokenizer + CLIPTextModel 的默认实现
    - 引擎子类可覆盖 _encode_impl 实现自定义编码逻辑

v0.2.5 ComfyUI 风格改造（B1）：
    - ``clip`` 输入端口改为可选（``required=False``），画布无需再画 CLIP 线
    - CLIP 模型通过 ``ctx.extra["loaded_clip"]`` 全局共享（由 ModelLoader /
      ErnieGGUFModelLoader 加载完成后写入），与 ComfyUI 风格一致
    - 双轨 fallback：优先 ctx.extra，回退 inputs.clip（向后兼容旧 yaml）
    - 旧工作流（仍画 clip 线的）可继续工作

平台基础 CLIPTextEncodeNode 提供：
    - 标准输入端口（text / negative_text + 可选 clip）
    - 标准输出端口（embeds / negative_embeds）
    - 标准参数（max_length / batch_size / truncate）
    - execute 流程（参数校验 → 进度上报 → 调用 _encode_impl → 返回）
    - 默认 _encode_impl 实现（基于 transformers CLIPTextModel + CLIPTokenizer）

支持 prompt + negative_prompt 两路编码，符合 OpenAI / diffusers 标准。

参考：
    - PR/architecture.md §12.2 / §12.2.6（节点示例）
    - PR/development-guide.md §28.5（数据类型规范）
    - v0.2-tasks.md 任务 3.5
    - v0.2.5-universal-nodes-analysis.md §五（3）
    - v0.2.5 ComfyUI 风格改造：ctx.extra 全局 CLIP 共享（方向 B）
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...types import Tensor
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class CLIPTextEncodeNode(BaseNode):
    """CLIP 文本编码节点（v0.2.5 通用节点改造后含默认实现）。

    平台基础节点库提供：
        - 标准化输入输出端口（text / 可选 clip → embeds / negative_embeds）
        - 标准化编码流程（参数校验 → 进度上报 → _encode_impl）
        - **v0.2.5 默认 _encode_impl 实现**（基于 transformers CLIPTextModel + CLIPTokenizer）

    v0.2.5 ComfyUI 风格改造（B1）：
        - ``clip`` 端口改为可选（``required=False``），画布无需画 CLIP 线
        - CLIP 通过 ``ctx.extra["loaded_clip"]`` 全局共享（由 ModelLoader
          / ErnieGGUFModelLoader 加载完成后写入）
        - 双轨 fallback：优先 ``ctx.extra["loaded_clip"]``，回退 ``inputs["clip"]``
          （向后兼容旧 yaml）

    引擎子类只需实现 `_encode_impl(text, negative_text, clip, ctx) -> dict`：
        - 返回 dict，必须包含 "embeds" 字段（Tensor 类型）
        - 可选包含 "negative_embeds" 字段（None 表示无负向编码）

    参数（params）：
        max_length (int)：最大 token 长度，默认 77（CLIP 标准）。
        batch_size (int)：batch 大小，默认 1。
        truncate (bool)：超长文本是否截断，默认 True。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
        pool_output (bool)：是否输出 pooled embeddings
            （用于 SDXL/Flux 等 pipeline），默认 False。
        return_pooled (bool)：同 pool_output（兼容命名），默认 False。
    """

    node_type = "CLIPTextEncode"
    # v0.2.5 P-CategoryFix：显式声明分类
    category = "encode"
    subcategory = "text"
    title = "CLIP 文本编码"
    icon = "🔤"
    description = "用 CLIP 文本编码器编码 prompt（基于 transformers CLIPTextModel + CLIPTokenizer）"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="text",
            type="text",
            required=True,
            description="正向文本 prompt",
        ),
        PortSpec(
            name="negative_text",
            type="text",
            required=False,
            description="负向文本 prompt（可选，None 表示无负向编码）",
        ),
        PortSpec(
            name="clip",
            type="model",
            required=False,
            description=(
                "CLIP 文本编码器模型（可选）。"
                "v0.2.5 ComfyUI 风格改造：推荐通过 ctx.extra['loaded_clip'] "
                "全局共享，无需显式连线；保留此端口仅为向后兼容旧工作流。"
            ),
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="embeds",
            type="embeds",
            description="正向 prompt 编码后的 embedding（Tensor）",
        ),
        PortSpec(
            name="negative_embeds",
            type="embeds",
            required=False,
            description="负向 prompt 编码后的 embedding（Tensor），None 表示无负向编码",
        ),
    ]
    params_schema = {
        "max_length": {
            "type": "integer", "required": False, "default": 77, "min": 1, "label": "最大长度",
            "description": "最大 token 长度（CLIP 标准为 77）",
        },
        "batch_size": {
            "type": "integer", "required": False, "default": 1, "min": 1, "label": "批量大小",
            "advanced": True,
            "description": "batch 大小",
        },
        "truncate": {
            "type": "boolean", "required": False, "default": True, "label": "截断超长文本",
            "advanced": True,
            "description": "超长文本是否截断",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device",
        },
        "pool_output": {
            "type": "boolean", "required": False, "default": False, "label": "输出 pooled",
            "advanced": True,
            "description": "是否输出 pooled embeddings（用于 SDXL/Flux）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._max_length: int = int(self.get_param("max_length", 77))
        self._batch_size: int = int(self.get_param("batch_size", 1))
        # 不强制 bool() 转换：保留原值以便 validate 检查类型
        self._truncate = self.get_param("truncate", True)
        self._device_override: Optional[str] = self.get_param("device")
        self._pool_output = self.get_param("pool_output", False)
        self._return_pooled = self.get_param("return_pooled", False)

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not isinstance(self._max_length, int) or self._max_length <= 0:
            errors.append(f"params.max_length 必须为正整数，当前值：{self._max_length}")
        if not isinstance(self._batch_size, int) or self._batch_size <= 0:
            errors.append(f"params.batch_size 必须为正整数，当前值：{self._batch_size}")
        if not isinstance(self._truncate, bool):
            errors.append(f"params.truncate 必须为 bool，当前值：{self._truncate!r}")
        if not isinstance(self._pool_output, bool):
            errors.append(f"params.pool_output 必须为 bool，当前值：{self._pool_output!r}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：编码 prompt + negative_prompt 并返回 embeds / negative_embeds。

        流程：
            1. 参数校验
            2. 提取 inputs.text
            3. **v0.2.5 ComfyUI 风格**：调用 _resolve_clip(inputs, ctx) 双轨查找 CLIP
               - 优先 ctx.extra["loaded_clip"]（全局共享，由 ModelLoader 写入）
               - 回退 inputs["clip"]（向后兼容旧工作流）
            4. 上报开始进度
            5. 调用 _encode_impl（子类实现）
            6. 检查返回结果，标准化为 {embeds, negative_embeds}
            7. 上报完成进度
            8. 返回

        注意：
            - inputs.negative_text 是可选的（None 表示无负向编码）
            - inputs.clip 现在是可选的（v0.2.5 B1），可由 ctx.extra 提供
            - 子类 _encode_impl 接收 negative_text 参数，None 时只编码正向
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        text = inputs.get("text")
        negative_text = inputs.get("negative_text")  # 可选

        if not text:
            raise NodeValidationError(
                self.node_type,
                ["inputs.text 必填且必须为非空字符串"],
            )

        # v0.2.5 ComfyUI 风格改造（B1）：双轨查找 CLIP
        # 优先 ctx.extra["loaded_clip"]（全局共享，无需画线）
        # 回退 inputs["clip"]（向后兼容旧工作流）
        clip, clip_source = self._resolve_clip(inputs, ctx)
        if clip is None:
            raise NodeValidationError(
                self.node_type,
                [
                    "未找到 CLIP 文本编码器。请：\n"
                    "  1. 在工作流中放置 ModelLoader 节点加载模型（会自动写入 ctx.extra['loaded_clip']）；\n"
                    "  2. 或通过 inputs.clip 端口显式连接 CLIP 输出（向后兼容模式）。"
                ],
            )

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="clip_encode.start",
            message="开始 CLIP 文本编码",
            text_preview=str(text)[:100],
            has_negative=negative_text is not None,
            max_length=self._max_length,
            device=device,
            clip_source=clip_source,
        )

        # 调用子类实现
        try:
            result = await self._encode_impl(text, negative_text, clip, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _encode_impl 方法，"
                "请继承 CLIPTextEncodeNode 并实现该方法"
            )
        except Exception as exc:
            logger.exception("CLIPTextEncode: 编码失败")
            raise RuntimeError(f"CLIP 文本编码失败: {exc}") from exc

        if not isinstance(result, dict) or "embeds" not in result:
            raise TypeError(
                f"_encode_impl 必须返回包含 'embeds' 字段的 dict，"
                f"实际返回：{type(result).__name__}"
            )

        # 标准化输出
        standardized = {
            "embeds": result["embeds"],
            "negative_embeds": result.get("negative_embeds"),
        }

        await ctx.report_progress(
            1.0,
            stage="clip_encode.done",
            message="CLIP 文本编码完成",
        )
        return standardized

    def _resolve_clip(
        self, inputs: dict, ctx: WorkflowContext
    ) -> tuple[Any, str]:
        """双轨查找 CLIP 文本编码器（v0.2.5 ComfyUI 风格改造 B1）。

        优先级：
            1. ``ctx.extra["loaded_clip"]``（全局共享，由 ModelLoader /
               ErnieGGUFModelLoader 加载完成后写入，无需画线，ComfyUI 风格）
            2. ``inputs["clip"]``（显式端口连接，向后兼容旧工作流）

        Args:
            inputs: 节点输入字典。
            ctx: 工作流上下文（提供 extra 全局状态）。

        Returns:
            元组 ``(clip, source)``：
                - ``clip``：CLIP 对象（None 表示未找到）
                - ``source``：来源标识，"ctx.extra" / "inputs.clip" / "none"

        Note:
            旧工作流迁移指南：若 yaml 仍显式连接 ``clip: main_model.output.clip``，
            本方法会回退到 inputs.clip；若 yaml 已删除该引用，则从 ctx.extra 读取。
            推荐删除 yaml 中的 clip 引用以简化画布。
        """
        # 优先级 1：ctx.extra["loaded_clip"]（ComfyUI 风格全局共享）
        ctx_clip = ctx.extra.get("loaded_clip")
        if ctx_clip is not None:
            logger.debug("CLIPTextEncode: 从 ctx.extra['loaded_clip'] 获取 CLIP")
            return ctx_clip, "ctx.extra"

        # 优先级 2：inputs["clip"]（向后兼容旧工作流）
        inputs_clip = inputs.get("clip")
        if inputs_clip is not None:
            logger.debug("CLIPTextEncode: 从 inputs.clip 获取 CLIP（向后兼容模式）")
            return inputs_clip, "inputs.clip"

        # 都没有
        return None, "none"

    async def _encode_impl(
        self,
        text: str,
        negative_text: Optional[str],
        clip: Any,
        ctx: WorkflowContext,
    ) -> dict:
        """默认编码实现（v0.2.5 通用节点改造）。

        基于 transformers CLIPTokenizer + CLIPTextModel，适用于标准 SD/SDXL/Flux 等
        diffusers 模型场景。支持多种 clip 输入形式：
            - diffusers pipeline 对象（提取 .tokenizer / .text_encoder）
            - tuple (tokenizer, text_model)
            - dict {"tokenizer": ..., "text_model": ...}
            - 目录路径（str / Path）：用 CLIPTokenizer.from_pretrained +
              CLIPTextModel.from_pretrained 加载

        Args:
            text: 正向 prompt 文本。
            negative_text: 负向 prompt 文本（None 表示无负向编码）。
            clip: CLIP 文本编码器（pipeline / tuple / dict / 路径）。
            ctx: 工作流上下文。

        Returns:
            dict 包含 "embeds" 字段（Tensor）和可选 "negative_embeds" 字段。

        Raises:
            ImportError: transformers 未安装时抛出（带安装指引）。
            RuntimeError: 编码失败时抛出。
        """
        tokenizer, text_model = self._extract_tokenizer_and_model(clip, ctx)
        device = self._device_override or ctx.device

        # 延迟导入 transformers
        try:
            import torch  # noqa: F401  用于类型检查与设备转移
        except ImportError as exc:
            raise ImportError(
                "默认 CLIPTextEncode 实现需要 torch 库，但未安装。"
                "请执行 `pip install torch transformers`，"
                "或在引擎中继承 CLIPTextEncodeNode 覆盖 _encode_impl。"
            ) from exc

        # 将 text_model 移至设备
        if device and hasattr(text_model, "to"):
            try:
                await asyncio.to_thread(text_model.to, device)
            except Exception as exc:
                logger.warning("将 CLIPTextModel 移至设备 %s 失败：%s", device, exc)

        # 编码正向
        embeds = await asyncio.to_thread(
            self._encode_one, text, tokenizer, text_model
        )
        # 编码负向（可选）
        negative_embeds = None
        if negative_text:
            negative_embeds = await asyncio.to_thread(
                self._encode_one, negative_text, tokenizer, text_model
            )

        return {
            "embeds": embeds,
            "negative_embeds": negative_embeds,
        }

    def _extract_tokenizer_and_model(
        self, clip: Any, ctx: WorkflowContext
    ) -> tuple[Any, Any]:
        """从 clip 输入中提取 (tokenizer, text_model)。

        支持多种 clip 形式：
            - diffusers pipeline 对象（有 tokenizer / text_encoder 属性）
            - tuple/list (tokenizer, text_model)
            - dict {"tokenizer": ..., "text_model": ...}
            - 目录路径（str / Path）：用 transformers 加载

        Raises:
            NodeValidationError: 无法识别 clip 形式或加载失败。
        """
        # 情况 1：pipeline 对象（diffusers 各类 Pipeline 都有这俩属性）
        if hasattr(clip, "tokenizer") and hasattr(clip, "text_encoder"):
            return clip.tokenizer, clip.text_encoder

        # 情况 2：tuple/list (tokenizer, text_model)
        if isinstance(clip, (tuple, list)) and len(clip) == 2:
            tok, mdl = clip
            if tok is not None and mdl is not None:
                return tok, mdl

        # 情况 3：dict
        if isinstance(clip, dict):
            tok = clip.get("tokenizer")
            mdl = clip.get("text_model") or clip.get("text_encoder")
            if tok is not None and mdl is not None:
                return tok, mdl

        # 情况 4：目录路径
        if isinstance(clip, (str, Path)):
            try:
                from transformers import CLIPTokenizer, CLIPTextModel
            except ImportError as exc:
                raise ImportError(
                    "默认 CLIPTextEncode 实现需要 transformers 库，但未安装。"
                    "请执行 `pip install transformers`。"
                ) from exc
            clip_path = str(clip)
            try:
                tokenizer = CLIPTokenizer.from_pretrained(clip_path)
                text_model = CLIPTextModel.from_pretrained(clip_path)
                return tokenizer, text_model
            except Exception as exc:
                raise NodeValidationError(
                    self.node_type,
                    [f"从路径加载 CLIP 失败 ({clip_path}): {exc}"],
                )

        raise NodeValidationError(
            self.node_type,
            [
                "无法识别 inputs.clip 的形式。期望："
                "diffusers pipeline / (tokenizer, text_model) tuple / "
                "dict {tokenizer, text_model} / clip 目录路径。"
                f"实际类型：{type(clip).__name__}"
            ],
        )

    def _encode_one(
        self, text: str, tokenizer: Any, text_model: Any
    ) -> Any:
        """编码单条文本为 embedding。

        Args:
            text: 待编码文本。
            tokenizer: CLIPTokenizer 实例。
            text_model: CLIPTextModel 实例。

        Returns:
            embedding Tensor（shape: [batch, seq_len, hidden_dim]）。
        """
        # tokenize
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=self._max_length,
            truncation=self._truncate,
            return_tensors="pt",
        )
        # encode
        outputs = text_model(**tokens)
        # 优先取 last_hidden_state（标准 CLIP 编码输出）
        embeds = getattr(outputs, "last_hidden_state", None)
        if embeds is None and isinstance(outputs, dict):
            embeds = outputs.get("last_hidden_state")
        if embeds is None and isinstance(outputs, (tuple, list)):
            embeds = outputs[0]
        if embeds is None:
            raise RuntimeError(
                "CLIPTextModel 输出无 last_hidden_state 字段，"
                f"输出类型：{type(outputs).__name__}"
            )
        return embeds

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_length={self._max_length}, "
            f"batch_size={self._batch_size}, truncate={self._truncate!r})"
        )

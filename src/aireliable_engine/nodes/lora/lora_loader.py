"""LoRA 加载节点（v0.2.5 新增）。

提供两种 LoRA 加载节点，参考 ComfyUI 设计：

1. **LoRALoaderNode（简单版）**
    - 输入：model（diffusers Pipeline）
    - 输出：model（已加载 LoRA 的 Pipeline）
    - 实现：调用 `pipe.load_lora_weights(lora_path)`
    - 用途：单一模型 LoRA 加载场景

2. **LoRALoaderAdvancedNode（ComfyUI 风格高级版）**
    - 输入：model + clip（可选）
    - 输出：model + clip（已加载 LoRA）
    - 实现：同时支持 model LoRA 和 CLIP LoRA 加载
    - 用途：需要分别加载 model 和 CLIP LoRA 的场景（如 SDXL + 文本编码器 LoRA）

工作流位置（参考 ComfyUI）：
    ModelLoader → LoRALoader → CLIPTextEncode → Sampler → VAEDecode → ImageSave
    或
    ModelLoader → LoRALoaderAdvanced → CLIPTextEncode → Sampler → VAEDecode → ImageSave

参考：
    - ComfyUI LoraLoader / LoraLoaderAdvanced 节点
    - v0.2.5-universal-nodes-analysis.md §六（2）
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class LoRALoaderNode(BaseNode):
    """LoRA 加载节点（简单版，v0.2.5 新增）。

    平台基础节点库提供：
        - 标准化输入输出端口（model → model）
        - 标准化加载流程（参数校验 → 路径解析 → 进度上报 → 加载 LoRA）
        - 默认实现：调用 diffusers Pipeline.load_lora_weights()

    工作流示例：
        ModelLoader → LoRALoader(lora_name="add_detail") → CLIPTextEncode → Sampler

    参数（params）：
        lora_name (str)：LoRA 名称（如 "add_detail" / "anime_style"）。
        lora_path (Optional[str])：LoRA 文件路径（.safetensors / .pt / .bin）。
            None 时按 lora_name 自动构造默认路径：
                models/shared/lora/<lora_name>.safetensors
        strength (float)：LoRA 强度，默认 1.0。
            1.0 = 完全应用，0.0 = 不应用，2.0 = 双倍强度。
        strength_clip (Optional[float])：CLIP LoRA 强度（仅当 model 有 text_encoder 时生效）。
            None 表示与 strength 相同。
        source (str)：LoRA 来源，"shared"（默认，跨引擎复用）或 "engine_specific"。
        device (Optional[str])：覆盖 ctx.device（仅用于显存检查，不实际移动 LoRA）。
    """

    node_type = "LoRALoader"
    # v0.2.5 P-CategoryFix：lora 分类
    category = "lora"
    subcategory = "user"
    title = "LoRA 加载器（简单版）"
    icon = "🎀"
    description = "加载 LoRA 权重到模型（简单版，model → model）"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            required=True,
            description="主模型对象（diffusers Pipeline，由 ModelLoader 输出）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            description="加载 LoRA 后的模型对象（同一 Pipeline 实例）",
        ),
    ]
    params_schema = {
        "lora_name": {
            "type": "string", "required": True, "label": "LoRA 名称",
            "format": "enum-from-dir",
            "dir": "image/loras/",
            "extensions": [".safetensors", ".pt", ".bin", ".ckpt"],
            "description": "LoRA 名称（从 image/loras/ 目录扫描，如 add_detail / anime_style）",
        },
        "lora_path": {
            "type": "string", "required": False, "format": "file-paths", "label": "LoRA 文件路径",
            "advanced": True,
            "description": "LoRA 文件路径（None 时按 lora_name 自动构造，点击 📁 按钮选择文件）",
        },
        "strength": {
            "type": "number", "required": False, "default": 1.0, "min": 0, "label": "LoRA 强度",
            "description": "LoRA 强度（1.0=完全应用，0.0=不应用）",
        },
        "strength_clip": {
            "type": "number", "required": False, "label": "CLIP LoRA 强度",
            "advanced": True,
            "description": "CLIP LoRA 强度（None=与 strength 相同）",
        },
        "source": {
            "type": "enum", "required": False, "default": "shared", "label": "来源",
            "options": ["shared", "engine_specific"],
            "description": "shared=跨引擎复用，engine_specific=引擎专属",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device（仅用于显存检查）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._lora_name: str = self.get_param("lora_name", "")
        self._lora_path: Optional[str] = self.get_param("lora_path")
        self._strength: float = float(self.get_param("strength", 1.0))
        self._strength_clip: Optional[float] = self.get_param("strength_clip")
        self._source: str = self.get_param("source", "shared")
        self._device_override: Optional[str] = self.get_param("device")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._lora_name or not isinstance(self._lora_name, str):
            errors.append("params.lora_name 必填且必须为非空字符串")
        if not isinstance(self._strength, (int, float)) or self._strength < 0:
            errors.append(f"params.strength 必须为非负数，当前值：{self._strength}")
        if self._strength_clip is not None and (
            not isinstance(self._strength_clip, (int, float)) or self._strength_clip < 0
        ):
            errors.append(f"params.strength_clip 必须为非负数或 None，当前值：{self._strength_clip}")
        if self._source not in {"shared", "engine_specific"}:
            errors.append(
                f"params.source 必须为 'shared' 或 'engine_specific'，当前值：{self._source!r}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载 LoRA 到模型。

        流程：
            1. 参数校验
            2. 提取 model 输入
            3. 解析 LoRA 路径
            4. 上报开始进度
            5. 调用 model.load_lora_weights() 加载 LoRA
            6. 设置 LoRA scale（fuse_lora / set_adapters）
            7. 上报完成进度
            8. 返回 {"model": model}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        model = inputs.get("model")
        if model is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.model 必填（LoRALoader 需要主模型）"],
            )

        # 解析 LoRA 路径
        lora_path = self._resolve_lora_path(ctx)
        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="lora_loader.start",
            message=f"开始加载 LoRA {self._lora_name}",
            lora_name=self._lora_name,
            lora_path=str(lora_path),
            strength=self._strength,
            strength_clip=self._strength_clip,
            device=device,
        )

        # 在线程中加载 LoRA（阻塞调用）
        def _load_lora():
            # diffusers Pipeline.load_lora_weights 支持文件路径加载
            if not hasattr(model, "load_lora_weights"):
                raise NotImplementedError(
                    f"模型 {type(model).__name__} 不支持 load_lora_weights 方法，"
                    "需要 diffusers 0.25+ 的 Pipeline 实例"
                )
            try:
                # 尝试新版 API（diffusers 0.27+）
                model.load_lora_weights(str(lora_path))
                # 设置 LoRA 强度
                self._set_lora_scale(model)
            except TypeError as exc:
                # 兼容老版 API：可能需要 adapter_name 参数
                if "adapter_name" in str(exc):
                    adapter_name = f"lora_{self._lora_name}"
                    model.load_lora_weights(
                        str(lora_path),
                        adapter_name=adapter_name,
                    )
                    self._set_lora_scale(model, adapter_name=adapter_name)
                else:
                    raise
            return model

        try:
            model = await asyncio.to_thread(_load_lora)
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.exception(
                "LoRALoader: 加载 %s 失败 (path=%s)",
                self._lora_name,
                lora_path,
            )
            raise RuntimeError(
                f"LoRA 加载失败 ({self._lora_name}): {exc}"
            ) from exc

        await ctx.report_progress(
            1.0,
            stage="lora_loader.done",
            message=f"LoRA 加载完成 {self._lora_name}",
            lora_name=self._lora_name,
            strength=self._strength,
        )
        return {"model": model}

    def _set_lora_scale(self, model: Any, adapter_name: Optional[str] = None) -> None:
        """设置 LoRA 强度。

        Args:
            model: diffusers Pipeline 实例。
            adapter_name: 适配器名称（多 LoRA 加载时区分）。
        """
        try:
            # diffusers 0.27+：使用 set_adapters
            if hasattr(model, "set_adapters"):
                if adapter_name:
                    model.set_adapters(
                        [adapter_name],
                        adapter_weights=[self._strength],
                    )
                else:
                    # 单 LoRA 场景：取已加载的 adapter 名称
                    adapters = getattr(model, "lora_adapter_names", []) or ["default"]
                    if adapters:
                        model.set_adapters(
                            adapters,
                            adapter_weights=[self._strength] * len(adapters),
                        )
            # 兼容老版：fuse_lora + scale
            elif hasattr(model, "fuse_lora"):
                model.fuse_lora(lora_scale=self._strength)
        except Exception as exc:
            logger.warning(
                "LoRALoader: 设置 LoRA scale 失败（继续但强度可能不生效）：%s",
                exc,
            )

    def _resolve_lora_path(self, ctx: WorkflowContext) -> Path:
        """解析 LoRA 文件路径。

        规则：
            1. lora_path 显式指定：直接使用（相对路径相对于 models_root）
            2. lora_path 未指定：按 source 自动构造（v0.2.5 规范）
                - source="shared"：models/shared/lora/<lora_name>.safetensors
                - source="engine_specific"：models/image/loras/<engine_id>/<lora_name>.safetensors
                  （与 engines/e_909469ed21ed/config.py 的 LORA_SUBDIR=`image/loras/<engine_name>/` 对齐）

        Raises:
            NodeValidationError: 路径不存在。
        """
        if self._lora_path:
            lora_path = Path(self._lora_path)
            if not lora_path.is_absolute() and ctx.models_root is not None:
                lora_path = ctx.models_root / lora_path
        else:
            if ctx.models_root is None:
                raise NodeValidationError(
                    self.node_type,
                    ["未指定 lora_path 且 ctx.models_root 为 None，无法自动构造路径"],
                )
            if self._source == "shared":
                lora_path = ctx.models_root / "shared" / "lora" / f"{self._lora_name}.safetensors"
            else:
                # v0.2.5：与 config.py 的 lora_dir 一致
                # 默认格式：models/image/loras/<engine_id>/<lora_name>.safetensors
                # engine_id 从 ctx.engine_id 读取（由 WorkflowContext 注入）
                engine_id = ctx.engine_id or "unknown_engine"
                lora_path = (
                    ctx.models_root / "image" / "loras" / engine_id / f"{self._lora_name}.safetensors"
                )

        if not lora_path.exists():
            # 尝试其他常见扩展名
            for ext in [".pt", ".bin", ".ckpt"]:
                alt_path = lora_path.with_suffix(ext)
                if alt_path.exists():
                    return alt_path
            raise NodeValidationError(
                self.node_type,
                [f"LoRA 路径不存在: {lora_path}（也尝试了 .pt / .bin / .ckpt 扩展名）"],
            )
        return lora_path

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：返回 None 不缓存（每次加载可能有副作用）。

        实际场景：LoRA 加载会原地修改 model，缓存无意义。
        """
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lora_name={self._lora_name!r}, "
            f"strength={self._strength}, source={self._source!r})"
        )


@register_node
class LoRALoaderAdvancedNode(BaseNode):
    """LoRA 加载节点（高级版，v0.2.5 新增，参考 ComfyUI LoraLoaderAdvanced）。

    与 LoRALoaderNode 的差异：
        - 额外支持 clip 输入（可加载文本编码器 LoRA）
        - 同时输出 model 和 clip
        - 可分别设置 model LoRA 强度和 CLIP LoRA 强度

    平台基础节点库提供：
        - 标准化输入输出端口（model+clip → model+clip）
        - 标准化加载流程（参数校验 → 路径解析 → 加载 LoRA → 设置强度）
        - 默认实现：调用 diffusers load_lora_weights + 兼容多种 Pipeline 形态

    工作流示例：
        ModelLoader → LoRALoaderAdvanced(lora_name="style_x") → CLIPTextEncode → Sampler

    参数（params）：
        lora_name (str)：LoRA 名称。
        lora_path (Optional[str])：LoRA 文件路径。None 时按 lora_name 自动构造。
        strength_model (float)：model LoRA 强度，默认 1.0。
        strength_clip (float)：CLIP LoRA 强度，默认 1.0。
        source (str)：LoRA 来源，"shared"（默认）或 "engine_specific"。
        device (Optional[str])：覆盖 ctx.device（仅用于显存检查）。
    """

    node_type = "LoRALoaderAdvanced"
    # v0.2.5 P-CategoryFix：lora 分类
    category = "lora"
    subcategory = "user"
    title = "LoRA 加载器（高级版）"
    icon = "🎀"
    description = "加载 LoRA 权重到 model 和 clip（参考 ComfyUI LoraLoaderAdvanced）"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            required=True,
            description="主模型对象（diffusers Pipeline 或 transformer/UNet）",
        ),
        PortSpec(
            name="clip",
            type="model",
            required=False,
            description="CLIP 文本编码器（可选，由 ModelLoader / CLIPTextEncodeLoader 输出）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            description="加载 LoRA 后的模型对象",
        ),
        PortSpec(
            name="clip",
            type="model",
            required=False,
            description="加载 LoRA 后的 CLIP 对象（无 clip 输入时为 None）",
        ),
    ]
    params_schema = {
        "lora_name": {
            "type": "string", "required": True, "label": "LoRA 名称",
            "format": "enum-from-dir",
            "dir": "image/loras/",
            "extensions": [".safetensors", ".pt", ".bin", ".ckpt"],
            "description": "LoRA 名称（从 image/loras/ 目录扫描，如 add_detail / anime_style）",
        },
        "lora_path": {
            "type": "string", "required": False, "format": "file-paths", "label": "LoRA 文件路径",
            "advanced": True,
            "description": "LoRA 文件路径（None 时按 lora_name 自动构造，点击 📁 按钮选择文件）",
        },
        "strength_model": {
            "type": "number", "required": False, "default": 1.0, "min": 0, "label": "model LoRA 强度",
            "description": "model LoRA 强度（1.0=完全应用，0.0=不应用）",
        },
        "strength_clip": {
            "type": "number", "required": False, "default": 1.0, "min": 0, "label": "CLIP LoRA 强度",
            "description": "CLIP LoRA 强度（1.0=完全应用，0.0=不应用）",
        },
        "source": {
            "type": "enum", "required": False, "default": "shared", "label": "来源",
            "options": ["shared", "engine_specific"],
            "description": "shared=跨引擎复用，engine_specific=引擎专属",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device（仅用于显存检查）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._lora_name: str = self.get_param("lora_name", "")
        self._lora_path: Optional[str] = self.get_param("lora_path")
        self._strength_model: float = float(self.get_param("strength_model", 1.0))
        self._strength_clip: float = float(self.get_param("strength_clip", 1.0))
        self._source: str = self.get_param("source", "shared")
        self._device_override: Optional[str] = self.get_param("device")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._lora_name or not isinstance(self._lora_name, str):
            errors.append("params.lora_name 必填且必须为非空字符串")
        if not isinstance(self._strength_model, (int, float)) or self._strength_model < 0:
            errors.append(
                f"params.strength_model 必须为非负数，当前值：{self._strength_model}"
            )
        if not isinstance(self._strength_clip, (int, float)) or self._strength_clip < 0:
            errors.append(
                f"params.strength_clip 必须为非负数，当前值：{self._strength_clip}"
            )
        if self._source not in {"shared", "engine_specific"}:
            errors.append(
                f"params.source 必须为 'shared' 或 'engine_specific'，当前值：{self._source!r}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载 LoRA 到 model 和 clip。

        流程：
            1. 参数校验
            2. 提取 model / clip 输入
            3. 解析 LoRA 路径
            4. 上报开始进度
            5. 加载 model LoRA（pipeline.load_lora_weights 或 PEFT 加载）
            6. 加载 CLIP LoRA（若 clip 输入存在，且有 load_lora_weights 方法）
            7. 设置 LoRA scale
            8. 上报完成进度
            9. 返回 {"model": model, "clip": clip}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        model = inputs.get("model")
        clip = inputs.get("clip")  # 可选

        if model is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.model 必填"],
            )

        # 解析 LoRA 路径
        lora_path = self._resolve_lora_path(ctx)
        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="lora_loader_advanced.start",
            message=f"开始加载 LoRA {self._lora_name}（model+clip）",
            lora_name=self._lora_name,
            lora_path=str(lora_path),
            strength_model=self._strength_model,
            strength_clip=self._strength_clip,
            has_clip=clip is not None,
            device=device,
        )

        # 加载 model LoRA
        def _load_model_lora():
            if hasattr(model, "load_lora_weights"):
                try:
                    model.load_lora_weights(str(lora_path))
                    self._set_lora_scale(
                        model,
                        strength=self._strength_model,
                        adapter_name=f"lora_{self._lora_name}",
                    )
                except TypeError as exc:
                    if "adapter_name" in str(exc):
                        model.load_lora_weights(
                            str(lora_path),
                            adapter_name=f"lora_{self._lora_name}",
                        )
                        self._set_lora_scale(
                            model,
                            strength=self._strength_model,
                            adapter_name=f"lora_{self._lora_name}",
                        )
                    else:
                        raise
            else:
                # 非pipeline对象（如纯transformer），尝试 PEFT 加载
                logger.warning(
                    "LoRALoaderAdvanced: model %s 不支持 load_lora_weights，"
                    "跳过 model LoRA 加载",
                    type(model).__name__,
                )

        try:
            await asyncio.to_thread(_load_model_lora)
        except Exception as exc:
            logger.exception(
                "LoRALoaderAdvanced: 加载 model LoRA %s 失败",
                self._lora_name,
            )
            raise RuntimeError(
                f"model LoRA 加载失败 ({self._lora_name}): {exc}"
            ) from exc

        # 加载 CLIP LoRA（若 clip 输入存在）
        if clip is not None:
            def _load_clip_lora():
                # diffusers Pipeline 中的 text_encoder 通常支持 load_lora_weights
                # 独立 CLIPTextModel 不直接支持，需要通过 PEFT
                clip_actual = clip
                # 若 clip 是 dict 包装，解包
                if isinstance(clip, dict):
                    clip_actual = clip.get("text_model") or clip.get("text_encoder") or clip
                if hasattr(clip_actual, "load_lora_weights"):
                    try:
                        clip_actual.load_lora_weights(str(lora_path))
                        self._set_lora_scale(
                            clip_actual,
                            strength=self._strength_clip,
                            adapter_name=f"clip_lora_{self._lora_name}",
                        )
                    except Exception as exc:
                        logger.warning(
                            "LoRALoaderAdvanced: 加载 CLIP LoRA 失败（继续）："
                            f"{exc}"
                        )
                else:
                    logger.info(
                        "LoRALoaderAdvanced: clip 对象 %s 不支持 load_lora_weights，"
                        "跳过 CLIP LoRA 加载",
                        type(clip_actual).__name__,
                    )

            try:
                await asyncio.to_thread(_load_clip_lora)
            except Exception as exc:
                logger.warning(
                    "LoRALoaderAdvanced: CLIP LoRA 加载失败（继续）：%s",
                    exc,
                )

        await ctx.report_progress(
            1.0,
            stage="lora_loader_advanced.done",
            message=f"LoRA 加载完成 {self._lora_name}",
            lora_name=self._lora_name,
            strength_model=self._strength_model,
            strength_clip=self._strength_clip,
        )
        return {"model": model, "clip": clip}

    def _set_lora_scale(
        self,
        target: Any,
        *,
        strength: float,
        adapter_name: Optional[str] = None,
    ) -> None:
        """设置 LoRA 强度。

        Args:
            target: 目标模型对象（Pipeline / transformer / text_encoder）。
            strength: LoRA 强度。
            adapter_name: 适配器名称。
        """
        try:
            if hasattr(target, "set_adapters"):
                if adapter_name:
                    target.set_adapters(
                        [adapter_name],
                        adapter_weights=[strength],
                    )
            elif hasattr(target, "fuse_lora"):
                target.fuse_lora(lora_scale=strength)
        except Exception as exc:
            logger.warning(
                "LoRALoaderAdvanced: 设置 LoRA scale 失败：%s",
                exc,
            )

    def _resolve_lora_path(self, ctx: WorkflowContext) -> Path:
        """解析 LoRA 文件路径（v0.2.5 规范，与 LoRALoaderNode 保持一致）。

        规则：
            1. lora_path 显式指定：直接使用（相对路径相对于 models_root）
            2. lora_path 未指定：按 source 自动构造
                - source="shared"：models/shared/lora/<lora_name>.safetensors
                - source="engine_specific"：models/image/loras/<engine_id>/<lora_name>.safetensors
                  （与 engines/e_909469ed21ed/config.py 的 lora_dir 一致）
        """
        if self._lora_path:
            lora_path = Path(self._lora_path)
            if not lora_path.is_absolute() and ctx.models_root is not None:
                lora_path = ctx.models_root / lora_path
        else:
            if ctx.models_root is None:
                raise NodeValidationError(
                    self.node_type,
                    ["未指定 lora_path 且 ctx.models_root 为 None，无法自动构造路径"],
                )
            if self._source == "shared":
                lora_path = ctx.models_root / "shared" / "lora" / f"{self._lora_name}.safetensors"
            else:
                # v0.2.5：与 config.py 的 lora_dir 一致
                engine_id = ctx.engine_id or "unknown_engine"
                lora_path = (
                    ctx.models_root / "image" / "loras" / engine_id / f"{self._lora_name}.safetensors"
                )

        if not lora_path.exists():
            for ext in [".pt", ".bin", ".ckpt"]:
                alt_path = lora_path.with_suffix(ext)
                if alt_path.exists():
                    return alt_path
            raise NodeValidationError(
                self.node_type,
                [f"LoRA 路径不存在: {lora_path}（也尝试了 .pt / .bin / .ckpt 扩展名）"],
            )
        return lora_path

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：返回 None 不缓存。"""
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lora_name={self._lora_name!r}, "
            f"strength_model={self._strength_model}, "
            f"strength_clip={self._strength_clip}, source={self._source!r})"
        )

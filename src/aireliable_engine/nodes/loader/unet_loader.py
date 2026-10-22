"""UNetLoader 节点：单独加载 UNet/Transformer 模型（v0.2.5 ComfyUI 风格节点补充）。

ComfyUI 风格的组件分离加载节点，与 ModelLoader（一体化加载）并存：
    - ModelLoader：一次加载所有组件（model + vae + clip + scheduler），适合 SD/SDXL 标准模型
    - UNetLoader：仅加载 UNet/Transformer，适合 Flux/AuraFlow 等组件分离式架构

参考 ComfyUI 的 UNetLoader 节点设计：
    - 无输入端口（独立加载器）
    - 1 个输出端口（model）
    - 1 个核心参数（unet_name，从目录扫描）
    - 自动检测模型类型（UNet2DConditionModel / Transformer2DModel）

与 ModelLoader 的共存机制：
    - node_type 不同（"UNetLoader" vs "ModelLoader"），互不覆盖
    - 输出端口名相同（"model"），Sampler 节点接哪个都行
    - 加载完成后写 ctx.extra["loaded_model"]（与 ModelLoader 行为一致），
      让后续 ModelSamplingAuraFlow 等变换器节点可通过 ctx.extra 获取 model

参考：
    - v0.2.5-comfyui-nodes-analysis.md §二（1）
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
class UNetLoaderNode(BaseNode):
    """UNet/Transformer 加载节点（ComfyUI 风格组件分离加载器）。

    平台基础节点库提供的独立 UNet 加载器，与 ModelLoader 并存：
        - ModelLoader 一次加载所有组件（model + vae + clip + scheduler）
        - UNetLoader 仅加载 UNet/Transformer，其他组件由 VAELoader / CLIPLoader 独立加载

    自动检测模型类型：
        - 若路径是 .safetensors / .ckpt / .bin 文件 → UNet2DConditionModel.from_single_file
        - 若路径是目录且含 transformer/ 子目录 → Transformer2DModel.from_pretrained
        - 若路径是目录且含 diffusion_pytorch_model.safetensors → UNet2DConditionModel.from_pretrained
        - 默认 → UNet2DConditionModel.from_pretrained（兼容老模型）

    参数（params）：
        unet_name (str)：UNet 模型名称（从目录扫描，必填）。
        cache_dir (str)：UNet 模型所在目录（相对 models_root 或绝对路径）。
            默认 "image/unet/"。
        dtype (str)：加载精度（fp16/bf16/fp32/auto），默认 "fp16"。
        device (Optional[str])：覆盖 ctx.device 的设备选择。

    输出端口：
        model (model)：加载后的 UNet/Transformer 实例。

    ctx.extra 集成：
        加载完成后写 ctx.extra["loaded_model"] = model，
        让后续 ModelSamplingAuraFlow 等变换器节点可通过 ctx.extra 获取 model。
    """

    node_type = "UNetLoader"
    category = "loader"
    subcategory = "model"
    title = "UNet加载器"
    icon = "📦"
    description = "单独加载 UNet/Transformer 模型（组件分离模式，参考 ComfyUI UNetLoader）"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            description="加载后的 UNet/Transformer 实例",
        ),
    ]
    params_schema = {
        "unet_name": {
            "type": "string", "required": True, "label": "UNet名称",
            "format": "enum-from-dir",
            "dir": "image/unet/",
            "extensions": [".safetensors", ".gguf", ".pt", ".ckpt", ".bin"],
            "description": "UNet 模型名称（从 cache_dir 目录扫描，dir 字段指明扫描相对 models_root 的子目录）",
        },
        "cache_dir": {
            "type": "string", "required": False, "default": "image/unet/",
            "label": "缓存目录",
            "format": "file-paths",
            "advanced": True,
            "description": "UNet 模型所在目录（相对 models_root 或绝对路径，点击 📁 按钮选择目录）",
        },
        "dtype": {
            "type": "enum", "required": False, "default": "auto",
            "options": ["auto", "fp16", "bf16", "fp32"],
            "label": "精度",
            "advanced": True,
            "description": "加载精度（auto=根据文件名自动推断，fp16=半精度，bf16=bfloat16，fp32=全精度）",
        },
        "device": {
            "type": "string", "required": False,
            "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device（如 cuda:0 / cpu）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._unet_name: str = self.get_param("unet_name", "") or ""
        self._cache_dir: str = self.get_param("cache_dir", "image/unet/") or "image/unet/"
        self._dtype: str = self.get_param("dtype", "fp16") or "fp16"
        self._device_override: Optional[str] = self.get_param("device")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._unet_name or not isinstance(self._unet_name, str):
            errors.append("params.unet_name 必填且必须为非空字符串")
        if not self._cache_dir or not isinstance(self._cache_dir, str):
            errors.append("params.cache_dir 必须为非空字符串")
        if self._dtype not in {"fp16", "bf16", "fp32", "auto"}:
            errors.append(f"params.dtype 必须为 fp16/bf16/fp32/auto 之一，当前值：{self._dtype}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载 UNet/Transformer 模型并返回 model。

        流程：
            1. 参数校验
            2. 解析模型路径
            3. 上报开始进度
            4. 调用 _load_impl（延迟导入 diffusers/torch）
            5. 写 ctx.extra["loaded_model"]（全局共享）
            6. 上报完成进度
            7. 返回 model
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # v0.2.5 dtype 自动推断：dtype 为 "auto" 时按文件名推断
        if self._dtype == "auto":
            from ..._dtype_utils import infer_dtype
            self._dtype = infer_dtype(self._unet_name, default="fp16")
            logger.info(
                "UNetLoader: dtype 自动推断为 %s（文件名 %r）",
                self._dtype, self._unet_name,
            )

        # 解析模型路径
        unet_path = self._resolve_unet_path(ctx)
        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="unet_loader.start",
            message=f"开始加载 UNet {self._unet_name}",
            unet_name=self._unet_name,
            unet_path=str(unet_path),
            device=device,
            dtype=self._dtype,
        )

        # 调用 _load_impl
        try:
            result = await self._load_impl(unet_path, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _load_impl 方法，"
                "且默认实现不可用（可能 diffusers 未安装）"
            )
        except Exception as exc:
            logger.exception(
                "UNetLoader: 加载 %s 失败 (path=%s)",
                self._unet_name,
                unet_path,
            )
            raise RuntimeError(
                f"UNet 加载失败 ({self._unet_name}): {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise TypeError(
                f"_load_impl 必须返回 dict，实际返回：{type(result).__name__}"
            )

        model = result.get("model")
        if model is None:
            raise RuntimeError(f"UNet 加载结果 model 为 None（unet_name={self._unet_name}）")

        # 写 ctx.extra（与 ModelLoader 行为一致）
        ctx.extra["loaded_model"] = model
        logger.debug(
            "UNetLoader: 已写入 ctx.extra['loaded_model']（供后续节点全局共享）"
        )

        await ctx.report_progress(
            1.0,
            stage="unet_loader.done",
            message=f"UNet 加载完成 {self._unet_name}",
            unet_name=self._unet_name,
        )
        return {"model": model}

    async def _load_impl(self, unet_path: Path, ctx: WorkflowContext) -> dict:
        """默认加载实现：基于 diffusers，自动检测模型类型。

        自动检测逻辑：
            - unet_path 是文件（.safetensors/.ckpt/.bin）→ UNet2DConditionModel.from_single_file
            - unet_path/transformer/ 子目录存在 → Transformer2DModel.from_pretrained
            - unet_path/diffusion_pytorch_model.safetensors 存在 → UNet2DConditionModel.from_pretrained
            - 默认 → UNet2DConditionModel.from_pretrained（兼容老模型）

        Args:
            unet_path: UNet 模型文件/目录绝对路径。
            ctx: 工作流上下文。

        Returns:
            dict 包含 model 字段。

        Raises:
            ImportError: diffusers 未安装时抛出（带安装指引）。
            RuntimeError: 加载失败时抛出。
        """
        try:
            import torch
            from diffusers import UNet2DConditionModel
        except ImportError as exc:
            raise ImportError(
                "默认 UNetLoader 实现需要 diffusers + torch 库，但未安装。"
                "请执行 `pip install diffusers torch` 或在引擎中继承 UNetLoaderNode 覆盖 _load_impl。"
            ) from exc

        torch_dtype = self._get_torch_dtype()
        device = self._device_override or ctx.device
        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        # 自动检测模型类型并加载
        if unet_path.is_file():
            # 单文件加载（.safetensors / .ckpt / .bin）
            model = await asyncio.to_thread(
                UNet2DConditionModel.from_single_file,
                str(unet_path),
                **kwargs,
            )
            model_type = "UNet2DConditionModel (from_single_file)"
        elif (unet_path / "transformer").is_dir():
            # Transformer2DModel（Flux/AuraFlow 风格）
            from diffusers import Transformer2DModel
            transformer_path = unet_path / "transformer"
            model = await asyncio.to_thread(
                Transformer2DModel.from_pretrained,
                str(transformer_path),
                **kwargs,
            )
            model_type = "Transformer2DModel"
        else:
            # 标准 UNet2DConditionModel（SD/SDXL 风格）
            model = await asyncio.to_thread(
                UNet2DConditionModel.from_pretrained,
                str(unet_path),
                **kwargs,
            )
            model_type = "UNet2DConditionModel"

        logger.info(
            "UNetLoader: 加载完成（type=%s, path=%s, dtype=%s）",
            model_type, unet_path, self._dtype,
        )

        # 移到设备
        if device and hasattr(model, "to"):
            try:
                await asyncio.to_thread(model.to, device)
            except Exception as exc:
                logger.warning("将 UNet 移至设备 %s 失败：%s", device, exc)

        return {"model": model}

    def _get_torch_dtype(self):
        """解析 dtype 参数为 torch.dtype 对象。"""
        if not self._dtype or self._dtype == "auto":
            return None
        try:
            import torch
        except ImportError:
            return None
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        return dtype_map.get(self._dtype)

    def _resolve_unet_path(self, ctx: WorkflowContext) -> Path:
        """解析 UNet 模型路径。

        规则：
            1. cache_dir 为绝对路径：直接使用
            2. cache_dir 为相对路径：相对于 ctx.models_root 拼接
            3. unet_name 指定：
                a. 若 cache_dir 下有同名目录（diffusers 结构）→ 返回目录
                b. 若 cache_dir 下有同名文件（.safetensors/.ckpt/.bin）→ 返回文件
                c. 若 cache_dir 本身就是模型目录（含 transformer/ 子目录）→ 返回 cache_dir

        Raises:
            NodeValidationError: 路径不存在或文件未找到。
        """
        cache_dir_path = Path(self._cache_dir)
        if not cache_dir_path.is_absolute() and ctx.models_root is not None:
            cache_dir_path = ctx.models_root / cache_dir_path

        if not cache_dir_path.exists():
            raise NodeValidationError(
                self.node_type,
                [f"UNet 模型目录不存在: {cache_dir_path}"],
            )

        # 若 cache_dir 本身就是模型目录（含 transformer/ 子目录或 diffusion_pytorch_model.safetensors）
        if (cache_dir_path / "transformer").is_dir() or \
           (cache_dir_path / "diffusion_pytorch_model.safetensors").exists():
            return cache_dir_path

        # 查找 unet_name 对应的文件或目录
        # 优先匹配目录（diffusers 结构）
        unet_dir = cache_dir_path / self._unet_name
        if unet_dir.is_dir():
            return unet_dir

        # 匹配文件（.safetensors / .ckpt / .bin）
        for ext in [".safetensors", ".ckpt", ".bin", ".pt"]:
            unet_file = cache_dir_path / f"{self._unet_name}{ext}"
            if unet_file.is_file():
                return unet_file

        # 找不到匹配的文件/目录，报错
        raise NodeValidationError(
            self.node_type,
            [
                f"未在 {cache_dir_path} 中找到匹配 '{self._unet_name}' 的 UNet 模型文件或目录，"
                "请检查 params.unet_name 是否正确，"
                "或通过 params.cache_dir 指定正确的模型目录。"
            ],
        )

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：基于 unet_name + cache_dir mtime + dtype。

        返回确定性键（可缓存，避免重复加载大模型）。
        路径不存在或异常时返回 None（不缓存）。
        """
        try:
            cache_dir_path = Path(self._cache_dir)
            if cache_dir_path.exists():
                mtime = int(cache_dir_path.stat().st_mtime)
            else:
                return None
            return f"unet:{self._unet_name}:{mtime}:{self._dtype}"
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(unet_name={self._unet_name!r}, "
            f"cache_dir={self._cache_dir!r}, dtype={self._dtype!r})"
        )

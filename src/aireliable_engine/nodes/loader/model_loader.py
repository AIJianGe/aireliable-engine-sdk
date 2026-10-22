"""ModelLoader 节点：主模型加载节点（v0.2 + v0.2.5 通用节点改造 + 架构类型联动目录改造）。

v0.2.5 架构类型联动目录改造（最新）：
    - architecture 字段重新定义为"模型架构类型"，options: checkpoint/diffusers/unet
    - 每个架构类型对应一个模型存放目录（image/checkpoints/ 或 image/unet/）
    - variant 字段的 dir 不再硬编码，通过 dir_depends_on + dir_map 动态联动
    - 移除 model_name / model_filename / subfolder 三个冗余字段
    - cache_dir 改为可选，留空时由 architecture 推导默认目录
    - _load_impl 按 architecture 分发到 _load_single_file / _load_pipeline_dir / _load_unet
    - 向后兼容：老 yaml 的 model_name/model_filename/旧 architecture 值（auto/pipeline/components）通过 __init__ 映射

v0.2.5 通用节点改造（U1-1）：
    - 不再是纯抽象基类，提供基于 diffusers 的默认实现
    - 支持 `architecture` 参数切换 checkpoint/diffusers/unet 三种加载模式

平台基础 ModelLoaderNode 提供：
    - 标准输出端口（model / vae / clip / scheduler）
    - 标准参数（architecture / variant / cache_dir / precision / device / torch_dtype）
    - 通用工具（路径解析、cache_key 计算、validate）
    - execute 流程（参数校验 → 路径解析 → 上报进度 → 调用 _load_impl → 返回结果）
    - 默认 _load_impl 实现（基于 diffusers，按 architecture 分发）

引擎开发者使用方式：

    from aireliable_engine.nodes.loader.model_loader import ModelLoaderNode

    class ErnieGGUFModelLoaderNode(ModelLoaderNode):
        node_type = "ErnieGGUFModelLoader"

        async def _load_impl(self, model_path, ctx):
            # 覆盖默认实现，实际加载 GGUF 模型
            from gguf import GGUFReader
            reader = GGUFReader(str(model_path))
            return {"model": ..., "vae": ..., "clip": ..., "scheduler": ...}

    - v0.2.5-loader-architecture-analysis.md
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


# v0.2.5 架构类型 → 默认模型目录映射
# 用于 cache_dir 为空时推导默认目录，以及 variant 下拉的 dir_map
_ARCH_DEFAULT_DIR: dict[str, str] = {
    "checkpoint": "image/checkpoints/",
    "diffusers": "image/checkpoints/",
    "unet": "image/unet/",
}


@register_node
class ModelLoaderNode(BaseNode):
    """主模型加载节点（v0.2.5 架构类型联动目录改造）。

    平台基础节点库提供：
        - 标准化 4 个输出端口（model / vae / clip / scheduler）
        - 标准化加载流程（参数校验 → 路径解析 → 进度上报 → _load_impl）
        - 标准化 cache_key（基于 model_path + variant + mtime）
        - v0.2.5 默认 _load_impl 实现（按 architecture 分发到三种加载方式）

    v0.2.5 架构类型联动目录改造：
        - architecture 字段：checkpoint / diffusers / unet 三选一
        - 每个架构类型对应一个模型存放目录（image/checkpoints/ 或 image/unet/）
        - variant 字段的 dir 通过 dir_depends_on + dir_map 动态联动
        - 移除 model_name / model_filename / subfolder 冗余字段
        - 向后兼容：老字段通过 __init__ 映射

    参数（params）：
        architecture (str)：模型架构类型，enum["checkpoint", "diffusers", "unet"]。
            必填，默认 "checkpoint"。
            - "checkpoint"：早期 SD/SDXL 单文件（image/checkpoints/，from_single_file）
            - "diffusers"：HF 标准 diffusers 目录（image/checkpoints/ 子目录，from_pretrained）
            - "unet"：组件分离式主模型（image/unet/，仅加载 UNet/Transformer）
        variant (str)：模型文件名或子目录名（从对应架构目录下拉选择）。
            必填。
        cache_dir (Optional[str])：模型所在目录。
            可选，留空时由 architecture 推导默认目录。
        precision (Optional[str])：精度（"auto" / "fp16" / "bf16" / "fp32"）。
            "auto" 时由 infer_dtype 按文件名自动推断。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
        torch_dtype (Optional[str])：torch dtype 字符串
            （"auto" / "fp16" / "bf16" / "fp32"）。
        low_vram_mode (bool)：低显存模式（sequential offload），默认 False。
    """

    node_type = "ModelLoader"
    # v0.2.5 P-CategoryFix：显式声明分类
    category = "loader"
    subcategory = "model"
    title = "主模型加载器"
    icon = "📦"
    description = "加载主模型（架构类型驱动目录选择：checkpoint/diffusers/unet）"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            description="主模型对象（pipeline / transformer 实例）",
        ),
        PortSpec(
            name="vae",
            type="model",
            required=False,
            description="VAE 模型（若与主模型分离加载）",
        ),
        PortSpec(
            name="clip",
            type="model",
            required=False,
            description="CLIP 文本编码器（若与主模型分离加载）",
        ),
        PortSpec(
            name="scheduler",
            type="model",
            required=False,
            description="采样调度器（diffusers scheduler）",
        ),
    ]
    params_schema = {
        # 架构类型：决定模型存放目录与加载方式
        "architecture": {
            "type": "enum", "required": True, "default": "checkpoint",
            "options": ["checkpoint", "diffusers", "unet"], "label": "架构类型",
            "description": "checkpoint=早期SD/SDXL单文件（image/checkpoints/），"
                           "diffusers=HF标准目录（image/checkpoints/子目录），"
                           "unet=组件分离式主模型（image/unet/）",
        },
        # 模型文件名：dir 由 architecture 联动决定
        "variant": {
            "type": "string", "required": True, "label": "模型文件",
            "format": "enum-from-dir",
            # v0.2.5：dir 不再硬编码，由 dir_depends_on + dir_map 动态决定
            "dir_depends_on": "architecture",
            "dir_map": {
                "checkpoint": "image/checkpoints/",
                "diffusers": "image/checkpoints/",
                "unet": "image/unet/",
            },
            "extensions": [".safetensors", ".gguf", ".pt", ".ckpt", ".bin"],
            "description": "模型文件名（根据架构类型从对应目录下拉选择）",
        },
        # 模型目录：可选，留空时由 architecture 推导
        "cache_dir": {
            "type": "string", "required": False, "label": "模型目录",
            "format": "file-paths",
            "advanced": True,
            "description": "模型所在目录（留空则根据架构类型自动选择："
                           "checkpoint/diffusers→image/checkpoints/，unet→image/unet/）",
        },
        # 精度：auto 时由 infer_dtype 推断
        "precision": {
            "type": "enum", "required": False, "default": "auto", "label": "精度",
            "options": ["auto", "fp16", "bf16", "fp32"],
            "advanced": True,
            "description": "精度（auto=根据文件名自动推断，fp16/bf16/fp32 手动指定）",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device（如 cuda:0 / cpu）",
        },
        "torch_dtype": {
            "type": "enum", "required": False, "default": "auto", "label": "torch dtype",
            "advanced": True,
            "options": ["auto", "fp16", "bf16", "fp32"],
            "description": "torch dtype（auto=自动推断）",
        },
        "low_vram_mode": {
            "type": "boolean", "required": False, "default": False, "label": "低显存模式",
            "advanced": True,
            "description": "低显存模式（sequential offload）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # 架构类型（默认 checkpoint）
        self._architecture: str = self.get_param("architecture", "checkpoint") or "checkpoint"

        # v0.2.5 向后兼容：老 architecture 值映射
        # auto/pipeline → checkpoint（单文件加载场景）
        # components → unet（组件分离场景）
        if self._architecture in ("auto", "pipeline"):
            self._architecture = "checkpoint"
            logger.debug("ModelLoader: architecture 老值映射 auto/pipeline → checkpoint")
        elif self._architecture == "components":
            self._architecture = "unet"
            logger.debug("ModelLoader: architecture 老值映射 components → unet")

        # variant 优先，回退老字段 model_name / model_filename（向后兼容）
        self._variant: str = (
            self.get_param("variant", "")
            or self.get_param("model_name", "")
            or self.get_param("model_filename", "")
        )

        # cache_dir 可选，留空时由 _resolve_model_path 推导
        self._cache_dir: str = self.get_param("cache_dir", "") or ""

        # v0.2.5 向后兼容：subfolder 拼接到 cache_dir
        subfolder = self.get_param("subfolder", "") or ""
        if subfolder and self._cache_dir:
            self._cache_dir = str(Path(self._cache_dir) / subfolder)
            logger.debug("ModelLoader: subfolder 兼容拼接 → cache_dir=%s", self._cache_dir)

        self._precision: Optional[str] = self.get_param("precision")
        self._device_override: Optional[str] = self.get_param("device")
        self._torch_dtype: Optional[str] = self.get_param("torch_dtype")
        self._low_vram_mode: bool = bool(self.get_param("low_vram_mode", False))

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._variant or not isinstance(self._variant, str):
            errors.append("params.variant 必填且必须为非空字符串")
        if self._architecture not in {"checkpoint", "diffusers", "unet"}:
            errors.append(
                f"params.architecture 必须为 checkpoint/diffusers/unet 之一，"
                f"当前值：{self._architecture}"
            )
        if self._torch_dtype is not None and self._torch_dtype not in {
            "fp16", "bf16", "fp32", "auto", "float16", "bfloat16", "float32"
        }:
            errors.append(f"params.torch_dtype 不支持的值：{self._torch_dtype}")
        if self._precision is not None and self._precision not in {
            "auto", "fp16", "bf16", "fp32"
        }:
            errors.append(f"params.precision 不支持的值：{self._precision}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载主模型并返回 4 个标准输出。

        流程：
            1. 参数校验
            2. v0.2.5 dtype 自动推断（precision / torch_dtype 为 "auto" 时按文件名推断）
            3. 解析模型路径（cache_dir 为空时由 architecture 推导）
            4. 上报开始进度
            5. 调用 _load_impl（按 architecture 分发）
            6. 检查返回 dict，填充缺失字段为 None
            7. 写 ctx.extra（clip 全局共享）
            8. 上报完成进度
            9. 返回标准化输出
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # v0.2.5 dtype 自动推断：precision / torch_dtype 为 "auto" 时按文件名推断
        from ..._dtype_utils import infer_dtype
        infer_filename = self._variant or ""
        if self._precision == "auto":
            self._precision = infer_dtype(infer_filename, default="fp16")
            logger.info(
                "ModelLoader: precision 自动推断为 %s（文件名 %r）",
                self._precision, infer_filename,
            )
        if self._torch_dtype == "auto":
            self._torch_dtype = infer_dtype(infer_filename, default="fp16")
            logger.info(
                "ModelLoader: torch_dtype 自动推断为 %s（文件名 %r）",
                self._torch_dtype, infer_filename,
            )

        # 解析模型路径
        model_path = self._resolve_model_path(ctx)
        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="model_loader.start",
            message=f"开始加载 {self._variant}（架构：{self._architecture}）",
            variant=self._variant,
            architecture=self._architecture,
            model_path=str(model_path),
            device=device,
        )

        # 调用 _load_impl（默认实现或子类覆盖）
        try:
            result = await self._load_impl(model_path, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _load_impl 方法，"
                "且默认实现不可用（可能 diffusers 未安装）"
            )
        except Exception as exc:
            logger.exception(
                "ModelLoader: 加载 %s 失败 (path=%s, architecture=%s)",
                self._variant, model_path, self._architecture,
            )
            raise RuntimeError(
                f"模型加载失败 ({self._variant}, architecture={self._architecture}): {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise TypeError(
                f"_load_impl 必须返回 dict，实际返回：{type(result).__name__}"
            )

        # 标准化输出：补全 4 个字段
        standardized = {
            "model": result.get("model"),
            "vae": result.get("vae"),
            "clip": result.get("clip"),
            "scheduler": result.get("scheduler"),
        }

        # v0.2.5 ComfyUI 风格改造（B2）：将 clip 写入 ctx.extra 全局共享
        # 让后续 CLIPTextEncode 节点无需显式连线即可获取 CLIP 模型
        loaded_clip = standardized.get("clip")
        if loaded_clip is not None:
            ctx.extra["loaded_clip"] = loaded_clip
            logger.debug(
                "ModelLoader: 已写入 ctx.extra['loaded_clip']（供 CLIPTextEncode 全局共享）"
            )

        await ctx.report_progress(
            1.0,
            stage="model_loader.done",
            message=f"加载完成 {self._variant}（架构：{self._architecture}）",
            variant=self._variant,
            architecture=self._architecture,
        )
        return standardized

    async def _load_impl(self, model_path: Path, ctx: WorkflowContext) -> dict:
        """默认加载实现（v0.2.5 架构类型联动目录改造）。

        按 architecture 分发到三种加载方式：
            - checkpoint：AutoPipelineForText2Image.from_single_file（SD/SDXL 单文件）
            - diffusers：AutoPipelineForText2Image.from_pretrained（HF 标准目录）
            - unet：仅加载 UNet/Transformer（组件分离式，VAE/CLIP 由独立节点加载）

        Args:
            model_path: 模型文件/目录绝对路径。
            ctx: 工作流上下文。

        Returns:
            dict 包含 model/vae/clip/scheduler 四个字段。

        Raises:
            ImportError: diffusers 未安装时抛出（带安装指引）。
            RuntimeError: 加载失败时抛出。
            ValueError: 未知 architecture 时抛出。
        """
        arch = self._architecture
        if arch == "checkpoint":
            return await self._load_single_file(model_path, ctx)
        elif arch == "diffusers":
            return await self._load_pipeline_dir(model_path, ctx)
        elif arch == "unet":
            return await self._load_unet(model_path, ctx)
        else:
            raise ValueError(
                f"未知 architecture：{arch}（必须为 checkpoint/diffusers/unet）"
            )

    async def _load_single_file(self, model_path: Path, ctx: WorkflowContext) -> dict:
        """checkpoint 模式：单文件加载（SD/SDXL .safetensors/.ckpt）。

        使用 diffusers AutoPipelineForText2Image.from_single_file 加载单文件模型。
        适用于早期 SD 1.5 / SDXL 等一体化 checkpoint 文件。
        """
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ImportError(
                "默认 ModelLoader 实现需要 diffusers 库，但未安装。"
                "请执行 `pip install diffusers` 或在引擎中继承 ModelLoaderNode 覆盖 _load_impl。"
            ) from exc

        torch_dtype = self._get_torch_dtype()

        def _load():
            kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}
            return AutoPipelineForText2Image.from_single_file(
                str(model_path), **kwargs,
            )

        pipe = await asyncio.to_thread(_load)
        return await self._apply_device_and_return(pipe, ctx)

    async def _load_pipeline_dir(self, model_path: Path, ctx: WorkflowContext) -> dict:
        """diffusers 模式：HF 标准目录加载。

        使用 diffusers AutoPipelineForText2Image.from_pretrained 加载 HF 仓库结构的目录。
        适用于 SD/SDXL/Flux 等标准 diffusers 模型（含 model_index.json 的目录）。
        """
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ImportError(
                "默认 ModelLoader 实现需要 diffusers 库，但未安装。"
                "请执行 `pip install diffusers` 或在引擎中继承 ModelLoaderNode 覆盖 _load_impl。"
            ) from exc

        torch_dtype = self._get_torch_dtype()

        def _load():
            kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}
            return AutoPipelineForText2Image.from_pretrained(
                str(model_path), **kwargs,
            )

        pipe = await asyncio.to_thread(_load)
        return await self._apply_device_and_return(pipe, ctx)

    async def _load_unet(self, model_path: Path, ctx: WorkflowContext) -> dict:
        """unet 模式：仅加载 UNet/Transformer（组件分离式）。

        适用于 Flux / AuraFlow 等组件分离式架构，主模型存放在 image/unet/ 目录。
        VAE / CLIP / Scheduler 由独立的 VAELoader / CLIPLoader 节点加载。

        自动检测模型类型：
            - model_path 是文件 → from_single_file（.safetensors / .gguf）
            - model_path 是目录 → from_pretrained
        """
        try:
            import torch
            from diffusers import Transformer2DModel, UNet2DConditionModel
        except ImportError as exc:
            raise ImportError(
                "默认 ModelLoader unet 模式需要 diffusers + torch 库，但未安装。"
                "请执行 `pip install diffusers torch` 或在引擎中继承 UNetLoaderNode 覆盖 _load_impl。"
            ) from exc

        torch_dtype = self._get_torch_dtype()
        device = self._device_override or ctx.device
        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        # 自动检测模型类型并加载
        if model_path.is_file():
            # 单文件加载（.safetensors / .gguf）
            model = await asyncio.to_thread(
                UNet2DConditionModel.from_single_file,
                str(model_path), **kwargs,
            )
        else:
            # 目录加载
            # 优先尝试 Transformer2DModel（Flux 风格），失败回退 UNet2DConditionModel
            if (model_path / "transformer").is_dir() or (model_path / "config.json").exists():
                try:
                    model = await asyncio.to_thread(
                        Transformer2DModel.from_pretrained,
                        str(model_path), **kwargs,
                    )
                except Exception:
                    model = await asyncio.to_thread(
                        UNet2DConditionModel.from_pretrained,
                        str(model_path), **kwargs,
                    )
            else:
                model = await asyncio.to_thread(
                    UNet2DConditionModel.from_pretrained,
                    str(model_path), **kwargs,
                )

        # 移到设备
        if device and hasattr(model, "to"):
            try:
                await asyncio.to_thread(model.to, device)
            except Exception as exc:
                logger.warning("将 UNet 移至设备 %s 失败：%s", device, exc)

        # unet 模式只返回 model，其他字段为 None（由独立节点加载）
        return {
            "model": model,
            "vae": None,
            "clip": None,
            "scheduler": None,
        }

    async def _apply_device_and_return(self, pipe, ctx: WorkflowContext) -> dict:
        """应用设备迁移与低显存模式，返回标准化输出 dict。

        checkpoint / diffusers 模式共用的后处理逻辑。
        """
        # 应用低显存模式
        if self._low_vram_mode:
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception as exc:
                logger.warning("启用 sequential_cpu_offload 失败：%s", exc)
        else:
            device = self._device_override or ctx.device
            if device:
                try:
                    await asyncio.to_thread(pipe.to, device)
                except Exception as exc:
                    logger.warning("将 pipeline 移至设备 %s 失败：%s", device, exc)

        return {
            "model": pipe,
            "vae": getattr(pipe, "vae", None),
            "clip": getattr(pipe, "text_encoder", None),
            "scheduler": getattr(pipe, "scheduler", None),
        }

    def _get_torch_dtype(self):
        """解析 torch_dtype 参数为 torch.dtype 对象。"""
        if not self._torch_dtype or self._torch_dtype == "auto":
            return None
        try:
            import torch
        except ImportError:
            return None
        dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        return dtype_map.get(self._torch_dtype)

    def _resolve_model_path(self, ctx: WorkflowContext) -> Path:
        """解析模型路径（v0.2.5 架构类型联动目录改造简化版）。

        规则：
            1. cache_dir 为空时，根据 architecture 推导默认目录
               - checkpoint / diffusers → image/checkpoints/
               - unet → image/unet/
            2. cache_dir 为绝对路径：直接使用
            3. cache_dir 为相对路径：相对于 ctx.models_root 拼接
            4. variant 直接作为文件名或子目录名拼接（不再模糊匹配）
            5. diffusers 模式：若 cache_dir 本身是 diffusers 目录（含 model_index.json），
               且 variant 为空，直接返回 cache_dir

        Raises:
            NodeValidationError: 路径不存在或文件未找到。
        """
        # 1. cache_dir 为空时，根据 architecture 推导默认目录
        if not self._cache_dir:
            self._cache_dir = _ARCH_DEFAULT_DIR.get(
                self._architecture, "image/checkpoints/"
            )
            logger.debug(
                "ModelLoader: cache_dir 为空，根据 architecture=%s 推导 → %s",
                self._architecture, self._cache_dir,
            )

        # 2. 解析为绝对路径
        cache_dir_path = Path(self._cache_dir)
        if not cache_dir_path.is_absolute() and ctx.models_root is not None:
            cache_dir_path = ctx.models_root / cache_dir_path

        if not cache_dir_path.exists():
            raise NodeValidationError(
                self.node_type,
                [f"模型目录不存在: {cache_dir_path}"],
            )

        # 3. diffusers 模式：若 cache_dir 本身就是 diffusers 目录且 variant 为空，直接返回
        if self._architecture == "diffusers" and not self._variant:
            if (cache_dir_path / "model_index.json").exists() or \
               (cache_dir_path / "transformer").is_dir():
                return cache_dir_path

        # 4. variant 直接作为文件名或子目录名拼接
        if self._variant:
            model_path = cache_dir_path / self._variant
            if model_path.exists():
                return model_path
            # 文件名带扩展名但不存在 → 报错
            if Path(self._variant).suffix:
                raise NodeValidationError(
                    self.node_type,
                    [f"模型文件不存在: {model_path}"],
                )
            # 无扩展名，可能是 diffusers 子目录但不存在 → 报错
            raise NodeValidationError(
                self.node_type,
                [
                    f"未在 {cache_dir_path} 中找到匹配 '{self._variant}' 的模型文件或子目录。"
                    f"请检查 architecture={self._architecture} 与 variant 是否对应。"
                ],
            )

        # 5. variant 为空且 cache_dir 不是 diffusers 目录 → 报错
        raise NodeValidationError(
            self.node_type,
            [
                f"params.variant 必填（当前 architecture={self._architecture}，"
                f"cache_dir={cache_dir_path}）"
            ],
        )

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：基于 variant + model_path + mtime + architecture。

        子类可重写以自定义缓存策略。
        """
        try:
            # 尝试解析路径以计算 mtime（失败时返回 None，不缓存）
            cache_dir_path = Path(self._cache_dir)
            if cache_dir_path.exists():
                mtime = int(cache_dir_path.stat().st_mtime)
            else:
                return None
            return f"model:{self._variant}:{mtime}:{self._architecture}:{self._torch_dtype or 'auto'}"
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(architecture={self._architecture!r}, "
            f"variant={self._variant!r}, cache_dir={self._cache_dir!r}, "
            f"precision={self._precision!r})"
        )

"""CLIPLoader 节点：单独加载 CLIP 文本编码器（v0.2.5 ComfyUI 风格节点补充）。

ComfyUI 风格的组件分离加载节点，与 ModelLoader（一体化加载）并存：
    - ModelLoader：一次加载所有组件（model + vae + clip + scheduler），适合 SD/SDXL 标准模型
    - CLIPLoader：仅加载 CLIP 文本编码器（tokenizer + text_model），适合 Flux/AuraFlow 等组件分离式架构

参考 ComfyUI 的 CLIPLoader 节点设计：
    - 无输入端口（独立加载器）
    - 1 个输出端口（clip，dict 格式 {"tokenizer", "text_model"}）
    - 2 个核心参数（clip_name + type），type 决定加载分支

支持的 CLIP 类型（与 ComfyUI 截图对齐）：
    - stable_diffusion：SD 1.5 / SD 2.x 的 CLIP-L（CLIPTextModel + CLIPTokenizer）
    - stable_diffusion_xl：SDXL 的 CLIP-L + CLIP-G（CLIPTextModel + CLIPTokenizer）
    - lumina2：Lumina2 的 Qwen 编码器（AutoModel + AutoTokenizer，trust_remote_code）
    - wan：Wan 视频模型的 CLIP（CLIPTextModel + CLIPTokenizer）
    - t5：T5 文本编码器（T5EncoderModel + T5Tokenizer，FLUX/T5 系列）
    - clip_l：独立 CLIP-L（CLIPTextModel + CLIPTokenizer）
    - clip_g：独立 CLIP-G（SDXL bigG，CLIPTextModel + CLIPTokenizer）

与 CLIPTextEncode 的集成（v0.2.5 B1 改造）：
    - 加载完成后写 ctx.extra["loaded_clip"] = {"tokenizer": ..., "text_model": ...}
    - CLIPTextEncode._resolve_clip 优先从 ctx.extra 读取，画布无需画 clip 线
    - dict 格式与 CLIPTextEncode._extract_tokenizer_and_model 的"情况 3：dict"分支完美兼容

参考：
    - v0.2.5-comfyui-nodes-analysis.md §二（2）
    - PR/architecture.md §20.19.10（ComfyUI 风格节点补充）
    - PR/development-guide.md §28.17.2（CLIPLoader 节点规范）
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


# 合法的 CLIP type 枚举值（与 ComfyUI 截图对齐）
_VALID_CLIP_TYPES = frozenset({
    "stable_diffusion",
    "stable_diffusion_xl",
    "lumina2",
    "wan",
    "t5",
    "clip_l",
    "clip_g",
})

# 使用 CLIPTextModel + CLIPTokenizer 的类型集合
_CLIP_LIKE_TYPES = frozenset({
    "stable_diffusion", "stable_diffusion_xl", "wan", "clip_l", "clip_g",
})


@register_node
class CLIPLoaderNode(BaseNode):
    """CLIP 文本编码器加载节点（ComfyUI 风格组件分离加载器）。

    平台基础节点库提供的独立 CLIP 加载器，与 ModelLoader 并存：
        - ModelLoader 一次加载所有组件（含 clip）
        - CLIPLoader 仅加载 CLIP（tokenizer + text_model），其他组件由
          UNetLoader / VAELoader 独立加载

    按 ``type`` 参数选择加载分支：
        - stable_diffusion / clip_l / clip_g / wan：用 CLIPTextModel + CLIPTokenizer
        - stable_diffusion_xl：用 CLIPTextModel + CLIPTokenizer（SDXL 双路 CLIP）
        - lumina2：用 AutoModel + AutoTokenizer（Qwen 编码器，trust_remote_code）
        - t5：用 T5EncoderModel + T5Tokenizer（FLUX/T5 系列）

    参数（params）：
        clip_name (str)：CLIP 模型名称（从目录扫描，必填）。
        type (str)：CLIP 类型枚举（必填，影响加载分支）。
        cache_dir (str)：CLIP 模型所在目录（相对 models_root 或绝对路径）。
            默认 "shared/text_encoders/"（跨引擎复用）。
        dtype (str)：加载精度（fp16/bf16/fp32/auto），默认 "fp16"。
        device (Optional[str])：覆盖 ctx.device 的设备选择。

    输出端口：
        clip (model)：CLIP 对象（dict 格式 ``{"tokenizer": ..., "text_model": ...}``）。

    ctx.extra 集成：
        加载完成后写 ``ctx.extra["loaded_clip"] = clip``，让后续 CLIPTextEncode
        节点可通过 ctx.extra 获取 CLIP，无需显式连线（v0.2.5 B1 改造）。
    """

    node_type = "CLIPLoader"
    category = "loader"
    subcategory = "clip"  # 新增子分类
    title = "加载CLIP"
    icon = "🔤"
    description = "单独加载 CLIP 文本编码器（支持 stable_diffusion / sdxl / lumina2 / t5 等多种类型）"

    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="clip",
            type="model",
            description="CLIP 文本编码器（dict: {tokenizer, text_model}）",
        ),
    ]
    params_schema = {
        "clip_name": {
            "type": "string", "required": True, "label": "CLIP名称",
            "format": "enum-from-dir",
            "dir": "shared/text_encoders/",
            "extensions": [".safetensors", ".gguf", ".pt", ".bin"],
            "description": "CLIP 模型名称（从 cache_dir 目录扫描，dir 字段指明扫描相对 models_root 的子目录）",
        },
        "type": {
            "type": "enum", "required": True, "default": "stable_diffusion",
            "options": [
                "stable_diffusion", "stable_diffusion_xl", "lumina2",
                "wan", "t5", "clip_l", "clip_g",
            ],
            "label": "类型", "format": "enum",
            "description": "CLIP 类型（影响加载分支：stable_diffusion/clip_l/clip_g/wan→CLIPTextModel，lumina2→AutoModel，t5→T5EncoderModel）",
        },
        "cache_dir": {
            "type": "string", "required": False, "default": "shared/text_encoders/",
            "label": "缓存目录",
            "format": "file-paths",
            "advanced": True,
            "description": "CLIP 模型所在目录（相对 models_root 或绝对路径，点击 📁 按钮选择目录）",
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
        self._clip_name: str = self.get_param("clip_name", "") or ""
        self._clip_type: str = self.get_param("type", "stable_diffusion") or "stable_diffusion"
        self._cache_dir: str = (
            self.get_param("cache_dir", "shared/text_encoders/")
            or "shared/text_encoders/"
        )
        self._dtype: str = self.get_param("dtype", "fp16") or "fp16"
        self._device_override: Optional[str] = self.get_param("device")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._clip_name or not isinstance(self._clip_name, str):
            errors.append("params.clip_name 必填且必须为非空字符串")
        if not self._cache_dir or not isinstance(self._cache_dir, str):
            errors.append("params.cache_dir 必须为非空字符串")
        if self._clip_type not in _VALID_CLIP_TYPES:
            errors.append(
                f"params.type 必须为 {sorted(_VALID_CLIP_TYPES)} 之一，当前值：{self._clip_type}"
            )
        if self._dtype not in {"fp16", "bf16", "fp32", "auto"}:
            errors.append(f"params.dtype 必须为 fp16/bf16/fp32/auto 之一，当前值：{self._dtype}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载 CLIP 文本编码器并返回 clip。

        流程：
            1. 参数校验
            2. 解析 CLIP 路径
            3. 上报开始进度
            4. 调用 _load_impl（延迟导入 transformers/torch，按 type 分支加载）
            5. 写 ctx.extra["loaded_clip"]（全局共享）
            6. 上报完成进度
            7. 返回 clip
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # v0.2.5 dtype 自动推断：dtype 为 "auto" 时按文件名推断
        if self._dtype == "auto":
            from ..._dtype_utils import infer_dtype
            self._dtype = infer_dtype(self._clip_name, default="fp16")
            logger.info(
                "CLIPLoader: dtype 自动推断为 %s（文件名 %r）",
                self._dtype, self._clip_name,
            )

        # 解析 CLIP 路径
        clip_path = self._resolve_clip_path(ctx)
        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="clip_loader.start",
            message=f"开始加载 CLIP {self._clip_name}（type={self._clip_type}）",
            clip_name=self._clip_name,
            clip_type=self._clip_type,
            clip_path=str(clip_path),
            device=device,
            dtype=self._dtype,
        )

        # 调用 _load_impl
        try:
            result = await self._load_impl(clip_path, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _load_impl 方法，"
                "且默认实现不可用（可能 transformers 未安装）"
            )
        except Exception as exc:
            logger.exception(
                "CLIPLoader: 加载 %s 失败 (path=%s, type=%s)",
                self._clip_name, clip_path, self._clip_type,
            )
            raise RuntimeError(
                f"CLIP 加载失败 ({self._clip_name}, type={self._clip_type}): {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise TypeError(
                f"_load_impl 必须返回 dict，实际返回：{type(result).__name__}"
            )

        clip = result.get("clip")
        if clip is None:
            raise RuntimeError(
                f"CLIP 加载结果 clip 为 None（clip_name={self._clip_name}, type={self._clip_type}）"
            )

        # 写 ctx.extra（与 v0.2.5 B1 改造后的 CLIPTextEncode 自动兼容）
        ctx.extra["loaded_clip"] = clip
        logger.debug(
            "CLIPLoader: 已写入 ctx.extra['loaded_clip']（供 CLIPTextEncode 全局共享）"
        )

        await ctx.report_progress(
            1.0,
            stage="clip_loader.done",
            message=f"CLIP 加载完成 {self._clip_name}（type={self._clip_type}）",
            clip_name=self._clip_name,
            clip_type=self._clip_type,
        )
        return {"clip": clip}

    async def _load_impl(self, clip_path: Path, ctx: WorkflowContext) -> dict:
        """默认加载实现：按 type 分支加载 CLIP 模型（7 种分支）。

        Args:
            clip_path: CLIP 模型文件/目录绝对路径。
            ctx: 工作流上下文。

        Returns:
            dict 包含 clip 字段（dict 格式 {"tokenizer": ..., "text_model": ...}）。

        Raises:
            ImportError: transformers 未安装时抛出（带安装指引）。
            RuntimeError: 加载失败时抛出。
            ValueError: 未知 type 时抛出。
        """
        try:
            import torch  # noqa: F401  用于 torch_dtype 解析
        except ImportError as exc:
            raise ImportError(
                "默认 CLIPLoader 实现需要 torch 库，但未安装。"
                "请执行 `pip install torch transformers` 或在引擎中继承 CLIPLoaderNode 覆盖 _load_impl。"
            ) from exc

        torch_dtype = self._get_torch_dtype()
        device = self._device_override or ctx.device
        clip_type = self._clip_type

        # 按 type 分支加载
        if clip_type in _CLIP_LIKE_TYPES:
            # stable_diffusion / stable_diffusion_xl / wan / clip_l / clip_g
            # 都用 CLIPTextModel + CLIPTokenizer
            tokenizer, text_model = await self._load_clip_like(clip_path, torch_dtype)
        elif clip_type == "lumina2":
            # Lumina2 用 Qwen 编码器，需要 trust_remote_code
            tokenizer, text_model = await self._load_lumina2(clip_path, torch_dtype)
        elif clip_type == "t5":
            # T5 文本编码器
            tokenizer, text_model = await self._load_t5(clip_path, torch_dtype)
        else:
            # 理论上 validate 已拦截，这里是防御性编程
            raise ValueError(f"未知 CLIP type: {clip_type}")

        # 移到设备
        if device and hasattr(text_model, "to"):
            try:
                await asyncio.to_thread(text_model.to, device)
            except Exception as exc:
                logger.warning("将 CLIP text_model 移至设备 %s 失败：%s", device, exc)

        logger.info(
            "CLIPLoader: 加载完成（type=%s, path=%s, dtype=%s）",
            clip_type, clip_path, self._dtype,
        )

        clip = {"tokenizer": tokenizer, "text_model": text_model}
        return {"clip": clip}

    async def _load_clip_like(
        self, clip_path: Path, torch_dtype: Any
    ) -> tuple[Any, Any]:
        """加载 stable_diffusion / sdxl / wan / clip_l / clip_g 类型的 CLIP。

        都用 CLIPTextModel + CLIPTokenizer 加载。
        """
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                "默认 CLIPLoader 实现需要 transformers 库，但未安装。"
                "请执行 `pip install transformers` 或在引擎中继承 CLIPLoaderNode 覆盖 _load_impl。"
            ) from exc

        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        # CLIPTokenizer 不接受 torch_dtype
        tokenizer = await asyncio.to_thread(
            CLIPTokenizer.from_pretrained, str(clip_path)
        )
        text_model = await asyncio.to_thread(
            CLIPTextModel.from_pretrained, str(clip_path), **kwargs,
        )
        return tokenizer, text_model

    async def _load_lumina2(
        self, clip_path: Path, torch_dtype: Any
    ) -> tuple[Any, Any]:
        """加载 lumina2 类型的 CLIP（Qwen 编码器，trust_remote_code）。"""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "默认 CLIPLoader 实现需要 transformers 库，但未安装。"
                "请执行 `pip install transformers` 或在引擎中继承 CLIPLoaderNode 覆盖 _load_impl。"
            ) from exc

        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        # Lumina2 用 Qwen 编码器，需要 trust_remote_code=True
        # 注意：trust_remote_code 存在安全风险，用户需信任模型源
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            str(clip_path),
            trust_remote_code=True,
        )
        text_model = await asyncio.to_thread(
            AutoModel.from_pretrained,
            str(clip_path),
            trust_remote_code=True,
            **kwargs,
        )
        return tokenizer, text_model

    async def _load_t5(
        self, clip_path: Path, torch_dtype: Any
    ) -> tuple[Any, Any]:
        """加载 t5 类型的 CLIP（T5EncoderModel + T5Tokenizer）。"""
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "默认 CLIPLoader 实现需要 transformers 库，但未安装。"
                "请执行 `pip install transformers` 或在引擎中继承 CLIPLoaderNode 覆盖 _load_impl。"
            ) from exc

        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        tokenizer = await asyncio.to_thread(
            T5Tokenizer.from_pretrained, str(clip_path)
        )
        text_model = await asyncio.to_thread(
            T5EncoderModel.from_pretrained, str(clip_path), **kwargs,
        )
        return tokenizer, text_model

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

    def _resolve_clip_path(self, ctx: WorkflowContext) -> Path:
        """解析 CLIP 模型路径。

        规则：
            1. cache_dir 为绝对路径：直接使用
            2. cache_dir 为相对路径：相对于 ctx.models_root 拼接
            3. clip_name 指定：
                a. 若 cache_dir 下有同名目录（HF 仓库结构）→ 返回目录
                b. 若 cache_dir 下有同名文件（.safetensors/.bin）→ 返回文件
                c. 若 cache_dir 本身就是 CLIP 目录（含 config.json）→ 返回 cache_dir

        Raises:
            NodeValidationError: 路径不存在或文件未找到。
        """
        cache_dir_path = Path(self._cache_dir)
        if not cache_dir_path.is_absolute() and ctx.models_root is not None:
            cache_dir_path = ctx.models_root / cache_dir_path

        if not cache_dir_path.exists():
            raise NodeValidationError(
                self.node_type,
                [f"CLIP 模型目录不存在: {cache_dir_path}"],
            )

        # 若 cache_dir 本身就是 CLIP 目录（含 config.json 或 tokenizer_config.json）
        if (cache_dir_path / "config.json").exists() or \
           (cache_dir_path / "tokenizer_config.json").exists():
            return cache_dir_path

        # 查找 clip_name 对应的目录
        clip_dir = cache_dir_path / self._clip_name
        if clip_dir.is_dir():
            return clip_dir

        # 匹配文件（.safetensors / .bin）
        for ext in [".safetensors", ".bin", ".pt"]:
            clip_file = cache_dir_path / f"{self._clip_name}{ext}"
            if clip_file.is_file():
                return clip_file

        # 找不到匹配的文件/目录，报错
        raise NodeValidationError(
            self.node_type,
            [
                f"未在 {cache_dir_path} 中找到匹配 '{self._clip_name}' 的 CLIP 模型文件或目录，"
                "请检查 params.clip_name 是否正确，"
                "或通过 params.cache_dir 指定正确的模型目录。"
            ],
        )

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：基于 clip_name + type + cache_dir mtime + dtype。

        返回确定性键（可缓存，避免重复加载大模型）。
        路径不存在或异常时返回 None（不缓存）。
        """
        try:
            cache_dir_path = Path(self._cache_dir)
            if cache_dir_path.exists():
                mtime = int(cache_dir_path.stat().st_mtime)
            else:
                return None
            return f"clip:{self._clip_name}:{self._clip_type}:{mtime}:{self._dtype}"
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(clip_name={self._clip_name!r}, "
            f"type={self._clip_type!r}, dtype={self._dtype!r})"
        )

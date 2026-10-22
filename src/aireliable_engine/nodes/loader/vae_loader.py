"""VAELoader 节点：VAE 加载节点（v0.2 + v0.2.5 通用节点改造）。

v0.2.5 通用节点改造（U1-2）：
    - 不再是纯抽象基类，提供基于 diffusers AutoencoderKL 的默认实现
    - 引擎子类可覆盖 _load_impl 实现自定义加载逻辑

平台基础 VAELoaderNode 提供：
    - 标准输出端口（vae）
    - 标准参数（vae_name / vae_path / source / subfolder / dtype）
    - 通用工具（路径解析、diffusers API 探测、cache_key）
    - execute 流程（参数校验 → 路径解析 → 进度上报 → 调用 _load_impl → 返回）
    - 默认 _load_impl 实现（基于 diffusers AutoencoderKL.from_pretrained）

支持两种 VAE 来源：
    - source="shared"：从 models/shared/vae/ 加载（跨引擎复用，如 sdxl-vae / flux-vae）
    - source="engine_specific"：从引擎专属目录加载

    - v0.2-tasks.md 任务 3.4
    - v0.2.5-universal-nodes-analysis.md §五（2）
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
class VAELoaderNode(BaseNode):
    """VAE 加载节点（v0.2.5 通用节点改造后含默认实现）。

    平台基础节点库提供：
        - 标准化输出端口（vae）
        - 标准化加载流程（参数校验 → 路径解析 → 进度上报 → _load_impl）
        - **v0.2.5 默认 _load_impl 实现**（基于 diffusers AutoencoderKL）

    引擎子类只需实现 `_load_impl(model_path, ctx) -> dict`：
        - 返回 dict，必须包含 "vae" 字段
        - vae 字段为加载后的 VAE 模型对象

    参数（params）：
        vae_name (str)：必填，VAE 名称（如 "sdxl-vae" / "flux-vae"）。
            若 None，按 source 自动构造默认路径：
                - source="shared"：models/shared/vae/<vae_name>/
                - source="engine_specific"：models/<type>/vae/<engine_id>/<vae_name>/
        source (str)：VAE 来源，"shared"（默认）或 "engine_specific"。
        dtype (Optional[str])：torch dtype 字符串
            （"fp16" / "bf16" / "fp32" / "auto"）。
        device (Optional[str])：覆盖 ctx.device 的设备选择。

    v0.2.5 简化（对标 ComfyUI）：
        - 移除 vae_path（由 source + vae_name 自动构造路径）
        - 移除 subfolder（VAE 通常在根目录）
        - 移除 force_reload（cache 机制自动处理）
        - 旧参数向后兼容：读取但忽略，日志 warning
    """

    node_type = "VAELoader"
    # v0.2.5 P-CategoryFix：显式声明分类
    category = "loader"
    subcategory = "vae"
    title = "VAE 加载器"
    icon = "📦"
    description = "加载 VAE 模型（基于 diffusers AutoencoderKL，支持独立 VAE 加载）"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="vae",
            type="model",
            description="VAE 模型对象",
        )
    ]
    params_schema = {
        "vae_name": {
            "type": "string", "required": True, "label": "VAE 名称",
            "format": "enum-from-dir",
            # v0.2.5：dir 不再硬编码，由 source 字段值联动决定（与前端 node_definitions.js 对齐）
            # - source=shared           → shared/vae/
            # - source=engine_specific  → image/vae/（递归扫描子目录 image/vae/<engine_id>/）
            "dir_depends_on": "source",
            "dir_map": {
                "shared": "shared/vae/",
                "engine_specific": "image/vae/",
            },
            "recursive": True,  # 递归扫描子目录（image/vae/<engine_id>/）
            "extensions": [".safetensors", ".gguf", ".pt", ".bin", ".ckpt"],
            "description": "VAE 名称（根据来源扫描 shared/vae/ 或 image/vae/ 目录，含子目录）",
        },
        "source": {
            "type": "enum", "required": False, "default": "shared",
            "options": ["shared", "engine_specific"], "label": "来源",
            "advanced": True,
            "description": "shared=跨引擎复用（shared/vae/），engine_specific=引擎专属（image/vae/）",
        },
        "dtype": {
            "type": "enum", "required": False, "default": "auto", "label": "精度",
            "advanced": True,
            "options": ["auto", "fp16", "bf16", "fp32"],
            "description": "加载精度（auto=根据文件名自动推断）",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._vae_name: str = self.get_param("vae_name", "")
        self._source: str = self.get_param("source", "shared")
        self._dtype: Optional[str] = self.get_param("dtype")
        self._device_override: Optional[str] = self.get_param("device")

        # v0.2.5 向后兼容：旧参数读取但忽略，日志 warning
        _deprecated = []
        if self.get_param("vae_path") is not None:
            _deprecated.append("vae_path")
        if self.get_param("subfolder") is not None:
            _deprecated.append("subfolder")
        if self.get_param("force_reload") is not None:
            _deprecated.append("force_reload")
        if _deprecated:
            logger.warning(
                "VAELoader: 以下废弃参数已忽略：%s（v0.2.5 已移除，由 source+vae_name 自动构造路径）",
                ", ".join(_deprecated),
            )

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._vae_name or not isinstance(self._vae_name, str):
            errors.append("params.vae_name 必填且必须为非空字符串")
        if self._source not in {"shared", "engine_specific"}:
            errors.append(
                f"params.source 必须为 'shared' 或 'engine_specific'，当前值：{self._source!r}"
            )
        if self._dtype is not None and self._dtype not in {
            "fp16", "bf16", "fp32", "auto"
        }:
            errors.append(f"params.dtype 不支持的值：{self._dtype}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载 VAE 并返回 vae 输出。

        流程：
            1. 参数校验
            2. 解析 VAE 路径（共享 / 引擎特定）
            3. 探测 diffusers API 版本（用于子类兼容性处理）
            4. 上报开始进度
            5. 调用 _load_impl（子类实现）
            6. 上报完成进度
            7. 返回 {"vae": vae_obj}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # v0.2.5 dtype 自动推断：dtype 为 "auto" 时按文件名推断
        if self._dtype == "auto":
            from ..._dtype_utils import infer_dtype
            self._dtype = infer_dtype(self._vae_name, default="fp16")
            logger.info(
                "VAELoader: dtype 自动推断为 %s（文件名 %r）",
                self._dtype, self._vae_name,
            )

        # 解析 VAE 路径
        vae_path = self._resolve_vae_path(ctx)
        device = self._device_override or ctx.device

        # 探测 diffusers API 版本（子类可用此信息做兼容处理）
        api_version = self._detect_diffusers_api_version()

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="vae_loader.start",
            message=f"开始加载 VAE {self._vae_name}",
            vae_name=self._vae_name,
            vae_path=str(vae_path),
            source=self._source,
            device=device,
            diffusers_api=api_version,
        )

        # 调用子类实现
        try:
            result = await self._load_impl(vae_path, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _load_impl 方法，"
                "请继承 VAELoaderNode 并实现该方法"
            )
        except Exception as exc:
            logger.exception(
                "VAELoader: 加载 %s 失败 (path=%s)",
                self._vae_name,
                vae_path,
            )
            raise RuntimeError(
                f"VAE 加载失败 ({self._vae_name}): {exc}"
            ) from exc

        if not isinstance(result, dict) or "vae" not in result:
            raise TypeError(
                f"_load_impl 必须返回包含 'vae' 字段的 dict，"
                f"实际返回：{type(result).__name__}"
            )

        await ctx.report_progress(
            1.0,
            stage="vae_loader.done",
            message=f"VAE 加载完成 {self._vae_name}",
            vae_name=self._vae_name,
        )
        return {"vae": result["vae"]}

    async def _load_impl(self, vae_path: Path, ctx: WorkflowContext) -> dict:
        """默认加载实现（v0.2.5 通用节点改造）。

        基于 diffusers AutoencoderKL.from_pretrained，适用于标准 diffusers VAE 模型。

        Args:
            vae_path: VAE 文件/目录绝对路径。
            ctx: 工作流上下文。

        Returns:
            dict 必须包含 "vae" 字段。

        Raises:
            ImportError: diffusers 未安装时抛出（带安装指引）。
            RuntimeError: 加载失败时抛出。
        """
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "默认 VAELoader 实现需要 diffusers 库，但未安装。"
                "请执行 `pip install diffusers` 或在引擎中继承 VAELoaderNode 覆盖 _load_impl。"
            ) from exc

        torch_dtype = self._get_torch_dtype()
        device = self._device_override or ctx.device
        kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        def _load():
            return AutoencoderKL.from_pretrained(str(vae_path), **kwargs)

        vae = await asyncio.to_thread(_load)

        # 移到设备
        if device and hasattr(vae, "to"):
            try:
                await asyncio.to_thread(vae.to, device)
            except Exception as exc:
                logger.warning("将 VAE 移至设备 %s 失败：%s", device, exc)

        return {"vae": vae}

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

    def _resolve_vae_path(self, ctx: WorkflowContext) -> Path:
        """解析 VAE 路径（v0.2.5 简化：仅按 source 自动构造）。

        规则：
            - source="shared"：models/shared/vae/<vae_name>
            - source="engine_specific"：models/<capability>/vae/<vae_name>

        vae_name 支持递归相对路径（如 ``ernie-image/diffusion_pytorch_model.safetensors``）。
        若 vae_name 指向文件（含扩展名），自动取父目录作为 diffusers 模型目录。

        Raises:
            NodeValidationError: 路径不存在。
        """
        if ctx.models_root is None:
            raise NodeValidationError(
                self.node_type,
                ["ctx.models_root 为 None，无法自动构造 VAE 路径"],
            )
        if self._source == "shared":
            vae_path = ctx.models_root / "shared" / "vae" / self._vae_name
        else:
            # engine_specific：vae_name 已包含引擎子目录路径
            # （如 ernie-image/diffusion_pytorch_model.safetensors）
            capability = ctx.capability or "image"
            vae_path = ctx.models_root / capability / "vae" / self._vae_name

        # 若 vae_name 指向文件（含扩展名），取父目录作为 diffusers 模型目录
        if vae_path.is_file():
            vae_path = vae_path.parent

        if not vae_path.exists():
            raise NodeValidationError(
                self.node_type,
                [f"VAE 路径不存在: {vae_path}"],
            )
        return vae_path

    def _detect_diffusers_api_version(self) -> str:
        """探测 diffusers API 版本（用于子类兼容性处理）。

        返回值：
            - ">=0.31"：新版 diffusers（用 scaling_factor）
            - "<0.31"：老版 diffusers（用 scale_factor）
            - "unknown"：未安装 diffusers 或探测失败

        子类可根据此值选择正确的 API 调用方式。
        """
        try:
            import diffusers  # type: ignore
            version_str = getattr(diffusers, "__version__", "0.0.0")
            # 简单字符串比较：major.minor
            parts = version_str.split(".")
            major = int(parts[0]) if parts else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            if (major, minor) >= (0, 31):
                return ">=0.31"
            return "<0.31"
        except ImportError:
            return "unknown"
        except Exception:
            return "unknown"

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：基于 vae_name + source + dtype。"""
        try:
            return f"vae:{self._vae_name}:{self._source}:{self._dtype or 'auto'}"
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(vae_name={self._vae_name!r}, "
            f"source={self._source!r}, dtype={self._dtype!r})"
        )

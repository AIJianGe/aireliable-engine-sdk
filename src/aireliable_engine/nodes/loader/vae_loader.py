"""VAELoader 节点：VAE 加载节点（v0.2）。

设计为**抽象基类节点**：实际加载逻辑由引擎子类实现。
平台基础 VAELoaderNode 提供：
    - 标准输出端口（vae）
    - 标准参数（vae_name / vae_path / source / subfolder / dtype）
    - 通用工具（路径解析、diffusers API 探测、cache_key）
    - execute 流程（参数校验 → 路径解析 → 进度上报 → 调用 _load_impl → 返回）

支持两种 VAE 来源：
    - source="shared"：从 models/shared/vae/ 加载（跨引擎复用，如 sdxl-vae / flux-vae）
    - source="engine_specific"：从引擎专属目录加载

参考：
    - PR/architecture.md §3.6（models/shared/vae/ 目录）
    - PR/development-guide.md §28（节点规范）
    - v0.2-tasks.md 任务 3.4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class VAELoaderNode(BaseNode):
    """VAE 加载节点（抽象基类）。

    引擎子类只需实现 `_load_impl(model_path, ctx) -> dict`：
        - 返回 dict，必须包含 "vae" 字段
        - vae 字段为加载后的 VAE 模型对象

    参数（params）：
        vae_name (str)：必填，VAE 名称（如 "sdxl-vae" / "flux-vae"）。
        vae_path (Optional[str])：VAE 路径（相对 models_root 或绝对路径）。
            若 None，按 source 自动构造默认路径：
                - source="shared"：models/shared/vae/<vae_name>/
                - source="engine_specific"：models/<type>/vae/<engine_id>/<vae_name>/
        source (str)：VAE 来源，"shared"（默认）或 "engine_specific"。
        subfolder (Optional[str])：diffusers 子目录（如 ""）。
        dtype (Optional[str])：torch dtype 字符串
            （"fp16" / "bf16" / "fp32" / "auto"）。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
        diffusers_subfolder_pattern (str)：diffusers VAE 子目录模式，
            默认 ""（VAE 通常在根目录）。
        force_reload (bool)：强制重新加载（绕过 cache），默认 False。
    """

    node_type = "VAELoader"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="vae",
            type="model",
            description="VAE 模型对象",
        )
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._vae_name: str = self.get_param("vae_name", "")
        self._vae_path: Optional[str] = self.get_param("vae_path")
        self._source: str = self.get_param("source", "shared")
        self._subfolder: Optional[str] = self.get_param("subfolder")
        self._dtype: Optional[str] = self.get_param("dtype")
        self._device_override: Optional[str] = self.get_param("device")
        self._diffusers_subfolder_pattern: str = self.get_param(
            "diffusers_subfolder_pattern", ""
        )
        self._force_reload: bool = bool(self.get_param("force_reload", False))

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
            "fp16", "bf16", "fp32", "auto", "float16", "bfloat16", "float32"
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
        """实际加载逻辑（子类必须实现）。

        Args:
            vae_path: VAE 文件/目录绝对路径。
            ctx: 工作流上下文。

        Returns:
            dict 必须包含 "vae" 字段。

        Raises:
            NotImplementedError: 基类未实现此方法。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 _load_impl 方法"
        )

    def _resolve_vae_path(self, ctx: WorkflowContext) -> Path:
        """解析 VAE 路径。

        规则：
            1. vae_path 显式指定：直接使用（相对路径相对于 models_root）
            2. vae_path 未指定：按 source 自动构造默认路径
                - source="shared"：models/shared/vae/<vae_name>/
                - source="engine_specific"：models/<type>/vae/<engine_id>/<vae_name>/

        Raises:
            NodeValidationError: 路径不存在。
        """
        if self._vae_path:
            vae_path = Path(self._vae_path)
            if not vae_path.is_absolute() and ctx.models_root is not None:
                vae_path = ctx.models_root / vae_path
        else:
            # 按 source 自动构造
            if ctx.models_root is None:
                raise NodeValidationError(
                    self.node_type,
                    ["未指定 vae_path 且 ctx.models_root 为 None，无法自动构造路径"],
                )
            if self._source == "shared":
                vae_path = ctx.models_root / "shared" / "vae" / self._vae_name
            else:
                # engine_specific：使用 ctx.engine_id 推断引擎专属目录
                engine_id = ctx.engine_id or "unknown_engine"
                # capability/type 从 ctx 获取（如 image / voice）
                capability = ctx.capability or "image"
                vae_path = (
                    ctx.models_root / capability / "vae" / engine_id / self._vae_name
                )

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

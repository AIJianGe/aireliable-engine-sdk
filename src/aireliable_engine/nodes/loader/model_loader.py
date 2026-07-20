"""ModelLoader 节点：主模型加载节点（v0.2）。

设计为**抽象基类节点**：实际加载逻辑由引擎的子类实现。
平台基础 ModelLoaderNode 提供：
    - 标准输出端口（model / vae / clip / scheduler）
    - 标准参数（variant / cache_dir / device / precision）
    - 通用工具（路径解析、cache_key 计算、validate）
    - execute 流程（参数校验 → 路径解析 → 上报进度 → 调用 _load_impl → 返回结果）

引擎开发者使用方式：

    from aireliable_engine.nodes.loader.model_loader import ModelLoaderNode

    class ErnieGGUFModelLoaderNode(ModelLoaderNode):
        node_type = "ErnieGGUFModelLoader"

        async def _load_impl(self, model_path, ctx):
            # 实际加载 GGUF 模型
            from gguf import GGUFReader
            reader = GGUFReader(str(model_path))
            return {"model": ..., "vae": ..., "clip": ..., "scheduler": ...}

参考：
    - v0.2-analysis.md §11.2 / §4.2
    - PR/architecture.md §12.2.5（节点 yaml 示例）
    - PR/development-guide.md §28.4（节点契约）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class ModelLoaderNode(BaseNode):
    """主模型加载节点（抽象基类）。

    平台基础节点库提供此抽象基类：
        - 标准化 4 个输出端口（model / vae / clip / scheduler）
        - 标准化加载流程（参数校验 → 路径解析 → 进度上报 → _load_impl）
        - 标准化 cache_key（基于 model_path + variant + mtime）

    引擎子类只需实现 `_load_impl(model_path, ctx) -> dict`：
        - 返回 dict，至少包含 "model" 字段
        - 可选包含 "vae" / "clip" / "scheduler" 字段
        - 未提供的字段会被填为 None

    参数（params）：
        variant (str)：变体 ID（如 "ernie-image-gguf-q4-k-m"）。
            必填，决定加载哪个具体模型文件。
        cache_dir (str)：模型所在目录（相对 models_root 或绝对路径）。
            必填。
        model_filename (Optional[str])：模型文件名。
            若 None，则从 cache_dir 扫描匹配 variant 的文件。
        precision (Optional[str])：精度（如 "q4_k_m" / "fp16" / "bf16"）。
            用于 cache_key 和加载日志，None 表示不限制。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
            None 时使用 ctx.device。
        torch_dtype (Optional[str])：torch dtype 字符串
            （"fp16" / "bf16" / "fp32" / "auto"）。
        subfolder (Optional[str])：diffusers 子目录（如 "unet/" / "vae/"）。
        low_vram_mode (bool)：低显存模式（sequential offload），默认 False。
    """

    node_type = "ModelLoader"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            description="主模型对象（pipeline / model 实例）",
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

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._variant: str = self.get_param("variant", "")
        self._cache_dir: str = self.get_param("cache_dir", "")
        self._model_filename: Optional[str] = self.get_param("model_filename")
        self._precision: Optional[str] = self.get_param("precision")
        self._device_override: Optional[str] = self.get_param("device")
        self._torch_dtype: Optional[str] = self.get_param("torch_dtype")
        self._subfolder: Optional[str] = self.get_param("subfolder")
        self._low_vram_mode: bool = bool(self.get_param("low_vram_mode", False))

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not self._variant or not isinstance(self._variant, str):
            errors.append("params.variant 必填且必须为非空字符串")
        if not self._cache_dir or not isinstance(self._cache_dir, str):
            errors.append("params.cache_dir 必填且必须为非空字符串")
        if self._torch_dtype is not None and self._torch_dtype not in {
            "fp16", "bf16", "fp32", "auto", "float16", "bfloat16", "float32"
        }:
            errors.append(f"params.torch_dtype 不支持的值：{self._torch_dtype}")
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：加载主模型并返回 4 个标准输出。

        流程：
            1. 参数校验
            2. 解析模型路径（相对路径转绝对路径，扫描文件名）
            3. 上报开始进度
            4. 调用 _load_impl（子类实现）
            5. 检查返回 dict，填充缺失字段为 None
            6. 上报完成进度
            7. 返回标准化输出
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 解析模型路径
        model_path = self._resolve_model_path(ctx)
        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="model_loader.start",
            message=f"开始加载 {self._variant}",
            variant=self._variant,
            model_path=str(model_path),
            device=device,
        )

        # 调用子类实现
        try:
            result = await self._load_impl(model_path, ctx)
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _load_impl 方法，"
                "无法直接使用 ModelLoaderNode 基类，请继承并实现该方法"
            )
        except Exception as exc:
            logger.exception(
                "ModelLoader: 加载 %s 失败 (path=%s)",
                self._variant,
                model_path,
            )
            raise RuntimeError(
                f"模型加载失败 ({self._variant}): {exc}"
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

        await ctx.report_progress(
            1.0,
            stage="model_loader.done",
            message=f"加载完成 {self._variant}",
            variant=self._variant,
        )
        return standardized

    async def _load_impl(self, model_path: Path, ctx: WorkflowContext) -> dict:
        """实际加载逻辑（子类必须实现）。

        Args:
            model_path: 模型文件绝对路径。
            ctx: 工作流上下文。

        Returns:
            dict 至少包含 "model" 字段，可选 "vae" / "clip" / "scheduler"。
            未提供的字段会被填为 None。

        Raises:
            NotImplementedError: 基类未实现此方法，必须由子类重写。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 _load_impl 方法"
        )

    def _resolve_model_path(self, ctx: WorkflowContext) -> Path:
        """解析模型路径。

        规则：
            1. cache_dir 为绝对路径：直接使用
            2. cache_dir 为相对路径：相对于 ctx.models_root 拼接
            3. model_filename 指定：直接拼接
            4. model_filename 未指定：在 cache_dir 中扫描匹配 variant 的文件

        Raises:
            NodeValidationError: 路径不存在或文件未找到。
        """
        cache_dir_path = Path(self._cache_dir)
        if not cache_dir_path.is_absolute() and ctx.models_root is not None:
            cache_dir_path = ctx.models_root / cache_dir_path

        if not cache_dir_path.exists():
            raise NodeValidationError(
                self.node_type,
                [f"模型目录不存在: {cache_dir_path}"],
            )

        if self._model_filename:
            model_path = cache_dir_path / self._model_filename
            if not model_path.exists():
                raise NodeValidationError(
                    self.node_type,
                    [f"模型文件不存在: {model_path}"],
                )
            return model_path

        # 未指定文件名：扫描目录中匹配 variant 的文件
        candidates = list(cache_dir_path.iterdir())
        # 优先匹配包含 variant 的文件
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if self._variant in candidate.name:
                return candidate

        # 找不到匹配的文件，报错（提示用户）
        raise NodeValidationError(
            self.node_type,
            [
                f"未在 {cache_dir_path} 中找到匹配 variant '{self._variant}' 的模型文件，"
                "请通过 params.model_filename 显式指定文件名"
            ],
        )

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：基于 variant + model_path + mtime。

        子类可重写以自定义缓存策略。
        """
        try:
            # 尝试解析路径以计算 mtime（失败时返回 None，不缓存）
            cache_dir_path = Path(self._cache_dir)
            if cache_dir_path.exists():
                mtime = int(cache_dir_path.stat().st_mtime)
            else:
                return None
            return f"model:{self._variant}:{mtime}:{self._torch_dtype or 'auto'}"
        except Exception:
            return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(variant={self._variant!r}, "
            f"cache_dir={self._cache_dir!r}, precision={self._precision!r})"
        )

"""EmptyLatentImage 节点：空 latent 生成节点（v0.2.5 新增）。

参考 ComfyUI 的 EmptyLatentImage 节点设计：
    - 在文生图工作流中作为起始节点，生成指定尺寸的空 latent
    - 后续接入 Sampler 节点进行采样
    - 支持 fill="empty"（零张量）/ fill="noise"（随机噪声）两种填充模式

平台基础 EmptyLatentImageNode 提供：
    - 标准输出端口（latent，Latent 类型）
    - 标准参数（width / height / batch_size / channels / fill / vae_type / seed）
    - execute 流程（参数校验 → 进度上报 → 生成 latent → 返回）

工作流位置：
    EmptyLatentImage → Sampler → VAEDecode → ImageSave

参考：
    - ComfyUI EmptyLatentImage 节点（latent/image 分类）
    - v0.2.5-universal-nodes-analysis.md §六（1）
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...types import Latent
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class EmptyLatentImageNode(BaseNode):
    """空 latent 生成节点（v0.2.5 新增）。

    平台基础节点库提供：
        - 标准化输出端口（latent）
        - 标准化生成流程（参数校验 → 进度上报 → 生成 latent）
        - 两种填充模式（empty=零张量，noise=随机噪声）

    工作流示例：
        EmptyLatentImage(width=1024, height=1024) → Sampler(model, embeds)

    参数（params）：
        width (int)：latent 对应的像素宽度。默认 1024。
            必须为正数，建议为 8 的倍数（diffusers 标准）。
        height (int)：latent 对应的像素高度。默认 1024。
            必须为正数，建议为 8 的倍数。
        batch_size (int)：batch 大小，默认 1。
        channels (Optional[int])：latent 通道数。
            None 时按 vae_type 自动推断（SD/SDXL=4，Flux=16）。
        fill (str)：填充模式，"empty"（默认）或 "noise"。
            - "empty"：生成零张量（torch.zeros）
            - "noise"：生成随机噪声（torch.randn）
        vae_type (str)：VAE 类型标识，用于下游 VAEDecode 校验。
            默认 "sd15"。可选值："sd15" / "sdxl" / "flux" / "sd3" 等。
        seed (Optional[int])：随机种子（fill="noise" 时生效）。
            None 表示完全随机。
        scale (int)：latent 下采样倍率。默认 8（SD/SDXL）。
            Flux 通常为 16。channels 为 None 时同时用于推断 channels。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
    """

    node_type = "EmptyLatentImage"
    # v0.2.5 P-CategoryFix：latent 分类（参考 ComfyUI）
    category = "latent"
    subcategory = "image"
    title = "空 Latent 图像"
    icon = "⬜"
    description = "生成指定尺寸的空 latent（参考 ComfyUI EmptyLatentImage，支持 empty/noise 两种填充模式）"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(
            name="latent",
            type="latent",
            description="空 latent 张量（Latent 类型，vae_type 标记来源）",
        )
    ]
    params_schema = {
        "width": {
            "type": "integer", "required": False, "default": 1024, "min": 1, "label": "宽度",
            "description": "latent 对应的像素宽度（建议 8 的倍数）",
        },
        "height": {
            "type": "integer", "required": False, "default": 1024, "min": 1, "label": "高度",
            "description": "latent 对应的像素高度（建议 8 的倍数）",
        },
        "batch_size": {
            "type": "integer", "required": False, "default": 1, "min": 1, "label": "批量大小",
            "description": "batch 大小",
        },
        "channels": {
            "type": "integer", "required": False, "label": "通道数",
            "advanced": True,
            "description": "latent 通道数（None 时按 vae_type 自动推断：SD/SDXL=4，Flux=16）",
        },
        "fill": {
            "type": "enum", "required": False, "default": "empty", "label": "填充模式",
            "options": ["empty", "noise"],
            "description": "empty=零张量，noise=随机噪声",
        },
        "vae_type": {
            "type": "enum", "required": False, "default": "sd15", "label": "VAE 类型",
            "options": ["sd15", "sdxl", "flux", "sd3", "custom"],
            "description": "VAE 类型标识（下游 VAEDecode 校验用）",
        },
        "seed": {
            "type": "integer", "required": False, "label": "随机种子",
            "advanced": True,
            "description": "随机种子（fill=noise 时生效，None=完全随机）",
        },
        "scale": {
            "type": "integer", "required": False, "default": 8, "min": 1, "label": "下采样倍率",
            "advanced": True,
            "description": "latent 下采样倍率（SD/SDXL=8，Flux=16）",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device",
        },
    }

    # vae_type → 默认 channels 映射
    _VAE_TYPE_CHANNELS = {
        "sd15": 4,
        "sdxl": 4,
        "flux": 16,
        "sd3": 16,
        "custom": 4,
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._width: int = int(self.get_param("width", 1024))
        self._height: int = int(self.get_param("height", 1024))
        self._batch_size: int = int(self.get_param("batch_size", 1))
        self._channels: Optional[int] = self.get_param("channels")
        self._fill: str = self.get_param("fill", "empty")
        self._vae_type: str = self.get_param("vae_type", "sd15")
        self._seed: Optional[int] = self.get_param("seed")
        self._scale: int = int(self.get_param("scale", 8))
        self._device_override: Optional[str] = self.get_param("device")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not isinstance(self._width, int) or self._width <= 0:
            errors.append(f"params.width 必须为正整数，当前值：{self._width}")
        if not isinstance(self._height, int) or self._height <= 0:
            errors.append(f"params.height 必须为正整数，当前值：{self._height}")
        if not isinstance(self._batch_size, int) or self._batch_size <= 0:
            errors.append(f"params.batch_size 必须为正整数，当前值：{self._batch_size}")
        if self._channels is not None and (
            not isinstance(self._channels, int) or self._channels <= 0
        ):
            errors.append(f"params.channels 必须为正整数或 None，当前值：{self._channels}")
        if self._fill not in {"empty", "noise"}:
            errors.append(
                f"params.fill 必须为 'empty' 或 'noise'，当前值：{self._fill!r}"
            )
        if self._vae_type not in self._VAE_TYPE_CHANNELS:
            errors.append(
                f"params.vae_type 不支持的值：{self._vae_type!r}，"
                f"可选：{list(self._VAE_TYPE_CHANNELS.keys())}"
            )
        if self._seed is not None and (not isinstance(self._seed, int) or self._seed < 0):
            errors.append(f"params.seed 必须为非负整数或 None，当前值：{self._seed}")
        if not isinstance(self._scale, int) or self._scale <= 0:
            errors.append(f"params.scale 必须为正整数，当前值：{self._scale}")
        # 警告：宽高非 8 倍数
        if self._width % 8 != 0 or self._height % 8 != 0:
            logger.warning(
                "EmptyLatentImage: width=%d / height=%d 不是 8 的倍数，"
                "部分模型可能报错",
                self._width, self._height,
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：生成空 latent。

        流程：
            1. 参数校验
            2. 推断 channels（若未指定，按 vae_type 自动）
            3. 计算 latent 尺寸（width/scale, height/scale）
            4. 上报开始进度
            5. 在线程中生成 latent 张量（empty=zeros，noise=randn）
            6. 包装为 Latent 类型
            7. 上报完成进度
            8. 返回 {"latent": Latent}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 推断 channels
        channels = self._channels or self._VAE_TYPE_CHANNELS.get(self._vae_type, 4)

        # 计算 latent 尺寸
        latent_h = self._height // self._scale
        latent_w = self._width // self._scale

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="empty_latent.start",
            message=f"生成空 latent {self._width}x{self._height} → {latent_w}x{latent_h}",
            width=self._width,
            height=self._height,
            latent_width=latent_w,
            latent_height=latent_h,
            channels=channels,
            batch_size=self._batch_size,
            fill=self._fill,
            vae_type=self._vae_type,
            device=device,
        )

        # 在线程中生成张量
        def _generate():
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "EmptyLatentImage 实现需要 torch 库，但未安装。"
                    "请执行 `pip install torch`。"
                ) from exc

            generator = None
            if self._fill == "noise" and self._seed is not None:
                try:
                    generator = torch.Generator(device=device or "cpu")
                    generator = generator.manual_seed(self._seed)
                except Exception as exc:
                    logger.warning("创建 torch.Generator 失败：%s", exc)

            shape = (self._batch_size, channels, latent_h, latent_w)
            if self._fill == "noise":
                tensor = torch.randn(
                    shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                # empty
                tensor = torch.zeros(
                    shape,
                    device=device,
                    dtype=torch.float32,
                )
            return tensor

        try:
            tensor = await asyncio.to_thread(_generate)
        except ImportError:
            raise
        except Exception as exc:
            logger.exception("EmptyLatentImage: 生成 latent 失败")
            raise RuntimeError(f"生成空 latent 失败: {exc}") from exc

        # 包装为 Latent
        latent_obj = Latent(
            tensor=tensor,
            vae_type=self._vae_type,
            metadata={
                "source": "empty_latent_image",
                "width": self._width,
                "height": self._height,
                "latent_width": latent_w,
                "latent_height": latent_h,
                "channels": channels,
                "batch_size": self._batch_size,
                "fill": self._fill,
                "seed": self._seed,
                "scale": self._scale,
            },
        )

        await ctx.report_progress(
            1.0,
            stage="empty_latent.done",
            message="空 latent 生成完成",
            shape=list(tensor.shape),
        )
        return {"latent": latent_obj}

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：相同参数的空 latent 可缓存（fill=empty 时确定性）。

        fill=noise 时若 seed 指定也可缓存，否则不缓存。
        """
        if self._fill == "empty":
            return (
                f"empty_latent:{self._width}x{self._height}:"
                f"c{self._channels or self._VAE_TYPE_CHANNELS.get(self._vae_type, 4)}:"
                f"b{self._batch_size}:{self._vae_type}"
            )
        if self._fill == "noise" and self._seed is not None:
            return (
                f"empty_latent_noise:{self._width}x{self._height}:"
                f"c{self._channels or self._VAE_TYPE_CHANNELS.get(self._vae_type, 4)}:"
                f"b{self._batch_size}:{self._vae_type}:seed{self._seed}"
            )
        return None  # noise + 无 seed：不缓存

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(width={self._width}, "
            f"height={self._height}, batch_size={self._batch_size}, "
            f"fill={self._fill!r}, vae_type={self._vae_type!r})"
        )

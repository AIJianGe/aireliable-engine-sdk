"""VAEEncode 节点：VAE 编码节点（v0.2.5 VAE 编码器对标 ComfyUI）。

将 Images（PIL 列表）通过 VAE 编码为 Latent（潜空间张量）。
支持多图批量编码：N 张 PIL → [N, 3, H, W] → vae.encode → [N, C, H/8, W/8] latent。

工作流位置（图生图链路）：
    ImagesInput ──images──→ VAEEncode ──latent──→ Sampler ──→ VAEDecode ──→ ImageSave
                               ↑
                               vae
                               ↑
                            VAELoader

与 ComfyUI VAEEncode 的对比：
    | 维度       | ComfyUI                | 本项目                     |
    |-----------|------------------------|----------------------------|
    | 输入端口   | pixels (IMAGE tensor)  | images (Images PIL 列表)   |
    | VAE 输入   | vae (VAE 对象)         | vae (model 类型)           |
    | 输出       | LATENT (tensor)        | latent (Latent 类型)       |
    | 批量支持   | 支持 (tensor batch)    | 支持 (Images 长度=N)       |

参考：
    - PR/architecture.md §3.6（models/shared/vae/ 目录）
    - PR/development-guide.md §28（节点规范）
    - v0.2.5-vae-encode-analysis.md §四（节点设计）
    - VAEDecodeNode（对称实现，decode/vae_decode.py）
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...types import Image, Images, Latent
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class VAEEncodeNode(BaseNode):
    """VAE 编码节点（v0.2.5 对标 ComfyUI VAEEncode）。

    将像素空间 Images 通过 VAE 编码为潜空间 Latent。
    支持单图编码和多图批量编码。

    平台基础节点提供：
        - 标准化输入输出端口（images+vae → latent）
        - 标准化编码流程（参数校验 → 进度上报 → _encode_impl → 标准化输出）
        - 默认 _encode_impl 实现（基于 diffusers VAE.encode）

    引擎子类可覆盖 `_encode_impl(images, vae, ctx, params) -> dict`：
        - 返回 dict，必须包含 "latent" 字段（Latent 类型或 torch.Tensor）
        - 基类会自动包装 torch.Tensor 为 Latent

    参数（params）：
        scaling_factor (Optional[float])：覆盖 VAE 的 scaling_factor。
            None 表示使用 VAE 自带的 scaling_factor（推荐）。
            编码时：latent = latent * scaling_factor
        shift_factor (Optional[float])：覆盖 VAE 的 shift_factor。
            None 表示使用 VAE 自带的 shift_factor（Flux VAE 用）。
            编码时：latent = latent + shift_factor
        device (Optional[str])：覆盖 ctx.device 的设备选择。
        batch_split (Optional[int])：分批编码的 batch 大小，避免 OOM。
            None 表示不分批（一次性编码全部图片）。
            建议值：1（逐张编码）/ 4 / 8。
    """

    node_type = "VAEEncode"
    category = "encode"
    subcategory = "vae"
    title = "VAE 编码器"
    icon = "🔒"
    description = "将 images 通过 VAE 编码为 latent（支持多图批量编码）"

    input_specs: list[PortSpec] = [
        PortSpec(
            name="images",
            type="images",
            required=True,
            description="输入图片（Images 类型，支持单图/多图）",
        ),
        PortSpec(
            name="vae",
            type="model",
            required=True,
            description="VAE 模型对象（由 VAELoader 输出）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="latent",
            type="latent",
            description="编码后的潜空间张量",
        )
    ]
    params_schema = {
        "scaling_factor": {
            "type": "number", "required": False, "label": "scaling_factor",
            "advanced": True,
            "description": "覆盖 VAE 的 scaling_factor（None=使用 VAE 自带）",
        },
        "shift_factor": {
            "type": "number", "required": False, "label": "shift_factor",
            "advanced": True,
            "description": "覆盖 VAE 的 shift_factor（Flux VAE 用，None=使用 VAE 自带）",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device",
        },
        "batch_split": {
            "type": "integer", "required": False, "label": "分批大小",
            "advanced": True,
            "description": "分批编码的 batch 大小（None=不分批）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._scaling_factor: Optional[float] = self.get_param("scaling_factor")
        self._shift_factor: Optional[float] = self.get_param("shift_factor")
        self._device_override: Optional[str] = self.get_param("device")
        self._batch_split: Optional[int] = self.get_param("batch_split")

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if self._scaling_factor is not None and (
            not isinstance(self._scaling_factor, (int, float))
            or self._scaling_factor <= 0
        ):
            errors.append(
                f"params.scaling_factor 必须为正数或 None，当前值：{self._scaling_factor}"
            )
        if self._shift_factor is not None and not isinstance(
            self._shift_factor, (int, float)
        ):
            errors.append(
                f"params.shift_factor 必须为数字或 None，当前值：{self._shift_factor}"
            )
        if self._batch_split is not None and (
            not isinstance(self._batch_split, int) or self._batch_split <= 0
        ):
            errors.append(
                f"params.batch_split 必须为正整数或 None，当前值：{self._batch_split}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：将 images 通过 VAE 编码为 latent。

        流程：
            1. 参数校验
            2. 提取输入（images / vae），校验必填
            3. 上报开始进度
            4. 调用 _encode_impl（子类实现），传入参数 dict
            5. 标准化返回值（Latent / torch.Tensor → Latent）
            6. 上报完成进度
            7. 返回 {"latent": Latent}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        images_input = inputs.get("images")
        vae = inputs.get("vae")

        if images_input is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.images 必填（VAEEncode 需要图片作为输入）"],
            )
        if vae is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.vae 必填（VAEEncode 需要 VAE 模型对象）"],
            )

        # 标准化 images 输入为 Images 包装类
        images = self._normalize_images_input(images_input)
        if len(images) == 0:
            raise NodeValidationError(
                self.node_type,
                ["inputs.images 不能为空（至少需要一张图片）"],
            )

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="vae_encode.start",
            message=f"开始 VAE 编码（{len(images)} 张图片）",
            image_count=len(images),
            device=device,
            has_scaling_factor=self._scaling_factor is not None,
            has_shift_factor=self._shift_factor is not None,
            batch_split=self._batch_split,
        )

        # 调用子类实现
        try:
            result = await self._encode_impl(
                images=images,
                vae=vae,
                ctx=ctx,
                params={
                    "scaling_factor": self._scaling_factor,
                    "shift_factor": self._shift_factor,
                    "device": device,
                    "batch_split": self._batch_split,
                },
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _encode_impl 方法，"
                "请继承 VAEEncodeNode 并实现该方法"
            )
        except Exception as exc:
            logger.exception("VAEEncode: 编码失败")
            raise RuntimeError(f"VAE 编码失败: {exc}") from exc

        # 标准化返回值为 Latent
        latent = self._normalize_latent_output(result)

        # 安全提取 latent shape 信息（用于日志和进度上报）
        try:
            tensor = latent.tensor
            shape = list(getattr(tensor, "shape", []))
            batch_size = shape[0] if shape else 1
        except Exception:
            shape, batch_size = [], 1

        await ctx.report_progress(
            1.0,
            stage="vae_encode.done",
            message=f"VAE 编码完成，latent batch={batch_size}",
            latent_shape=shape,
        )
        return {"latent": latent}

    async def _encode_impl(
        self,
        images: Images,
        vae: Any,
        ctx: WorkflowContext,
        params: dict,
    ) -> dict:
        """默认编码实现（v0.2.5 对标 ComfyUI VAEEncode）。

        基于 diffusers VAE.encode 方法，支持：
            - 多图批量编码（N 张 PIL → [N, C, H/8, W/8] latent）
            - scaling_factor / shift_factor 覆盖（Flux VAE 兼容）
            - batch_split 分批编码避免 OOM
            - 图片尺寸自动对齐到 8 的倍数

        Args:
            images: Images 包装类（PIL 列表）。
            vae: VAE 模型对象（diffusers AutoencoderKL 或类似对象）。
            ctx: 工作流上下文。
            params: 已解析的参数 dict，包含：
                - scaling_factor: Optional[float]
                - shift_factor: Optional[float]
                - device: str
                - batch_split: Optional[int]

        Returns:
            dict 包含 "latent" 字段（Latent 类型或 torch.Tensor）。

        Raises:
            ImportError: torch 未安装时抛出（带安装指引）。
            RuntimeError: 编码失败时抛出。
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "默认 VAEEncode 实现需要 torch 库，但未安装。"
                "请执行 `pip install torch diffusers`，"
                "或在引擎中继承 VAEEncodeNode 覆盖 _encode_impl。"
            ) from exc

        import numpy as np

        # PIL 列表 → [N, 3, H, W] tensor，归一化到 [-1, 1]
        pixel_tensor = self._images_to_tensor(images)

        # 提取 scaling_factor / shift_factor（优先用参数覆盖，否则从 VAE 推断）
        scaling_factor = params.get("scaling_factor")
        if scaling_factor is None:
            scaling_factor = self._infer_scaling_factor(vae)
        shift_factor = params.get("shift_factor")
        if shift_factor is None:
            shift_factor = self._infer_shift_factor(vae)

        # 移到设备
        device = params.get("device")
        if device and hasattr(pixel_tensor, "to"):
            try:
                pixel_tensor = pixel_tensor.to(device)
            except Exception as exc:
                logger.warning("将 pixel_tensor 移至设备 %s 失败：%s", device, exc)
        if device and hasattr(vae, "to"):
            try:
                vae.to(device)
            except Exception as exc:
                logger.warning("将 VAE 移至设备 %s 失败：%s", device, exc)

        # 分批处理
        batch_size = pixel_tensor.shape[0]
        batch_split = params.get("batch_split")
        if batch_split and batch_split > 0 and batch_size > batch_split:
            logger.info(
                "VAEEncode: batch=%d, split_size=%d，分批编码",
                batch_size, batch_split,
            )
            latent_list = []
            for i in range(0, batch_size, batch_split):
                chunk = pixel_tensor[i:i + batch_split]
                latent_chunk = await asyncio.to_thread(
                    self._encode_one_batch, vae, chunk
                )
                latent_list.append(latent_chunk)
            # 拼接结果
            try:
                latents = torch.cat(latent_list, dim=0)
            except Exception:
                latents = latent_list
        else:
            latents = await asyncio.to_thread(
                self._encode_one_batch, vae, pixel_tensor
            )

        # 应用 scaling_factor（编码时乘以）
        if scaling_factor is not None and scaling_factor != 0:
            latents = latents * scaling_factor

        # 应用 shift_factor（编码时加，Flux VAE 用）
        if shift_factor is not None and shift_factor != 0:
            latents = latents + shift_factor

        # 包装为 Latent
        latent_obj = Latent(
            tensor=latents,
            vae_type="encoded",
            metadata={
                "source": "vae_encode",
                "image_count": batch_size,
                "scaling_factor": scaling_factor,
                "shift_factor": shift_factor,
            },
        )
        return {"latent": latent_obj}

    def _images_to_tensor(self, images: Images) -> Any:
        """将 Images（PIL 列表）转换为 [N, 3, H, W] tensor，归一化到 [-1, 1]。

        Args:
            images: Images 包装类。

        Returns:
            torch.Tensor [N, 3, H, W]，范围 [-1, 1]。

        Raises:
            ImportError: torch / numpy / PIL 未安装时抛出。
            RuntimeError: 图片转换失败时抛出。
        """
        try:
            import torch
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                f"VAEEncode._images_to_tensor 需要 torch / numpy：{exc}"
            ) from exc

        tensors = []
        for img in images.items:
            # 转 RGB（VAE 要求 3 通道输入）
            pil = img.pil.convert("RGB")
            arr = np.array(pil)  # [H, W, 3] uint8 [0, 255]
            # [H, W, 3] → [3, H, W]，归一化到 [-1, 1]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
            tensors.append(tensor)

        # 堆叠为 [N, 3, H, W]
        pixel_tensor = torch.stack(tensors, dim=0)

        # 确保尺寸是 8 的倍数（VAE 下采样要求）
        _, _, h, w = pixel_tensor.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            logger.warning(
                "VAEEncode: 图片尺寸 %dx%d 不是 8 的倍数，裁剪到 %dx%d",
                w, h, new_w, new_h,
            )
            pixel_tensor = pixel_tensor[:, :, :new_h, :new_w]

        return pixel_tensor

    def _encode_one_batch(self, vae: Any, pixel_tensor: Any) -> Any:
        """单批编码（阻塞调用，由 asyncio.to_thread 包装）。

        Args:
            vae: VAE 模型对象。
            pixel_tensor: 像素张量 [B, 3, H, W]，范围 [-1, 1]。

        Returns:
            潜空间张量 [B, C, H/8, W/8]。
        """
        try:
            output = vae.encode(pixel_tensor)
            # diffusers VAE.encode 返回 AutoencoderKLOutput（含 .latent_dist 字段）
            if hasattr(output, "latent_dist"):
                return output.latent_dist.sample()
            # 部分实现返回 dict
            if isinstance(output, dict):
                if "latent_dist" in output:
                    return output["latent_dist"].sample()
                if "latents" in output:
                    return output["latents"]
            # 直接返回 tensor
            if hasattr(output, "shape"):
                return output
            # tuple 形式
            if isinstance(output, tuple) and len(output) > 0:
                return output[0]
            raise RuntimeError(
                f"VAE.encode 返回了无法识别的类型：{type(output).__name__}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"VAE.encode 调用失败: {exc}。"
                f"VAE 类型: {type(vae).__name__}，"
                f"pixel_tensor shape: {getattr(pixel_tensor, 'shape', '?')}"
            ) from exc

    def _infer_scaling_factor(self, vae: Any) -> Optional[float]:
        """从 VAE 对象推断 scaling_factor。

        优先级：vae.config.scaling_factor → vae.scaling_factor → 默认 0.18215。

        Args:
            vae: VAE 模型对象。

        Returns:
            scaling_factor 数值。
        """
        try:
            config = getattr(vae, "config", None)
            if config is not None:
                factor = getattr(config, "scaling_factor", None)
                if factor is not None and isinstance(factor, (int, float)):
                    return float(factor)
            factor = getattr(vae, "scaling_factor", None)
            if factor is not None and isinstance(factor, (int, float)):
                return float(factor)
        except (AttributeError, TypeError):
            pass
        # SD/SDXL VAE 默认值
        return 0.18215

    def _infer_shift_factor(self, vae: Any) -> Optional[float]:
        """从 VAE 对象推断 shift_factor（Flux VAE 用）。

        Args:
            vae: VAE 模型对象。

        Returns:
            shift_factor 数值，None 表示无 shift_factor。
        """
        try:
            config = getattr(vae, "config", None)
            if config is not None:
                factor = getattr(config, "shift_factor", None)
                if factor is not None and isinstance(factor, (int, float)):
                    return float(factor)
            factor = getattr(vae, "shift_factor", None)
            if factor is not None and isinstance(factor, (int, float)):
                return float(factor)
        except (AttributeError, TypeError):
            pass
        return None

    def _normalize_images_input(self, images_input: Any) -> Images:
        """将各种形式的 images 输入标准化为 Images 包装类。

        支持的输入：
            - Images 实例（直接返回）
            - list[Image]（包装为 Images）
            - 单个 Image（包装为 Images 长度=1）
            - list[PIL.Image]（包装为 Image 再包装为 Images）

        Args:
            images_input: 输入对象。

        Returns:
            Images 包装类。

        Raises:
            TypeError: 无法标准化时抛出。
        """
        if isinstance(images_input, Images):
            return images_input
        if isinstance(images_input, Image):
            return Images.from_single(images_input)
        if isinstance(images_input, list):
            items = []
            for item in images_input:
                if isinstance(item, Image):
                    items.append(item)
                elif hasattr(item, "size"):  # PIL.Image
                    items.append(Image(pil=item))
                else:
                    raise TypeError(
                        f"images 列表中的元素类型不支持：{type(item).__name__}"
                    )
            return Images(items=items)
        if hasattr(images_input, "size"):  # 单个 PIL.Image
            return Images.from_single(Image(pil=images_input))
        raise TypeError(
            f"images 输入必须是 Images / list[Image] / Image / PIL.Image，"
            f"实际类型：{type(images_input).__name__}"
        )

    def _normalize_latent_output(self, result: Any) -> Latent:
        """将 _encode_impl 返回值标准化为 Latent 类型。

        支持的输入：
            - dict 含 "latent" 字段，值为 Latent / torch.Tensor
            - Latent 实例（直接返回）
            - torch.Tensor（包装为 Latent）

        Args:
            result: _encode_impl 返回值。

        Returns:
            Latent 实例。

        Raises:
            TypeError: 无法标准化时抛出。
        """
        # 解包 dict
        if isinstance(result, dict):
            if "latent" not in result:
                raise TypeError(
                    f"_encode_impl 返回的 dict 必须包含 'latent' 字段，"
                    f"实际 keys：{list(result.keys())}"
                )
            latent_value = result["latent"]
        else:
            latent_value = result

        # Latent 实例
        if isinstance(latent_value, Latent):
            return latent_value
        # torch.Tensor → 包装为 Latent
        if hasattr(latent_value, "dim"):  # torch.Tensor
            return Latent(
                tensor=latent_value,
                vae_type="encoded",
                metadata={"source": "vae_encode"},
            )
        raise TypeError(
            f"_encode_impl 返回的 latent 字段必须是 Latent / torch.Tensor，"
            f"实际类型：{type(latent_value).__name__}"
        )

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：VAEEncode 通常不缓存（每次编码结果可能不同）。

        子类可覆盖以启用缓存。
        """
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scaling_factor={self._scaling_factor!r}, "
            f"shift_factor={self._shift_factor!r}, "
            f"batch_split={self._batch_split!r})"
        )

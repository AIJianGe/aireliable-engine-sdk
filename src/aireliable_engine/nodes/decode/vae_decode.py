"""VAEDecode 节点：VAE 解码节点（v0.2 + v0.2.5 通用节点改造）。

v0.2.5 通用节点改造（U1-5）：
    - 不再是纯抽象基类，提供基于 diffusers VAE 的默认实现
    - 支持 scaling_factor / shift_factor（Flux VAE 兼容）
    - 支持 batch_split 分批解码避免 OOM
    - 张量自动转 PIL Image

平台基础 VAEDecodeNode 提供：
    - 标准输入端口（latent / vae）
    - 标准输出端口（images，Images 类型，variadic）
    - 标准参数（dtype / output_mode / clamp / scaling_factor / shift_factor / batch_split）
    - execute 流程（参数校验 → 输入校验 → 进度上报 → 调用 _decode_impl → 标准化输出）
    - 默认 _decode_impl 实现（基于 diffusers AutoencoderKL.decode）

工作流位置：
    Sampler → **VAEDecode** → ImageSave

将 Sampler 输出的 Latent 解码为像素空间 Images。
输出为 Images 包装类（验收标准要求），下游 ImageSave 节点消费。

    - v0.2-tasks.md 任务 3.7
    - v0.2.5-universal-nodes-analysis.md §五（5）
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Union

from ...node import BaseNode, NodeValidationError, PortSpec
from ...node_registry import register_node
from ...types import Image, Images, Latent
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


@register_node
class VAEDecodeNode(BaseNode):
    """VAE 解码节点（v0.2.5 通用节点改造后含默认实现）。

    平台基础节点库提供：
        - 标准化输入输出端口（latent+vae → images）
        - 标准化解码流程（参数校验 → 进度上报 → _decode_impl）
        - **v0.2.5 默认 _decode_impl 实现**（基于 diffusers VAE.decode）

    引擎子类只需实现 `_decode_impl(latent, vae, ctx, params) -> dict`：
        - 返回 dict，必须包含 "images" 字段
        - "images" 字段可为 Images 包装类 或 list[Image]（基类会自动包装）

    参数（params）：
        dtype (str)：输出像素 dtype，"uint8"（默认）或 "float32"。
            uint8：像素值范围 [0, 255]（标准 PIL 模式）
            float32：像素值范围 [0.0, 1.0]（用于后续浮点处理）
        output_mode (str)：PIL 颜色模式，"RGB"（默认）/ "RGBA" / "L"。
            仅对 uint8 输出生效；float32 输出忽略此参数。
        clamp (bool)：是否将输出像素值 clamp 到合法范围。默认 True。
            uint8：clamp 到 [0, 255]
            float32：clamp 到 [0.0, 1.0]
        scaling_factor (Optional[float])：覆盖 VAE 的 scaling_factor。
            None 表示使用 VAE 自带的 scaling_factor（推荐）。
            用于老版 VAE 或自定义解码缩放。
        shift_factor (Optional[float])：覆盖 VAE 的 shift_factor。
            None 表示使用 VAE 自带的 shift_factor。
            diffusers 0.30+ 的 Flux VAE 使用 shift_factor。
        device (Optional[str])：覆盖 ctx.device 的设备选择。
            None 表示使用 ctx.device。
        batch_split (Optional[int])：分批解码的 batch 大小，避免 OOM。
            None 表示不分批（一次性解码全部 latent）。
            建议值：1（逐张解码）/ 4 / 8。
        call_timeout (Optional[float])：解码超时秒数，None 不限制。
            超时会触发 asyncio.TimeoutError。
        vae_type_check (bool)：是否校验 latent.vae_type 与 VAE 兼容。默认 True。
            True 时若 latent.vae_type 非空且与 vae 不匹配会警告（不报错）。
    """

    node_type = "VAEDecode"
    # v0.2.5 P-CategoryFix：显式声明分类
    category = "decode"
    subcategory = "vae"
    title = "VAE 解码器"
    icon = "🖼️"
    description = "将 latent 通过 VAE 解码为像素空间图像（基于 diffusers VAE.decode）"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="latent",
            type="latent",
            required=True,
            description="采样器输出的潜空间张量（Latent 类型）",
        ),
        PortSpec(
            name="vae",
            type="model",
            required=True,
            description="VAE 模型对象（由 VAELoader 或 ModelLoader 输出）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="images",
            type="images",
            variadic=True,
            description="解码后的图片数组（Images 包装类）",
        )
    ]
    params_schema = {
        # v0.2.5 简化（对标 ComfyUI VAEDecode）：移除所有 params
        # 像素输出默认值（uint8 / RGB / clamp=True / scaling_factor 取 VAE 自带 / device=ctx.device）
        # 由实现内部硬编码（与前端 node_definitions.js 同步保持空）
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # v0.2.5 简化（对标 ComfyUI VAEDecode）：所有参数硬编码为默认值
        # 向后兼容：旧 yaml 仍可能写有这些字段，静默忽略（仅 debug 日志）
        if params:
            _deprecated = (
                "dtype", "output_mode", "clamp", "scaling_factor",
                "shift_factor", "device", "batch_split",
                "call_timeout", "vae_type_check",
            )
            for _name in _deprecated:
                if _name in params:
                    logger.debug(
                        "VAEDecode: params.%s 已废弃（v0.2.5 对标 ComfyUI 移除），忽略",
                        _name,
                    )
        # 默认值（与原 init 保持一致，确保行为不变）
        self._dtype: str = "uint8"
        self._output_mode: str = "RGB"
        self._clamp: bool = True
        self._scaling_factor: Optional[float] = None
        self._shift_factor: Optional[float] = None
        self._device_override: Optional[str] = None
        self._batch_split: Optional[int] = None
        self._call_timeout: Optional[float] = None
        self._vae_type_check: bool = True

    def validate(self) -> list[str]:
        """参数校验。

        v0.2.5 简化：所有参数硬编码默认值，无需校验。保留方法签名以兼容基类。
        """
        return []

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：将 latent 通过 VAE 解码为 Images。

        流程：
            1. 参数校验
            2. 提取输入（latent / vae），校验必填
            3. 可选 VAE 类型一致性检查（警告级别）
            4. 上报开始进度
            5. 调用 _decode_impl（子类实现），传入参数 dict
            6. 标准化返回值（list[Image] / Images → Images）
            7. 上报完成进度
            8. 返回 {"images": Images}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        latent = inputs.get("latent")
        vae = inputs.get("vae")

        if latent is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.latent 必填（VAEDecode 需要潜空间张量作为输入）"],
            )
        if vae is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.vae 必填（VAEDecode 需要 VAE 模型对象）"],
            )

        # VAE 类型一致性检查（警告级别，不阻止执行）
        if isinstance(self._vae_type_check, bool) and self._vae_type_check:
            self._check_vae_type_compatibility(latent, vae)

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="vae_decode.start",
            message="开始 VAE 解码",
            dtype=self._dtype,
            output_mode=self._output_mode,
            clamp=self._clamp if isinstance(self._clamp, bool) else True,
            device=device,
            has_scaling_factor=self._scaling_factor is not None,
            has_shift_factor=self._shift_factor is not None,
            batch_split=self._batch_split,
        )

        # 调用子类实现
        try:
            result = await self._decode_impl(
                latent=latent,
                vae=vae,
                ctx=ctx,
                params={
                    "dtype": self._dtype,
                    "output_mode": self._output_mode,
                    "clamp": self._clamp if isinstance(self._clamp, bool) else True,
                    "scaling_factor": self._scaling_factor,
                    "shift_factor": self._shift_factor,
                    "device": device,
                    "batch_split": self._batch_split,
                    "call_timeout": self._call_timeout,
                },
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _decode_impl 方法，"
                "请继承 VAEDecodeNode 并实现该方法"
            )
        except Exception as exc:
            logger.exception("VAEDecode: 解码失败")
            raise RuntimeError(f"VAE 解码失败: {exc}") from exc

        # 标准化返回值
        images = self._normalize_images_output(result)

        await ctx.report_progress(
            1.0,
            stage="vae_decode.done",
            message=f"VAE 解码完成，共 {len(images)} 张图片",
            image_count=len(images),
        )
        return {"images": images}

    async def _decode_impl(
        self,
        latent: Latent,
        vae: Any,
        ctx: WorkflowContext,
        params: dict,
    ) -> dict:
        """默认解码实现（v0.2.5 通用节点改造）。

        基于 diffusers VAE.decode 方法，支持：
            - scaling_factor / shift_factor 覆盖（Flux VAE 兼容）
            - batch_split 分批解码避免 OOM
            - 张量自动转 PIL Image

        Args:
            latent: 潜空间张量（Latent 类型，或 torch.Tensor）。
            vae: VAE 模型对象（diffusers AutoencoderKL 或类似对象）。
            ctx: 工作流上下文。
            params: 已解析的参数 dict，包含：
                - dtype: "uint8" / "float32"
                - output_mode: "RGB" / "RGBA" / "L"
                - clamp: bool
                - scaling_factor: Optional[float]
                - shift_factor: Optional[float]
                - device: str
                - batch_split: Optional[int]
                - call_timeout: Optional[float]

        Returns:
            dict 包含 "images" 字段（list[Image]）。

        Raises:
            ImportError: torch 未安装时抛出（带安装指引）。
            RuntimeError: 解码失败时抛出。
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "默认 VAEDecode 实现需要 torch 库，但未安装。"
                "请执行 `pip install torch diffusers`，"
                "或在引擎中继承 VAEDecodeNode 覆盖 _decode_impl。"
            ) from exc

        import numpy as np

        # 解包 latent tensor
        latent_tensor = self._unwrap_latent(latent)

        # 提取 scaling_factor / shift_factor（优先用参数覆盖，否则从 VAE 推断）
        scaling_factor = params.get("scaling_factor")
        if scaling_factor is None:
            scaling_factor = getattr(vae, "scaling_factor", None) or getattr(vae, "config", None)
            if scaling_factor is not None and hasattr(scaling_factor, "scaling_factor"):
                scaling_factor = scaling_factor.scaling_factor
            elif scaling_factor is not None and not isinstance(scaling_factor, (int, float)):
                # config 对象，重新尝试
                scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        shift_factor = params.get("shift_factor")
        if shift_factor is None:
            try:
                shift_factor = getattr(vae.config, "shift_factor", None)
            except (AttributeError, TypeError):
                shift_factor = None

        # 应用 scaling_factor（除以 scaling_factor 缩放回 VAE 输入范围）
        if scaling_factor is not None and scaling_factor != 0:
            latent_tensor = latent_tensor / scaling_factor

        # 应用 shift_factor（Flux VAE）
        if shift_factor is not None and shift_factor != 0:
            latent_tensor = latent_tensor - shift_factor

        # 移到设备
        device = params.get("device")
        if device and hasattr(latent_tensor, "to"):
            try:
                latent_tensor = latent_tensor.to(device)
            except Exception as exc:
                logger.warning("将 latent 移至设备 %s 失败：%s", device, exc)
        if device and hasattr(vae, "to"):
            try:
                vae.to(device)
            except Exception as exc:
                logger.warning("将 VAE 移至设备 %s 失败：%s", device, exc)

        # 分批处理
        batch_size = latent_tensor.shape[0] if hasattr(latent_tensor, "shape") else 1
        batch_split = params.get("batch_split")
        if batch_split and batch_split > 0 and batch_size > batch_split:
            logger.info(
                "VAEDecode: batch=%d, split_size=%d，分批解码",
                batch_size, batch_split,
            )
            decoded_list = []
            for i in range(0, batch_size, batch_split):
                chunk = latent_tensor[i:i + batch_split]
                decoded_chunk = await asyncio.to_thread(
                    self._decode_one_batch, vae, chunk
                )
                decoded_list.append(decoded_chunk)
            # 拼接结果
            try:
                decoded = torch.cat(decoded_list, dim=0)
            except Exception:
                # fallback：保留 list 形式
                decoded = decoded_list
        else:
            decoded = await asyncio.to_thread(
                self._decode_one_batch, vae, latent_tensor
            )

        # 转换为 list[Image]
        clamp = params.get("clamp", True)
        dtype = params.get("dtype", "uint8")
        output_mode = params.get("output_mode", "RGB")
        images = self._tensor_to_images(decoded, dtype=dtype, mode=output_mode, clamp=clamp)

        return {"images": images}

    def _unwrap_latent(self, latent: Any) -> Any:
        """从 Latent 包装类解包为 torch.Tensor。

        支持的输入：
            - Latent dataclass（返回 .tensor）
            - torch.Tensor（直接返回）
            - list[Tensor]（堆叠）
            - dict 含 "tensor" 字段

        Args:
            latent: 输入对象。

        Returns:
            torch.Tensor。
        """
        if latent is None:
            raise NodeValidationError(
                self.node_type,
                ["latent 不能为 None"],
            )
        # Latent dataclass
        if hasattr(latent, "tensor") and hasattr(latent, "vae_type"):
            return latent.tensor
        # dict
        if isinstance(latent, dict) and "tensor" in latent:
            return latent["tensor"]
        # list of Tensor
        if isinstance(latent, (list, tuple)) and latent:
            try:
                import torch
                unwrapped = [self._unwrap_latent(x) for x in latent]
                return torch.cat(unwrapped, dim=0)
            except Exception:
                return self._unwrap_latent(latent[0])
        # 已经是 torch.Tensor
        return latent

    def _decode_one_batch(self, vae: Any, latent_tensor: Any) -> Any:
        """单批解码（阻塞调用，由 asyncio.to_thread 包装）。

        Args:
            vae: VAE 模型对象。
            latent_tensor: latent tensor。

        Returns:
            解码后的像素张量（shape: [B, C, H, W]，范围 [-1, 1] 或 [0, 1]）。
        """
        try:
            # diffusers VAE.decode 返回 DecoderOutput（含 .sample 字段）
            output = vae.decode(latent_tensor)
            if hasattr(output, "sample"):
                return output.sample
            if isinstance(output, tuple) and len(output) > 0:
                return output[0]
            if isinstance(output, dict) and "sample" in output:
                return output["sample"]
            # 直接返回 tensor（部分实现返回裸 tensor）
            return output
        except Exception as exc:
            raise RuntimeError(
                f"VAE.decode 调用失败: {exc}。"
                f"VAE 类型: {type(vae).__name__}，"
                f"latent shape: {getattr(latent_tensor, 'shape', '?')}"
            ) from exc

    def _tensor_to_images(
        self,
        tensor: Any,
        *,
        dtype: str = "uint8",
        mode: str = "RGB",
        clamp: bool = True,
    ) -> list[Image]:
        """将解码后的张量转换为 list[Image]。

        Args:
            tensor: 解码后张量（shape: [B, C, H, W] 或 list[张量]）。
                范围 [-1, 1] 或 [0, 1]，由 VAE 输出决定。
            dtype: "uint8" 或 "float32"。
            mode: PIL 颜色模式（"RGB" / "RGBA" / "L"）。
            clamp: 是否 clamp 到合法范围。

        Returns:
            list[Image] 实例。
        """
        try:
            import torch
            import numpy as np
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError(
                "VAEDecode._tensor_to_images 需要 torch / numpy / PIL："
                f"{exc}"
            ) from exc

        # 处理 list[张量] 形式
        if isinstance(tensor, list):
            results = []
            for t in tensor:
                results.extend(self._tensor_to_images(t, dtype=dtype, mode=mode, clamp=clamp))
            return results

        # 单个张量
        if not hasattr(tensor, "shape"):
            raise TypeError(
                f"_tensor_to_images 期望 torch.Tensor，得到 {type(tensor).__name__}"
            )

        # 确保是 4D：[B, C, H, W]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError(
                f"解码张量维度异常：{tensor.dim()}D，期望 4D [B,C,H,W]"
            )

        images: list[Image] = []
        for i in range(tensor.shape[0]):
            single = tensor[i].detach().cpu()
            # 假设输入范围 [-1, 1]，转 [0, 1]
            if single.min() < 0 or single.max() > 1.0 + 1e-3:
                # 范围 [-1, 1] → [0, 1]
                single = (single + 1.0) / 2.0
            if clamp:
                single = single.clamp(0.0, 1.0)
            # 转换为 numpy HWC
            if single.shape[0] == 1:
                # 灰度图
                arr = (single.squeeze(0).numpy() * 255 if dtype == "uint8"
                       else single.squeeze(0).numpy())
                arr = arr
                pil_mode = "L" if dtype == "uint8" else "F"
                pil = PILImage.fromarray(arr.astype(np.uint8) if dtype == "uint8" else arr, mode=pil_mode)
                if mode != "L" and dtype == "uint8":
                    pil = pil.convert(mode)
            else:
                # RGB / RGBA
                arr = single.permute(1, 2, 0).numpy()
                if dtype == "uint8":
                    arr = (arr * 255).astype(np.uint8)
                    if arr.shape[-1] == 3 and mode == "RGB":
                        pil = PILImage.fromarray(arr, mode="RGB")
                    elif arr.shape[-1] == 4 and mode == "RGBA":
                        pil = PILImage.fromarray(arr, mode="RGBA")
                    else:
                        pil = PILImage.fromarray(arr).convert(mode)
                else:
                    pil = PILImage.fromarray(arr.astype(np.float32), mode="F")
            images.append(Image(pil=pil, metadata={"source": "vae_decode"}))
        return images

    def _normalize_images_output(self, result: Any) -> Images:
        """将 _decode_impl 返回值标准化为 Images 包装类。

        支持的输入形式：
            - dict 含 "images" 字段，值为 Images / list[Image] / 单个 Image
            - Images 实例（直接返回）
            - list[Image]（包装为 Images）
            - 单个 Image（包装为 Images 长度=1）

        Raises:
            TypeError: 返回值无法标准化为 Images。
        """
        # 解包 dict
        if isinstance(result, dict):
            if "images" not in result:
                raise TypeError(
                    f"_decode_impl 返回的 dict 必须包含 'images' 字段，"
                    f"实际 keys：{list(result.keys())}"
                )
            images_value = result["images"]
        else:
            images_value = result

        # 标准化
        if isinstance(images_value, Images):
            return images_value
        if isinstance(images_value, Image):
            return Images.from_single(images_value)
        if isinstance(images_value, list):
            # 校验列表元素类型
            for i, item in enumerate(images_value):
                if not isinstance(item, Image):
                    raise TypeError(
                        f"_decode_impl 返回的 list[{i}] 不是 Image 类型，"
                        f"实际类型：{type(item).__name__}"
                    )
            return Images.from_list(images_value)
        raise TypeError(
            f"_decode_impl 返回的 images 字段必须是 Images / list[Image] / Image，"
            f"实际类型：{type(images_value).__name__}"
        )

    def _check_vae_type_compatibility(self, latent: Any, vae: Any) -> None:
        """检查 latent.vae_type 与 VAE 类型的一致性（警告级别）。

        若 latent 是 Latent 实例且 vae_type 非空，与 vae 的类型标识对比。
        不一致时仅记录 warning，不抛异常（兼容性优先）。

        Args:
            latent: 输入的 latent 对象。
            vae: VAE 模型对象。
        """
        try:
            if not isinstance(latent, Latent):
                return
            latent_vae_type = getattr(latent, "vae_type", "") or ""
            if not latent_vae_type:
                return
            # 尝试从 vae 获取类型标识（不同 diffusers 版本字段名不同）
            vae_type_id = (
                getattr(vae, "vae_type", None)
                or getattr(vae, "name", None)
                or getattr(vae, "_name", None)
                or ""
            )
            if vae_type_id and latent_vae_type and vae_type_id != latent_vae_type:
                logger.warning(
                    "VAEDecode: latent.vae_type=%s 与 vae 标识=%s 不一致，"
                    "可能导致解码结果异常",
                    latent_vae_type,
                    vae_type_id,
                )
        except Exception:
            # 类型检查失败不影响主流程
            pass

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：VAEDecode 通常不缓存（每次解码结果可能不同）。

        子类可覆盖以启用缓存（如纯确定性解码）。
        """
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(dtype={self._dtype!r}, "
            f"output_mode={self._output_mode!r}, "
            f"clamp={self._clamp!r})"
        )

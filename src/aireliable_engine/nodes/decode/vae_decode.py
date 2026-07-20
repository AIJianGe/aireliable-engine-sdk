"""VAEDecode 节点：VAE 解码节点（v0.2）。

设计为**抽象基类节点**：实际解码逻辑由引擎子类实现。
平台基础 VAEDecodeNode 提供：
    - 标准输入端口（latent / vae）
    - 标准输出端口（images，Images 类型，variadic）
    - 标准参数（dtype / output_mode / clamp / scaling_factor / shift_factor / batch_split）
    - execute 流程（参数校验 → 输入校验 → 进度上报 → 调用 _decode_impl → 标准化输出）

工作流位置：
    Sampler → **VAEDecode** → ImageSave

将 Sampler 输出的 Latent 解码为像素空间 Images。
输出为 Images 包装类（验收标准要求），下游 ImageSave 节点消费。

参考：
    - PR/architecture.md §12.2 / §12.4（工作流示例）
    - PR/development-guide.md §28.5（数据类型规范）
    - v0.2-tasks.md 任务 3.7
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from ...node import BaseNode, NodeValidationError, PortSpec
from ...types import Image, Images, Latent
from ...workflow_context import WorkflowContext

logger = logging.getLogger(__name__)


class VAEDecodeNode(BaseNode):
    """VAE 解码节点（抽象基类）。

    引擎子类只需实现 `_decode_impl(latent, vae, ctx) -> dict`：
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

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._dtype: str = self.get_param("dtype", "uint8")
        self._output_mode: str = self.get_param("output_mode", "RGB")
        # 不强制 bool() 转换，保留原值供 validate() 检查类型
        self._clamp: Any = self.get_param("clamp", True)
        self._scaling_factor: Optional[float] = self.get_param("scaling_factor")
        self._shift_factor: Optional[float] = self.get_param("shift_factor")
        self._device_override: Optional[str] = self.get_param("device")
        self._batch_split: Optional[int] = self.get_param("batch_split")
        self._call_timeout: Optional[float] = self.get_param("call_timeout")
        self._vae_type_check: Any = self.get_param("vae_type_check", True)

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if self._dtype not in {"uint8", "float32"}:
            errors.append(
                f"params.dtype 必须为 'uint8' 或 'float32'，当前值：{self._dtype!r}"
            )
        if self._output_mode not in {"RGB", "RGBA", "L"}:
            errors.append(
                f"params.output_mode 必须为 'RGB' / 'RGBA' / 'L'，"
                f"当前值：{self._output_mode!r}"
            )
        if not isinstance(self._clamp, bool):
            errors.append(
                f"params.clamp 必须为 bool 类型，当前值：{self._clamp!r} "
                f"（类型 {type(self._clamp).__name__}）"
            )
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
        if self._call_timeout is not None and (
            not isinstance(self._call_timeout, (int, float)) or self._call_timeout <= 0
        ):
            errors.append(
                f"params.call_timeout 必须为正数或 None，当前值：{self._call_timeout}"
            )
        if not isinstance(self._vae_type_check, bool):
            errors.append(
                f"params.vae_type_check 必须为 bool 类型，当前值：{self._vae_type_check!r}"
            )
        return errors

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
        """实际解码逻辑（子类必须实现）。

        Args:
            latent: 潜空间张量（Latent 类型）。
            vae: VAE 模型对象（diffusers AutoencoderKL / VaeImageProcessor 等）。
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
            dict 必须包含 "images" 字段，值为以下任一形式：
                - Images 包装类（推荐）
                - list[Image]（基类自动包装为 Images）
                - 单个 Image（基类自动包装为 Images 长度=1）

        Raises:
            NotImplementedError: 基类未实现此方法。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 _decode_impl 方法"
        )

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

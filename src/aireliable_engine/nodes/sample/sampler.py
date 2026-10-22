"""Sampler 节点：通用采样器节点（v0.2 + v0.2.5 通用节点改造）。

v0.2.5 通用节点改造（U1-4）：
    - 不再是纯抽象基类，提供基于 diffusers 的默认实现
    - 支持 pipeline / components 两种架构模式
    - pipeline 模式：model 是 diffusers Pipeline，直接调用 __call__
    - components 模式：model 是 transformer/UNet，配合 scheduler 做手动采样循环
    - 新增 latent 可选输入端口（用于 EmptyLatentImage 接入）

平台基础 SamplerNode 提供：
    - 标准输入端口（model / embeds / negative_embeds / images / scheduler / latent）
    - 标准输出端口（latent）
    - 标准参数（steps / guidance / size / seed / batch_size / eta / sampler_name）
    - execute 流程（参数校验 → 进度上报 → 调用 _sample_impl → 回调进度）
    - 默认 _sample_impl 实现（pipeline/components 双架构）

参考：
    - PR/architecture.md §12.2 / §12.2.6（节点示例）
    - PR/development-guide.md §28.5（数据类型规范）
    - v0.2-tasks.md 任务 3.6
    - v0.2.5-universal-nodes-analysis.md §五（4）
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
class SamplerNode(BaseNode):
    """通用采样器节点（v0.2.5 通用节点改造后含默认实现）。

    平台基础节点库提供：
        - 标准化输入输出端口（model/embeds/...→latent）
        - 标准化采样流程（参数校验 → 进度上报 → _sample_impl）
        - **v0.2.5 默认 _sample_impl 实现**（pipeline/components 双架构）

    v0.2.5 通用节点改造：
        - 默认实现支持 pipeline / components 两种模式自动切换
        - pipeline 模式：model 是 diffusers Pipeline，直接 __call__ 一步出图
        - components 模式：model 是 transformer/UNet + scheduler 做手动采样循环
        - 新增 `latent` 可选输入端口（接入 EmptyLatentImage 做图生图 / inpaint）

    引擎子类可覆盖 `_sample_impl(inputs, params, ctx, progress_callback) -> dict`：
        - 返回 dict，必须包含 "latent" 字段（Latent 类型）
        - 通过 progress_callback(step, total, **kwargs) 上报每步进度

    参数（params）：
        steps (int)：采样步数，默认 30。
        guidance (float)：CFG guidance scale，默认 7.5。
        size (str)：输出尺寸（"1024x1024"），默认 "1024x1024"。
            components 模式下用于初始化 latent（pipeline 模式忽略，由 pipeline 推断）
        seed (Optional[int])：随机种子，None 随机。
        batch_size (int)：batch 大小，默认 1。
        eta (float)：DDIM eta（控制随机性），默认 0.0。
        sampler_name (Optional[str])：采样器名称（如 "euler" / "dpm++"）。
        cfg_scheduler (Optional[str])：CFG 调度策略（linear / constant 等）。
        device (Optional[str])：覆盖 ctx.device。
        call_timeout (Optional[float])：采样超时秒数，None 不限制。
        denoise (float)：components 模式下去噪强度（图生图用），默认 1.0。
            1.0 = 完全重绘（纯文生图），0.0 = 不采样。
    """

    node_type = "Sampler"
    # v0.2.5 P-CategoryFix：显式声明分类
    category = "sample"
    subcategory = "sampler"
    title = "通用采样器"
    icon = "🎯"
    description = "文生图/图生图采样器（支持 diffusers pipeline 与 components 两种架构自动切换）"
    input_specs: list[PortSpec] = [
        PortSpec(
            name="model",
            type="model",
            required=True,
            description="主模型（diffusers Pipeline 或 transformer/UNet，由 ModelLoader 输出）",
        ),
        PortSpec(
            name="embeds",
            type="embeds",
            required=True,
            description="正向 prompt 编码（由 CLIPTextEncode 输出）",
        ),
        PortSpec(
            name="negative_embeds",
            type="embeds",
            required=False,
            description="负向 prompt 编码（None 表示无负向）",
        ),
        PortSpec(
            name="images",
            type="images",
            required=False,
            description="输入图片（图生图 / IPAdapter / ControlNet 用）",
        ),
        PortSpec(
            name="scheduler",
            type="model",
            required=False,
            description="调度器（components 模式必填；pipeline 模式忽略）",
        ),
        PortSpec(
            name="latent",
            type="latent",
            required=False,
            description="可选 latent 输入（接入 EmptyLatentImage；components 模式下若无则随机初始化）",
        ),
    ]
    output_specs: list[PortSpec] = [
        PortSpec(
            name="latent",
            type="latent",
            description="采样后的 latent（Latent 类型）",
        )
    ]
    params_schema = {
        "steps": {
            "type": "integer", "required": False, "default": 30, "min": 1, "label": "采样步数",
            "description": "采样步数（建议 20-50）",
        },
        "guidance": {
            "type": "number", "required": False, "default": 7.5, "min": 0, "label": "CFG 引导强度",
            "description": "CFG guidance scale（典型 7-9）",
        },
        "size": {
            "type": "string", "required": False, "default": "1024x1024", "label": "输出尺寸",
            "description": "输出尺寸 WxH（如 1024x1024 / 512x512）",
        },
        "seed": {
            "type": "integer", "required": False, "label": "随机种子",
            "advanced": True,
            "description": "随机种子（None=随机）",
        },
        "batch_size": {
            "type": "integer", "required": False, "default": 1, "min": 1, "label": "批量大小",
            "advanced": True,
            "description": "batch 大小",
        },
        "eta": {
            "type": "number", "required": False, "default": 0.0, "min": 0, "max": 1, "label": "DDIM eta",
            "advanced": True,
            "description": "DDIM eta（控制随机性，0=确定性，1=完全随机）",
        },
        "sampler_name": {
            "type": "string", "required": False, "label": "采样器名称",
            "advanced": True,
            "description": "采样器名称（如 euler / dpm++），可选",
        },
        "cfg_scheduler": {
            "type": "string", "required": False, "label": "CFG 调度策略",
            "advanced": True,
            "description": "CFG 调度策略（linear / constant 等）",
        },
        "device": {
            "type": "string", "required": False, "label": "设备",
            "advanced": True,
            "description": "覆盖 ctx.device（如 cuda:0 / cpu）",
        },
        "call_timeout": {
            "type": "number", "required": False, "label": "采样超时（秒）",
            "advanced": True,
            "description": "采样超时秒数，None 不限制",
        },
        "denoise": {
            "type": "number", "required": False, "default": 1.0, "min": 0, "max": 1, "label": "去噪强度",
            "advanced": True,
            "description": "components 模式下去噪强度（1.0=完全重绘，0.0=不采样）",
        },
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._steps: int = int(self.get_param("steps", 30))
        self._guidance: float = float(self.get_param("guidance", 7.5))
        self._size: str = self.get_param("size", "1024x1024")
        self._seed: Optional[int] = self.get_param("seed")
        self._batch_size: int = int(self.get_param("batch_size", 1))
        self._eta: float = float(self.get_param("eta", 0.0))
        self._sampler_name: Optional[str] = self.get_param("sampler_name")
        self._cfg_scheduler: Optional[str] = self.get_param("cfg_scheduler")
        self._device_override: Optional[str] = self.get_param("device")
        self._call_timeout: Optional[float] = self.get_param("call_timeout")
        self._denoise: float = float(self.get_param("denoise", 1.0))

    def validate(self) -> list[str]:
        """参数校验。"""
        errors: list[str] = []
        if not isinstance(self._steps, int) or self._steps <= 0:
            errors.append(f"params.steps 必须为正整数，当前值：{self._steps}")
        if not isinstance(self._guidance, (int, float)) or self._guidance < 0:
            errors.append(f"params.guidance 必须为非负数，当前值：{self._guidance}")
        if not isinstance(self._size, str) or "x" not in self._size:
            errors.append(f"params.size 必须为 'WxH' 格式字符串，当前值：{self._size!r}")
        if self._seed is not None and (not isinstance(self._seed, int) or self._seed < 0):
            errors.append(f"params.seed 必须为非负整数或 None，当前值：{self._seed}")
        if not isinstance(self._batch_size, int) or self._batch_size <= 0:
            errors.append(f"params.batch_size 必须为正整数，当前值：{self._batch_size}")
        if not isinstance(self._eta, (int, float)) or self._eta < 0 or self._eta > 1:
            errors.append(f"params.eta 必须在 [0, 1] 范围内，当前值：{self._eta}")
        if not isinstance(self._denoise, (int, float)) or self._denoise < 0 or self._denoise > 1:
            errors.append(f"params.denoise 必须在 [0, 1] 范围内，当前值：{self._denoise}")
        if self._call_timeout is not None and (
            not isinstance(self._call_timeout, (int, float)) or self._call_timeout <= 0
        ):
            errors.append(
                f"params.call_timeout 必须为正数或 None，当前值：{self._call_timeout}"
            )
        return errors

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        """执行节点：调用 diffusers pipeline 的 __call__ 方法。

        流程：
            1. 参数校验
            2. 提取输入（model / embeds / negative_embeds / images / scheduler / latent）
            3. 解析 size 字符串为 (width, height)
            4. 上报开始进度
            5. 调用 _sample_impl（默认实现或子类覆盖），传入 progress_callback
            6. 检查返回结果
            7. 上报完成进度
            8. 返回 {"latent": Latent}
        """
        errors = self.validate()
        if errors:
            raise NodeValidationError(self.node_type, errors)

        # 提取输入
        model = inputs.get("model")
        embeds = inputs.get("embeds")
        negative_embeds = inputs.get("negative_embeds")
        images = inputs.get("images")
        scheduler = inputs.get("scheduler")
        latent = inputs.get("latent")  # v0.2.5 新增：可选 latent 输入

        if model is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.model 必填（采样器需要主模型）"],
            )
        if embeds is None:
            raise NodeValidationError(
                self.node_type,
                ["inputs.embeds 必填（采样器需要正向 prompt 编码）"],
            )

        # 解析 size
        try:
            width, height = self._parse_size(self._size)
        except ValueError as exc:
            raise NodeValidationError(
                self.node_type,
                [f"params.size 解析失败：{exc}"],
            )

        device = self._device_override or ctx.device

        # 上报开始
        await ctx.report_progress(
            0.0,
            stage="sampler.start",
            message=f"开始采样 {self._steps} 步",
            steps=self._steps,
            guidance=self._guidance,
            size=self._size,
            seed=self._seed,
            device=device,
            has_negative=negative_embeds is not None,
            has_images=images is not None,
            has_latent=latent is not None,
        )

        # 构造 progress_callback（子类调用）
        async def progress_callback(step: int, total: int, **kwargs: Any) -> None:
            if total <= 0:
                return
            progress = step / total
            await ctx.report_progress(
                progress,
                stage="sampler.step",
                message=f"采样 {step}/{total}",
                step=step,
                total=total,
                **kwargs,
            )

        # 调用 _sample_impl（默认实现或子类覆盖）
        try:
            result = await self._sample_impl(
                inputs={
                    "model": model,
                    "embeds": embeds,
                    "negative_embeds": negative_embeds,
                    "images": images,
                    "scheduler": scheduler,
                    "latent": latent,
                },
                params={
                    "steps": self._steps,
                    "guidance": self._guidance,
                    "width": width,
                    "height": height,
                    "seed": self._seed,
                    "batch_size": self._batch_size,
                    "eta": self._eta,
                    "sampler_name": self._sampler_name,
                    "cfg_scheduler": self._cfg_scheduler,
                    "device": device,
                    "denoise": self._denoise,
                },
                ctx=ctx,
                progress_callback=progress_callback,
            )
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} 未实现 _sample_impl 方法，"
                "且默认实现不可用（可能 diffusers 未安装）"
            )
        except Exception as exc:
            logger.exception("Sampler: 采样失败")
            raise RuntimeError(f"采样失败: {exc}") from exc

        if not isinstance(result, dict) or "latent" not in result:
            raise TypeError(
                f"_sample_impl 必须返回包含 'latent' 字段的 dict，"
                f"实际返回：{type(result).__name__}"
            )

        await ctx.report_progress(
            1.0,
            stage="sampler.done",
            message="采样完成",
        )
        return {"latent": result["latent"]}

    async def _sample_impl(
        self,
        inputs: dict,
        params: dict,
        ctx: WorkflowContext,
        progress_callback,
    ) -> dict:
        """默认采样实现（v0.2.5 通用节点改造）。

        自动检测 model 形态并切换 pipeline / components 模式：
            - pipeline 模式：model 是 diffusers Pipeline（有 __call__ 且非 Transformer）
              → 调用 model(prompt_embeds=..., negative_prompt_embeds=..., ...)
            - components 模式：model 是 transformer/UNet（无 __call__ 或具 forward 但
              非完整 pipeline）→ 配合 scheduler 做手动采样循环

        v0.2.5 ComfyUI 风格改造（C3-2）：
            - 采样前读取 ``ctx.extra["model_shift"]``（由 ModelSamplingAuraFlow 写入）
            - 应用 shift 到 ``model.scheduler``（与 ModelSamplingAuraFlow 的方案 A 一致）
            - try-except 包裹：shift 注入失败不影响采样，仅 warning 日志

        Args:
            inputs: 包含 model / embeds / negative_embeds / images / scheduler / latent。
            params: 已解析的参数 dict（steps / guidance / width / height / seed / ...）。
            ctx: 工作流上下文。
            progress_callback: async 回调，签名 (step, total, **kwargs) -> None。

        Returns:
            dict 必须包含 "latent" 字段（Latent 类型）。

        Raises:
            ImportError: diffusers 未安装时抛出（带安装指引）。
            RuntimeError: 采样失败时抛出。
        """
        model = inputs["model"]
        embeds = inputs["embeds"]
        negative_embeds = inputs.get("negative_embeds")
        scheduler = inputs.get("scheduler")
        latent = inputs.get("latent")
        images = inputs.get("images")  # v0.2.5 latent 优先改造：pipeline fallback 用

        # v0.2.5 ComfyUI 风格改造（C3-2）：读取 ctx.extra["model_shift"]
        # 应用 shift 到 model.scheduler（与 ModelSamplingAuraFlow 的方案 A 一致）
        # try-except 包裹：shift 注入失败不影响采样，仅 warning 日志
        shift = ctx.extra.get("model_shift")
        if shift is not None:
            try:
                target_scheduler = scheduler or getattr(model, "scheduler", None)
                if target_scheduler is not None:
                    # 修改 config.shift
                    config = getattr(target_scheduler, "config", None)
                    if config is not None and hasattr(config, "shift"):
                        try:
                            config.shift = shift
                        except (AttributeError, TypeError):
                            try:
                                config["shift"] = shift
                            except Exception:
                                logger.warning(
                                    "Sampler: scheduler.config.shift 不可修改"
                                    "（model_shift=%s 将由 set_shift 应用）", shift
                                )
                    # diffusers 0.27+ 的 FlowMatchEulerDiscreteScheduler 支持 set_shift
                    if hasattr(target_scheduler, "set_shift"):
                        target_scheduler.set_shift(shift)
                    logger.debug(
                        "Sampler: 已应用 model_shift=%s 到 scheduler", shift
                    )
            except Exception as exc:
                logger.warning(
                    "Sampler: 应用 model_shift 失败（忽略，继续采样）：%s", exc
                )

        # 自动检测架构模式
        if self._is_pipeline_mode(model):
            # v0.2.5 latent 优先改造：把 images 注入 params 供 pipeline fallback
            params_with_images = dict(params)
            params_with_images["images"] = images
            return await self._sample_pipeline(
                model, embeds, negative_embeds, latent,
                params_with_images, ctx, progress_callback,
            )
        return await self._sample_components(
            model, scheduler, latent, embeds, negative_embeds,
            params, ctx, progress_callback,
        )

    def _is_pipeline_mode(self, model: Any) -> bool:
        """检测是否为 pipeline 模式（diffusers Pipeline 实例）。

        判定规则：
            - 有 __call__ 方法（且签名包含 prompt/prompt_embeds 等典型参数）
            - 有 scheduler / unet 等子组件属性（典型 pipeline 特征）
            - 不是单纯 transformer/UNet（避免误判）

        Args:
            model: 主模型对象。

        Returns:
            True=按 pipeline 调用，False=按 components 模式。
        """
        # diffusers Pipeline 都有这些属性
        pipeline_indicators = ("scheduler", "unet", "vae", "config")
        has_pipeline_attrs = all(
            hasattr(model, attr) for attr in pipeline_indicators[:2]
        )
        if not has_pipeline_attrs:
            return False
        # 排除 transformer/UNet：它们没有 scheduler 属性
        # diffusers Pipeline 有 scheduler，transformer 没有
        if not hasattr(model, "scheduler"):
            return False
        # 进一步检查：pipeline 通常有 __call__ 接受 prompt/prompt_embeds
        return callable(getattr(model, "__call__", None))

    async def _sample_pipeline(
        self,
        pipeline: Any,
        embeds: Any,
        negative_embeds: Any,
        latent: Any,
        params: dict,
        ctx: WorkflowContext,
        progress_callback,
    ) -> dict:
        """pipeline 模式采样：调用 diffusers Pipeline __call__。

        v0.2.5 latent 优先改造：
            优先级 latent > images > 随机初始化：
                1. latent 输入不为 None → 使用 latents= 参数（VAEEncode 编码的 latent）
                   配合 strength=denoise 控制去噪强度
                2. latent 为 None 但 images 不为 None → 使用 image= 参数（pipeline 内部 VAE 编码）
                3. 两者都为 None → 纯文生图（pipeline 内部随机初始化）

        适用于 SD/SDXL/Flux 等一体化 pipeline 场景。
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "默认 Sampler 实现需要 torch 库，但未安装。"
                "请执行 `pip install torch diffusers`，"
                "或在引擎中继承 SamplerNode 覆盖 _sample_impl。"
            ) from exc

        device = params.get("device")
        steps = params.get("steps", 30)
        guidance = params.get("guidance", 7.5)
        seed = params.get("seed")
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        batch_size = params.get("batch_size", 1)
        eta = params.get("eta", 0.0)
        denoise = params.get("denoise", 1.0)
        images = params.get("images")  # 图生图用（fallback）

        # 构造 generator
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=device or "cpu")
                generator = generator.manual_seed(seed)
            except Exception as exc:
                logger.warning("创建 torch.Generator 失败：%s", exc)

        # 从 Tensor 包装类解包 embeds
        prompt_embeds = self._unwrap_tensor(embeds)
        negative_prompt_embeds = self._unwrap_tensor(negative_embeds)

        # 准备调用 kwargs
        call_kwargs: dict[str, Any] = {
            "prompt_embeds": prompt_embeds,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "height": height,
            "width": width,
            "num_images_per_prompt": batch_size,
            "eta": eta,
        }
        if negative_prompt_embeds is not None:
            call_kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        if generator is not None:
            call_kwargs["generator"] = generator

        # v0.2.5 latent 优先逻辑：latent > images > 随机初始化
        latent_tensor = self._unwrap_latent_tensor(latent)
        if latent_tensor is not None:
            # 优先使用外部 VAEEncode 编码的 latent
            call_kwargs["latents"] = latent_tensor
            # 配合 denoise 控制去噪强度（diffusers pipeline 的 strength 参数）
            if denoise is not None and denoise < 1.0:
                call_kwargs["strength"] = denoise
            logger.info(
                "Sampler.pipeline: 使用外部 latent（shape=%s, denoise=%s）",
                list(latent_tensor.shape) if hasattr(latent_tensor, "shape") else "?",
                denoise,
            )
        elif images is not None:
            # fallback：使用 images 走 pipeline 内部 VAE 编码（仅取第一张）
            init_image = self._extract_first_pil(images)
            if init_image is not None:
                call_kwargs["image"] = init_image
                # img2img 模式下 denoise 对应 strength
                if denoise is not None and denoise < 1.0:
                    call_kwargs["strength"] = denoise
                logger.info(
                    "Sampler.pipeline: 使用 image fallback（denoise=%s）", denoise
                )

        # diffusers Pipeline 支持 callback_on_step_end（0.22+）
        step_counter = {"current": 0}

        def _sync_callback(pipe, step, timestep, callback_kwargs):
            step_counter["current"] = step
            return callback_kwargs

        if hasattr(pipeline, "_callback_tensor_inputs"):
            # 新版 diffusers 支持 callback_on_step_end
            call_kwargs["callback_on_step_end"] = _sync_callback
            call_kwargs["callback_on_step_end_tensor_inputs"] = [
                "latents"
            ]

        async def _run_pipeline():
            # 异步上报进度
            def _call():
                return pipeline(**call_kwargs)

            # 在线程中执行阻塞调用
            result = await asyncio.to_thread(_call)
            return result

        # 启动进度上报协程（轻量）
        async def _report_progress():
            while step_counter["current"] < steps:
                try:
                    await progress_callback(step_counter["current"], steps)
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        progress_task = asyncio.create_task(_report_progress())
        try:
            output = await _run_pipeline()
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass

        # 最终上报
        try:
            await progress_callback(steps, steps)
        except Exception:
            pass

        # 从 pipeline 输出提取 latents（而非最终图像）
        # diffusers Pipeline 输出通常是 ImagePipelineOutput 或 FluxPipelineOutput
        # 若 output.images 存在，则用 VAE 反推 latent 不合适
        # 这里我们返回 output 直接的 latents（若 pipeline 提供）
        latents = getattr(output, "latents", None)
        if latents is None and isinstance(output, tuple) and len(output) > 0:
            latents = output[0] if not hasattr(output[0], "images") else None

        if latents is None:
            # pipeline 没有暴露 latents，直接用最终图像转回 Latent
            # （后续 VAE decode 会跳过，直接使用 images）
            # 这种情况比较少，但兼容处理
            logger.warning(
                "Sampler.pipeline: pipeline 未返回 latents，"
                "将使用 images 作为 Latent 的 fallback（下游 VAE decode 会跳过解码）"
            )
            images_obj = getattr(output, "images", None)
            if images_obj is None and isinstance(output, tuple):
                images_obj = output[0].images if hasattr(output[0], "images") else None
            if images_obj is None:
                raise RuntimeError("pipeline 输出无 latents 也无 images，无法构造 Latent")
            # 用第一张图作为张量包装（标记为已解码像素空间）
            try:
                import numpy as np
                arr = np.array(images_obj[0]) if isinstance(images_obj, list) else np.array(images_obj)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                latents = tensor
                if device:
                    latents = latents.to(device)
            except Exception as exc:
                raise RuntimeError(f"转换 pipeline images 为 Latent 失败：{exc}") from exc

        # 包装为 Latent 类型
        latent_obj = Latent(
            tensor=latents,
            vae_type="pipeline",  # 标记来自 pipeline，下游 VAE 可识别
            metadata={
                "source": "pipeline",
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            },
        )
        return {"latent": latent_obj}

    async def _sample_components(
        self,
        model: Any,
        scheduler: Any,
        latent: Any,
        embeds: Any,
        negative_embeds: Any,
        params: dict,
        ctx: WorkflowContext,
        progress_callback,
    ) -> dict:
        """components 模式采样：手动循环 + scheduler.step。

        适用于 Flux / 自定义 transformer 架构。需要：
            - model: transformer/UNet（支持 forward 接受 encoder_hidden_states）
            - scheduler: diffusers SchedulerInstance
            - latent: 初始 latent（来自 EmptyLatentImage 或随机初始化）

        若 scheduler 为 None，尝试从 model.scheduler 提取；都失败则报错。
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "默认 Sampler components 实现需要 torch 库，但未安装。"
                "请执行 `pip install torch diffusers`。"
            ) from exc

        if scheduler is None:
            scheduler = getattr(model, "scheduler", None)
        if scheduler is None:
            raise NodeValidationError(
                self.node_type,
                ["components 模式需要 scheduler 输入或 model.scheduler 属性，两者都缺失"],
            )

        device = params.get("device")
        steps = params.get("steps", 30)
        guidance = params.get("guidance", 7.5)
        seed = params.get("seed")
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        batch_size = params.get("batch_size", 1)
        eta = params.get("eta", 0.0)
        denoise = params.get("denoise", 1.0)

        # 提取初始 latent（v0.2.5：优先用 _unwrap_latent_tensor 支持 Latent 包装类）
        latents_tensor = self._unwrap_latent_tensor(latent)
        if latents_tensor is None:
            # fallback：尝试 _unwrap_tensor（兼容旧版 Tensor 包装类）
            latents_tensor = self._unwrap_tensor(latent)
        if latents_tensor is not None:
            latent_source = "外部输入（VAEEncode 或 EmptyLatentImage）"
            logger.info(
                "Sampler.components: 使用外部 latent（shape=%s）",
                list(latents_tensor.shape) if hasattr(latents_tensor, "shape") else "?",
            )
        else:
            # 随机初始化（text2image 场景）
            latent_source = "随机初始化"
            logger.info("Sampler.components: 无外部 latent，随机初始化")
            # 根据模型配置推断 channels：SD=4, SDXL=4, Flux=16
            in_channels = getattr(model, "config", None)
            in_channels = getattr(in_channels, "in_channels", 4) if in_channels else 4
            # latent 尺寸通常是像素 / 8（SD/SDXL）或 / 16（Flux）
            scale = 8  # 默认 SD/SDXL
            latent_h, latent_w = height // scale, width // scale
            generator = None
            if seed is not None:
                try:
                    generator = torch.Generator(device=device or "cpu")
                    generator = generator.manual_seed(seed)
                except Exception as exc:
                    logger.warning("创建 torch.Generator 失败：%s", exc)
            latents_tensor = torch.randn(
                (batch_size, in_channels, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

        # 移到设备
        if device and hasattr(latents_tensor, "to"):
            try:
                latents_tensor = latents_tensor.to(device)
            except Exception as exc:
                logger.warning("将 latents 移至设备 %s 失败：%s", device, exc)

        # 设置 scheduler timesteps
        try:
            scheduler.set_timesteps(steps, device=device)
        except Exception as exc:
            logger.warning("scheduler.set_timesteps 失败：%s", exc)

        # 应用 denoise（图生图：从 denoise 比例处开始采样）
        timesteps = scheduler.timesteps
        if denoise < 1.0 and len(timesteps) > 0:
            start_idx = int(len(timesteps) * (1.0 - denoise))
            timesteps = timesteps[start_idx:]
            if len(timesteps) == 0:
                timesteps = scheduler.timesteps  # fallback

        # 提取 embeds
        prompt_embeds = self._unwrap_tensor(embeds)
        negative_prompt_embeds = self._unwrap_tensor(negative_embeds)

        # CFG：拼接 prompt + negative
        if negative_prompt_embeds is not None:
            latent_model_input = torch.cat([latents_tensor] * 2)
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
        else:
            latent_model_input = latents_tensor
            prompt_embeds_input = prompt_embeds

        # 采样循环
        total = len(timesteps) if timesteps is not None else steps
        for i, t in enumerate(timesteps):
            # model forward（UNet/Transformer）
            with torch.no_grad():
                # 不同模型 forward 签名不同
                try:
                    noise_pred = model(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_input,
                    )
                    if isinstance(noise_pred, tuple):
                        noise_pred = noise_pred[0]
                except TypeError:
                    # 旧版 UNet 不接受 encoder_hidden_states，直接传 sample/timestep
                    noise_pred = model(latent_model_input, t)
                    if isinstance(noise_pred, tuple):
                        noise_pred = noise_pred[0]

            # CFG 分离预测
            if negative_prompt_embeds is not None:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (
                    noise_pred_cond - noise_pred_uncond
                )

            # scheduler step
            try:
                latents_tensor = scheduler.step(
                    noise_pred, t, latents_tensor, eta=eta
                ).prev_sample
            except Exception as exc:
                logger.warning("scheduler.step 失败（step %d）：%s", i, exc)
                break

            # 上报进度
            try:
                await progress_callback(i + 1, total)
            except Exception:
                pass

        # 包装为 Latent
        latent_obj = Latent(
            tensor=latents_tensor,
            vae_type="components",
            metadata={
                "source": "components",
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "denoise": denoise,
                "scheduler": type(scheduler).__name__,
            },
        )
        return {"latent": latent_obj}

    def _unwrap_latent_tensor(self, latent: Any) -> Any:
        """从 Latent 包装类解包为 torch.Tensor（v0.2.5 新增）。

        支持的输入：
            - Latent 包装类（返回 .tensor）
            - torch.Tensor（直接返回）
            - None（返回 None）

        Args:
            latent: Latent 包装类 / torch.Tensor / None。

        Returns:
            torch.Tensor 或 None。
        """
        if latent is None:
            return None
        # Latent 包装类（types.Latent 有 .tensor 字段）
        if hasattr(latent, "tensor") and not hasattr(latent, "dim"):
            tensor = latent.tensor
            if tensor is None:
                return None
            # 移到目标设备（如果 latent_tensor 有 to 方法）
            return tensor
        # 已经是 torch.Tensor
        if hasattr(latent, "dim"):
            return latent
        logger.warning(
            "Sampler._unwrap_latent_tensor: 无法识别的 latent 类型：%s",
            type(latent).__name__,
        )
        return None

    def _unwrap_tensor(self, obj: Any) -> Any:
        """从 Tensor 包装类解包为 torch.Tensor。

        支持的输入：
            - Tensor 包装类（返回 .data）
            - torch.Tensor（直接返回）
            - list[Tensor]（堆叠为单个 tensor）
            - None（返回 None）

        Args:
            obj: 输入对象。

        Returns:
            torch.Tensor 或 None。
        """
        if obj is None:
            return None
        # Tensor 包装类
        if hasattr(obj, "data") and hasattr(obj, "device"):
            return obj.data
        # 已经是 torch.Tensor
        if hasattr(obj, "dim"):  # torch.Tensor 才有 dim 方法
            return obj
        # list of Tensor
        if isinstance(obj, (list, tuple)) and obj:
            unwrapped = [self._unwrap_tensor(x) for x in obj]
            try:
                import torch
                return torch.cat(unwrapped, dim=0)
            except Exception:
                return unwrapped[0]
        return obj

    def _extract_first_pil(self, images: Any) -> Any:
        """从 Images / list[Image] / PIL.Image 中提取第一张 PIL.Image。

        Args:
            images: Images 包装类 / list[Image] / 单个 Image / PIL.Image。

        Returns:
            PIL.Image.Image 或 None（无法提取）。
        """
        try:
            # Images 包装类
            if hasattr(images, "items"):
                items = images.items
                if items:
                    return items[0].pil
                return None
            # list[Image]
            if isinstance(images, list) and images:
                first = images[0]
                if hasattr(first, "pil"):
                    return first.pil
                return first
            # 单个 Image
            if hasattr(images, "pil"):
                return images.pil
            # PIL.Image 直接返回
            return images
        except Exception as exc:
            logger.warning("提取第一张 PIL 图片失败：%s", exc)
            return None

    def _parse_size(self, size: str) -> tuple[int, int]:
        """解析 'WxH' 字符串为 (width, height)。

        支持格式：
            - "1024x1024"
            - "1024X1024"（大写也接受）
            - "1024*1024"
            - "1024×1024"（Unicode 乘号）

        Raises:
            ValueError: 格式不正确。
        """
        # 统一替换分隔符
        normalized = size.lower().replace("×", "x").replace("*", "x")
        parts = normalized.split("x")
        if len(parts) != 2:
            raise ValueError(f"size 格式不正确：{size!r}，应为 'WxH' 如 '1024x1024'")
        try:
            width = int(parts[0])
            height = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"size 宽高必须为整数：{size!r}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"size 宽高必须为正数：{size!r}")
        # 检查是否为 8 的倍数（diffusers 标准）
        if width % 8 != 0 or height % 8 != 0:
            logger.warning(
                "Sampler: size %dx%d 不是 8 的倍数，部分模型可能报错",
                width, height,
            )
        return width, height

    def cache_key(self, inputs: dict) -> Optional[str]:
        """缓存键：Sampler 不缓存（每次采样结果不同）。"""
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(steps={self._steps}, "
            f"guidance={self._guidance}, size={self._size!r}, "
            f"seed={self._seed})"
        )

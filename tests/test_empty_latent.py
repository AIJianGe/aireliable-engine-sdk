"""EmptyLatentImage 节点单元 + 端到端测试。

覆盖任务 v0.2.5-universal-nodes-analysis.md §6（待办）:
    - 补 EmptyLatentImage 端到端测试（fill=empty/noise, vae_type 推断）

参考：
    - ComfyUI EmptyLatentImage 节点
    - v0.2.5-universal-nodes-analysis.md §六（1）
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.sample.empty_latent import EmptyLatentImageNode

# 检测 torch 是否可用
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# 跳过标记
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH,
    reason="EmptyLatentImage 需要 torch 库",
)


# ===== 基本属性测试 =====


class TestEmptyLatentImageNodeBasic:
    """EmptyLatentImageNode 基本属性测试。"""

    def test_node_type(self):
        node = EmptyLatentImageNode({})
        assert node.node_type == "EmptyLatentImage"

    def test_category_and_subcategory(self):
        node = EmptyLatentImageNode({})
        assert node.category == "latent"
        assert node.subcategory == "image"

    def test_title_is_chinese(self):
        node = EmptyLatentImageNode({})
        assert node.title == "空 Latent 图像"

    def test_input_specs_is_empty(self):
        """EmptyLatentImage 无输入端口（独立生成节点）。"""
        node = EmptyLatentImageNode({})
        assert node.input_specs == []

    def test_output_specs_has_latent(self):
        """EmptyLatentImage 仅 1 个输出端口 latent。"""
        node = EmptyLatentImageNode({})
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "latent"
        assert port.type == "latent"

    def test_default_width_height(self):
        node = EmptyLatentImageNode({})
        assert node._width == 1024
        assert node._height == 1024

    def test_default_batch_size(self):
        node = EmptyLatentImageNode({})
        assert node._batch_size == 1

    def test_default_fill_is_empty(self):
        node = EmptyLatentImageNode({})
        assert node._fill == "empty"

    def test_default_vae_type_is_sd15(self):
        node = EmptyLatentImageNode({})
        assert node._vae_type == "sd15"

    def test_default_scale_is_8(self):
        node = EmptyLatentImageNode({})
        assert node._scale == 8

    def test_repr_contains_key_params(self):
        node = EmptyLatentImageNode({"width": 512, "height": 768, "fill": "noise"})
        repr_str = repr(node)
        assert "EmptyLatentImageNode" in repr_str
        assert "512" in repr_str
        assert "768" in repr_str
        assert "noise" in repr_str


# ===== params_schema 测试 =====


class TestEmptyLatentImageNodeParamsSchema:
    """params_schema 测试。"""

    def test_params_schema_contains_core_fields(self):
        node = EmptyLatentImageNode({})
        for field_name in ["width", "height", "batch_size", "channels",
                           "fill", "vae_type", "seed", "scale", "device"]:
            assert field_name in node.params_schema

    def test_fill_options(self):
        node = EmptyLatentImageNode({})
        assert node.params_schema["fill"]["options"] == ["empty", "noise"]

    def test_vae_type_options(self):
        node = EmptyLatentImageNode({})
        assert set(node.params_schema["vae_type"]["options"]) >= {
            "sd15", "sdxl", "flux", "sd3", "custom"
        }


# ===== 参数校验测试 =====


class TestEmptyLatentImageNodeValidate:
    """参数校验测试。"""

    def test_validate_default_params(self):
        node = EmptyLatentImageNode({})
        assert node.validate() == []

    def test_validate_rejects_negative_width(self):
        node = EmptyLatentImageNode({"width": -1})
        errors = node.validate()
        assert any("width" in e for e in errors)

    def test_validate_rejects_zero_height(self):
        node = EmptyLatentImageNode({"height": 0})
        errors = node.validate()
        assert any("height" in e for e in errors)

    def test_validate_rejects_zero_batch_size(self):
        node = EmptyLatentImageNode({"batch_size": 0})
        errors = node.validate()
        assert any("batch_size" in e for e in errors)

    def test_validate_rejects_invalid_fill(self):
        node = EmptyLatentImageNode({"fill": "invalid"})
        errors = node.validate()
        assert any("fill" in e for e in errors)

    def test_validate_rejects_invalid_vae_type(self):
        node = EmptyLatentImageNode({"vae_type": "unknown_vae"})
        errors = node.validate()
        assert any("vae_type" in e for e in errors)

    def test_validate_rejects_negative_seed(self):
        node = EmptyLatentImageNode({"fill": "noise", "seed": -1})
        errors = node.validate()
        assert any("seed" in e for e in errors)

    def test_validate_accepts_valid_custom_channels(self):
        node = EmptyLatentImageNode({"channels": 8})
        assert node.validate() == []

    def test_validate_rejects_negative_channels(self):
        node = EmptyLatentImageNode({"channels": -4})
        errors = node.validate()
        assert any("channels" in e for e in errors)


# ===== vae_type → channels 推断测试 =====


class TestEmptyLatentImageVaeTypeInference:
    """vae_type → channels 推断测试。"""

    def test_sd15_channels_is_4(self):
        assert EmptyLatentImageNode._VAE_TYPE_CHANNELS["sd15"] == 4

    def test_sdxl_channels_is_4(self):
        assert EmptyLatentImageNode._VAE_TYPE_CHANNELS["sdxl"] == 4

    def test_flux_channels_is_16(self):
        assert EmptyLatentImageNode._VAE_TYPE_CHANNELS["flux"] == 16

    def test_sd3_channels_is_16(self):
        assert EmptyLatentImageNode._VAE_TYPE_CHANNELS["sd3"] == 16

    def test_custom_channels_is_4(self):
        assert EmptyLatentImageNode._VAE_TYPE_CHANNELS["custom"] == 4


# ===== cache_key 测试 =====


class TestEmptyLatentImageNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_for_empty_fill(self):
        """fill=empty 时 cache_key 确定性。"""
        node = EmptyLatentImageNode({"width": 1024, "height": 1024, "fill": "empty"})
        key = node.cache_key({})
        assert key is not None
        assert "empty_latent" in key
        assert "1024x1024" in key
        assert "sd15" in key

    def test_cache_key_for_noise_with_seed(self):
        """fill=noise + seed 指定时 cache_key 含 seed。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "fill": "noise", "seed": 42,
        })
        key = node.cache_key({})
        assert key is not None
        assert "empty_latent_noise" in key
        assert "seed42" in key

    def test_cache_key_for_noise_without_seed_returns_none(self):
        """fill=noise + 无 seed 时不缓存。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "fill": "noise",
        })
        key = node.cache_key({})
        assert key is None

    def test_cache_key_differs_for_different_vae_type(self):
        """不同 vae_type 的 cache_key 不同（channels 不同）。"""
        node_sd15 = EmptyLatentImageNode({"width": 1024, "height": 1024, "vae_type": "sd15"})
        node_flux = EmptyLatentImageNode({"width": 1024, "height": 1024, "vae_type": "flux"})
        assert node_sd15.cache_key({}) != node_flux.cache_key({})

    def test_cache_key_differs_for_different_batch_size(self):
        """不同 batch_size 的 cache_key 不同。"""
        node1 = EmptyLatentImageNode({"width": 1024, "height": 1024, "batch_size": 1})
        node2 = EmptyLatentImageNode({"width": 1024, "height": 1024, "batch_size": 2})
        assert node1.cache_key({}) != node2.cache_key({})


# ===== 端到端测试：fill=empty =====


@requires_torch
class TestEmptyLatentImageExecuteEmpty:
    """fill=empty 端到端测试。"""

    @pytest.mark.asyncio
    async def test_execute_empty_default_1024(self):
        """默认参数：1024x1024 → 128x128 latent (scale=8, channels=4)。"""
        node = EmptyLatentImageNode({})
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        assert latent is not None
        # shape: [1, 4, 128, 128]
        assert latent.tensor.shape == (1, 4, 128, 128)
        # 零张量
        assert torch.all(latent.tensor == 0)
        # vae_type 标识
        assert latent.vae_type == "sd15"

    @pytest.mark.asyncio
    async def test_execute_empty_custom_size(self):
        """自定义尺寸：512x512 → 64x64 latent。"""
        node = EmptyLatentImageNode({"width": 512, "height": 512})
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        assert latent.tensor.shape == (1, 4, 64, 64)
        assert torch.all(latent.tensor == 0)

    @pytest.mark.asyncio
    async def test_execute_empty_batch_size(self):
        """batch_size=4 → [4, 4, 128, 128]。"""
        node = EmptyLatentImageNode({"width": 1024, "height": 1024, "batch_size": 4})
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        assert latent.tensor.shape == (4, 4, 128, 128)
        assert torch.all(latent.tensor == 0)

    @pytest.mark.asyncio
    async def test_execute_empty_flux_channels_16(self):
        """vae_type=flux → channels=16。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "vae_type": "flux",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        # Flux 通常 scale=16，所以 1024/16=64
        assert latent.tensor.shape == (1, 16, 128, 128)  # scale 仍为默认 8

    @pytest.mark.asyncio
    async def test_execute_empty_custom_channels(self):
        """显式 channels 覆盖 vae_type 推断。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "channels": 8,
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        assert latent.tensor.shape == (1, 8, 128, 128)

    @pytest.mark.asyncio
    async def test_execute_empty_metadata_complete(self):
        """execute 写入的 Latent metadata 完整。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "batch_size": 2, "fill": "empty",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        meta = latent.metadata
        assert meta["source"] == "empty_latent_image"
        assert meta["width"] == 1024
        assert meta["height"] == 1024
        assert meta["latent_width"] == 128
        assert meta["latent_height"] == 128
        assert meta["channels"] == 4
        assert meta["batch_size"] == 2
        assert meta["fill"] == "empty"
        assert meta["scale"] == 8


# ===== 端到端测试：fill=noise =====


@requires_torch
class TestEmptyLatentImageExecuteNoise:
    """fill=noise 端到端测试。"""

    @pytest.mark.asyncio
    async def test_execute_noise_default(self):
        """fill=noise 默认（无 seed）：生成随机张量。"""
        node = EmptyLatentImageNode({"fill": "noise"})
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        assert latent.tensor.shape == (1, 4, 128, 128)
        # 噪声张量 - 应有非零值
        assert not torch.all(latent.tensor == 0)
        # 噪声值在合理范围
        assert latent.tensor.min() > -10 and latent.tensor.max() < 10

    @pytest.mark.asyncio
    async def test_execute_noise_with_seed_deterministic(self):
        """fill=noise + seed 确定性（相同 seed → 相同结果）。"""
        node1 = EmptyLatentImageNode({"fill": "noise", "seed": 42})
        node2 = EmptyLatentImageNode({"fill": "noise", "seed": 42})
        ctx = make_default_context()

        result1 = await node1.execute({}, ctx)
        result2 = await node2.execute({}, ctx)

        # 相同 seed 应产生相同结果
        assert torch.equal(result1["latent"].tensor, result2["latent"].tensor)

    @pytest.mark.asyncio
    async def test_execute_noise_different_seeds_differ(self):
        """fill=noise + 不同 seed → 不同结果。"""
        node1 = EmptyLatentImageNode({"fill": "noise", "seed": 1})
        node2 = EmptyLatentImageNode({"fill": "noise", "seed": 2})
        ctx = make_default_context()

        result1 = await node1.execute({}, ctx)
        result2 = await node2.execute({}, ctx)

        # 不同 seed 应产生不同结果
        assert not torch.equal(result1["latent"].tensor, result2["latent"].tensor)

    @pytest.mark.asyncio
    async def test_execute_noise_statistics(self):
        """fill=noise 生成的标准正态分布。"""
        node = EmptyLatentImageNode({
            "fill": "noise", "width": 256, "height": 256, "seed": 0,
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        # 大量样本下，均值接近 0，标准差接近 1
        tensor = latent.tensor
        flat = tensor.flatten()
        assert abs(flat.mean().item()) < 0.1
        assert abs(flat.std().item() - 1.0) < 0.1

    @pytest.mark.asyncio
    async def test_execute_noise_metadata_includes_seed(self):
        """fill=noise 时 metadata 包含 seed。"""
        node = EmptyLatentImageNode({"fill": "noise", "seed": 42})
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        meta = result["latent"].metadata
        assert meta["fill"] == "noise"
        assert meta["seed"] == 42


# ===== vae_type 推断端到端测试 =====


@requires_torch
class TestEmptyLatentImageVaeTypeExecution:
    """vae_type 推断在 execute 时的端到端测试。"""

    @pytest.mark.asyncio
    async def test_sd15_implies_4_channels(self):
        """vae_type=sd15 → channels=4。"""
        node = EmptyLatentImageNode({"width": 512, "height": 512, "vae_type": "sd15"})
        ctx = make_default_context()
        result = await node.execute({}, ctx)
        assert result["latent"].tensor.shape == (1, 4, 64, 64)
        assert result["latent"].vae_type == "sd15"

    @pytest.mark.asyncio
    async def test_sdxl_implies_4_channels(self):
        """vae_type=sdxl → channels=4。"""
        node = EmptyLatentImageNode({"width": 1024, "height": 1024, "vae_type": "sdxl"})
        ctx = make_default_context()
        result = await node.execute({}, ctx)
        assert result["latent"].tensor.shape == (1, 4, 128, 128)
        assert result["latent"].vae_type == "sdxl"

    @pytest.mark.asyncio
    async def test_flux_implies_16_channels(self):
        """vae_type=flux → channels=16。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "vae_type": "flux", "scale": 16,
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)
        # scale=16 → 1024/16=64
        assert result["latent"].tensor.shape == (1, 16, 64, 64)
        assert result["latent"].vae_type == "flux"

    @pytest.mark.asyncio
    async def test_sd3_implies_16_channels(self):
        """vae_type=sd3 → channels=16。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "vae_type": "sd3", "scale": 16,
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)
        assert result["latent"].tensor.shape == (1, 16, 64, 64)
        assert result["latent"].vae_type == "sd3"

    @pytest.mark.asyncio
    async def test_custom_vae_type_implies_4_channels(self):
        """vae_type=custom → channels=4（默认）。"""
        node = EmptyLatentImageNode({"width": 512, "height": 512, "vae_type": "custom"})
        ctx = make_default_context()
        result = await node.execute({}, ctx)
        assert result["latent"].tensor.shape == (1, 4, 64, 64)
        assert result["latent"].vae_type == "custom"


# ===== 错误处理测试 =====


@requires_torch
class TestEmptyLatentImageErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_execute_invalid_params_raises(self):
        """参数校验失败抛 NodeValidationError。"""
        node = EmptyLatentImageNode({"width": -1})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await node.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_execute_zero_width_raises(self):
        """width=0 抛 NodeValidationError。"""
        node = EmptyLatentImageNode({"width": 0})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "width" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_invalid_vae_type_raises(self):
        """未知 vae_type 抛 NodeValidationError。"""
        node = EmptyLatentImageNode({"vae_type": "unknown"})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "vae_type" in str(exc_info.value)


# ===== 进度上报测试 =====


@requires_torch
class TestEmptyLatentImageProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self):
        """execute 上报开始和完成进度。"""
        node = EmptyLatentImageNode({"width": 512, "height": 512})
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await node.execute({}, ctx)

        assert len(progress_calls) >= 2
        assert progress_calls[0][0] == 0.0
        assert progress_calls[0][1] == "empty_latent.start"
        assert progress_calls[-1][0] == 1.0
        assert progress_calls[-1][1] == "empty_latent.done"

    @pytest.mark.asyncio
    async def test_progress_includes_dimensions(self):
        """进度上报包含 width/height/latent 尺寸信息。"""
        node = EmptyLatentImageNode({"width": 512, "height": 512})
        progress_kwargs: list[dict] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_kwargs.append(kwargs)

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await node.execute({}, ctx)

        # 第一个进度帧应包含 width/height/latent_width/latent_height
        start_kwargs = progress_kwargs[0]
        assert start_kwargs["width"] == 512
        assert start_kwargs["height"] == 512
        assert start_kwargs["latent_width"] == 64
        assert start_kwargs["latent_height"] == 64
        assert start_kwargs["channels"] == 4


# ===== 与其他节点协同的端到端测试 =====


@requires_torch
class TestEmptyLatentImageIntegration:
    """EmptyLatentImage 在完整工作流中的端到端测试。"""

    @pytest.mark.asyncio
    async def test_typical_text2image_workflow_size(self):
        """典型文生图工作流的尺寸（SDXL 1024x1024）。"""
        node = EmptyLatentImageNode({
            "width": 1024, "height": 1024,
            "batch_size": 1, "fill": "empty", "vae_type": "sdxl",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        # SDXL: 1024/8=128, channels=4
        assert latent.tensor.shape == (1, 4, 128, 128)
        assert latent.vae_type == "sdxl"

    @pytest.mark.asyncio
    async def test_image2image_latent_size(self):
        """图生图场景的 latent 尺寸（应与原图 VAE 编码后一致）。"""
        # 假设输入图 512x512 → VAE 编码后 latent 为 64x64 (scale=8)
        node = EmptyLatentImageNode({
            "width": 512, "height": 512, "vae_type": "sd15",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        latent = result["latent"]
        assert latent.tensor.shape == (1, 4, 64, 64)

    @pytest.mark.asyncio
    async def test_multiple_empty_latents_with_different_vae_types(self):
        """同一工作流中不同 vae_type 的 EmptyLatentImage 节点并存。"""
        ctx = make_default_context()

        node_sd15 = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "vae_type": "sd15",
        })
        node_flux = EmptyLatentImageNode({
            "width": 1024, "height": 1024, "vae_type": "flux", "scale": 16,
        })

        result_sd15 = await node_sd15.execute({}, ctx)
        result_flux = await node_flux.execute({}, ctx)

        # SD15: channels=4
        assert result_sd15["latent"].tensor.shape[1] == 4
        # Flux: channels=16
        assert result_flux["latent"].tensor.shape[1] == 16
        # vae_type 标识不同
        assert result_sd15["latent"].vae_type == "sd15"
        assert result_flux["latent"].vae_type == "flux"

    @pytest.mark.asyncio
    async def test_empty_latent_can_connect_to_sampler(self):
        """EmptyLatentImage 输出可作为 Sampler 的 latent 输入。"""
        node = EmptyLatentImageNode({"width": 1024, "height": 1024, "vae_type": "sdxl"})
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        # 模拟 Sampler 接收 latent
        latent = result["latent"]
        # Sampler 需要的字段：tensor / vae_type
        assert hasattr(latent, "tensor")
        assert hasattr(latent, "vae_type")
        assert latent.tensor.dim() == 4  # [B, C, H, W]
        assert latent.vae_type == "sdxl"

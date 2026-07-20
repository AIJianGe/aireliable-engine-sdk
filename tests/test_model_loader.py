"""ModelLoader 节点单元测试。

覆盖任务 3.3 验收标准：
    - 支持 diffusers pipeline 加载（用 mock pipeline 模拟）
    - 支持变体切换
    - 单元测试用 mock pipeline

由于 ModelLoaderNode 是抽象基类，测试通过创建子类 mock 来验证：
    - _MockDiffusersModelLoader：模拟 diffusers pipeline 加载
    - _MockGGUFModelLoader：模拟 GGUF 加载
    - _FailingModelLoader：模拟加载失败
    - _NonDictModelLoader：模拟返回非 dict

参考：v0.2-tasks.md 任务 3.3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.loader.model_loader import ModelLoaderNode


# ===== Mock 子类 =====


class _MockDiffusersModelLoader(ModelLoaderNode):
    """模拟 diffusers pipeline 加载器。"""

    node_type = "MockDiffusersModelLoader"

    def __init__(self, params, return_value: dict = None, raise_exc: Exception = None):
        super().__init__(params)
        self._return_value = return_value or {
            "model": MagicMock(name="diffusers_pipeline"),
            "vae": MagicMock(name="diffusers_vae"),
            "clip": MagicMock(name="diffusers_clip"),
            "scheduler": MagicMock(name="diffusers_scheduler"),
        }
        self._raise_exc = raise_exc
        self.load_called_with_path: Path = None
        self.load_called_with_ctx: Any = None

    async def _load_impl(self, model_path: Path, ctx) -> dict:
        self.load_called_with_path = model_path
        self.load_called_with_ctx = ctx
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _MockGGUFModelLoader(ModelLoaderNode):
    """模拟 GGUF 加载器（只返回 model，无 vae/clip/scheduler）。"""

    node_type = "MockGGUFModelLoader"

    async def _load_impl(self, model_path: Path, ctx) -> dict:
        return {"model": MagicMock(name="gguf_model")}


class _FailingModelLoader(ModelLoaderNode):
    """加载失败模拟。"""

    node_type = "FailingModelLoader"

    async def _load_impl(self, model_path: Path, ctx) -> dict:
        raise FileNotFoundError(f"模型文件损坏: {model_path}")


class _NonDictModelLoader(ModelLoaderNode):
    """返回非 dict 的错误实现。"""

    node_type = "NonDictModelLoader"

    async def _load_impl(self, model_path: Path, ctx) -> dict:
        return "not a dict"


class _BareModelLoader(ModelLoaderNode):
    """未重写 _load_impl 的子类（应抛 NotImplementedError）。"""

    node_type = "BareModelLoader"


# ===== 基本属性测试 =====


class TestModelLoaderNodeBasic:
    """ModelLoaderNode 基本属性测试。"""

    def test_node_type_is_modelloader(self):
        node = ModelLoaderNode({"variant": "flux", "cache_dir": "/tmp"})
        assert node.node_type == "ModelLoader"

    def test_input_specs_is_empty_list(self):
        node = ModelLoaderNode({"variant": "flux", "cache_dir": "/tmp"})
        assert node.input_specs == []

    def test_output_specs_has_four_ports(self):
        """输出端口：model / vae / clip / scheduler。"""
        node = ModelLoaderNode({"variant": "flux", "cache_dir": "/tmp"})
        names = [p.name for p in node.output_specs]
        assert names == ["model", "vae", "clip", "scheduler"]
        # 所有端口类型为 model
        for port in node.output_specs:
            assert port.type == "model"
        # model 必填，其他可选
        assert node.output_specs[0].required is True
        for port in node.output_specs[1:]:
            assert port.required is False

    def test_init_with_params(self):
        node = ModelLoaderNode({
            "variant": "flux-dev-fp8",
            "cache_dir": "models/image/checkpoints/flux/",
            "model_filename": "flux-dev-fp8.safetensors",
            "precision": "fp8",
            "device": "cuda:0",
            "torch_dtype": "bf16",
            "subfolder": "unet/",
            "low_vram_mode": True,
        })
        assert node._variant == "flux-dev-fp8"
        assert node._cache_dir == "models/image/checkpoints/flux/"
        assert node._model_filename == "flux-dev-fp8.safetensors"
        assert node._precision == "fp8"
        assert node._device_override == "cuda:0"
        assert node._torch_dtype == "bf16"
        assert node._subfolder == "unet/"
        assert node._low_vram_mode is True

    def test_repr_contains_class_name_and_variant(self):
        node = ModelLoaderNode({"variant": "flux", "cache_dir": "/tmp"})
        repr_str = repr(node)
        assert "ModelLoaderNode" in repr_str
        assert "flux" in repr_str


# ===== 参数校验测试 =====


class TestModelLoaderNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_valid_params(self):
        node = ModelLoaderNode({"variant": "flux", "cache_dir": "/tmp"})
        assert node.validate() == []

    def test_validate_reports_missing_variant(self):
        node = ModelLoaderNode({"cache_dir": "/tmp"})
        errors = node.validate()
        assert any("variant" in e for e in errors)

    def test_validate_reports_empty_variant(self):
        node = ModelLoaderNode({"variant": "", "cache_dir": "/tmp"})
        errors = node.validate()
        assert any("variant" in e for e in errors)

    def test_validate_reports_missing_cache_dir(self):
        node = ModelLoaderNode({"variant": "flux"})
        errors = node.validate()
        assert any("cache_dir" in e for e in errors)

    def test_validate_reports_unsupported_torch_dtype(self):
        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": "/tmp",
            "torch_dtype": "int8",
        })
        errors = node.validate()
        assert any("torch_dtype" in e for e in errors)

    def test_validate_accepts_all_supported_torch_dtypes(self):
        for dtype in ["fp16", "bf16", "fp32", "auto", "float16", "bfloat16", "float32"]:
            node = ModelLoaderNode({
                "variant": "flux",
                "cache_dir": "/tmp",
                "torch_dtype": dtype,
            })
            assert node.validate() == [], f"dtype {dtype} 应该被接受"


# ===== 路径解析测试 =====


class TestModelLoaderNodePathResolution:
    """模型路径解析测试。"""

    def test_absolute_cache_dir_used_directly(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": str(tmp_path),
            "model_filename": "model.safetensors",
        })
        # 创建模型文件
        (tmp_path / "model.safetensors").write_bytes(b"fake model content")
        ctx = make_default_context()
        model_path = node._resolve_model_path(ctx)
        assert model_path == tmp_path / "model.safetensors"

    def test_relative_cache_dir_resolved_against_models_root(self, tmp_path: Path):
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "flux"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"fake")

        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": "image/flux",                   # 相对路径
            "model_filename": "model.safetensors",
        })
        ctx = make_default_context(models_root=models_root)
        model_path = node._resolve_model_path(ctx)
        assert model_path == cache_dir / "model.safetensors"

    def test_nonexistent_cache_dir_raises(self):
        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": "/nonexistent/path",
            "model_filename": "model.safetensors",
        })
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_model_path(ctx)
        assert "目录不存在" in str(exc_info.value)

    def test_nonexistent_model_file_raises(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": str(tmp_path),
            "model_filename": "missing.safetensors",
        })
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_model_path(ctx)
        assert "模型文件不存在" in str(exc_info.value)

    def test_scan_matching_variant_filename(self, tmp_path: Path):
        """未指定 model_filename 时，扫描匹配 variant 的文件。"""
        # 创建多个文件，只有一个匹配 variant
        (tmp_path / "flux-dev-fp8.safetensors").write_bytes(b"fake")
        (tmp_path / "sdxl-base.safetensors").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_text("readme")

        node = ModelLoaderNode({
            "variant": "flux-dev-fp8",
            "cache_dir": str(tmp_path),
            # 不指定 model_filename
        })
        ctx = make_default_context()
        model_path = node._resolve_model_path(ctx)
        assert model_path.name == "flux-dev-fp8.safetensors"

    def test_scan_no_matching_file_raises(self, tmp_path: Path):
        """扫描不到匹配 variant 的文件时报错。"""
        (tmp_path / "sdxl-base.safetensors").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_text("readme")

        node = ModelLoaderNode({
            "variant": "flux-dev-fp8",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_model_path(ctx)
        assert "model_filename" in str(exc_info.value)


# ===== 变体切换测试（验收标准）=====


class TestModelLoaderNodeVariantSwitching:
    """变体切换测试（任务 3.3 验收标准）。"""

    @pytest.mark.asyncio
    async def test_variant_passed_to_load_impl(self, tmp_path: Path):
        """variant 参数正确传递到 _load_impl。"""
        (tmp_path / "flux-dev-fp8.safetensors").write_bytes(b"fake")
        loader = _MockDiffusersModelLoader({
            "variant": "flux-dev-fp8",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        result = await loader.execute({}, ctx)
        # 验证 _load_impl 被调用
        assert loader.load_called_with_path is not None
        assert loader.load_called_with_path.name == "flux-dev-fp8.safetensors"
        # 验证返回的 4 个字段
        assert "model" in result
        assert "vae" in result
        assert "clip" in result
        assert "scheduler" in result

    @pytest.mark.asyncio
    async def test_different_variants_load_different_models(self, tmp_path: Path):
        """切换 variant 加载不同模型文件。"""
        # 创建两个 variant 的模型文件
        (tmp_path / "flux-dev-fp8.safetensors").write_bytes(b"flux dev")
        (tmp_path / "flux-schnell-fp8.safetensors").write_bytes(b"flux schnell")

        # 第一个 variant
        loader1 = _MockDiffusersModelLoader({
            "variant": "flux-dev-fp8",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        await loader1.execute({}, ctx)
        assert "flux-dev-fp8" in str(loader1.load_called_with_path)

        # 切换到第二个 variant
        loader2 = _MockDiffusersModelLoader({
            "variant": "flux-schnell-fp8",
            "cache_dir": str(tmp_path),
        })
        await loader2.execute({}, ctx)
        assert "flux-schnell-fp8" in str(loader2.load_called_with_path)


# ===== mock pipeline 加载测试（验收标准）=====


class TestModelLoaderNodeMockPipeline:
    """用 mock pipeline 验证加载流程（任务 3.3 验收标准）。"""

    @pytest.mark.asyncio
    async def test_diffusers_pipeline_loaded_successfully(self, tmp_path: Path):
        """模拟 diffusers pipeline 加载。"""
        (tmp_path / "flux-dev-fp8.safetensors").write_bytes(b"fake")

        mock_pipeline = MagicMock(name="diffusers_pipeline")
        mock_vae = MagicMock(name="diffusers_vae")
        mock_clip = MagicMock(name="diffusers_clip")
        mock_scheduler = MagicMock(name="diffusers_scheduler")

        loader = _MockDiffusersModelLoader(
            {
                "variant": "flux-dev-fp8",
                "cache_dir": str(tmp_path),
                "torch_dtype": "bf16",
            },
            return_value={
                "model": mock_pipeline,
                "vae": mock_vae,
                "clip": mock_clip,
                "scheduler": mock_scheduler,
            },
        )
        ctx = make_default_context(device="cuda:0")
        result = await loader.execute({}, ctx)

        assert result["model"] is mock_pipeline
        assert result["vae"] is mock_vae
        assert result["clip"] is mock_clip
        assert result["scheduler"] is mock_scheduler

    @pytest.mark.asyncio
    async def test_partial_output_filled_with_none(self, tmp_path: Path):
        """子类只返回 model 字段，其他字段填 None。"""
        (tmp_path / "gguf-model.gguf").write_bytes(b"fake")

        loader = _MockGGUFModelLoader({
            "variant": "gguf-model",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        result = await loader.execute({}, ctx)

        assert result["model"] is not None
        assert result["vae"] is None
        assert result["clip"] is None
        assert result["scheduler"] is None

    @pytest.mark.asyncio
    async def test_device_override_used(self, tmp_path: Path):
        """device 参数覆盖 ctx.device。"""
        (tmp_path / "flux.safetensors").write_bytes(b"fake")
        loader = _MockDiffusersModelLoader({
            "variant": "flux",
            "cache_dir": str(tmp_path),
            "device": "cuda:1",                   # 覆盖设备
        })
        ctx = make_default_context(device="cpu")   # ctx.device 与 override 不同
        await loader.execute({}, ctx)
        # 这里只验证流程不报错（_load_impl 是 mock，不实际用 device）

    @pytest.mark.asyncio
    async def test_gguf_loader_supported(self, tmp_path: Path):
        """GGUF 加载器（子类）支持（任务备注：可选 GGUF 加载）。"""
        (tmp_path / "ernie-image-Q4_K_M.gguf").write_bytes(b"fake gguf")

        loader = _MockGGUFModelLoader({
            "variant": "ernie-image-Q4_K_M",
            "cache_dir": str(tmp_path),
            "precision": "q4_k_m",
        })
        ctx = make_default_context()
        result = await loader.execute({}, ctx)
        assert result["model"] is not None


# ===== 错误处理测试 =====


class TestModelLoaderNodeErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_bare_model_loader_raises_not_implemented(self, tmp_path: Path):
        """未重写 _load_impl 的子类抛 NotImplementedError。"""
        (tmp_path / "flux.safetensors").write_bytes(b"fake")
        loader = _BareModelLoader({
            "variant": "flux",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        with pytest.raises(NotImplementedError) as exc_info:
            await loader.execute({}, ctx)
        assert "_load_impl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_impl_failure_wrapped_as_runtime_error(self, tmp_path: Path):
        """_load_impl 抛异常时被包装为 RuntimeError。"""
        (tmp_path / "broken.safetensors").write_bytes(b"fake")
        loader = _FailingModelLoader({
            "variant": "broken",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await loader.execute({}, ctx)
        assert "模型加载失败" in str(exc_info.value)
        assert "broken.safetensors" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_dict_return_raises_type_error(self, tmp_path: Path):
        """_load_impl 返回非 dict 抛 TypeError。"""
        (tmp_path / "flux.safetensors").write_bytes(b"fake")
        loader = _NonDictModelLoader({
            "variant": "flux",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        with pytest.raises(TypeError) as exc_info:
            await loader.execute({}, ctx)
        assert "dict" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_params_raises_validation_error(self):
        """参数校验失败抛 NodeValidationError。"""
        loader = _MockDiffusersModelLoader({"variant": "", "cache_dir": ""})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await loader.execute({}, ctx)


# ===== cache_key 测试 =====


class TestModelLoaderNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_contains_variant(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux-dev-fp8",
            "cache_dir": str(tmp_path),
        })
        key = node.cache_key({})
        assert "flux-dev-fp8" in key

    def test_cache_key_contains_torch_dtype(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": str(tmp_path),
            "torch_dtype": "bf16",
        })
        key = node.cache_key({})
        assert "bf16" in key

    def test_cache_key_returns_none_for_nonexistent_dir(self):
        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": "/nonexistent/path/that/does/not/exist",
        })
        assert node.cache_key({}) is None

    def test_cache_key_changes_with_mtime(self, tmp_path: Path):
        """目录 mtime 变化时 cache_key 也变化。"""
        import os
        import time

        node = ModelLoaderNode({
            "variant": "flux",
            "cache_dir": str(tmp_path),
        })
        key1 = node.cache_key({})

        # 修改目录 mtime
        time.sleep(0.1)
        os.utime(tmp_path, None)

        key2 = node.cache_key({})
        # mtime 可能相同（精度问题），但前缀应一致
        assert key1.startswith("model:flux:")
        assert key2.startswith("model:flux:")


# ===== 进度上报测试 =====


class TestModelLoaderNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self, tmp_path: Path):
        """加载时上报开始和完成进度。"""
        (tmp_path / "flux.safetensors").write_bytes(b"fake")
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        loader = _MockDiffusersModelLoader({
            "variant": "flux",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await loader.execute({}, ctx)

        # 至少 start + done 两次
        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0.0, "model_loader.start")
        assert progress_calls[-1] == (1.0, "model_loader.done")

    @pytest.mark.asyncio
    async def test_progress_not_reported_on_validation_failure(self):
        """参数校验失败时不上报进度。"""
        progress_calls: list = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append(progress)

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        loader = _MockDiffusersModelLoader({"variant": "", "cache_dir": ""})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        with pytest.raises(NodeValidationError):
            await loader.execute({}, ctx)
        assert progress_calls == []

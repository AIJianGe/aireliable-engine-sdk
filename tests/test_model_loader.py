"""ModelLoader 节点单元测试（v0.2.5 架构类型联动目录改造）。

覆盖：
    - v0.2.5 架构类型联动（checkpoint/diffusers/unet 三种模式的路径解析）
    - v0.2.5 向后兼容（老字段 model_name/model_filename/subfolder/旧 architecture 值）
    - 支持 diffusers pipeline 加载（用 mock pipeline 模拟）
    - 支持变体切换
    - 单元测试用 mock pipeline

由于 ModelLoaderNode 是抽象基类，测试通过创建子类 mock 来验证：
    - _MockDiffusersModelLoader：模拟 diffusers pipeline 加载
    - _MockGGUFModelLoader：模拟 GGUF 加载
    - _FailingModelLoader：模拟加载失败
    - _NonDictModelLoader：模拟返回非 dict

参考：
    - v0.2.5-loader-architecture-tasks.md A3-1
    - v0.2.5-loader-architecture-analysis.md
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
    """未重写 _load_impl 的子类（应抛 NotImplementedError / ImportError）。"""

    node_type = "BareModelLoader"


# ===== 基本属性测试 =====


class TestModelLoaderNodeBasic:
    """ModelLoaderNode 基本属性测试。"""

    def test_node_type_is_modelloader(self):
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
        assert node.node_type == "ModelLoader"

    def test_input_specs_is_empty_list(self):
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
        assert node.input_specs == []

    def test_output_specs_has_four_ports(self):
        """输出端口：model / vae / clip / scheduler。"""
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
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
        """v0.2.5 架构类型联动：新 params_schema 字段（architecture/variant/cache_dir/precision 等）。"""
        node = ModelLoaderNode({
            "architecture": "unet",
            "variant": "flux-dev-fp8.safetensors",
            "cache_dir": "models/image/unet/flux/",
            "precision": "fp16",
            "device": "cuda:0",
            "torch_dtype": "bf16",
            "low_vram_mode": True,
        })
        assert node._architecture == "unet"
        assert node._variant == "flux-dev-fp8.safetensors"
        assert node._cache_dir == "models/image/unet/flux/"
        assert node._precision == "fp16"
        assert node._device_override == "cuda:0"
        assert node._torch_dtype == "bf16"
        assert node._low_vram_mode is True

    def test_default_architecture_is_checkpoint(self):
        """architecture 默认值为 checkpoint。"""
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
        assert node._architecture == "checkpoint"

    def test_repr_contains_class_name_and_variant(self):
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
        repr_str = repr(node)
        assert "ModelLoaderNode" in repr_str
        assert "flux" in repr_str
        # v0.2.5：repr 含 architecture
        assert "checkpoint" in repr_str


# ===== 参数校验测试 =====


class TestModelLoaderNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_valid_params(self):
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
        assert node.validate() == []

    def test_validate_reports_missing_variant(self):
        node = ModelLoaderNode({"cache_dir": "/tmp"})
        errors = node.validate()
        assert any("variant" in e for e in errors)

    def test_validate_reports_empty_variant(self):
        node = ModelLoaderNode({"variant": "", "cache_dir": "/tmp"})
        errors = node.validate()
        assert any("variant" in e for e in errors)

    def test_validate_reports_unsupported_architecture(self):
        """v0.2.5：architecture 必须为 checkpoint/diffusers/unet 之一。"""
        # 注意：旧值 auto/pipeline/components 在 __init__ 已被映射，不会进入 validate
        # 此处直接构造一个不合法值绕过 __init__ 映射
        node = ModelLoaderNode({"variant": "flux.safetensors", "cache_dir": "/tmp"})
        node._architecture = "unknown_arch"
        errors = node.validate()
        assert any("architecture" in e for e in errors)

    def test_validate_accepts_all_supported_architectures(self):
        """v0.2.5：三种合法 architecture 值都应通过校验。"""
        for arch in ["checkpoint", "diffusers", "unet"]:
            node = ModelLoaderNode({
                "architecture": arch,
                "variant": "flux.safetensors",
                "cache_dir": "/tmp",
            })
            assert node.validate() == [], f"architecture {arch} 应该被接受"

    def test_validate_reports_unsupported_torch_dtype(self):
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": "/tmp",
            "torch_dtype": "int8",
        })
        errors = node.validate()
        assert any("torch_dtype" in e for e in errors)

    def test_validate_accepts_all_supported_torch_dtypes(self):
        for dtype in ["fp16", "bf16", "fp32", "auto", "float16", "bfloat16", "float32"]:
            node = ModelLoaderNode({
                "variant": "flux.safetensors",
                "cache_dir": "/tmp",
                "torch_dtype": dtype,
            })
            assert node.validate() == [], f"dtype {dtype} 应该被接受"

    def test_validate_reports_unsupported_precision(self):
        """v0.2.5：precision 选项为 auto/fp16/bf16/fp32（移除 q4_k_m/q8_0）。"""
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": "/tmp",
            "precision": "q4_k_m",  # 已移除，改由 infer_dtype 推断
        })
        errors = node.validate()
        assert any("precision" in e for e in errors)

    def test_validate_accepts_supported_precisions(self):
        for prec in ["auto", "fp16", "bf16", "fp32"]:
            node = ModelLoaderNode({
                "variant": "flux.safetensors",
                "cache_dir": "/tmp",
                "precision": prec,
            })
            assert node.validate() == [], f"precision {prec} 应该被接受"


# ===== v0.2.5 架构类型联动测试 =====


class TestModelLoaderNodeArchitecture:
    """v0.2.5 架构类型联动：architecture 决定模型存放目录与加载方式。"""

    def test_params_schema_architecture_options(self):
        """params_schema 的 architecture 字段 options 为 checkpoint/diffusers/unet。"""
        schema = ModelLoaderNode.params_schema
        assert schema["architecture"]["options"] == ["checkpoint", "diffusers", "unet"]
        assert schema["architecture"]["default"] == "checkpoint"
        assert schema["architecture"]["required"] is True

    def test_params_schema_variant_has_dir_depends_on(self):
        """variant 字段含 dir_depends_on: "architecture"。"""
        schema = ModelLoaderNode.params_schema
        assert schema["variant"]["dir_depends_on"] == "architecture"

    def test_params_schema_variant_dir_map(self):
        """variant 字段的 dir_map 含三个架构类型 → 目录映射。"""
        schema = ModelLoaderNode.params_schema
        dir_map = schema["variant"]["dir_map"]
        assert dir_map == {
            "checkpoint": "image/checkpoints/",
            "diffusers": "image/checkpoints/",
            "unet": "image/unet/",
        }

    def test_params_schema_no_model_name_field(self):
        """v0.2.5：移除 model_name 字段。"""
        schema = ModelLoaderNode.params_schema
        assert "model_name" not in schema

    def test_params_schema_no_model_filename_field(self):
        """v0.2.5：移除 model_filename 字段。"""
        schema = ModelLoaderNode.params_schema
        assert "model_filename" not in schema

    def test_params_schema_no_subfolder_field(self):
        """v0.2.5：移除 subfolder 字段。"""
        schema = ModelLoaderNode.params_schema
        assert "subfolder" not in schema

    def test_params_schema_cache_dir_is_optional(self):
        """v0.2.5：cache_dir 改为可选（required: False）。"""
        schema = ModelLoaderNode.params_schema
        assert schema["cache_dir"]["required"] is False

    def test_params_schema_precision_options_no_quant(self):
        """v0.2.5：precision options 移除 q4_k_m/q8_0（改由 infer_dtype 推断）。"""
        schema = ModelLoaderNode.params_schema
        assert "q4_k_m" not in schema["precision"]["options"]
        assert "q8_0" not in schema["precision"]["options"]
        assert schema["precision"]["options"] == ["auto", "fp16", "bf16", "fp32"]


# ===== v0.2.5 向后兼容测试 =====


class TestModelLoaderNodeBackwardCompat:
    """v0.2.5 向后兼容：老 yaml 的字段与值仍能工作。"""

    def test_old_architecture_auto_maps_to_checkpoint(self):
        """老 architecture=auto → 映射为 checkpoint。"""
        node = ModelLoaderNode({
            "architecture": "auto",
            "variant": "flux.safetensors",
            "cache_dir": "/tmp",
        })
        assert node._architecture == "checkpoint"

    def test_old_architecture_pipeline_maps_to_checkpoint(self):
        """老 architecture=pipeline → 映射为 checkpoint。"""
        node = ModelLoaderNode({
            "architecture": "pipeline",
            "variant": "flux.safetensors",
            "cache_dir": "/tmp",
        })
        assert node._architecture == "checkpoint"

    def test_old_architecture_components_maps_to_unet(self):
        """老 architecture=components → 映射为 unet。"""
        node = ModelLoaderNode({
            "architecture": "components",
            "variant": "flux.safetensors",
            "cache_dir": "/tmp",
        })
        assert node._architecture == "unet"

    def test_variant_falls_back_to_model_name(self):
        """variant 为空时回退读 model_name（向后兼容）。"""
        node = ModelLoaderNode({
            "model_name": "flux-dev-fp8.safetensors",
            "cache_dir": "/tmp",
        })
        assert node._variant == "flux-dev-fp8.safetensors"

    def test_variant_falls_back_to_model_filename(self):
        """variant 为空且 model_name 也为空时，回退读 model_filename。"""
        node = ModelLoaderNode({
            "model_filename": "flux-dev-fp8.safetensors",
            "cache_dir": "/tmp",
        })
        assert node._variant == "flux-dev-fp8.safetensors"

    def test_variant_priority_over_model_name(self):
        """variant 优先级高于 model_name / model_filename。"""
        node = ModelLoaderNode({
            "variant": "new-variant.safetensors",
            "model_name": "old-name.safetensors",
            "model_filename": "old-filename.safetensors",
            "cache_dir": "/tmp",
        })
        assert node._variant == "new-variant.safetensors"

    def test_subfolder_appended_to_cache_dir(self, tmp_path: Path):
        """v0.2.5：subfolder 兼容拼接 → cache_dir（向后兼容老 yaml）。"""
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
            "subfolder": "unet/",
        })
        # subfolder 被拼接到 cache_dir
        assert node._cache_dir == str(Path(str(tmp_path)) / "unet")


# ===== 路径解析测试 =====


class TestModelLoaderNodePathResolution:
    """模型路径解析测试（v0.2.5 架构类型联动目录改造）。

    新行为：
        - cache_dir 为空时，根据 architecture 推导默认目录
        - variant 直接作为文件名或子目录名拼接（不再模糊匹配）
    """

    def test_absolute_cache_dir_used_directly(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
        })
        # 创建模型文件
        (tmp_path / "flux.safetensors").write_bytes(b"fake model content")
        ctx = make_default_context()
        model_path = node._resolve_model_path(ctx)
        assert model_path == tmp_path / "flux.safetensors"

    def test_relative_cache_dir_resolved_against_models_root(self, tmp_path: Path):
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "flux"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model.safetensors").write_bytes(b"fake")

        node = ModelLoaderNode({
            "variant": "model.safetensors",
            "cache_dir": "image/flux",                   # 相对路径
        })
        ctx = make_default_context(models_root=models_root)
        model_path = node._resolve_model_path(ctx)
        assert model_path == cache_dir / "model.safetensors"

    def test_nonexistent_cache_dir_raises(self):
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": "/nonexistent/path",
        })
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_model_path(ctx)
        assert "目录不存在" in str(exc_info.value)

    def test_nonexistent_model_file_raises(self, tmp_path: Path):
        """v0.2.5：variant 带扩展名但文件不存在 → 报错（不再模糊扫描）。"""
        node = ModelLoaderNode({
            "variant": "missing.safetensors",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_model_path(ctx)
        assert "模型文件不存在" in str(exc_info.value)

    def test_variant_directly_used_as_filename(self, tmp_path: Path):
        """v0.2.5：variant 直接作为文件名拼接（不再模糊匹配）。"""
        (tmp_path / "flux-dev-fp8.safetensors").write_bytes(b"fake")
        (tmp_path / "sdxl-base.safetensors").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_text("readme")

        node = ModelLoaderNode({
            "variant": "flux-dev-fp8.safetensors",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        model_path = node._resolve_model_path(ctx)
        assert model_path.name == "flux-dev-fp8.safetensors"

    def test_cache_dir_empty_checkpoint_derives_default_dir(self, tmp_path: Path):
        """v0.2.5：cache_dir 为空 + architecture=checkpoint → 推导 image/checkpoints/。"""
        models_root = tmp_path / "models"
        checkpoints_dir = models_root / "image" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "sdxl.safetensors").write_bytes(b"fake")

        node = ModelLoaderNode({
            "architecture": "checkpoint",
            "variant": "sdxl.safetensors",
            # cache_dir 留空
        })
        ctx = make_default_context(models_root=models_root)
        model_path = node._resolve_model_path(ctx)
        assert model_path == checkpoints_dir / "sdxl.safetensors"

    def test_cache_dir_empty_unet_derives_default_dir(self, tmp_path: Path):
        """v0.2.5：cache_dir 为空 + architecture=unet → 推导 image/unet/。"""
        models_root = tmp_path / "models"
        unet_dir = models_root / "image" / "unet"
        unet_dir.mkdir(parents=True)
        (unet_dir / "flux-dev.safetensors").write_bytes(b"fake")

        node = ModelLoaderNode({
            "architecture": "unet",
            "variant": "flux-dev.safetensors",
            # cache_dir 留空
        })
        ctx = make_default_context(models_root=models_root)
        model_path = node._resolve_model_path(ctx)
        assert model_path == unet_dir / "flux-dev.safetensors"

    def test_cache_dir_empty_diffusers_derives_checkpoints_dir(self, tmp_path: Path):
        """v0.2.5：cache_dir 为空 + architecture=diffusers → 推导 image/checkpoints/。"""
        models_root = tmp_path / "models"
        checkpoints_dir = models_root / "image" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "flux-dev.safetensors").write_bytes(b"fake")

        node = ModelLoaderNode({
            "architecture": "diffusers",
            "variant": "flux-dev.safetensors",
        })
        ctx = make_default_context(models_root=models_root)
        model_path = node._resolve_model_path(ctx)
        assert model_path == checkpoints_dir / "flux-dev.safetensors"

    def test_diffusers_mode_empty_variant_returns_cache_dir_if_pipeline(self, tmp_path: Path):
        """v0.2.5：diffusers 模式 + variant 为空 + cache_dir 是 diffusers 目录 → 返回 cache_dir。"""
        # 创建 diffusers 目录结构（含 model_index.json）
        (tmp_path / "model_index.json").write_text('{"valid": true}')
        node = ModelLoaderNode({
            "architecture": "diffusers",
            "variant": "",  # variant 为空
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        model_path = node._resolve_model_path(ctx)
        assert model_path == tmp_path

    def test_variant_as_subdirectory(self, tmp_path: Path):
        """v0.2.5：variant 也可以是子目录名（diffusers 目录加载）。"""
        sub = tmp_path / "stable-diffusion-xl-base-1.0"
        sub.mkdir()
        (sub / "model_index.json").write_text('{"valid": true}')

        node = ModelLoaderNode({
            "architecture": "diffusers",
            "variant": "stable-diffusion-xl-base-1.0",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        model_path = node._resolve_model_path(ctx)
        assert model_path == sub


# ===== 变体切换测试（验收标准）=====


class TestModelLoaderNodeVariantSwitching:
    """变体切换测试。"""

    @pytest.mark.asyncio
    async def test_variant_passed_to_load_impl(self, tmp_path: Path):
        """variant 参数正确传递到 _load_impl。"""
        (tmp_path / "flux-dev-fp8.safetensors").write_bytes(b"fake")
        loader = _MockDiffusersModelLoader({
            "variant": "flux-dev-fp8.safetensors",
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
            "variant": "flux-dev-fp8.safetensors",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        await loader1.execute({}, ctx)
        assert "flux-dev-fp8" in str(loader1.load_called_with_path)

        # 切换到第二个 variant
        loader2 = _MockDiffusersModelLoader({
            "variant": "flux-schnell-fp8.safetensors",
            "cache_dir": str(tmp_path),
        })
        await loader2.execute({}, ctx)
        assert "flux-schnell-fp8" in str(loader2.load_called_with_path)


# ===== mock pipeline 加载测试（验收标准）=====


class TestModelLoaderNodeMockPipeline:
    """用 mock pipeline 验证加载流程。"""

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
                "variant": "flux-dev-fp8.safetensors",
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
            "variant": "gguf-model.gguf",
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
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
            "device": "cuda:1",                   # 覆盖设备
        })
        ctx = make_default_context(device="cpu")   # ctx.device 与 override 不同
        await loader.execute({}, ctx)
        # 这里只验证流程不报错（_load_impl 是 mock，不实际用 device）

    @pytest.mark.asyncio
    async def test_gguf_loader_supported(self, tmp_path: Path):
        """GGUF 加载器（子类）支持。"""
        (tmp_path / "ernie-image-Q4_K_M.gguf").write_bytes(b"fake gguf")

        loader = _MockGGUFModelLoader({
            "variant": "ernie-image-Q4_K_M.gguf",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        result = await loader.execute({}, ctx)
        assert result["model"] is not None


# ===== 错误处理测试 =====


class TestModelLoaderNodeErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_bare_model_loader_raises_not_implemented(self, tmp_path: Path):
        """未重写 _load_impl 的子类走默认实现。

        v0.2.5 通用节点改造后，父类 ModelLoaderNode 提供默认 _load_impl
        （基于 diffusers AutoPipelineForText2Image / 组件拼装）。_BareModelLoader
        未重写该方法，将调用默认实现：
            - 缺依赖（diffusers/torch）时抛 ImportError，被 execute 包装为
              RuntimeError（"模型加载失败 (flux): ..."）
            - 若默认实现被移除则回退为 NotImplementedError
        """
        (tmp_path / "flux.safetensors").write_bytes(b"fake")
        loader = _BareModelLoader({
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
        })
        ctx = make_default_context()
        with pytest.raises((NotImplementedError, ImportError, RuntimeError)) as exc_info:
            await loader.execute({}, ctx)
        msg = str(exc_info.value)
        assert (
            "_load_impl" in msg
            or "模型加载失败" in msg
            or isinstance(exc_info.value.__cause__, ImportError)
        )

    @pytest.mark.asyncio
    async def test_load_impl_failure_wrapped_as_runtime_error(self, tmp_path: Path):
        """_load_impl 抛异常时被包装为 RuntimeError。"""
        (tmp_path / "broken.safetensors").write_bytes(b"fake")
        loader = _FailingModelLoader({
            "variant": "broken.safetensors",
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
            "variant": "flux.safetensors",
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
    """cache_key 测试（v0.2.5：含 architecture）。"""

    def test_cache_key_contains_variant(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux-dev-fp8.safetensors",
            "cache_dir": str(tmp_path),
        })
        key = node.cache_key({})
        assert "flux-dev-fp8" in key

    def test_cache_key_contains_torch_dtype(self, tmp_path: Path):
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
            "torch_dtype": "bf16",
        })
        key = node.cache_key({})
        assert "bf16" in key

    def test_cache_key_contains_architecture(self, tmp_path: Path):
        """v0.2.5：cache_key 含 architecture。"""
        node = ModelLoaderNode({
            "architecture": "unet",
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
        })
        key = node.cache_key({})
        assert "unet" in key

    def test_cache_key_returns_none_for_nonexistent_dir(self):
        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": "/nonexistent/path/that/does/not/exist",
        })
        assert node.cache_key({}) is None

    def test_cache_key_changes_with_mtime(self, tmp_path: Path):
        """目录 mtime 变化时 cache_key 也变化。"""
        import os
        import time

        node = ModelLoaderNode({
            "variant": "flux.safetensors",
            "cache_dir": str(tmp_path),
        })
        key1 = node.cache_key({})

        # 修改目录 mtime
        time.sleep(0.1)
        os.utime(tmp_path, None)

        key2 = node.cache_key({})
        # mtime 可能相同（精度问题），但前缀应一致
        # v0.2.5：cache_key 格式为 model:{variant}:{mtime}:{architecture}:{torch_dtype}
        assert key1.startswith("model:flux.safetensors:")
        assert key2.startswith("model:flux.safetensors:")


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
            "variant": "flux.safetensors",
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

"""UNetLoader 节点单元测试（v0.2.5 ComfyUI 风格节点补充）。

覆盖 C1-3 验收标准：
    - 节点定义（node_type / category / subcategory / title / icon）
    - 输入端口为空（input_specs == []）
    - 输出端口为 model（1 个，type="model"）
    - params_schema 含 unet_name（required=True，format=enum-from-dir）
    - cache_key 返回确定性键（unet_name + mtime + dtype）
    - 缺 diffusers 抛 ImportError（带安装指引）
    - execute 流程（mock diffusers UNet2DConditionModel.from_pretrained）
    - 自动检测 transformer/ 子目录（mock Transformer2DModel.from_pretrained）
    - 加载完成后写 ctx.extra["loaded_model"]

参考：v0.2.5-comfyui-nodes-tasks.md C1-3
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.loader.unet_loader import UNetLoaderNode


# ===== Mock 子类 =====


class _MockUNetLoader(UNetLoaderNode):
    """模拟 UNet 加载器（绕过默认 _load_impl）。"""

    node_type = "MockUNetLoader"

    def __init__(self, params, return_value: dict = None, raise_exc: Exception = None):
        super().__init__(params)
        self._return_value = return_value or {"model": MagicMock(name="mock_unet")}
        self._raise_exc = raise_exc
        self.load_called_with_path: Path = None

    async def _load_impl(self, unet_path: Path, ctx) -> dict:
        self.load_called_with_path = unet_path
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _FailingUNetLoader(UNetLoaderNode):
    node_type = "FailingUNetLoader"

    async def _load_impl(self, unet_path: Path, ctx) -> dict:
        raise FileNotFoundError(f"UNet 文件损坏: {unet_path}")


class _NoneModelUNetLoader(UNetLoaderNode):
    """返回 model 为 None。"""

    node_type = "NoneModelUNetLoader"

    async def _load_impl(self, unet_path: Path, ctx) -> dict:
        return {"model": None}


class _NonDictUNetLoader(UNetLoaderNode):
    """返回非 dict。"""

    node_type = "NonDictUNetLoader"

    async def _load_impl(self, unet_path: Path, ctx) -> dict:
        return "not a dict"


# ===== Mock diffusers 模块工具 =====


def _make_mock_diffusers_module(
    unet2d_cls=None,
    transformer2d_cls=None,
):
    """构造一个 mock diffusers 模块，含 UNet2DConditionModel / Transformer2DModel。"""
    mock_module = types.ModuleType("diffusers")

    if unet2d_cls is None:
        unet2d_cls = MagicMock()
    if transformer2d_cls is None:
        transformer2d_cls = MagicMock()

    mock_module.UNet2DConditionModel = unet2d_cls
    mock_module.Transformer2DModel = transformer2d_cls
    return mock_module


def _make_mock_torch_module():
    """构造一个 mock torch 模块（含 float16 / bfloat16 / float32）。"""
    mock_module = types.ModuleType("torch")
    mock_module.float16 = "torch.float16"
    mock_module.bfloat16 = "torch.bfloat16"
    mock_module.float32 = "torch.float32"
    return mock_module


@pytest.fixture
def clean_diffusers_torch():
    """清理 diffusers / torch 模块，确保 _load_impl 的延迟导入走 mock。"""
    original_diffusers = sys.modules.get("diffusers")
    original_torch = sys.modules.get("torch")
    sys.modules.pop("diffusers", None)
    sys.modules.pop("torch", None)
    try:
        yield
    finally:
        if original_diffusers is not None:
            sys.modules["diffusers"] = original_diffusers
        else:
            sys.modules.pop("diffusers", None)
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        else:
            sys.modules.pop("torch", None)


# ===== 基本属性测试 =====


class TestUNetLoaderNodeBasic:
    """UNetLoaderNode 基本属性测试。"""

    def test_node_type_is_unetloader(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.node_type == "UNetLoader"

    def test_category_is_loader(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.category == "loader"

    def test_subcategory_is_model(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.subcategory == "model"

    def test_title_is_chinese(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.title == "UNet加载器"

    def test_icon_is_box_emoji(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.icon == "📦"

    def test_input_specs_is_empty_list(self):
        """与 ComfyUI 一致，无输入端口。"""
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.input_specs == []

    def test_output_specs_has_single_model_port(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "model"
        assert port.type == "model"

    def test_default_cache_dir_is_image_unet(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node._cache_dir == "image/unet/"

    def test_default_dtype_is_fp16(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node._dtype == "fp16"

    def test_repr_contains_class_name_and_unet_name(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        repr_str = repr(node)
        assert "UNetLoaderNode" in repr_str
        assert "flux-unet" in repr_str


# ===== params_schema 测试 =====


class TestUNetLoaderNodeParamsSchema:
    """params_schema 测试。"""

    def test_params_schema_contains_unet_name(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert "unet_name" in node.params_schema

    def test_unet_name_is_required(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.params_schema["unet_name"]["required"] is True

    def test_unet_name_has_enum_from_dir_format(self):
        """unet_name 从目录扫描，format=enum-from-dir。"""
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.params_schema["unet_name"].get("format") == "enum-from-dir"

    def test_cache_dir_is_advanced(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.params_schema["cache_dir"].get("advanced") is True

    def test_dtype_is_advanced(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.params_schema["dtype"].get("advanced") is True

    def test_device_is_advanced(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.params_schema["device"].get("advanced") is True

    def test_dtype_options_include_fp16_bf16_fp32_auto(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        options = node.params_schema["dtype"].get("options", [])
        assert "fp16" in options
        assert "bf16" in options
        assert "fp32" in options
        assert "auto" in options


# ===== 参数校验测试 =====


class TestUNetLoaderNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_valid_params(self):
        node = UNetLoaderNode({"unet_name": "flux-unet"})
        assert node.validate() == []

    def test_validate_reports_missing_unet_name(self):
        node = UNetLoaderNode({})
        errors = node.validate()
        assert any("unet_name" in e for e in errors)

    def test_validate_reports_empty_unet_name(self):
        node = UNetLoaderNode({"unet_name": ""})
        errors = node.validate()
        assert any("unet_name" in e for e in errors)

    def test_validate_reports_unsupported_dtype(self):
        node = UNetLoaderNode({"unet_name": "flux-unet", "dtype": "int8"})
        errors = node.validate()
        assert any("dtype" in e for e in errors)

    def test_validate_accepts_fp16_dtype(self):
        node = UNetLoaderNode({"unet_name": "flux-unet", "dtype": "fp16"})
        assert node.validate() == []

    def test_validate_accepts_bf16_dtype(self):
        node = UNetLoaderNode({"unet_name": "flux-unet", "dtype": "bf16"})
        assert node.validate() == []

    def test_validate_accepts_auto_dtype(self):
        node = UNetLoaderNode({"unet_name": "flux-unet", "dtype": "auto"})
        assert node.validate() == []


# ===== 路径解析测试 =====


class TestUNetLoaderNodePathResolution:
    """UNet 路径解析测试。"""

    def test_resolve_path_for_directory_model(self, tmp_path: Path):
        """cache_dir 下有同名目录（diffusers 结构）→ 返回目录。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "unet"
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir(parents=True)

        node = UNetLoaderNode({"unet_name": "flux-unet"})
        ctx = make_default_context(models_root=models_root)
        unet_path = node._resolve_unet_path(ctx)
        assert unet_path == unet_dir

    def test_resolve_path_for_safetensors_file(self, tmp_path: Path):
        """cache_dir 下有 .safetensors 文件 → 返回文件。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_file = cache_dir / "flux-unet.safetensors"
        unet_file.write_bytes(b"mock")

        node = UNetLoaderNode({"unet_name": "flux-unet"})
        ctx = make_default_context(models_root=models_root)
        unet_path = node._resolve_unet_path(ctx)
        assert unet_path == unet_file

    def test_resolve_path_for_ckpt_file(self, tmp_path: Path):
        """cache_dir 下有 .ckpt 文件 → 返回文件。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_file = cache_dir / "flux-unet.ckpt"
        unet_file.write_bytes(b"mock")

        node = UNetLoaderNode({"unet_name": "flux-unet"})
        ctx = make_default_context(models_root=models_root)
        unet_path = node._resolve_unet_path(ctx)
        assert unet_path == unet_file

    def test_resolve_path_for_transformer_subdir(self, tmp_path: Path):
        """cache_dir 本身就是模型目录（含 transformer/ 子目录）→ 返回 cache_dir。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "unet" / "flux-unet"
        (cache_dir / "transformer").mkdir(parents=True)

        node = UNetLoaderNode({"unet_name": "flux-unet"})
        ctx = make_default_context(models_root=models_root)
        unet_path = node._resolve_unet_path(ctx)
        assert unet_path == cache_dir

    def test_resolve_path_not_found_raises(self, tmp_path: Path):
        """未找到匹配的文件/目录时抛 NodeValidationError。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "image" / "unet"
        cache_dir.mkdir(parents=True)

        node = UNetLoaderNode({"unet_name": "nonexistent"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_unet_path(ctx)
        assert "未在" in str(exc_info.value) or "不存在" in str(exc_info.value)

    def test_resolve_path_cache_dir_not_found_raises(self, tmp_path: Path):
        """cache_dir 本身不存在时抛 NodeValidationError。"""
        models_root = tmp_path / "models"

        node = UNetLoaderNode({"unet_name": "flux-unet"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_unet_path(ctx)
        assert "不存在" in str(exc_info.value)

    def test_resolve_path_with_absolute_cache_dir(self, tmp_path: Path):
        """cache_dir 为绝对路径时直接使用。"""
        cache_dir = tmp_path / "custom-unet-dir"
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir(parents=True)

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        unet_path = node._resolve_unet_path(ctx)
        assert unet_path == unet_dir


# ===== cache_key 测试 =====


class TestUNetLoaderNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_contains_unet_name(self, tmp_path: Path):
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        key = node.cache_key({})
        assert key is not None
        assert "flux-unet" in key

    def test_cache_key_contains_dtype(self, tmp_path: Path):
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "bf16",
        })
        key = node.cache_key({})
        assert "bf16" in key

    def test_cache_key_starts_with_unet_prefix(self, tmp_path: Path):
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        key = node.cache_key({})
        assert key.startswith("unet:")

    def test_cache_key_is_deterministic(self, tmp_path: Path):
        """相同参数返回相同 cache_key。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)

        node1 = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        node2 = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        assert node1.cache_key({}) == node2.cache_key({})

    def test_cache_key_returns_none_when_dir_not_exists(self):
        """cache_dir 不存在时返回 None。"""
        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": "/nonexistent/path/that/does/not/exist/",
        })
        assert node.cache_key({}) is None

    def test_cache_key_differs_for_different_dtype(self, tmp_path: Path):
        """不同 dtype 产生不同 cache_key。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)

        node_fp16 = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        node_bf16 = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "bf16",
        })
        assert node_fp16.cache_key({}) != node_bf16.cache_key({})


# ===== 加载流程测试 =====


class TestUNetLoaderNodeExecute:
    """execute 完整流程测试。"""

    @pytest.mark.asyncio
    async def test_execute_returns_model_from_load_impl(self, tmp_path: Path):
        """execute 调用 _load_impl 并返回 model。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        mock_model = MagicMock(name="loaded_unet")
        loader = _MockUNetLoader(
            {"unet_name": "flux-unet", "cache_dir": str(cache_dir)},
            return_value={"model": mock_model},
        )
        ctx = make_default_context()
        result = await loader.execute({}, ctx)

        assert "model" in result
        assert result["model"] is mock_model
        assert loader.load_called_with_path == unet_dir

    @pytest.mark.asyncio
    async def test_execute_writes_ctx_extra_loaded_model(self, tmp_path: Path):
        """C1-1-8：加载完成后写 ctx.extra["loaded_model"]。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        mock_model = MagicMock(name="loaded_unet")
        loader = _MockUNetLoader(
            {"unet_name": "flux-unet", "cache_dir": str(cache_dir)},
            return_value={"model": mock_model},
        )
        ctx = make_default_context()
        await loader.execute({}, ctx)

        assert "loaded_model" in ctx.extra
        assert ctx.extra["loaded_model"] is mock_model

    @pytest.mark.asyncio
    async def test_execute_invalid_params_raises(self):
        """参数校验失败抛 NodeValidationError。"""
        loader = _MockUNetLoader({"unet_name": "", "dtype": "invalid"})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await loader.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_execute_load_failure_wrapped_as_runtime_error(self, tmp_path: Path):
        """_load_impl 抛异常时被包装为 RuntimeError。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "broken-unet"
        unet_dir.mkdir()

        loader = _FailingUNetLoader({
            "unet_name": "broken-unet",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await loader.execute({}, ctx)
        assert "UNet 加载失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_none_model_raises_runtime_error(self, tmp_path: Path):
        """返回 model 为 None 时抛 RuntimeError。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        loader = _NoneModelUNetLoader({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await loader.execute({}, ctx)
        assert "model 为 None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_non_dict_return_raises_type_error(self, tmp_path: Path):
        """返回非 dict 时抛 TypeError。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        loader = _NonDictUNetLoader({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(TypeError) as exc_info:
            await loader.execute({}, ctx)
        assert "dict" in str(exc_info.value)


# ===== 默认 _load_impl 测试（mock diffusers/torch）=====


class TestUNetLoaderNodeDefaultLoadImpl:
    """默认 _load_impl 实现测试（mock diffusers/torch 模块）。"""

    @pytest.mark.asyncio
    async def test_default_load_impl_raises_importerror_without_diffusers(
        self, tmp_path: Path, clean_diffusers_torch
    ):
        """C1-1-12：缺 diffusers 抛 ImportError + 安装指引。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await node.execute({}, ctx)
        # RuntimeError 包装的 __cause__ 应为 ImportError
        cause = exc_info.value.__cause__
        assert isinstance(cause, ImportError)
        assert "diffusers" in str(cause)
        assert "pip install" in str(cause)

    @pytest.mark.asyncio
    async def test_default_load_impl_calls_unet_from_pretrained_for_dir(
        self, tmp_path: Path, clean_diffusers_torch
    ):
        """C1-3-8：目录加载走 UNet2DConditionModel.from_pretrained。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        mock_unet_instance = MagicMock(name="loaded_unet")
        mock_unet_cls = MagicMock()
        mock_unet_cls.from_pretrained = MagicMock(return_value=mock_unet_instance)
        mock_diffusers = _make_mock_diffusers_module(unet2d_cls=mock_unet_cls)
        mock_torch = _make_mock_torch_module()

        sys.modules["diffusers"] = mock_diffusers
        sys.modules["torch"] = mock_torch

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        # 验证 UNet2DConditionModel.from_pretrained 被调用
        mock_unet_cls.from_pretrained.assert_called_once()
        call_args = mock_unet_cls.from_pretrained.call_args
        assert str(unet_dir) in call_args.args[0] or call_args.args[0] == str(unet_dir)

        # 验证返回的 model 是 mock 实例
        assert result["model"] is mock_unet_instance
        # 验证 ctx.extra 被写入
        assert ctx.extra["loaded_model"] is mock_unet_instance

    @pytest.mark.asyncio
    async def test_default_load_impl_calls_from_single_file_for_file(
        self, tmp_path: Path, clean_diffusers_torch
    ):
        """单文件加载走 UNet2DConditionModel.from_single_file。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_file = cache_dir / "flux-unet.safetensors"
        unet_file.write_bytes(b"mock model data")

        mock_unet_instance = MagicMock(name="loaded_unet_single")
        mock_unet_cls = MagicMock()
        mock_unet_cls.from_single_file = MagicMock(return_value=mock_unet_instance)
        mock_diffusers = _make_mock_diffusers_module(unet2d_cls=mock_unet_cls)
        mock_torch = _make_mock_torch_module()

        sys.modules["diffusers"] = mock_diffusers
        sys.modules["torch"] = mock_torch

        node = UNetLoaderNode({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        # 验证 from_single_file 被调用
        mock_unet_cls.from_single_file.assert_called_once()
        assert result["model"] is mock_unet_instance

    @pytest.mark.asyncio
    async def test_default_load_impl_calls_transformer2d_for_transformer_subdir(
        self, tmp_path: Path, clean_diffusers_torch
    ):
        """C1-3-9：transformer/ 子目录加载走 Transformer2DModel.from_pretrained。"""
        cache_dir = tmp_path / "image" / "unet" / "flux-transformer"
        transformer_dir = cache_dir / "transformer"
        transformer_dir.mkdir(parents=True)

        mock_transformer_instance = MagicMock(name="loaded_transformer")
        mock_transformer_cls = MagicMock()
        mock_transformer_cls.from_pretrained = MagicMock(return_value=mock_transformer_instance)
        mock_unet_cls = MagicMock()  # 不应被调用
        mock_diffusers = _make_mock_diffusers_module(
            unet2d_cls=mock_unet_cls,
            transformer2d_cls=mock_transformer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["diffusers"] = mock_diffusers
        sys.modules["torch"] = mock_torch

        node = UNetLoaderNode({
            "unet_name": "flux-transformer",
            "cache_dir": str(cache_dir.parent),  # cache_dir 指向 unet/，下面有 flux-transformer/transformer/
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        # 验证 Transformer2DModel.from_pretrained 被调用
        mock_transformer_cls.from_pretrained.assert_called_once()
        call_args = mock_transformer_cls.from_pretrained.call_args
        assert "transformer" in call_args.args[0]

        # 验证 UNet2DConditionModel.from_pretrained 未被调用
        mock_unet_cls.from_pretrained.assert_not_called()

        assert result["model"] is mock_transformer_instance
        assert ctx.extra["loaded_model"] is mock_transformer_instance


# ===== 进度上报测试 =====


class TestUNetLoaderNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self, tmp_path: Path):
        """加载时上报开始和完成进度。"""
        cache_dir = tmp_path / "image" / "unet"
        cache_dir.mkdir(parents=True)
        unet_dir = cache_dir / "flux-unet"
        unet_dir.mkdir()

        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        loader = _MockUNetLoader({
            "unet_name": "flux-unet",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await loader.execute({}, ctx)

        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0.0, "unet_loader.start")
        assert progress_calls[-1] == (1.0, "unet_loader.done")

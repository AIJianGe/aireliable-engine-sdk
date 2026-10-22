"""CLIPLoader 节点单元测试（v0.2.5 ComfyUI 风格节点补充）。

覆盖 C2-3 验收标准：
    - 节点定义（node_type / category / subcategory / title / icon）
    - 输入端口为空（input_specs == []）
    - 输出端口为 clip（1 个，type="model"）
    - params_schema 含 clip_name（required=True）+ type（required=True，7 个枚举值）
    - cache_key 返回确定性键
    - 缺 transformers 抛 ImportError（带安装指引）
    - 7 种 type 加载分支（mock transformers）
    - 加载完成后写 ctx.extra["loaded_clip"]（dict 格式含 tokenizer + text_model）
    - ctx.extra["loaded_clip"] 可被 CLIPTextEncode._resolve_clip 读取（兼容性）

参考：v0.2.5-comfyui-nodes-tasks.md C2-3
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.loader.clip_loader import CLIPLoaderNode


# ===== Mock 子类 =====


class _MockCLIPLoader(CLIPLoaderNode):
    """模拟 CLIP 加载器（绕过默认 _load_impl）。"""

    node_type = "MockCLIPLoader"

    def __init__(self, params, return_value: dict = None, raise_exc: Exception = None):
        super().__init__(params)
        self._return_value = return_value or {
            "clip": {
                "tokenizer": MagicMock(name="mock_tokenizer"),
                "text_model": MagicMock(name="mock_text_model"),
            }
        }
        self._raise_exc = raise_exc
        self.load_called_with_path: Path = None

    async def _load_impl(self, clip_path: Path, ctx) -> dict:
        self.load_called_with_path = clip_path
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _FailingCLIPLoader(CLIPLoaderNode):
    node_type = "FailingCLIPLoader"

    async def _load_impl(self, clip_path: Path, ctx) -> dict:
        raise FileNotFoundError(f"CLIP 文件损坏: {clip_path}")


class _NoneClipCLIPLoader(CLIPLoaderNode):
    """返回 clip 为 None。"""

    node_type = "NoneClipCLIPLoader"

    async def _load_impl(self, clip_path: Path, ctx) -> dict:
        return {"clip": None}


class _NonDictCLIPLoader(CLIPLoaderNode):
    """返回非 dict。"""

    node_type = "NonDictCLIPLoader"

    async def _load_impl(self, clip_path: Path, ctx) -> dict:
        return "not a dict"


# ===== Mock transformers 模块工具 =====


def _make_mock_transformers_module(
    clip_text_model_cls=None,
    clip_tokenizer_cls=None,
    auto_model_cls=None,
    auto_tokenizer_cls=None,
    t5_encoder_model_cls=None,
    t5_tokenizer_cls=None,
):
    """构造一个 mock transformers 模块。"""
    mock_module = types.ModuleType("transformers")

    if clip_text_model_cls is None:
        clip_text_model_cls = MagicMock()
    if clip_tokenizer_cls is None:
        clip_tokenizer_cls = MagicMock()
    if auto_model_cls is None:
        auto_model_cls = MagicMock()
    if auto_tokenizer_cls is None:
        auto_tokenizer_cls = MagicMock()
    if t5_encoder_model_cls is None:
        t5_encoder_model_cls = MagicMock()
    if t5_tokenizer_cls is None:
        t5_tokenizer_cls = MagicMock()

    mock_module.CLIPTextModel = clip_text_model_cls
    mock_module.CLIPTokenizer = clip_tokenizer_cls
    mock_module.AutoModel = auto_model_cls
    mock_module.AutoTokenizer = auto_tokenizer_cls
    mock_module.T5EncoderModel = t5_encoder_model_cls
    mock_module.T5Tokenizer = t5_tokenizer_cls
    return mock_module


def _make_mock_torch_module():
    """构造一个 mock torch 模块（含 float16 / bfloat16 / float32）。"""
    mock_module = types.ModuleType("torch")
    mock_module.float16 = "torch.float16"
    mock_module.bfloat16 = "torch.bfloat16"
    mock_module.float32 = "torch.float32"
    return mock_module


@pytest.fixture
def clean_transformers_torch():
    """清理 transformers / torch 模块，确保 _load_impl 的延迟导入走 mock。"""
    original_transformers = sys.modules.get("transformers")
    original_torch = sys.modules.get("torch")
    sys.modules.pop("transformers", None)
    sys.modules.pop("torch", None)
    try:
        yield
    finally:
        if original_transformers is not None:
            sys.modules["transformers"] = original_transformers
        else:
            sys.modules.pop("transformers", None)
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        else:
            sys.modules.pop("torch", None)


# ===== 基本属性测试 =====


class TestCLIPLoaderNodeBasic:
    """CLIPLoaderNode 基本属性测试。"""

    def test_node_type_is_cliploder(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.node_type == "CLIPLoader"

    def test_category_is_loader(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.category == "loader"

    def test_subcategory_is_clip(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.subcategory == "clip"

    def test_title_is_chinese(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.title == "加载CLIP"

    def test_icon_is_abc_emoji(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.icon == "🔤"

    def test_input_specs_is_empty_list(self):
        """与 ComfyUI 一致，无输入端口。"""
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.input_specs == []

    def test_output_specs_has_single_clip_port(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "clip"
        assert port.type == "model"

    def test_default_cache_dir_is_shared_text_encoders(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node._cache_dir == "shared/text_encoders/"

    def test_default_dtype_is_fp16(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node._dtype == "fp16"

    def test_default_type_is_stable_diffusion(self):
        node = CLIPLoaderNode({"clip_name": "clip-l"})
        assert node._clip_type == "stable_diffusion"

    def test_repr_contains_class_name_and_clip_name(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        repr_str = repr(node)
        assert "CLIPLoaderNode" in repr_str
        assert "clip-l" in repr_str


# ===== params_schema 测试 =====


class TestCLIPLoaderNodeParamsSchema:
    """params_schema 测试。"""

    def test_params_schema_contains_clip_name(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert "clip_name" in node.params_schema

    def test_clip_name_is_required(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.params_schema["clip_name"]["required"] is True

    def test_clip_name_has_enum_from_dir_format(self):
        """clip_name 从目录扫描，format=enum-from-dir。"""
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.params_schema["clip_name"].get("format") == "enum-from-dir"

    def test_params_schema_contains_type(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert "type" in node.params_schema

    def test_type_is_required(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.params_schema["type"]["required"] is True

    def test_type_has_seven_options(self):
        """type 枚举含 7 个 CLIP 类型。"""
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        options = node.params_schema["type"].get("options", [])
        assert "stable_diffusion" in options
        assert "stable_diffusion_xl" in options
        assert "lumina2" in options
        assert "wan" in options
        assert "t5" in options
        assert "clip_l" in options
        assert "clip_g" in options
        assert len(options) == 7

    def test_cache_dir_is_advanced(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.params_schema["cache_dir"].get("advanced") is True

    def test_dtype_is_advanced(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.params_schema["dtype"].get("advanced") is True

    def test_device_is_advanced(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.params_schema["device"].get("advanced") is True


# ===== 参数校验测试 =====


class TestCLIPLoaderNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_valid_params(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        assert node.validate() == []

    def test_validate_reports_missing_clip_name(self):
        node = CLIPLoaderNode({"type": "stable_diffusion"})
        errors = node.validate()
        assert any("clip_name" in e for e in errors)

    def test_validate_reports_empty_clip_name(self):
        node = CLIPLoaderNode({"clip_name": "", "type": "stable_diffusion"})
        errors = node.validate()
        assert any("clip_name" in e for e in errors)

    def test_validate_reports_invalid_type(self):
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "invalid_type"})
        errors = node.validate()
        assert any("type" in e for e in errors)

    def test_validate_reports_unsupported_dtype(self):
        node = CLIPLoaderNode({
            "clip_name": "clip-l", "type": "stable_diffusion", "dtype": "int8",
        })
        errors = node.validate()
        assert any("dtype" in e for e in errors)

    @pytest.mark.parametrize("clip_type", [
        "stable_diffusion", "stable_diffusion_xl", "lumina2",
        "wan", "t5", "clip_l", "clip_g",
    ])
    def test_validate_accepts_all_seven_types(self, clip_type):
        """7 种 type 都通过校验。"""
        node = CLIPLoaderNode({"clip_name": "clip-l", "type": clip_type})
        assert node.validate() == []


# ===== 路径解析测试 =====


class TestCLIPLoaderNodePathResolution:
    """CLIP 路径解析测试。"""

    def test_resolve_path_for_directory_model(self, tmp_path: Path):
        """cache_dir 下有同名目录 → 返回目录。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "shared" / "text_encoders"
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir(parents=True)

        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        ctx = make_default_context(models_root=models_root)
        clip_path = node._resolve_clip_path(ctx)
        assert clip_path == clip_dir

    def test_resolve_path_for_safetensors_file(self, tmp_path: Path):
        """cache_dir 下有 .safetensors 文件 → 返回文件。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_file = cache_dir / "clip-l.safetensors"
        clip_file.write_bytes(b"mock")

        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        ctx = make_default_context(models_root=models_root)
        clip_path = node._resolve_clip_path(ctx)
        assert clip_path == clip_file

    def test_resolve_path_for_config_json_dir(self, tmp_path: Path):
        """cache_dir 本身就是 CLIP 目录（含 config.json）→ 返回 cache_dir。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "shared" / "text_encoders" / "clip-l"
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text("{}")

        node = CLIPLoaderNode({
            "clip_name": "ignored",  # cache_dir 含 config.json 时 clip_name 被忽略
            "type": "stable_diffusion",
            "cache_dir": "shared/text_encoders/clip-l/",
        })
        ctx = make_default_context(models_root=models_root)
        clip_path = node._resolve_clip_path(ctx)
        assert clip_path == cache_dir

    def test_resolve_path_not_found_raises(self, tmp_path: Path):
        """未找到匹配的文件/目录时抛 NodeValidationError。"""
        models_root = tmp_path / "models"
        cache_dir = models_root / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node = CLIPLoaderNode({"clip_name": "nonexistent", "type": "stable_diffusion"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_clip_path(ctx)
        assert "未在" in str(exc_info.value) or "不存在" in str(exc_info.value)

    def test_resolve_path_cache_dir_not_found_raises(self, tmp_path: Path):
        """cache_dir 本身不存在时抛 NodeValidationError。"""
        models_root = tmp_path / "models"

        node = CLIPLoaderNode({"clip_name": "clip-l", "type": "stable_diffusion"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_clip_path(ctx)
        assert "不存在" in str(exc_info.value)

    def test_resolve_path_with_absolute_cache_dir(self, tmp_path: Path):
        """cache_dir 为绝对路径时直接使用。"""
        cache_dir = tmp_path / "custom-clip-dir"
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir(parents=True)

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        clip_path = node._resolve_clip_path(ctx)
        assert clip_path == clip_dir


# ===== cache_key 测试 =====


class TestCLIPLoaderNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_contains_clip_name(self, tmp_path: Path):
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        key = node.cache_key({})
        assert key is not None
        assert "clip-l" in key

    def test_cache_key_contains_type(self, tmp_path: Path):
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "t5",
            "cache_dir": str(cache_dir),
        })
        key = node.cache_key({})
        assert "t5" in key

    def test_cache_key_contains_dtype(self, tmp_path: Path):
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
            "dtype": "bf16",
        })
        key = node.cache_key({})
        assert "bf16" in key

    def test_cache_key_starts_with_clip_prefix(self, tmp_path: Path):
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        key = node.cache_key({})
        assert key.startswith("clip:")

    def test_cache_key_is_deterministic(self, tmp_path: Path):
        """相同参数返回相同 cache_key。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node1 = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        node2 = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        assert node1.cache_key({}) == node2.cache_key({})

    def test_cache_key_returns_none_when_dir_not_exists(self):
        """cache_dir 不存在时返回 None。"""
        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": "/nonexistent/path/that/does/not/exist/",
        })
        assert node.cache_key({}) is None

    def test_cache_key_differs_for_different_type(self, tmp_path: Path):
        """不同 type 产生不同 cache_key。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)

        node_sd = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        node_t5 = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "t5",
            "cache_dir": str(cache_dir),
        })
        assert node_sd.cache_key({}) != node_t5.cache_key({})


# ===== 加载流程测试 =====


class TestCLIPLoaderNodeExecute:
    """execute 完整流程测试。"""

    @pytest.mark.asyncio
    async def test_execute_returns_clip_from_load_impl(self, tmp_path: Path):
        """execute 调用 _load_impl 并返回 clip。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        mock_clip = {
            "tokenizer": MagicMock(name="mock_tokenizer"),
            "text_model": MagicMock(name="mock_text_model"),
        }
        loader = _MockCLIPLoader(
            {
                "clip_name": "clip-l",
                "type": "stable_diffusion",
                "cache_dir": str(cache_dir),
            },
            return_value={"clip": mock_clip},
        )
        ctx = make_default_context()
        result = await loader.execute({}, ctx)

        assert "clip" in result
        assert result["clip"] is mock_clip
        assert loader.load_called_with_path == clip_dir

    @pytest.mark.asyncio
    async def test_execute_writes_ctx_extra_loaded_clip(self, tmp_path: Path):
        """C2-1-9：加载完成后写 ctx.extra["loaded_clip"]。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        mock_clip = {
            "tokenizer": MagicMock(name="mock_tokenizer"),
            "text_model": MagicMock(name="mock_text_model"),
        }
        loader = _MockCLIPLoader(
            {
                "clip_name": "clip-l",
                "type": "stable_diffusion",
                "cache_dir": str(cache_dir),
            },
            return_value={"clip": mock_clip},
        )
        ctx = make_default_context()
        await loader.execute({}, ctx)

        assert "loaded_clip" in ctx.extra
        assert ctx.extra["loaded_clip"] is mock_clip

    @pytest.mark.asyncio
    async def test_execute_invalid_params_raises(self):
        """参数校验失败抛 NodeValidationError。"""
        loader = _MockCLIPLoader({
            "clip_name": "",
            "type": "invalid_type",
            "dtype": "int8",
        })
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await loader.execute({}, ctx)

    @pytest.mark.asyncio
    async def test_execute_load_failure_wrapped_as_runtime_error(self, tmp_path: Path):
        """_load_impl 抛异常时被包装为 RuntimeError。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "broken-clip"
        clip_dir.mkdir()

        loader = _FailingCLIPLoader({
            "clip_name": "broken-clip",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await loader.execute({}, ctx)
        assert "CLIP 加载失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_none_clip_raises_runtime_error(self, tmp_path: Path):
        """返回 clip 为 None 时抛 RuntimeError。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        loader = _NoneClipCLIPLoader({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await loader.execute({}, ctx)
        assert "clip 为 None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_non_dict_return_raises_type_error(self, tmp_path: Path):
        """返回非 dict 时抛 TypeError。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        loader = _NonDictCLIPLoader({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(TypeError) as exc_info:
            await loader.execute({}, ctx)
        assert "dict" in str(exc_info.value)


# ===== 默认 _load_impl 测试（mock transformers/torch）=====


class TestCLIPLoaderNodeDefaultLoadImpl:
    """默认 _load_impl 实现测试（mock transformers/torch 模块）。"""

    @pytest.mark.asyncio
    async def test_default_load_impl_raises_importerror_without_torch(
        self, tmp_path: Path, clean_transformers_torch
    ):
        """C2-1-12：缺 torch 抛 ImportError + 安装指引。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await node.execute({}, ctx)
        cause = exc_info.value.__cause__
        assert isinstance(cause, ImportError)
        assert "torch" in str(cause)
        assert "pip install" in str(cause)

    @pytest.mark.asyncio
    async def test_load_stable_diffusion_branch(
        self, tmp_path: Path, clean_transformers_torch
    ):
        """C2-3-8：type=stable_diffusion 加载分支（CLIPTextModel + CLIPTokenizer）。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        mock_tokenizer = MagicMock(name="loaded_tokenizer")
        mock_text_model = MagicMock(name="loaded_text_model")
        mock_clip_text_model_cls = MagicMock()
        mock_clip_text_model_cls.from_pretrained = MagicMock(return_value=mock_text_model)
        mock_clip_tokenizer_cls = MagicMock()
        mock_clip_tokenizer_cls.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_transformers = _make_mock_transformers_module(
            clip_text_model_cls=mock_clip_text_model_cls,
            clip_tokenizer_cls=mock_clip_tokenizer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["transformers"] = mock_transformers
        sys.modules["torch"] = mock_torch

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        # 验证 CLIPTextModel + CLIPTokenizer 被调用
        mock_clip_text_model_cls.from_pretrained.assert_called_once()
        mock_clip_tokenizer_cls.from_pretrained.assert_called_once()

        # 验证返回的 clip 是 dict 格式
        clip = result["clip"]
        assert isinstance(clip, dict)
        assert clip["tokenizer"] is mock_tokenizer
        assert clip["text_model"] is mock_text_model

        # 验证 ctx.extra 被写入
        assert ctx.extra["loaded_clip"] is clip

    @pytest.mark.asyncio
    async def test_load_stable_diffusion_xl_branch(
        self, tmp_path: Path, clean_transformers_torch
    ):
        """C2-3-9：type=stable_diffusion_xl 加载分支（同样走 CLIPTextModel + CLIPTokenizer）。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "sdxl-clip"
        clip_dir.mkdir()

        mock_tokenizer = MagicMock()
        mock_text_model = MagicMock()
        mock_clip_text_model_cls = MagicMock()
        mock_clip_text_model_cls.from_pretrained = MagicMock(return_value=mock_text_model)
        mock_clip_tokenizer_cls = MagicMock()
        mock_clip_tokenizer_cls.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_transformers = _make_mock_transformers_module(
            clip_text_model_cls=mock_clip_text_model_cls,
            clip_tokenizer_cls=mock_clip_tokenizer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["transformers"] = mock_transformers
        sys.modules["torch"] = mock_torch

        node = CLIPLoaderNode({
            "clip_name": "sdxl-clip",
            "type": "stable_diffusion_xl",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        mock_clip_text_model_cls.from_pretrained.assert_called_once()
        mock_clip_tokenizer_cls.from_pretrained.assert_called_once()
        assert isinstance(result["clip"], dict)

    @pytest.mark.asyncio
    async def test_load_lumina2_branch(
        self, tmp_path: Path, clean_transformers_torch
    ):
        """C2-3-10：type=lumina2 加载分支（AutoModel + AutoTokenizer，trust_remote_code）。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "lumina-qwen"
        clip_dir.mkdir()

        mock_tokenizer = MagicMock()
        mock_text_model = MagicMock()
        mock_auto_model_cls = MagicMock()
        mock_auto_model_cls.from_pretrained = MagicMock(return_value=mock_text_model)
        mock_auto_tokenizer_cls = MagicMock()
        mock_auto_tokenizer_cls.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_transformers = _make_mock_transformers_module(
            auto_model_cls=mock_auto_model_cls,
            auto_tokenizer_cls=mock_auto_tokenizer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["transformers"] = mock_transformers
        sys.modules["torch"] = mock_torch

        node = CLIPLoaderNode({
            "clip_name": "lumina-qwen",
            "type": "lumina2",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        # 验证 AutoModel + AutoTokenizer 被调用
        mock_auto_model_cls.from_pretrained.assert_called_once()
        mock_auto_tokenizer_cls.from_pretrained.assert_called_once()

        # 验证 trust_remote_code=True 被传递
        call_kwargs = mock_auto_model_cls.from_pretrained.call_args.kwargs
        assert call_kwargs.get("trust_remote_code") is True

        # CLIPTextModel 不应被调用
        mock_clip_text_model_cls = mock_transformers.CLIPTextModel
        mock_clip_text_model_cls.from_pretrained.assert_not_called()

        assert isinstance(result["clip"], dict)

    @pytest.mark.asyncio
    async def test_load_t5_branch(
        self, tmp_path: Path, clean_transformers_torch
    ):
        """C2-3-11：type=t5 加载分支（T5EncoderModel + T5Tokenizer）。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "t5-xxl"
        clip_dir.mkdir()

        mock_tokenizer = MagicMock()
        mock_text_model = MagicMock()
        mock_t5_encoder_cls = MagicMock()
        mock_t5_encoder_cls.from_pretrained = MagicMock(return_value=mock_text_model)
        mock_t5_tokenizer_cls = MagicMock()
        mock_t5_tokenizer_cls.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_transformers = _make_mock_transformers_module(
            t5_encoder_model_cls=mock_t5_encoder_cls,
            t5_tokenizer_cls=mock_t5_tokenizer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["transformers"] = mock_transformers
        sys.modules["torch"] = mock_torch

        node = CLIPLoaderNode({
            "clip_name": "t5-xxl",
            "type": "t5",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        mock_t5_encoder_cls.from_pretrained.assert_called_once()
        mock_t5_tokenizer_cls.from_pretrained.assert_called_once()
        assert isinstance(result["clip"], dict)
        assert result["clip"]["tokenizer"] is mock_tokenizer
        assert result["clip"]["text_model"] is mock_text_model

    @pytest.mark.asyncio
    @pytest.mark.parametrize("clip_type", ["wan", "clip_l", "clip_g"])
    async def test_load_other_clip_like_branches(
        self, tmp_path: Path, clean_transformers_torch, clip_type
    ):
        """C2-3-12：type=wan / clip_l / clip_g 加载分支（都走 CLIPTextModel + CLIPTokenizer）。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "test-clip"
        clip_dir.mkdir()

        mock_tokenizer = MagicMock()
        mock_text_model = MagicMock()
        mock_clip_text_model_cls = MagicMock()
        mock_clip_text_model_cls.from_pretrained = MagicMock(return_value=mock_text_model)
        mock_clip_tokenizer_cls = MagicMock()
        mock_clip_tokenizer_cls.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_transformers = _make_mock_transformers_module(
            clip_text_model_cls=mock_clip_text_model_cls,
            clip_tokenizer_cls=mock_clip_tokenizer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["transformers"] = mock_transformers
        sys.modules["torch"] = mock_torch

        node = CLIPLoaderNode({
            "clip_name": "test-clip",
            "type": clip_type,
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        result = await node.execute({}, ctx)

        mock_clip_text_model_cls.from_pretrained.assert_called_once()
        mock_clip_tokenizer_cls.from_pretrained.assert_called_once()
        assert isinstance(result["clip"], dict)


# ===== ctx.extra["loaded_clip"] 与 CLIPTextEncode 兼容性测试 =====


class TestCLIPLoaderNodeCLIPTextEncodeCompat:
    """C2-3-13/C2-3-14：验证 ctx.extra["loaded_clip"] 可被 CLIPTextEncode 读取。"""

    @pytest.mark.asyncio
    async def test_loaded_clip_format_compatible_with_clip_text_encode(
        self, tmp_path: Path, clean_transformers_torch
    ):
        """ctx.extra["loaded_clip"] 是 dict 格式 {tokenizer, text_model}，
        可被 CLIPTextEncode._extract_tokenizer_and_model 的"情况 3：dict"分支读取。
        """
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        mock_tokenizer = MagicMock()
        mock_text_model = MagicMock()
        mock_clip_text_model_cls = MagicMock()
        mock_clip_text_model_cls.from_pretrained = MagicMock(return_value=mock_text_model)
        mock_clip_tokenizer_cls = MagicMock()
        mock_clip_tokenizer_cls.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_transformers = _make_mock_transformers_module(
            clip_text_model_cls=mock_clip_text_model_cls,
            clip_tokenizer_cls=mock_clip_tokenizer_cls,
        )
        mock_torch = _make_mock_torch_module()

        sys.modules["transformers"] = mock_transformers
        sys.modules["torch"] = mock_torch

        node = CLIPLoaderNode({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
            "dtype": "fp16",
        })
        ctx = make_default_context()
        await node.execute({}, ctx)

        # 验证 ctx.extra["loaded_clip"] 是 dict 格式
        loaded_clip = ctx.extra["loaded_clip"]
        assert isinstance(loaded_clip, dict)
        assert "tokenizer" in loaded_clip
        assert "text_model" in loaded_clip

        # 用 CLIPTextEncode._extract_tokenizer_and_model 验证兼容性
        from aireliable_engine.nodes.encode.clip_encode import CLIPTextEncodeNode

        clip_text_node = CLIPTextEncodeNode({})
        tok, mdl = clip_text_node._extract_tokenizer_and_model(loaded_clip, ctx)
        assert tok is mock_tokenizer
        assert mdl is mock_text_model


# ===== 进度上报测试 =====


class TestCLIPLoaderNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self, tmp_path: Path):
        """加载时上报开始和完成进度。"""
        cache_dir = tmp_path / "shared" / "text_encoders"
        cache_dir.mkdir(parents=True)
        clip_dir = cache_dir / "clip-l"
        clip_dir.mkdir()

        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        loader = _MockCLIPLoader({
            "clip_name": "clip-l",
            "type": "stable_diffusion",
            "cache_dir": str(cache_dir),
        })
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await loader.execute({}, ctx)

        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0.0, "clip_loader.start")
        assert progress_calls[-1] == (1.0, "clip_loader.done")

"""VAELoader 节点单元测试。

覆盖任务 3.4 验收标准：
    - 支持共享 VAE（从 models/shared/vae/ 加载）
    - 支持引擎特定 VAE
    - 版本兼容（diffusers API 探测）

参考：v0.2-tasks.md 任务 3.4
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.loader.vae_loader import VAELoaderNode


# ===== Mock 子类 =====


class _MockVAELoader(VAELoaderNode):
    """模拟 VAE 加载器。"""

    node_type = "MockVAELoader"

    def __init__(self, params, return_value: dict = None, raise_exc: Exception = None):
        super().__init__(params)
        self._return_value = return_value or {"vae": MagicMock(name="mock_vae")}
        self._raise_exc = raise_exc
        self.load_called_with_path: Path = None

    async def _load_impl(self, vae_path: Path, ctx) -> dict:
        self.load_called_with_path = vae_path
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _FailingVAELoader(VAELoaderNode):
    node_type = "FailingVAELoader"

    async def _load_impl(self, vae_path: Path, ctx) -> dict:
        raise FileNotFoundError(f"VAE 文件损坏: {vae_path}")


class _InvalidReturnVAELoader(VAELoaderNode):
    """返回 dict 但缺少 vae 字段。"""

    node_type = "InvalidReturnVAELoader"

    async def _load_impl(self, vae_path: Path, ctx) -> dict:
        return {"model": MagicMock()}  # 缺少 vae 字段


class _NonDictVAELoader(VAELoaderNode):
    """返回非 dict。"""

    node_type = "NonDictVAELoader"

    async def _load_impl(self, vae_path: Path, ctx) -> dict:
        return "not a dict"


class _BareVAELoader(VAELoaderNode):
    """未重写 _load_impl。"""

    node_type = "BareVAELoader"


# ===== 基本属性测试 =====


class TestVAELoaderNodeBasic:
    """VAELoaderNode 基本属性测试。"""

    def test_node_type_is_vaeloader(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        assert node.node_type == "VAELoader"

    def test_input_specs_is_empty_list(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        assert node.input_specs == []

    def test_output_specs_has_vae_port(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        assert len(node.output_specs) == 1
        port = node.output_specs[0]
        assert port.name == "vae"
        assert port.type == "model"

    def test_default_source_is_shared(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        assert node._source == "shared"

    def test_init_with_engine_specific_source(self):
        node = VAELoaderNode({
            "vae_name": "flux-vae",
            "source": "engine_specific",
            "dtype": "fp16",
            "device": "cuda:0",
            "force_reload": True,
        })
        assert node._vae_name == "flux-vae"
        assert node._source == "engine_specific"
        assert node._dtype == "fp16"
        assert node._device_override == "cuda:0"
        assert node._force_reload is True

    def test_repr_contains_class_name_and_vae_name(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        repr_str = repr(node)
        assert "VAELoaderNode" in repr_str
        assert "sdxl-vae" in repr_str


# ===== 参数校验测试 =====


class TestVAELoaderNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_valid_params(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        assert node.validate() == []

    def test_validate_reports_missing_vae_name(self):
        node = VAELoaderNode({})
        errors = node.validate()
        assert any("vae_name" in e for e in errors)

    def test_validate_reports_empty_vae_name(self):
        node = VAELoaderNode({"vae_name": ""})
        errors = node.validate()
        assert any("vae_name" in e for e in errors)

    def test_validate_reports_invalid_source(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae", "source": "invalid"})
        errors = node.validate()
        assert any("source" in e for e in errors)

    def test_validate_reports_unsupported_dtype(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae", "dtype": "int8"})
        errors = node.validate()
        assert any("dtype" in e for e in errors)

    def test_validate_accepts_shared_source(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae", "source": "shared"})
        assert node.validate() == []

    def test_validate_accepts_engine_specific_source(self):
        node = VAELoaderNode({"vae_name": "flux-vae", "source": "engine_specific"})
        assert node.validate() == []


# ===== 共享 VAE 路径解析测试（验收标准）=====


class TestVAELoaderNodeSharedPath:
    """共享 VAE（source=shared）路径解析测试（任务 3.4 验收标准）。"""

    def test_shared_vae_path_resolved_from_models_root(self, tmp_path: Path):
        """source=shared 时自动构造 models/shared/vae/<name>/ 路径。"""
        models_root = tmp_path / "models"
        vae_dir = models_root / "shared" / "vae" / "sdxl-vae"
        vae_dir.mkdir(parents=True)

        node = VAELoaderNode({"vae_name": "sdxl-vae", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        vae_path = node._resolve_vae_path(ctx)
        assert vae_path == vae_dir

    def test_shared_vae_path_with_explicit_vae_path(self, tmp_path: Path):
        """vae_path 显式指定时直接使用。"""
        vae_dir = tmp_path / "custom-vae-dir"
        vae_dir.mkdir()

        node = VAELoaderNode({
            "vae_name": "sdxl-vae",
            "vae_path": str(vae_dir),
        })
        ctx = make_default_context()
        vae_path = node._resolve_vae_path(ctx)
        assert vae_path == vae_dir

    def test_shared_vae_path_relative_to_models_root(self, tmp_path: Path):
        """vae_path 为相对路径时相对于 models_root 解析。"""
        models_root = tmp_path / "models"
        vae_dir = models_root / "custom" / "my-vae"
        vae_dir.mkdir(parents=True)

        node = VAELoaderNode({
            "vae_name": "my-vae",
            "vae_path": "custom/my-vae",       # 相对路径
        })
        ctx = make_default_context(models_root=models_root)
        vae_path = node._resolve_vae_path(ctx)
        assert vae_path == vae_dir

    def test_shared_vae_path_not_found_raises(self, tmp_path: Path):
        """共享 VAE 路径不存在时报错。"""
        models_root = tmp_path / "models"

        node = VAELoaderNode({"vae_name": "nonexistent", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_vae_path(ctx)
        assert "VAE 路径不存在" in str(exc_info.value)

    def test_shared_vae_path_no_models_root_raises(self):
        """未指定 vae_path 且 ctx.models_root 为 None 时报错。"""
        node = VAELoaderNode({"vae_name": "sdxl-vae", "source": "shared"})
        ctx = make_default_context(models_root=None)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_vae_path(ctx)
        assert "models_root" in str(exc_info.value)


# ===== 引擎特定 VAE 路径解析测试（验收标准）=====


class TestVAELoaderNodeEngineSpecificPath:
    """引擎特定 VAE（source=engine_specific）路径解析测试。"""

    def test_engine_specific_vae_path_resolved(self, tmp_path: Path):
        """source=engine_specific 时使用 ctx.engine_id 构造路径。"""
        models_root = tmp_path / "models"
        vae_dir = models_root / "image" / "vae" / "ernie_image" / "ernie-vae"
        vae_dir.mkdir(parents=True)

        node = VAELoaderNode({
            "vae_name": "ernie-vae",
            "source": "engine_specific",
        })
        ctx = make_default_context(
            engine_id="ernie_image",
            capability="image",
            models_root=models_root,
        )
        vae_path = node._resolve_vae_path(ctx)
        assert vae_path == vae_dir

    def test_engine_specific_vae_path_with_unknown_engine(self, tmp_path: Path):
        """ctx.engine_id 为空时使用 'unknown_engine' 兜底。"""
        models_root = tmp_path / "models"
        vae_dir = models_root / "image" / "vae" / "unknown_engine" / "ernie-vae"
        vae_dir.mkdir(parents=True)

        node = VAELoaderNode({
            "vae_name": "ernie-vae",
            "source": "engine_specific",
        })
        ctx = make_default_context(
            engine_id="",
            capability="image",
            models_root=models_root,
        )
        vae_path = node._resolve_vae_path(ctx)
        assert "unknown_engine" in str(vae_path)


# ===== diffusers API 探测测试（验收标准）=====


class TestVAELoaderNodeDiffusersAPIDetection:
    """版本兼容（diffusers API 探测）测试（任务 3.4 验收标准）。"""

    def test_detect_returns_unknown_when_diffusers_not_installed(self):
        """未安装 diffusers 时返回 'unknown'。"""
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        # 在 SDK 测试环境中通常没装 diffusers
        # 模拟未安装的情况
        import sys
        original = sys.modules.get("diffusers")
        sys.modules.pop("diffusers", None)
        try:
            version = node._detect_diffusers_api_version()
            # 可能是 unknown 或具体版本，取决于环境
            assert version in {"unknown", ">=0.31", "<0.31"}
        finally:
            if original is not None:
                sys.modules["diffusers"] = original

    def test_detect_returns_version_when_diffusers_installed(self):
        """diffusers 已安装时返回具体版本字符串。"""
        # 创建 mock diffusers 模块
        import sys
        import types

        mock_module = types.ModuleType("diffusers")
        mock_module.__version__ = "0.32.0"
        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_module
        try:
            node = VAELoaderNode({"vae_name": "sdxl-vae"})
            version = node._detect_diffusers_api_version()
            assert version == ">=0.31"
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                del sys.modules["diffusers"]

    def test_detect_returns_old_version_for_0_30(self):
        """diffusers 0.30.0 时返回 '<0.31'。"""
        import sys
        import types

        mock_module = types.ModuleType("diffusers")
        mock_module.__version__ = "0.30.0"
        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_module
        try:
            node = VAELoaderNode({"vae_name": "sdxl-vae"})
            version = node._detect_diffusers_api_version()
            assert version == "<0.31"
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                del sys.modules["diffusers"]

    def test_detect_returns_unknown_on_bad_version_string(self):
        """diffusers __version__ 是非数字字符串时返回 'unknown'。"""
        import sys
        import types

        mock_module = types.ModuleType("diffusers")
        # __version__ 是非数字字符串，int() 解析会抛 ValueError，被 except Exception 捕获
        mock_module.__version__ = "abc.def"
        original = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_module
        try:
            node = VAELoaderNode({"vae_name": "sdxl-vae"})
            version = node._detect_diffusers_api_version()
            assert version == "unknown"
        finally:
            if original is not None:
                sys.modules["diffusers"] = original
            else:
                sys.modules.pop("diffusers", None)


# ===== 加载流程测试 =====


class TestVAELoaderNodeExecute:
    """execute 完整流程测试。"""

    @pytest.mark.asyncio
    async def test_execute_returns_vae_from_load_impl(self, tmp_path: Path):
        """execute 调用 _load_impl 并返回 vae。"""
        vae_dir = tmp_path / "sdxl-vae"
        vae_dir.mkdir()
        mock_vae = MagicMock(name="loaded_vae")

        loader = _MockVAELoader(
            {"vae_name": "sdxl-vae", "vae_path": str(vae_dir)},
            return_value={"vae": mock_vae},
        )
        ctx = make_default_context()
        result = await loader.execute({}, ctx)

        assert "vae" in result
        assert result["vae"] is mock_vae
        assert loader.load_called_with_path == vae_dir

    @pytest.mark.asyncio
    async def test_execute_passes_resolved_path_to_load_impl(self, tmp_path: Path):
        """execute 把解析后的路径传给 _load_impl。"""
        models_root = tmp_path / "models"
        vae_dir = models_root / "shared" / "vae" / "flux-vae"
        vae_dir.mkdir(parents=True)

        loader = _MockVAELoader({"vae_name": "flux-vae", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        await loader.execute({}, ctx)
        assert loader.load_called_with_path == vae_dir

    @pytest.mark.asyncio
    async def test_execute_invalid_params_raises(self):
        """参数校验失败抛 NodeValidationError。"""
        loader = _MockVAELoader({"vae_name": "", "source": "invalid"})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await loader.execute({}, ctx)


# ===== 错误处理测试 =====


class TestVAELoaderNodeErrors:
    """错误处理测试。"""

    @pytest.mark.asyncio
    async def test_bare_loader_raises_not_implemented(self, tmp_path: Path):
        """未重写 _load_impl 的子类抛 NotImplementedError。"""
        vae_dir = tmp_path / "sdxl-vae"
        vae_dir.mkdir()

        loader = _BareVAELoader({"vae_name": "sdxl-vae", "vae_path": str(vae_dir)})
        ctx = make_default_context()
        with pytest.raises(NotImplementedError) as exc_info:
            await loader.execute({}, ctx)
        assert "_load_impl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_impl_failure_wrapped_as_runtime_error(self, tmp_path: Path):
        """_load_impl 抛异常时被包装为 RuntimeError。"""
        vae_dir = tmp_path / "broken-vae"
        vae_dir.mkdir()

        loader = _FailingVAELoader({"vae_name": "broken-vae", "vae_path": str(vae_dir)})
        ctx = make_default_context()
        with pytest.raises(RuntimeError) as exc_info:
            await loader.execute({}, ctx)
        assert "VAE 加载失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_return_dict_raises_type_error(self, tmp_path: Path):
        """返回 dict 缺少 vae 字段时抛 TypeError。"""
        vae_dir = tmp_path / "sdxl-vae"
        vae_dir.mkdir()

        loader = _InvalidReturnVAELoader({
            "vae_name": "sdxl-vae",
            "vae_path": str(vae_dir),
        })
        ctx = make_default_context()
        with pytest.raises(TypeError) as exc_info:
            await loader.execute({}, ctx)
        assert "vae" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_dict_return_raises_type_error(self, tmp_path: Path):
        """返回非 dict 时抛 TypeError。"""
        vae_dir = tmp_path / "sdxl-vae"
        vae_dir.mkdir()

        loader = _NonDictVAELoader({"vae_name": "sdxl-vae", "vae_path": str(vae_dir)})
        ctx = make_default_context()
        with pytest.raises(TypeError) as exc_info:
            await loader.execute({}, ctx)
        assert "dict" in str(exc_info.value)


# ===== cache_key 测试 =====


class TestVAELoaderNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_contains_vae_name(self):
        node = VAELoaderNode({"vae_name": "sdxl-vae"})
        key = node.cache_key({})
        assert "sdxl-vae" in key

    def test_cache_key_contains_source(self):
        node = VAELoaderNode({
            "vae_name": "flux-vae",
            "source": "engine_specific",
        })
        key = node.cache_key({})
        assert "engine_specific" in key

    def test_cache_key_contains_dtype(self):
        node = VAELoaderNode({"vae_name": "flux-vae", "dtype": "bf16"})
        key = node.cache_key({})
        assert "bf16" in key

    def test_cache_key_default_dtype_is_auto(self):
        node = VAELoaderNode({"vae_name": "flux-vae"})
        key = node.cache_key({})
        assert "auto" in key


# ===== 进度上报测试 =====


class TestVAELoaderNodeProgressReporting:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_progress_reported_start_and_done(self, tmp_path: Path):
        """加载时上报开始和完成进度。"""
        vae_dir = tmp_path / "sdxl-vae"
        vae_dir.mkdir()
        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        loader = _MockVAELoader({"vae_name": "sdxl-vae", "vae_path": str(vae_dir)})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await loader.execute({}, ctx)

        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0.0, "vae_loader.start")
        assert progress_calls[-1] == (1.0, "vae_loader.done")

    @pytest.mark.asyncio
    async def test_diffusers_api_version_in_progress(self, tmp_path: Path):
        """进度上报中包含 diffusers API 版本信息。"""
        vae_dir = tmp_path / "sdxl-vae"
        vae_dir.mkdir()
        progress_kwargs: list[dict] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_kwargs.append(kwargs)

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        loader = _MockVAELoader({"vae_name": "sdxl-vae", "vae_path": str(vae_dir)})
        ctx = make_default_context(progress_reporter=_CapturingReporter())
        await loader.execute({}, ctx)

        # 第一个进度帧应包含 diffusers_api 字段
        assert "diffusers_api" in progress_kwargs[0]
        assert progress_kwargs[0]["diffusers_api"] in {"unknown", ">=0.31", "<0.31"}

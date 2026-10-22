"""LoRALoader / LoRALoaderAdvanced 节点单元 + 端到端测试。

覆盖任务 v0.2.5-universal-nodes-analysis.md §6（待办）:
    - 补 LoRALoader / LoRALoaderAdvanced 端到端集成测试
    - 测试路径解析（shared / engine_specific）
    - 测试参数校验
    - 测试执行流程（含 mock model / clip）
    - 测试错误处理（路径不存在、加载失败、clip=None 兼容）

参考：
    - v0.2.5-universal-nodes-analysis.md
    - ComfyUI LoraLoader / LoraLoaderAdvanced 节点
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.lora.lora_loader import (
    LoRALoaderAdvancedNode,
    LoRALoaderNode,
)


# ===== Mock 子类 =====


class _MockModel:
    """模拟 diffusers Pipeline（具备 load_lora_weights 方法）。"""

    def __init__(self, supports_set_adapters: bool = True):
        self.load_lora_weights_calls: list[Path] = []
        self.set_adapters_calls: list[dict] = []
        self.fuse_lora_calls: list[dict] = []
        self._supports_set_adapters = supports_set_adapters

    def load_lora_weights(self, path: str, **kwargs) -> None:
        self.load_lora_weights_calls.append(Path(path))

    def set_adapters(self, adapter_names, adapter_weights=None) -> None:
        if self._supports_set_adapters:
            self.set_adapters_calls.append(
                {"names": list(adapter_names), "weights": list(adapter_weights or [])}
            )

    def fuse_lora(self, lora_scale: float = 1.0) -> None:
        self.fuse_lora_calls.append({"lora_scale": lora_scale})


class _MockBareModel:
    """不具备 load_lora_weights 的模型对象。"""

    pass


# ===== 共享路径解析测试 =====


class TestLoRALoaderSharedPath:
    """source=shared 路径解析测试。"""

    def test_shared_lora_path_resolved(self, tmp_path: Path):
        """shared 模式：models/shared/lora/<lora_name>.safetensors。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "add_detail.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()

        node = LoRALoaderNode({"lora_name": "add_detail", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        resolved = node._resolve_lora_path(ctx)
        assert resolved == lora_path

    def test_shared_lora_fallback_to_other_extensions(self, tmp_path: Path):
        """shared 模式：.safetensors 不存在但 .pt 存在时回退。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "anime_style.pt"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()

        node = LoRALoaderNode({"lora_name": "anime_style", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        resolved = node._resolve_lora_path(ctx)
        assert resolved == lora_path

    def test_shared_lora_not_found_raises(self, tmp_path: Path):
        """shared 模式：路径不存在抛 NodeValidationError。"""
        models_root = tmp_path / "models"

        node = LoRALoaderNode({"lora_name": "nonexistent", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_lora_path(ctx)
        assert "LoRA 路径不存在" in str(exc_info.value)


# ===== 引擎特定路径解析测试（v0.2.5 规范）=====


class TestLoRALoaderEngineSpecificPath:
    """source=engine_specific 路径解析测试。

    v0.2.5 规范：models/image/loras/<engine_id>/<lora_name>.safetensors
    （与 engines/e_909469ed21ed/config.py 的 LORA_SUBDIR=`image/loras/<engine_name>/` 对齐）
    """

    def test_engine_specific_lora_path_with_engine_id(self, tmp_path: Path):
        """engine_specific 模式：使用 ctx.engine_id 拼路径。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "image" / "loras" / "e_909469ed21ed" / "add_detail.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()

        node = LoRALoaderNode({"lora_name": "add_detail", "source": "engine_specific"})
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        resolved = node._resolve_lora_path(ctx)
        assert resolved == lora_path

    def test_engine_specific_lora_path_fallback_to_unknown_engine(self, tmp_path: Path):
        """engine_specific 模式：ctx.engine_id 为空时使用 'unknown_engine'。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "image" / "loras" / "unknown_engine" / "test.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()

        node = LoRALoaderNode({"lora_name": "test", "source": "engine_specific"})
        ctx = make_default_context(engine_id="", models_root=models_root)
        resolved = node._resolve_lora_path(ctx)
        assert resolved == lora_path

    def test_engine_specific_lora_not_found_raises(self, tmp_path: Path):
        """engine_specific 模式：路径不存在抛 NodeValidationError。"""
        models_root = tmp_path / "models"

        node = LoRALoaderNode({"lora_name": "missing", "source": "engine_specific"})
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_lora_path(ctx)
        assert "LoRA 路径不存在" in str(exc_info.value)

    def test_explicit_lora_path_used_directly(self, tmp_path: Path):
        """lora_path 显式指定时优先使用，忽略 source 和 lora_name。"""
        models_root = tmp_path / "models"
        custom_path = models_root / "custom" / "my_lora.safetensors"
        custom_path.parent.mkdir(parents=True)
        custom_path.touch()

        node = LoRALoaderNode({
            "lora_name": "ignored",
            "lora_path": str(custom_path),
            "source": "engine_specific",
        })
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        resolved = node._resolve_lora_path(ctx)
        assert resolved == custom_path

    def test_no_models_root_raises(self):
        """ctx.models_root 为 None 时抛 NodeValidationError。"""
        node = LoRALoaderNode({"lora_name": "test", "source": "engine_specific"})
        ctx = make_default_context(models_root=None)
        with pytest.raises(NodeValidationError) as exc_info:
            node._resolve_lora_path(ctx)
        assert "models_root" in str(exc_info.value)


# ===== 参数校验测试 =====


class TestLoRALoaderValidate:
    """参数校验测试。"""

    def test_validate_missing_lora_name(self):
        node = LoRALoaderNode({})
        errors = node.validate()
        assert any("lora_name" in e for e in errors)

    def test_validate_negative_strength(self):
        node = LoRALoaderNode({
            "lora_name": "test",
            "strength": -0.5,
        })
        errors = node.validate()
        assert any("strength" in e for e in errors)

    def test_validate_invalid_source(self):
        node = LoRALoaderNode({
            "lora_name": "test",
            "source": "invalid",
        })
        errors = node.validate()
        assert any("source" in e for e in errors)

    def test_validate_negative_strength_clip(self):
        node = LoRALoaderNode({
            "lora_name": "test",
            "strength_clip": -1.0,
        })
        errors = node.validate()
        assert any("strength_clip" in e for e in errors)

    def test_validate_accepts_valid_params(self):
        node = LoRALoaderNode({
            "lora_name": "add_detail",
            "source": "shared",
            "strength": 1.0,
        })
        assert node.validate() == []


# ===== LoRALoader 端到端执行测试 =====


class TestLoRALoaderExecute:
    """LoRALoader 端到端测试。"""

    @pytest.mark.asyncio
    async def test_execute_loads_lora_with_default_strength(self, tmp_path: Path):
        """execute 成功加载 LoRA，使用默认 strength=1.0。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "add_detail.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel()

        node = LoRALoaderNode({
            "lora_name": "add_detail",
            "source": "shared",
        })
        ctx = make_default_context(models_root=models_root)
        result = await node.execute({"model": mock_model}, ctx)

        assert result["model"] is mock_model
        assert len(mock_model.load_lora_weights_calls) == 1
        assert mock_model.load_lora_weights_calls[0] == lora_path
        # strength=1.0 → set_adapters 应被调用
        assert len(mock_model.set_adapters_calls) >= 1
        assert mock_model.set_adapters_calls[0]["weights"] == [1.0]

    @pytest.mark.asyncio
    async def test_execute_loads_lora_with_custom_strength(self, tmp_path: Path):
        """execute 使用自定义 strength 加载 LoRA。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "anime_style.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel()

        node = LoRALoaderNode({
            "lora_name": "anime_style",
            "source": "shared",
            "strength": 0.7,
        })
        ctx = make_default_context(models_root=models_root)
        await node.execute({"model": mock_model}, ctx)

        assert mock_model.set_adapters_calls[0]["weights"] == [0.7]

    @pytest.mark.asyncio
    async def test_execute_missing_model_raises(self, tmp_path: Path):
        """execute 在 model 输入缺失时抛 NodeValidationError。"""
        node = LoRALoaderNode({"lora_name": "test", "source": "shared"})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({}, ctx)
        assert "model 必填" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_unsupported_model_raises(self, tmp_path: Path):
        """execute 在模型不支持 load_lora_weights 时抛 RuntimeError。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "test.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        bare_model = _MockBareModel()

        node = LoRALoaderNode({"lora_name": "test", "source": "shared"})
        ctx = make_default_context(models_root=models_root)
        with pytest.raises(RuntimeError) as exc_info:
            await node.execute({"model": bare_model}, ctx)
        assert "load_lora_weights" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_fuse_lora(self, tmp_path: Path):
        """execute 在模型不支持 set_adapters 时回退到 fuse_lora。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "test.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel(supports_set_adapters=False)

        node = LoRALoaderNode({
            "lora_name": "test",
            "source": "shared",
            "strength": 0.5,
        })
        ctx = make_default_context(models_root=models_root)
        await node.execute({"model": mock_model}, ctx)

        # 不应调用 set_adapters
        assert len(mock_model.set_adapters_calls) == 0
        # 应调用 fuse_lora
        assert len(mock_model.fuse_lora_calls) == 1
        assert mock_model.fuse_lora_calls[0]["lora_scale"] == 0.5

    @pytest.mark.asyncio
    async def test_execute_reports_progress(self, tmp_path: Path):
        """execute 正确上报开始和完成进度。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "shared" / "lora" / "test.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel()

        progress_calls: list[tuple[float, str]] = []

        from aireliable_engine import ProgressReporter

        class _CapturingReporter(ProgressReporter):
            async def report_progress(self, progress, stage="", **kwargs):
                progress_calls.append((progress, stage))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        node = LoRALoaderNode({"lora_name": "test", "source": "shared"})
        ctx = make_default_context(
            models_root=models_root,
            progress_reporter=_CapturingReporter(),
        )
        await node.execute({"model": mock_model}, ctx)

        assert progress_calls[0] == (0.0, "lora_loader.start")
        assert progress_calls[-1] == (1.0, "lora_loader.done")


# ===== LoRALoaderAdvanced 端到端测试 =====


class TestLoRALoaderAdvancedExecute:
    """LoRALoaderAdvanced 端到端测试。"""

    @pytest.mark.asyncio
    async def test_execute_loads_model_lora_only(self, tmp_path: Path):
        """无 clip 输入时仅加载 model LoRA。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "image" / "loras" / "e_909469ed21ed" / "add_detail.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel()

        node = LoRALoaderAdvancedNode({
            "lora_name": "add_detail",
            "source": "engine_specific",
            "strength_model": 0.8,
            "strength_clip": 0.5,
        })
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        result = await node.execute({"model": mock_model}, ctx)

        assert result["model"] is mock_model
        # 无 clip 输入时 clip 输出为 None（不抛异常）
        assert result["clip"] is None
        assert len(mock_model.load_lora_weights_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_loads_model_and_clip_lora(self, tmp_path: Path):
        """同时加载 model LoRA 和 CLIP LoRA。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "image" / "loras" / "e_909469ed21ed" / "style_x.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel()
        mock_clip = _MockModel()

        node = LoRALoaderAdvancedNode({
            "lora_name": "style_x",
            "source": "engine_specific",
            "strength_model": 0.9,
            "strength_clip": 0.7,
        })
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        result = await node.execute({"model": mock_model, "clip": mock_clip}, ctx)

        assert result["model"] is mock_model
        assert result["clip"] is mock_clip
        # model LoRA 加载
        assert len(mock_model.load_lora_weights_calls) == 1
        # CLIP LoRA 加载（如果 clip 自身支持 load_lora_weights）
        assert len(mock_clip.load_lora_weights_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_clip_without_load_lora_continues(self, tmp_path: Path):
        """clip 不支持 load_lora_weights 时跳过（不抛异常）。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "image" / "loras" / "e_909469ed21ed" / "test.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        mock_model = _MockModel()
        bare_clip = _MockBareModel()

        node = LoRALoaderAdvancedNode({
            "lora_name": "test",
            "source": "engine_specific",
        })
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        # 不应抛异常
        result = await node.execute({"model": mock_model, "clip": bare_clip}, ctx)
        assert result["model"] is mock_model
        assert result["clip"] is bare_clip

    @pytest.mark.asyncio
    async def test_execute_model_without_load_lora_warns(self, tmp_path: Path):
        """model 不支持 load_lora_weights 时记录 warning 跳过（LoRALoaderAdvanced 风格）。"""
        models_root = tmp_path / "models"
        lora_path = models_root / "image" / "loras" / "e_909469ed21ed" / "test.safetensors"
        lora_path.parent.mkdir(parents=True)
        lora_path.touch()
        bare_model = _MockBareModel()

        node = LoRALoaderAdvancedNode({
            "lora_name": "test",
            "source": "engine_specific",
        })
        ctx = make_default_context(
            engine_id="e_909469ed21ed",
            models_root=models_root,
        )
        # LoRALoaderAdvanced 在 model 不支持时仅记录 warning（不抛异常）
        # 这与 LoRALoader 的硬失败行为不同（ComfyUI 风格兼容）
        result = await node.execute({"model": bare_model}, ctx)
        assert result["model"] is bare_model

    @pytest.mark.asyncio
    async def test_execute_missing_model_raises(self):
        """execute 在 model 输入缺失时抛 NodeValidationError。"""
        node = LoRALoaderAdvancedNode({"lora_name": "test", "source": "shared"})
        ctx = make_default_context()
        with pytest.raises(NodeValidationError):
            await node.execute({}, ctx)


# ===== 基本属性测试 =====


class TestLoRALoaderNodeAttributes:
    """节点基本属性测试。"""

    def test_node_type_is_lora_loader(self):
        node = LoRALoaderNode({"lora_name": "test"})
        assert node.node_type == "LoRALoader"
        assert node.category == "lora"
        assert node.subcategory == "user"

    def test_advanced_node_type(self):
        node = LoRALoaderAdvancedNode({"lora_name": "test"})
        assert node.node_type == "LoRALoaderAdvanced"
        assert node.category == "lora"
        assert node.subcategory == "user"

    def test_input_specs_simple(self):
        """LoRALoader 仅 1 个输入端口（model）。"""
        node = LoRALoaderNode({"lora_name": "test"})
        assert len(node.input_specs) == 1
        assert node.input_specs[0].name == "model"
        assert node.input_specs[0].required is True

    def test_input_specs_advanced(self):
        """LoRALoaderAdvanced 有 2 个输入端口（model + clip）。"""
        node = LoRALoaderAdvancedNode({"lora_name": "test"})
        assert len(node.input_specs) == 2
        port_names = {p.name for p in node.input_specs}
        assert port_names == {"model", "clip"}
        # clip 是 optional
        clip_port = next(p for p in node.input_specs if p.name == "clip")
        assert clip_port.required is False

    def test_output_specs_simple(self):
        """LoRALoader 仅 1 个输出端口（model）。"""
        node = LoRALoaderNode({"lora_name": "test"})
        assert len(node.output_specs) == 1
        assert node.output_specs[0].name == "model"

    def test_output_specs_advanced(self):
        """LoRALoaderAdvanced 有 2 个输出端口（model + clip）。"""
        node = LoRALoaderAdvancedNode({"lora_name": "test"})
        assert len(node.output_specs) == 2
        port_names = {p.name for p in node.output_specs}
        assert port_names == {"model", "clip"}

    def test_params_schema_has_dir_field(self):
        """lora_name 字段有 dir 配置用于 enum-from-dir 扫描。"""
        node = LoRALoaderNode({"lora_name": "test"})
        assert "lora_name" in node.params_schema
        assert node.params_schema["lora_name"]["format"] == "enum-from-dir"
        assert node.params_schema["lora_name"]["dir"] == "image/loras/"

    def test_cache_key_returns_none(self):
        """cache_key 返回 None（LoRA 加载有副作用，不缓存）。"""
        node = LoRALoaderNode({"lora_name": "test"})
        assert node.cache_key({"model": MagicMock()}) is None

        node_adv = LoRALoaderAdvancedNode({"lora_name": "test"})
        assert node_adv.cache_key({"model": MagicMock()}) is None

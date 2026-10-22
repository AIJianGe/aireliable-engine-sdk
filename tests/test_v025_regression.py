"""v0.2.5 Loader 节点下拉选择与精度自动推断回归测试（架构类型联动目录改造版）。

验证 v0.2.5 改造的核心契约（防止后续改动导致回归）：
    - A3-2-1: Loader 节点的 params_schema 含 dir / dir_depends_on 字段
    - A3-2-2: 前端 fallback 节点定义（node_definitions.js）含 dir / dir_depends_on 字段
    - A3-2-3: ModelLoader 的 dir_map 含三个架构类型映射
    - D6-3-2: 有 dtype 参数的节点 options 含 "auto"
    - D6-3-4: `_dtype_utils.infer_dtype` 推断正确

参考：
    - v0.2.5-loader-architecture-tasks.md A3-2
    - v0.2.5-loader-architecture-analysis.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aireliable_engine._dtype_utils import infer_dtype
from aireliable_engine.nodes.loader.clip_loader import CLIPLoaderNode
from aireliable_engine.nodes.loader.model_loader import ModelLoaderNode
from aireliable_engine.nodes.loader.unet_loader import UNetLoaderNode
from aireliable_engine.nodes.loader.vae_loader import VAELoaderNode
from aireliable_engine.nodes.lora.lora_loader import (
    LoRALoaderAdvancedNode,
    LoRALoaderNode,
)


# 前端 node_definitions.js 路径：
#   tests/ -> aireliable-engine-sdk/ -> sdk/ -> AIReliable/
#   再进入 src-tauri/ui/workflow_editor/
_NODE_DEFINITIONS_JS = (
    Path(__file__).resolve().parents[3]
    / "src-tauri" / "ui" / "workflow_editor" / "node_definitions.js"
)


# ===== A3-2-1: Loader 节点的 params_schema 含 dir / dir_depends_on 字段 =====


class TestLoaderNodesDirField:
    """验证 Loader 节点的 params_schema 含目录字段（A3-2-1）。

    v0.2.5 架构类型联动改造后：
        - ModelLoader 的 variant 改用 dir_depends_on + dir_map（不再有 dir）
        - 其他 Loader 节点（UNetLoader/CLIPLoader/VAELoader/LoRALoader）保持 dir 字段
    """

    def test_model_loader_variant_has_dir_depends_on(self):
        """v0.2.5：ModelLoader.variant 含 dir_depends_on: "architecture"。"""
        schema = ModelLoaderNode.params_schema
        assert schema["variant"].get("dir_depends_on") == "architecture"

    def test_model_loader_variant_has_dir_map(self):
        """v0.2.5：ModelLoader.variant 含 dir_map，三个架构类型映射齐全。"""
        schema = ModelLoaderNode.params_schema
        dir_map = schema["variant"].get("dir_map")
        assert dir_map is not None
        assert dir_map["checkpoint"] == "image/checkpoints/"
        assert dir_map["diffusers"] == "image/checkpoints/"
        assert dir_map["unet"] == "image/unet/"

    def test_model_loader_variant_no_hardcoded_dir(self):
        """v0.2.5：ModelLoader.variant 不再硬编码 dir 字段（改由 dir_map 联动）。"""
        schema = ModelLoaderNode.params_schema
        assert "dir" not in schema["variant"]

    def test_model_loader_no_model_name_field(self):
        """v0.2.5：ModelLoader.params_schema 移除 model_name 字段。"""
        schema = ModelLoaderNode.params_schema
        assert "model_name" not in schema

    def test_model_loader_no_model_filename_field(self):
        """v0.2.5：ModelLoader.params_schema 移除 model_filename 字段。"""
        schema = ModelLoaderNode.params_schema
        assert "model_filename" not in schema

    def test_unet_loader_unet_name_has_dir(self):
        schema = UNetLoaderNode.params_schema
        assert schema["unet_name"].get("dir") == "image/unet/"

    def test_clip_loader_clip_name_has_dir(self):
        schema = CLIPLoaderNode.params_schema
        assert schema["clip_name"].get("dir") == "shared/text_encoders/"

    def test_vae_loader_vae_name_has_dir(self):
        schema = VAELoaderNode.params_schema
        # v0.2.5：VAE 目录随 source 联动，不再是单一 dir
        assert schema["vae_name"].get("dir_depends_on") == "source"
        assert schema["vae_name"]["dir_map"]["shared"] == "shared/vae/"

    def test_lora_loader_lora_name_has_dir(self):
        schema = LoRALoaderNode.params_schema
        assert schema["lora_name"].get("dir") == "image/loras/"

    def test_lora_advanced_lora_name_has_dir(self):
        schema = LoRALoaderAdvancedNode.params_schema
        assert schema["lora_name"].get("dir") == "image/loras/"

    def test_all_enum_from_dir_params_have_dir_or_depends_on(self):
        """v0.2.5：所有 format=enum-from-dir 的参数必须含 dir 或 dir_depends_on。"""
        nodes = [
            ModelLoaderNode, UNetLoaderNode, CLIPLoaderNode,
            VAELoaderNode, LoRALoaderNode, LoRALoaderAdvancedNode,
        ]
        for node_cls in nodes:
            schema = node_cls.params_schema
            for name, param in schema.items():
                if param.get("format") == "enum-from-dir":
                    has_dir = "dir" in param
                    has_depends = "dir_depends_on" in param
                    assert has_dir or has_depends, (
                        f"{node_cls.__name__}.params_schema['{name}'] "
                        f"是 enum-from-dir 但既无 dir 也无 dir_depends_on"
                    )


# ===== D6-3-2: 有 dtype 参数的节点 options 含 "auto" =====


class TestLoaderNodesDtypeAuto:
    """验证有 dtype 参数的节点 options 含 "auto"（D6-3-2）。

    dtype="auto" 触发各 Loader 节点 execute 中调用 infer_dtype 自动推断。
    """

    def test_model_loader_precision_has_auto(self):
        schema = ModelLoaderNode.params_schema
        assert "auto" in schema["precision"]["options"]
        assert schema["precision"]["default"] == "auto"

    def test_model_loader_torch_dtype_has_auto(self):
        schema = ModelLoaderNode.params_schema
        assert "auto" in schema["torch_dtype"]["options"]

    def test_unet_loader_dtype_has_auto(self):
        schema = UNetLoaderNode.params_schema
        assert "auto" in schema["dtype"]["options"]
        assert schema["dtype"]["default"] == "auto"

    def test_clip_loader_dtype_has_auto(self):
        schema = CLIPLoaderNode.params_schema
        assert "auto" in schema["dtype"]["options"]
        assert schema["dtype"]["default"] == "auto"

    def test_vae_loader_dtype_has_auto(self):
        schema = VAELoaderNode.params_schema
        assert "auto" in schema["dtype"]["options"]
        assert schema["dtype"]["default"] == "auto"


# ===== A3-2-2: 前端 fallback 节点定义含 dir / dir_depends_on 字段 =====


class TestFrontendFallbackDirField:
    """验证前端 node_definitions.js 的 fallback 节点含目录字段（A3-2-2）。

    前端 JS 文件无法直接 import，采用文本校验确保关键 Loader 节点
    的 fallback 定义包含 dir 或 dir_depends_on 字段（保证前端离线模式下
    的下拉框也能按正确目录拉取模型列表）。

    v0.2.5 架构类型联动改造后：
        - ModelLoader 的 variant 改用 dir_depends_on + dir_map
        - 其他 Loader 节点保持 dir 字段
    """

    @pytest.fixture(scope="module")
    def node_defs_content(self) -> str:
        """读取 node_definitions.js 文件内容。"""
        if not _NODE_DEFINITIONS_JS.exists():
            pytest.skip(f"前端节点定义文件不存在: {_NODE_DEFINITIONS_JS}")
        return _NODE_DEFINITIONS_JS.read_text(encoding="utf-8")

    def test_model_loader_fallback_has_dir_depends_on(self, node_defs_content: str):
        """v0.2.5：ModelLoader fallback 含 dir_depends_on: "architecture"。"""
        assert 'dir_depends_on: "architecture"' in node_defs_content

    def test_model_loader_fallback_has_dir_map(self, node_defs_content: str):
        """v0.2.5：ModelLoader fallback 含 dir_map 三个映射。

        注意：JS 对象字面量中 key 无引号（checkpoint: "..." 而非 "checkpoint": "..."）。
        """
        assert 'checkpoint: "image/checkpoints/"' in node_defs_content
        assert 'diffusers: "image/checkpoints/"' in node_defs_content
        assert 'unet: "image/unet/"' in node_defs_content

    def test_model_loader_fallback_no_model_name(self, node_defs_content: str):
        """v0.2.5：ModelLoader fallback 移除 model_name 字段。"""
        # 不应再出现 model_name 作为 ModelLoader 的 params 键
        #（注意：其他节点可能也有 model_name，此处用 ModelLoader 上下文限定）
        # 简单校验：ModelLoader 块内不应有 model_name:
        assert "model_name: {" not in node_defs_content

    def test_model_loader_fallback_architecture_options(self, node_defs_content: str):
        """v0.2.5：ModelLoader fallback 的 architecture options 为 checkpoint/diffusers/unet。"""
        assert '"checkpoint"' in node_defs_content
        assert '"diffusers"' in node_defs_content
        assert '"unet"' in node_defs_content

    def test_unet_loader_fallback_has_dir(self, node_defs_content: str):
        assert 'unet_name: { type: "string", required: true, format: "enum-from-dir", dir: "image/unet/"' in node_defs_content

    def test_clip_loader_fallback_has_dir(self, node_defs_content: str):
        assert 'clip_name: { type: "string", required: true, format: "enum-from-dir", dir: "shared/text_encoders/"' in node_defs_content

    def test_vae_loader_fallback_has_dir(self, node_defs_content: str):
        # v0.2.5：VAE 目录随 source 联动（dir_depends_on + dir_map）
        assert 'dir_depends_on: "source"' in node_defs_content
        assert 'shared: "shared/vae/"' in node_defs_content

    def test_lora_loader_fallback_has_dir(self, node_defs_content: str):
        # LoRALoader + LoRALoaderAdvanced 都应有 dir: "image/loras/"
        assert node_defs_content.count('dir: "image/loras/"') >= 2


# ===== A3-2-3: ModelLoader 的 dir_map 含三个架构类型映射 =====


class TestArchitectureDirMap:
    """验证 ModelLoader 的 dir_map 三个映射存在（A3-2-3）。"""

    def test_dir_map_has_checkpoint_mapping(self):
        schema = ModelLoaderNode.params_schema
        assert schema["variant"]["dir_map"]["checkpoint"] == "image/checkpoints/"

    def test_dir_map_has_diffusers_mapping(self):
        schema = ModelLoaderNode.params_schema
        assert schema["variant"]["dir_map"]["diffusers"] == "image/checkpoints/"

    def test_dir_map_has_unet_mapping(self):
        schema = ModelLoaderNode.params_schema
        assert schema["variant"]["dir_map"]["unet"] == "image/unet/"

    def test_dir_map_depends_on_architecture(self):
        """dir_depends_on 指向 architecture 字段。"""
        schema = ModelLoaderNode.params_schema
        assert schema["variant"]["dir_depends_on"] == "architecture"

    def test_architecture_field_has_three_options(self):
        """architecture 字段 options 为三个架构类型。"""
        schema = ModelLoaderNode.params_schema
        assert schema["architecture"]["options"] == ["checkpoint", "diffusers", "unet"]

    def test_architecture_field_required(self):
        """architecture 字段必填。"""
        schema = ModelLoaderNode.params_schema
        assert schema["architecture"]["required"] is True

    def test_architecture_field_default_is_checkpoint(self):
        """architecture 默认值为 checkpoint（早期 SD 架构）。"""
        schema = ModelLoaderNode.params_schema
        assert schema["architecture"]["default"] == "checkpoint"


# ===== D6-3-4: _dtype_utils.infer_dtype 推断正确（回归） =====


class TestInferDtypeRegression:
    """验证 infer_dtype 推断正确（D6-3-4，回归测试）。

    详细测试见 test_dtype_utils.py，此处仅做关键场景回归，
    防止正则模式被误改。
    """

    @pytest.mark.parametrize("filename,expected", [
        # GGUF 量化类型
        ("ernie-image-Q4_K_M.gguf", "q4_k_m"),
        ("model.Q8_0.gguf", "q8_0"),
        ("model.Q3_K_L.gguf", "q3_k_m"),
        # GGUF 内部精度
        ("model.F16.gguf", "f16"),
        ("model.BF16.GGUF", "bf16"),
        # safetensors 精度后缀
        ("model.fp16.safetensors", "fp16"),
        ("model.bf16.safetensors", "bf16"),
        ("model.fp32.safetensors", "fp32"),
        # 无匹配 → 默认
        ("model.unknown.safetensors", "fp16"),
    ])
    def test_infer_dtype_regression(self, filename, expected):
        assert infer_dtype(filename) == expected

    def test_infer_dtype_custom_default(self):
        assert infer_dtype("unknown.bin", default="bf16") == "bf16"

    def test_infer_dtype_none_and_empty(self):
        assert infer_dtype(None) == "fp16"
        assert infer_dtype("") == "fp16"

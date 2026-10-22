"""_dtype_utils.infer_dtype 单元测试（v0.2.5 dtype 自动推断）。

覆盖任务 D6-2 验收标准：
    - GGUF 量化类型推断（Q2_K / Q3_K / Q4_K_M / Q4_0 / Q5_K / Q5_0 / Q6_K / Q8_0）
    - GGUF 内部精度推断（F16 / F32）
    - safetensors/pt/ckpt 精度后缀推断（fp16 / bf16 / fp32）
    - 推断失败返回 default
    - None / 空字符串边界情况

参考：
    - v0.2.5-loader-dropdown-tasks.md D5-1
    - v0.2.5-loader-dropdown-analysis.md §3.3.1
"""

from __future__ import annotations

import pytest

from aireliable_engine._dtype_utils import infer_dtype


# ===== GGUF 量化类型测试 =====


class TestInferDtypeGGUFQuant:
    """GGUF 量化类型推断。"""

    @pytest.mark.parametrize("filename,expected", [
        ("ernie-image-Q4_K_M.gguf", "q4_k_m"),
        ("ernie-image-Q4_K_S.gguf", "q4_k_m"),
        ("ernie-image-q4_k_m.gguf", "q4_k_m"),
        ("ernie-image-q4_k_s.gguf", "q4_k_m"),
        ("model.Q4_K_M.gguf", "q4_k_m"),
    ])
    def test_q4_k_m(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.Q8_0.gguf", "q8_0"),
        ("model.q8_0.gguf", "q8_0"),
    ])
    def test_q8_0(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.Q2_K.gguf", "q2_k"),
        ("model.q2_k.gguf", "q2_k"),
    ])
    def test_q2_k(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.Q3_K_S.gguf", "q3_k_m"),
        ("model.Q3_K_M.gguf", "q3_k_m"),
        ("model.Q3_K_L.gguf", "q3_k_m"),
    ])
    def test_q3_k(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.Q4_0.gguf", "q4_0"),
        ("model.q4_0.gguf", "q4_0"),
    ])
    def test_q4_0(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.Q5_K_S.gguf", "q5_k_m"),
        ("model.Q5_K_M.gguf", "q5_k_m"),
        ("model.Q5_0.gguf", "q5_0"),
    ])
    def test_q5(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.Q6_K.gguf", "q6_k"),
    ])
    def test_q6_k(self, filename, expected):
        assert infer_dtype(filename) == expected


# ===== GGUF 内部精度测试 =====


class TestInferDtypeGGUFInternal:
    """GGUF 内部精度（F 前缀）推断。"""

    @pytest.mark.parametrize("filename,expected", [
        ("model.F16.gguf", "f16"),
        ("model.f16.gguf", "f16"),
    ])
    def test_f16(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.F32.gguf", "f32"),
        ("model.f32.gguf", "f32"),
    ])
    def test_f32(self, filename, expected):
        assert infer_dtype(filename) == expected


# ===== safetensors/pt/ckpt 精度后缀测试 =====


class TestInferDtypeSafetensors:
    """safetensors/pt/ckpt 精度后缀推断。"""

    @pytest.mark.parametrize("filename,expected", [
        ("model.fp16.safetensors", "fp16"),
        ("model.float16.safetensors", "fp16"),
        ("model.fp16.pt", "fp16"),
        ("model.fp16.ckpt", "fp16"),
        ("model.fp16.bin", "fp16"),
    ])
    def test_fp16(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.bf16.safetensors", "bf16"),
        ("model.bfloat16.safetensors", "bf16"),
        ("model.bf16.pt", "bf16"),
    ])
    def test_bf16(self, filename, expected):
        assert infer_dtype(filename) == expected

    @pytest.mark.parametrize("filename,expected", [
        ("model.fp32.safetensors", "fp32"),
        ("model.float32.safetensors", "fp32"),
        ("model.fp32.ckpt", "fp32"),
    ])
    def test_fp32(self, filename, expected):
        assert infer_dtype(filename) == expected


# ===== 默认值与边界情况测试 =====


class TestInferDtypeDefault:
    """推断失败与边界情况。"""

    def test_no_match_returns_default(self):
        """无精度后缀的文件名返回默认值 fp16。"""
        assert infer_dtype("model.safetensors") == "fp16"
        assert infer_dtype("flux-dev.safetensors") == "fp16"
        assert infer_dtype("checkpoint.ckpt") == "fp16"

    def test_custom_default(self):
        """自定义默认值。"""
        assert infer_dtype("model.safetensors", default="bf16") == "bf16"
        assert infer_dtype("unknown.gguf", default="fp32") == "fp32"

    def test_empty_string(self):
        """空字符串返回默认值。"""
        assert infer_dtype("") == "fp16"
        assert infer_dtype("", default="bf16") == "bf16"

    def test_none_input(self):
        """None 输入返回默认值。"""
        assert infer_dtype(None) == "fp16"
        assert infer_dtype(None, default="fp32") == "fp32"

    def test_case_insensitive(self):
        """大小写不敏感。"""
        assert infer_dtype("MODEL.FP16.SAFETENSORS") == "fp16"
        assert infer_dtype("Model.BF16.GGUF") == "bf16"
        assert infer_dtype("model.q4_k_m.GGUF") == "q4_k_m"

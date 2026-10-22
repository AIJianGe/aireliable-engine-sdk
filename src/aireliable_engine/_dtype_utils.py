"""dtype 自动推断工具（v0.2.5 引入）。

根据模型文件名后缀自动推断 torch dtype / GGUF 量化类型。
供各 Loader 节点在 dtype 参数为 "auto" 时调用。

推断规则（按优先级排序）：
    1. GGUF 量化类型（Q 前缀）：q2_k / q3_k_m / q4_k_m / q4_0 / q5_k_m / q5_0 / q6_k / q8_0
    2. GGUF 内部精度（F 前缀）：f16 / f32
    3. safetensors / pt / ckpt 精度后缀：fp16 / bf16 / fp32
    4. 无匹配 → 返回 default（默认 "fp16"）

参考：
    - v0.2.5-loader-dropdown-analysis.md §3.3.1 文件名后缀推断
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# 文件名后缀 → dtype 映射（按优先级排序，先匹配先返回）
# 说明：GGUF 量化标识前缀可能是 `-` / `_` / `.`（如 "ernie-image-Q4_K_M.gguf"），
#       故只匹配末尾的量化标识，不限制前缀分隔符。
_DTYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ===== GGUF 量化类型（Q 前缀，不区分大小写）=====
    # Q2_K
    (re.compile(r"q2_k\.gguf$", re.IGNORECASE), "q2_k"),
    # Q3_K_S/M/L（归一为 q3_k_m）
    (re.compile(r"q3_k_[sml]\.gguf$", re.IGNORECASE), "q3_k_m"),
    # Q4_K_S/M（归一为 q4_k_m）
    (re.compile(r"q4_k_[sm]\.gguf$", re.IGNORECASE), "q4_k_m"),
    # Q4_0
    (re.compile(r"q4_0\.gguf$", re.IGNORECASE), "q4_0"),
    # Q5_K_S/M（归一为 q5_k_m）
    (re.compile(r"q5_k_[sm]\.gguf$", re.IGNORECASE), "q5_k_m"),
    # Q5_0
    (re.compile(r"q5_0\.gguf$", re.IGNORECASE), "q5_0"),
    # Q6_K
    (re.compile(r"q6_k\.gguf$", re.IGNORECASE), "q6_k"),
    # Q8_0
    (re.compile(r"q8_0\.gguf$", re.IGNORECASE), "q8_0"),
    # ===== GGUF 内部精度格式（F/FP/BF 前缀）=====
    (re.compile(r"\.bf16\.gguf$", re.IGNORECASE), "bf16"),
    (re.compile(r"\.bfloat16\.gguf$", re.IGNORECASE), "bf16"),
    (re.compile(r"\.fp16\.gguf$", re.IGNORECASE), "fp16"),
    (re.compile(r"\.float16\.gguf$", re.IGNORECASE), "fp16"),
    (re.compile(r"\.fp32\.gguf$", re.IGNORECASE), "fp32"),
    (re.compile(r"\.float32\.gguf$", re.IGNORECASE), "fp32"),
    (re.compile(r"\.f16\.gguf$", re.IGNORECASE), "f16"),
    (re.compile(r"\.f32\.gguf$", re.IGNORECASE), "f32"),
    # ===== safetensors / pt / ckpt / bin 精度后缀 =====
    (re.compile(r"\.fp16\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE), "fp16"),
    (re.compile(r"\.float16\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE), "fp16"),
    (re.compile(r"\.bf16\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE), "bf16"),
    (re.compile(r"\.bfloat16\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE), "bf16"),
    (re.compile(r"\.fp32\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE), "fp32"),
    (re.compile(r"\.float32\.(safetensors|pt|ckpt|bin)$", re.IGNORECASE), "fp32"),
]


def infer_dtype(filename: str, default: str = "fp16") -> str:
    """根据模型文件名推断 dtype。

    Args:
        filename: 模型文件名（如 "ernie-image-Q4_K_M.gguf" / "model.fp16.safetensors"）。
                  传入 None 或空字符串时直接返回 default。
        default: 推断失败时的默认值（默认 "fp16"）。

    Returns:
        dtype 字符串，可能值：
        - GGUF 量化：q2_k / q3_k_m / q4_k_m / q4_0 / q5_k_m / q5_0 / q6_k / q8_0
        - GGUF 内部精度：f16 / f32
        - safetensors/pt/ckpt 精度：fp16 / bf16 / fp32
        - 推断失败：default

    Examples:
        >>> infer_dtype("ernie-image-Q4_K_M.gguf")
        'q4_k_m'
        >>> infer_dtype("model.fp16.safetensors")
        'fp16'
        >>> infer_dtype("model.safetensors")
        'fp16'
        >>> infer_dtype("", default="bf16")
        'bf16'
    """
    if not filename or not isinstance(filename, str):
        return default
    for pattern, dtype in _DTYPE_PATTERNS:
        if pattern.search(filename):
            return dtype
    logger.debug("infer_dtype: 文件名 %r 未匹配任何精度模式，返回默认 %s", filename, default)
    return default

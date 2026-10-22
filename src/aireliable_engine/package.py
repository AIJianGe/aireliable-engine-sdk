"""引擎包（.airengine）的规范化文件布局与内容哈希。

``aireliable-engine pack`` 与平台导入器共用本模块，保证**打包端与校验端**
对同一份内容算出同一个哈希，避免"本地通过、导入被拒"。

规范化规则（v1）：

1. 只对**相对路径 + 文件字节**计算，不依赖文件系统时间/权限；
2. 路径统一使用 POSIX 分隔符（``/``），按字典序排序；
3. 跳过开发期产物：``.git``、``.venv``、``__pycache__``、``.aireliable``、
   ``.pytest_cache``、``dist``、``models``、``*.pyc``；
4. 跳过 ``manifest.yaml`` 与 ``package.json``（前者承载哈希、后者是哈希清单，
   自引用会形成循环）；
5. 每个条目参与哈希的字节为 ``path(utf-8) + \\0 + content + \\0``。

包结构（与平台导入器一致）::

    manifest.yaml
    registry.yaml
    code/<engine_id>/...
    env/pyproject.toml          # 必需
    env/uv.lock                 # 可选
    docs/...                    # 可选
    package.json                # 由 pack 写入：文件清单 + 整包哈希
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

#: 包内必需文件（相对路径）
REQUIRED_FILES: tuple[str, ...] = ("manifest.yaml", "registry.yaml", "env/pyproject.toml")

#: 哈希计算时跳过的目录名（任意层级）
EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", "__pycache__", ".aireliable", ".pytest_cache", "dist", "models"}
)

#: 哈希计算时跳过的根级文件
EXCLUDE_ROOT_FILES: frozenset[str] = frozenset({"manifest.yaml", "package.json"})

#: 不应打进包的大文件后缀（模型权重必须放在平台 models/ 目录）
MODEL_WEIGHT_SUFFIXES: frozenset[str] = frozenset(
    {".safetensors", ".gguf", ".ckpt", ".bin", ".pt", ".pth", ".onnx", ".pkl"}
)

#: 单文件体积上限（字节）：超过视为误把模型/数据打进了包
MAX_FILE_BYTES = 20 * 1024 * 1024


def iter_files(root: Path, *, skip_hash_excluded: bool = True) -> list[Path]:
    """列出包内文件（绝对路径），已按相对路径排序。

    Args:
        root: 包根目录。
        skip_hash_excluded: 是否跳过哈希排除项（``manifest.yaml``/``package.json``）。
            做"大文件检查"时应传 False，以便也能检查 manifest 等文件。

    Returns:
        排序后的文件路径列表。
    """
    root = root.resolve()
    items: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
            continue
        if path.suffix == ".pyc":
            continue
        if skip_hash_excluded and len(rel.parts) == 1 and rel.name in EXCLUDE_ROOT_FILES:
            continue
        items.append(path)
    items.sort(key=lambda p: p.relative_to(root).as_posix())
    return items


def compute_content_hash(root: Path) -> str:
    """计算规范化内容哈希（64 位十六进制）。

    Args:
        root: 包根目录（打包前目录或解压后的包目录均可）。

    Returns:
        sha256 十六进制摘要。
    """
    root = root.resolve()
    digest = hashlib.sha256()
    for path in iter_files(root, skip_hash_excluded=True):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_manifest(root: Path) -> list[dict[str, object]]:
    """生成逐文件哈希清单（写入 ``package.json``）。"""
    root = root.resolve()
    rows: list[dict[str, object]] = []
    for path in iter_files(root, skip_hash_excluded=True):
        data = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return rows


def write_package_json(
    root: Path,
    *,
    engine_id: str,
    version: str,
    sha256: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """写入 ``package.json``（文件清单 + 整包哈希）。

    注意：``package.json`` 本身不参与内容哈希。
    """
    root = root.resolve()
    meta: dict[str, object] = {
        "package_format": 1,
        "engine_id": engine_id,
        "version": version,
        "sha256": sha256 or compute_content_hash(root),
        "files": file_manifest(root),
    }
    if extra:
        meta.update(extra)
    (root / "package.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def read_package_json(root: Path) -> dict[str, object] | None:
    """读取 ``package.json``；不存在或无法解析时返回 None。"""
    path = root / "package.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def oversized_files(root: Path, *, max_bytes: int = MAX_FILE_BYTES) -> list[Path]:
    """返回超过体积上限的文件（用于 ``validate`` 的大文件检查）。"""
    root = root.resolve()
    return [p for p in iter_files(root, skip_hash_excluded=False) if p.stat().st_size > max_bytes]


def model_weight_files(root: Path) -> list[Path]:
    """返回包内疑似模型权重文件（应改为平台 models/ 目录管理）。"""
    root = root.resolve()
    return [
        p
        for p in iter_files(root, skip_hash_excluded=False)
        if p.suffix.lower() in MODEL_WEIGHT_SUFFIXES
    ]


def missing_required(root: Path) -> list[str]:
    """返回包内缺失的必需文件（相对路径）。"""
    root = root.resolve()
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def iter_zip_members(names: Iterable[str]) -> list[str]:
    """过滤 zip 成员名（插件/归档侧复用）：去掉目录与排除项。"""
    kept: list[str] = []
    for name in names:
        if name.endswith("/"):
            continue
        parts = name.split("/")
        if any(part in EXCLUDE_DIRS for part in parts[:-1]):
            continue
        if name.endswith(".pyc"):
            continue
        if len(parts) == 1 and parts[0] in EXCLUDE_ROOT_FILES:
            continue
        kept.append(name)
    return sorted(kept)

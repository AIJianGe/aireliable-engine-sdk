"""manifest.yaml 结构模型（SDK 单一真相源）。

本模块是 AIReliable 引擎 ``manifest.yaml`` 的**结构契约**，随 SDK 发布：

- SDK 的 ``aireliable-engine validate`` 用它做本地校验；
- 平台 core 直接复用同一份模型，只在其上叠加平台策略（兼容矩阵、全局唯一性等），
  避免两端 schema 漂移。

顶层字段分三类：

1. 身份：``engine_id`` / ``engine_name`` / ``engine_version`` / ``engine_uid`` / ``entry``
2. 契约：``manifest_schema_version`` / ``engine_api_version`` / ``aireliable_compat`` / ``sdk_version``
3. 模型：``models`` 块（variants / model_files / model_layout / loras / downloader / pipeline_template）
4. 环境：``env_config`` 块（torch / CUDA / Python 约束）
5. 分发：``author`` / ``source`` / ``upgrade`` / ``package`` / ``license`` / ``repository`` / ``visibility``

用法::

    from aireliable_engine.manifest import (
        ManifestSchema,
        ManifestSchemaError,
        validate_manifest_dict,
        validate_manifest_yaml,
    )

    schema = validate_manifest_yaml(yaml_text)
    data = schema.to_dict()

设计要点：
    - 全部未知顶层字段按 ``extra="forbid"`` 拒绝（拼写保护），引擎私有字段放 ``extra`` 槽；
    - 新增字段一律可选，旧 manifest 不受影响；
    - ``validate_v021_strict()`` 是平台导入时的强制策略，SDK 侧由 ``validate --strict`` 触发。
"""

from __future__ import annotations

import re
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# ===========================================================================
# 契约版本（SDK 为单一真相源，平台 core 直接复用）
# ===========================================================================

# 引擎 HTTP 契约版本（破坏性变更才 +1）
ENGINE_API_VERSION = "1"

# manifest 结构版本（MAJOR.MINOR；MAJOR 变更表示不兼容）
CURRENT_MANIFEST_SCHEMA_VERSION = "0.4"


# ===========================================================================
# 版本区间解析（与平台语义一致的精简实现，避免 SDK 依赖平台代码）
# ===========================================================================

class SpecifierError(ValueError):
    """版本区间语法错误。"""


_OPERATORS = (">=", "<=", "==", "!=", ">", "<", "=")
_SPEC_ITEM_RE = re.compile(
    r"^(>=|<=|==|!=|>|<|=)?\s*v?"
    r"(\d+(?:\.\d+){0,2}"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?)$"
)
_ANY_SPECS = frozenset({"*", "any", "x", "X"})


def _normalize_version(version: str) -> str:
    """把 1–3 段版本号补齐为 3 段（保留预发布 / 构建元数据）。"""
    version = (version or "").strip().lstrip("v")
    if not version:
        raise SpecifierError("版本号为空")

    core, dash, rest = version.partition("-")
    if dash:
        suffix = "-" + rest
    else:
        core, plus, build = core.partition("+")
        suffix = "+" + build if plus else ""

    parts = core.split(".")
    if not 1 <= len(parts) <= 3:
        raise SpecifierError(
            f"版本号 {version!r} 格式无效（必须是 MAJOR[.MINOR[.PATCH]]，"
            f"最多 3 段，不能写成 0.2.14.8 这种 4 段形式）"
        )
    if any(not p.isdigit() for p in parts):
        raise SpecifierError(f"版本号 {version!r} 格式无效（主版本号必须为数字）")

    parts += ["0"] * (3 - len(parts))
    return ".".join(parts) + suffix


def parse_specifier(spec: str) -> list[tuple[str, str]]:
    """解析版本区间为 ``[(operator, normalized_version), ...]``。"""
    if spec is None or not str(spec).strip():
        raise SpecifierError("版本区间为空")

    raw = str(spec).strip()
    if raw in _ANY_SPECS:
        return []

    raw = re.sub(r"(>=|<=|==|!=|>|<|=)\s+", r"\1", raw)
    items_raw = [p for p in re.split(r"[,，\s]+", raw) if p]

    items: list[tuple[str, str]] = []
    for item in items_raw:
        m = _SPEC_ITEM_RE.match(item)
        if not m:
            raise SpecifierError(
                f"版本区间 {spec!r} 含非法片段 {item!r}"
                f"（支持 >= <= == != > < = 与 MAJOR[.MINOR[.PATCH]]）"
            )
        op = m.group(1) or "=="
        if op == "=":
            op = "=="
        items.append((op, _normalize_version(m.group(2))))
    return items


# ===========================================================================
# 常量
# ===========================================================================

# engine_uid 格式：aireliable://<author>/<family>/<name>
# author / family / name：小写字母 + 数字 + 下划线 + 连字符
# 任务 64（Schema 治理）：放行 `-`（对齐 OCI/Docker 仓库名与 K8s DNS-1123 —— 均允许连字符）；
# 仍禁止大写与非 ASCII（机器标识符业界惯例为 ASCII）——unicode 作者名（如 AI剑歌）须改 ASCII。
_ENGINE_UID_PATTERN = re.compile(
    r"^aireliable://[a-z0-9_-]+/[a-z0-9_-]+/[a-z0-9_-]+$"
)

# 语义化版本号：MAJOR.MINOR.PATCH（可选 -prerelease）
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)

# engine_id 格式：小写字母 + 数字 + 下划线（v0.2.1+ 统一为不允许 `-`，加长度限制 64）
# 与 UI（app.js）/ EngineScaffolder / engine_id_generator.py 保持一致
_ENGINE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

# GitHub repo 格式：owner/repo（owner / repo 不能含斜杠）
_GITHUB_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# 范式枚举
# v0.2.5：恢复 integrated 范式（全自定义模式），与 workflow 共存
# - workflow：工作流引擎，继承 WorkflowEngine，挂载工作流编辑器 UI
# - integrated：全自定义引擎，继承 CustomEngine，挂载 Gradio UI
# 已废弃：custom（v0.2.5 移除，由 integrated 替代）
# 注意：v0.2.5+ 的新引擎建议使用 workflow 范式，integrated 仅用于向后兼容已有引擎
_PARADIGMS = {"workflow", "integrated"}

# source.type 枚举
_SOURCE_TYPES = {"github", "local", "registry"}

# ===========================================================================
# v0.2.1 模型分类规范化常量（PR/architecture.md §16）
# ===========================================================================

# 13 类顶层目录（用于 target_dir / model_layout.base / lora.scan_dir 校验）
# 任务 64（Schema 治理）：新增 `audio` —— 与 HF「音频任务独立成族」一致，
# 用于音效 / 视频转音频（V2A/Foley）等非语音类引擎（如 foleycrafter）。
# 详见 PR/architecture.md §3.6 / docs/docs/appendix/file-paths.md
# 注意：本常量在 engine_scaffolder.py / model_placement_checker.py 另有副本，须同改。
_MODEL_TOP_LEVEL_DIRS = frozenset({
    "shared", "image", "video", "voice", "audio", "llm", "multimodal",
    "embedding", "reranker", "detection", "segmentation",
    "restoration", "misc",
})

# model_files[].role 枚举（16 项，详见 PR/architecture.md §16.3 / §18.2）
# v0.2.3 新增（ERNIE-Image 改造）：pe / scheduler / text_encoder
_MODEL_FILE_ROLE_VALUES = frozenset({
    "main", "mmproj", "vae", "clip", "t5", "clip_vision",
    "lora", "controlnet", "ipadapter", "embedding",
    "upscale", "config", "tokenizer",
    # v0.2.3 新增（ERNIE-Image 改造，详见 §18.2）
    "pe",              # Prompt Enhancer（图片专属，如 ERNIE-Image 的 Ministral3 PE）
    "scheduler",       # diffusers scheduler_config.json
    "text_encoder",   # 通用文本编码器（Mistral3 等，与 clip/t5 同层但更通用）
})

# v0.4.4+：旧版本 manifest 中常用 `role: weights`（主权重），与新枚举 `main` 同义。
# 允许以别名形式通过校验，跨版本兼容（避免老引擎导入失败）。
# - 校验通过时不改写原值（保留 manifest 原文，便于用户后续手工清理）
# - 内部如需识别"主权重"角色，应同时识别 main / weights 两个值
_MODEL_FILE_ROLE_ALIASES = {
    "weights": "main",   # 主权重的历史命名（部分老引擎如 voxcpm_tts）
}


def normalize_model_file_role(value: str) -> str:
    """将 model_files[].role 字段规范化为标准值。

    v0.4.4+：支持别名（如 `weights` → `main`），保持向后兼容。
    不在枚举也不在别名中的值原样返回，由 validator 抛错。
    """
    if value in _MODEL_FILE_ROLE_ALIASES:
        return _MODEL_FILE_ROLE_ALIASES[value]
    return value

# model_files[].source.type 枚举
_MODEL_SOURCE_TYPES = frozenset({
    "huggingface", "modelscope", "url", "local",
})


# ===========================================================================
# 异常
# ===========================================================================

class ManifestSchemaError(ValueError):
    """manifest.yaml schema 校验错误。

    Attributes:
        field_path: 错误字段路径（如 "author.name"）。空字符串表示顶层错误。
    """

    def __init__(self, message: str, *, field_path: str = ""):
        self.field_path = field_path
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.field_path:
            return f"[{self.field_path}] {msg}"
        return msg


# ===========================================================================
# 作者信息
# ===========================================================================

class AuthorInfo(BaseModel):
    """作者信息块（v0.2.7 重新定义语义为"创建引擎的作者"）。

    v0.2.6 此字段曾填第三方开源项目方（如 PaddlePaddle），v0.2.7 改为
    "在 AIReliable 平台上做引擎封装的作者"。第三方开源项目方请放入
    `third_party_references` 字段。

    Attributes:
        name: 创建引擎的作者名（必填）。默认 "Unknown"。如 "张三" / "CommunityUser"。
        email: 作者邮箱（选填）。
        github: 作者自己的 GitHub 主页 URL（选填，非第三方仓库）。
        website: 作者官网 URL（选填）。
        license: 作者封装代码的协议（v0.2.7 起可选，空字符串表示未开源）。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    email: str | None = None
    github: str | None = None
    website: str | None = None
    # v0.2.7：license 改为可选，空字符串表示作者封装代码未开源
    license: str = ""

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("author.name 不能为空")
        return v.strip()

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "@" not in v:
            raise ValueError(f"author.email {v!r} 格式无效（必须含 @）")
        return v

    @field_validator("github", "website")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL {v!r} 必须以 http:// 或 https:// 开头")
        return v


class ThirdPartyReference(BaseModel):
    """第三方开源引用项（v0.2.7 新增）。

    引擎引用的第三方开源模型/代码，由作者负责标注。

    Attributes:
        name: 第三方项目名（必填）。如 "FoleyCrafter" / "Auffusion"。
        author: 第三方项目方（必填）。如 "open-mmlab" / "auffusion"。
        github: 第三方项目仓库 URL（选填）。
        website: 第三方项目官网 URL（选填）。
        license: 第三方项目协议（必填）。如 "Apache-2.0" / "MIT"。
        description: 在本引擎中的作用描述（选填）。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    github: str | None = None
    website: str | None = None
    license: str = Field(..., min_length=1)
    description: str = ""
    # 任务 64：引擎私有扩展槽（承载非标准键，如 strata 曾用的 repo / usage）
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "author", "license")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("third_party_references 项的 name/author/license 不能为空")
        return v.strip()

    @field_validator("github", "website")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL {v!r} 必须以 http:// 或 https:// 开头")
        return v


# ===========================================================================
# 来源信息
# ===========================================================================

class SourceInfo(BaseModel):
    """来源信息块。

    Attributes:
        type: 来源类型（必填）。github / local / registry。
        github_repo: GitHub 仓库（type=github 时必填）。格式：owner/repo。
        branch: 分支名（选填）。
        commit: commit hash（选填）。
        download_url: 直接下载 URL（选填）。
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., min_length=1)
    github_repo: str | None = None
    branch: str | None = None
    commit: str | None = None
    download_url: str | None = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _SOURCE_TYPES:
            raise ValueError(
                f"source.type {v!r} 无效（必须是 {sorted(_SOURCE_TYPES)} 之一）"
            )
        return v

    @field_validator("github_repo")
    @classmethod
    def _validate_github_repo(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _GITHUB_REPO_PATTERN.match(v):
            raise ValueError(
                f"source.github_repo {v!r} 格式无效（必须是 owner/repo）"
            )
        return v

    @field_validator("download_url")
    @classmethod
    def _validate_download_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"source.download_url {v!r} 必须以 http:// 或 https:// 开头")
        return v

    @model_validator(mode="after")
    def _validate_github_repo_required(self) -> SourceInfo:
        """type=github 时 github_repo 必填。"""
        if self.type == "github" and not self.github_repo:
            raise ValueError("source.type=github 时 source.github_repo 必填")
        return self


# ===========================================================================
# 升级信息
# ===========================================================================

class UpgradeInfo(BaseModel):
    """升级信息块。

    Attributes:
        check_url: 检查更新的 URL（必填）。通常是 GitHub API releases/latest。
        auto_check: 是否自动检查更新（选填，默认 False）。
        min_compat_version: 最低兼容版本（选填）。升级时校验配置兼容性。
    """

    model_config = ConfigDict(extra="forbid")

    check_url: str = Field(..., min_length=1)
    auto_check: bool = False
    min_compat_version: str | None = None

    @field_validator("check_url")
    @classmethod
    def _validate_check_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"upgrade.check_url {v!r} 必须以 http:// 或 https:// 开头")
        return v

    @field_validator("min_compat_version")
    @classmethod
    def _validate_min_compat_version(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _SEMVER_PATTERN.match(v):
            raise ValueError(
                f"upgrade.min_compat_version {v!r} 不是合法的语义化版本号"
            )
        return v


# ===========================================================================
# 节点包依赖
# ===========================================================================

class NodePackageDep(BaseModel):
    """节点包依赖项。

    Attributes:
        name: 节点包名（必填）。如 "aireliable-node-ipadapter-plus"。
        version: 版本约束（选填）。如 ">=0.1.0" / "^0.2.0"。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    version: str | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("node_packages[].name 不能为空")
        return v.strip()


# ===========================================================================
# v0.2.1 模型分类规范化（manifest.models 块）
#
# 设计参考：
#   - PR/architecture.md §16.4 manifest.models 块（Pydantic schema）
#   - PR/development-guide.md §5.4-§5.5 模型文件预检与位置检测
#   - docs/docs/appendix/model-classification.md 枚举值清单
#   - docs/docs/appendix/model-placement-rules.md 位置放置规则
#
# 字段强制策略（v0.2.1）：
#   - models 块：schema 层 Optional（向后兼容 v0.1/v0.2），EngineImporter 强制必填
#   - models.model_layout.base：必填
#   - models.model_files[].role：必填（13 项枚举之一）
#   - models.variants[].model_file_ids：可选（v0.3 强制必填，废弃 mmproj_filename）
#
# v0.2.4 变更：
#   - models.model_files[].sha256：已移除（完整性校验改为 size_bytes 单一策略）
# ===========================================================================


def _validate_model_top_level_dir(value: str, field_name: str) -> str:
    """校验目录路径必须以 12 类顶层目录之一开头。

    Args:
        value: 待校验的目录路径（如 "image/checkpoints/ernie-image/"）。
        field_name: 字段名（用于错误信息）。

    Returns:
        标准化后的路径（去除尾部斜杠）。

    Raises:
        ValueError: 路径不以 12 类顶层目录开头。
    """
    if not value or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    v = value.strip()
    # 统一路径分隔符（Windows 反斜杠 → 正斜杠）
    v = v.replace("\\", "/")
    # 去除开头的 ./
    if v.startswith("./"):
        v = v[2:]
    # 去除尾部斜杠
    v = v.rstrip("/")
    if not v:
        raise ValueError(f"{field_name} 不能为空")
    # 取首段
    first_segment = v.split("/", 1)[0]
    if first_segment not in _MODEL_TOP_LEVEL_DIRS:
        raise ValueError(
            f"{field_name} {value!r} 必须以 12 类顶层目录之一开头："
            f"{sorted(_MODEL_TOP_LEVEL_DIRS)}（实际首段：{first_segment!r}）"
        )
    return v


# ---------------------------------------------------------------------------
# 模型文件来源信息（model_files[].source）
# ---------------------------------------------------------------------------

class ModelSourceSchema(BaseModel):
    """模型文件来源信息块（manifest.models.model_files[].source）。

    Attributes:
        type: 来源类型（必填）。huggingface / modelscope / url / local。
        repo_id: 仓库 ID（type=huggingface/modelscope 时必填）。如 "unsloth/ERNIE-Image-GGUF"。
        filename: 仓库内文件路径（选填，任务 64 起）。省略表示整仓 / subdir 子目录下载。
        url: 直接下载 URL（type=url 时必填，type=huggingface/modelscope 时可选镜像）。
        mirror_url: 镜像下载 URL（选填）。如 hf-mirror.com 镜像。
        subdir: 仓库内子目录（v0.2.3 新增，选填）。
            用于 diffusers 配置组件（如 vae / text_encoder / scheduler / pe / tokenizer），
            指明文件在仓库中的子目录路径。
            如 "vae" 表示文件在 "Baidu/ERNIE-Image/vae/" 下。
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., min_length=1)
    repo_id: str | None = None
    filename: str | None = None
    url: str | None = None
    mirror_url: str | None = None
    # v0.2.3 新增字段（ERNIE-Image 改造）
    subdir: str | None = None
    # 任务 64：引擎私有扩展槽（如 ernie 的 convert_required / convert_script）
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _MODEL_SOURCE_TYPES:
            raise ValueError(
                f"source.type {v!r} 无效（必须是 {sorted(_MODEL_SOURCE_TYPES)} 之一）"
            )
        return v

    @field_validator("url", "mirror_url")
    @classmethod
    def _validate_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL {v!r} 必须以 http:// 或 https:// 开头")
        return v

    @model_validator(mode="after")
    def _validate_repo_required(self) -> ModelSourceSchema:
        """type=huggingface/modelscope 时 repo_id 必填（filename 可选）。

        任务 64（Schema 治理）：filename 改可选 —— 省略表示整仓 / 子目录（subdir）
        下载，与 huggingface_hub.snapshot_download(allow_patterns=...) 的整仓语义一致；
        整仓引擎（ltx25 / minimax 等）不再被判错。
        """
        if self.type in ("huggingface", "modelscope"):
            if not self.repo_id:
                raise ValueError(
                    f"source.type={self.type!r} 时 source.repo_id 必填"
                )
        elif self.type == "url":
            if not self.url:
                raise ValueError(
                    "source.type='url' 时 source.url 必填"
                )
        return self


# ---------------------------------------------------------------------------
# 单个模型文件声明（model_files[]）
# ---------------------------------------------------------------------------

class ModelFileSchema(BaseModel):
    """单个模型文件声明（manifest.models.model_files[]）。

    v0.2.4 变更：
        - 移除 sha256 字段（完整性校验改为 size_bytes 单一策略）

    v0.2.13 变更：
        - 新增 key_files 字段（仅 format=marker 时有意义）：声明标志文件代表的关键权重
          用于预检整仓就绪性（标志文件存在 ≠ 整仓权重齐全）

    v0.4.x 变更：
        - 新增 exclusive_group 字段（互斥组标识，多 variant 引擎"下载一个就能跑"）
        - 新增 required_for 字段（此文件服务的 variant name 列表）
        - 两者均为可选，向后兼容；不填则视为共享文件（所有 variant 都需要）

    Attributes:
        id: 文件唯一标识（必填）。如 "main-q4-k-m" / "mmproj-qwen3.6-f16"。
        role: 文件角色（必填，13 项枚举之一）。
            main / mmproj / vae / clip / t5 / clip_vision /
            lora / controlnet / ipadapter / embedding / upscale / config / tokenizer
        filename: 文件名（必填）。如 "ernie-image-Q4_K_M.gguf"。
        target_dir: 目标目录（必填，必须以 12 类顶层开头）。
            如 "image/checkpoints/ernie-image"。
        size_bytes: 文件大小字节数（选填）。
        precision: 精度标识（选填）。如 "q4_k_m" / "fp16" / "bf16"。
        format: 文件格式（选填）。safetensors / gguf / pt / bin / marker。
            marker 表示"标志文件"——文件本身仅几 KB，但代表整个 HuggingFace 仓库
            （实际需 snapshot_download 整仓，约数 GB）。
        key_files: 关键权重文件列表（选填，仅 format=marker 时有意义）。
            声明标志文件代表的关键权重相对路径（相对于 target_dir）。
            预检时不仅检查标志文件存在，还要检查所有 key_files 存在。
            如 ["unet/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.bin"]
        exclusive_group: 互斥组标识（选填，v0.4.x 新增）。
            同组文件只需下载其中一个即可运行某个 variant。
            如 "ernie-image-main" / "minimax-h3-transformer" / "llama-gguf"。
            null 或不填表示不参与互斥（共享文件，所有 variant 都需要）。
            命名规范：{引擎简称}-{role}，如 "ernie-image-main"。
        required_for: 此文件服务的 variant name 列表（选填，v0.4.x 新增）。
            如 ["ernie-image-gguf-q4-k-m"] 或 ["H3-FL2VA-pruned-bf16", "H3-Ref2VA-pruned-bf16"]。
            空列表或省略表示所有 variant 都需要（共享文件）。
            用于前端"按 variant 下载"模式展示该 variant 需要的文件清单。
        source: 来源信息块（选填，含 HF/ModelScope repo_id + filename + url）。
        description: 描述（选填）。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    target_dir: str = Field(..., min_length=1)
    size_bytes: int | None = None
    precision: str | None = None
    format: str | None = None
    key_files: list[str] | None = None  # v0.2.13+ 新增
    exclusive_group: str | None = None  # v0.4.x 新增：互斥组标识
    required_for: list[str] | None = None  # v0.4.x 新增：服务的 variant name 列表
    # 任务 64（Schema 治理）新增：可选组件声明。
    # 类型 ⊥ 可选性 —— role 表达资产类型，optional 表达「缺失不阻断启动」，
    # 取代此前把可选性塞进 role 枚举（role: 'optional'）的越权用法。
    optional: bool = False
    # 任务 64 新增：该文件作为「可选依赖」所服务的 variant name 列表（ltx25 既有用法）。
    optional_dependency_for: list[str] = Field(default_factory=list)
    source: ModelSourceSchema | None = None
    description: str = ""
    # 任务 64 新增：引擎私有扩展槽（其余未知键仍按 extra="forbid" 报错）
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_files[].id 不能为空")
        return v.strip()

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        # v0.4.4+：先做别名规范化（如 weights → main），再校验枚举。
        # 这样老版本 manifest 中的 `role: weights` 可以跨版本导入。
        v = normalize_model_file_role(v)
        if v not in _MODEL_FILE_ROLE_VALUES:
            raise ValueError(
                f"model_files[].role {v!r} 无效（必须是 {len(_MODEL_FILE_ROLE_VALUES)} 项枚举之一，"
                f"或 {len(_MODEL_FILE_ROLE_ALIASES)} 个历史别名之一："
                f"枚举={sorted(_MODEL_FILE_ROLE_VALUES)}，"
                f"别名={dict(_MODEL_FILE_ROLE_ALIASES)}）"
            )
        return v

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_files[].filename 不能为空")
        return v.strip()

    @field_validator("target_dir")
    @classmethod
    def _validate_target_dir(cls, v: str) -> str:
        return _validate_model_top_level_dir(v, "model_files[].target_dir")

    @field_validator("size_bytes")
    @classmethod
    def _validate_size_bytes(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v < 0:
            raise ValueError(f"model_files[].size_bytes {v} 不能为负数")
        return v

    @field_validator("precision", "format")
    @classmethod
    def _validate_non_empty_str(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            return None
        return v.strip()


# ---------------------------------------------------------------------------
# 模型变体（variants[]）
# ---------------------------------------------------------------------------

class VariantCtxSchema(BaseModel):
    """per-variant ctx 配置（v0.3.3+ 新增，主要用于 llama_cpp 引擎）。

    声明每个 variant 的上下文长度（ctx-size）配置，供前端 UI 渲染滑块和
    引擎启动时生成 INI 文件使用。

    GGUF 元数据中的 `general.context_length`（即 n_ctx_train）是模型训练时的
    最大上下文长度，本字段的 `max` 不应超过该值。

    Attributes:
        recommended: 推荐的 ctx-size（默认值，平衡显存和性能）。如 32768。
        min: 最小允许的 ctx-size（防止过小导致功能异常）。如 4096。
        max: 最大允许的 ctx-size（受限于显存和 n_ctx_train）。如 262144。
    """

    model_config = ConfigDict(extra="forbid")

    recommended: int = Field(..., ge=1)
    min: int = Field(..., ge=1)
    max: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _validate_range(self) -> VariantCtxSchema:
        """校验 min <= recommended <= max。"""
        if self.min > self.recommended:
            raise ValueError(
                f"ctx.min ({self.min}) 不能大于 ctx.recommended ({self.recommended})"
            )
        if self.recommended > self.max:
            raise ValueError(
                f"ctx.recommended ({self.recommended}) 不能大于 ctx.max ({self.max})"
            )
        return self


class VariantSchema(BaseModel):
    """模型变体声明（manifest.models.variants[]）。

    替代 v0.2 的 mmproj_filename 隐式关联，v0.2.1 起使用 model_file_ids 显式关联。

    v0.3.3 新增字段：
        - ctx: per-variant 上下文长度配置（主要用于 llama_cpp 引擎的 --ctx-size 参数）
            含 recommended / min / max 三个值，前端 UI 渲染滑块使用

    Attributes:
        name: 变体名（必填）。如 "ernie-image-gguf-q4-k-m" / "qwen3.6-35b-a3b-q6-k"。
        alias: 短别名（选填，支持字符串或列表）。如 "ernie-q4" / ["qwen3.6-unc", "qwen-unc", "unc-q6"]（API model 字段可用）。
            v0.3.8+ 支持列表形式，用于为一个 variant 配置多个短别名，解决工具输入长度限制（32 字符）问题。
        base_model: 基座模型族（选填）。如 "ernie-image-sft" / "qwen3.6-35b-a3b"。
        precision: 精度（必填）。如 "q4_k_m" / "fp16" / "bf16"。
        loader: 加载器（选填）。gguf / diffusers / transformers。
        vram_required: 显存需求（选填）。如 "8GB" / "22GB"。
        path: 完整路径（必填，相对 models_root）。如 "image/checkpoints/ernie-image/ernie-image-Q4_K_M.gguf"。
        model_file_ids: 关联文件 ID 列表（v0.2.1 新增，替代 mmproj_filename）。
            如 ["main-q4-k-m"] / ["qwen3.6-q6-k-gguf", "mmproj-qwen3.6-f16-gguf"]。
        uv_env: uv 环境目录（选填）。如 "envs/image-ernie-image-gguf"。
        default_steps: 默认采样步数（选填）。
        default_guidance: 默认引导系数（选填）。
        supports_pe: 是否支持 PE 编码（选填，默认 False）。
        supports_mmproj: 是否支持多模态投影（选填，默认 False）。
        ctx: per-variant 上下文长度配置（v0.3.3+ 新增，选填）。
            含 recommended / min / max 三个值，主要用于 llama_cpp 引擎。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    alias: list[str] | str | None = None
    base_model: str | None = None
    precision: str = Field(..., min_length=1)
    loader: str | None = None
    vram_required: str | None = None
    path: str = Field(..., min_length=1)
    model_file_ids: list[str] = Field(default_factory=list)
    uv_env: str | None = None
    default_steps: int | None = None
    default_guidance: float | None = None
    supports_pe: bool = False
    supports_mmproj: bool = False
    # v0.3.3+ 新增：per-variant ctx 配置（主要用于 llama_cpp 引擎）
    ctx: VariantCtxSchema | None = None
    # ===== 任务 64（Schema 治理）：由引擎私有扩展「提升」为平台声明字段 =====
    description: str | None = None         # 变体说明（UI 展示）
    ram_required_gb: int | None = None     # 内存需求（GB，与 vram_required 区分；桌面 UI 消费）
    disk_required_gb: float | None = None  # 磁盘体积（GB）
    low_vram_note: str | None = None       # 低显存 ⚠️ 提示（仅低档非空，桌面 UI 渲染）
    # 方案 ID 列表（任务 64 决策 2：全平台统一），与 models.schemes[].id 双向关联
    schemes: list[str] = Field(default_factory=list)
    # 引擎私有扩展槽（如 strata 的 mtp / vision、foleycrafter 的 default_*_scale）
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variants[].name 不能为空")
        return v.strip()

    @field_validator("precision")
    @classmethod
    def _validate_precision(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variants[].precision 不能为空")
        return v.strip()

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variants[].path 不能为空")
        return v.strip()

    @field_validator("default_steps")
    @classmethod
    def _validate_default_steps(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v < 1:
            raise ValueError(f"variants[].default_steps {v} 必须 >= 1")
        return v

    @field_validator("default_guidance")
    @classmethod
    def _validate_default_guidance(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if v < 0:
            raise ValueError(f"variants[].default_guidance {v} 不能为负数")
        return v


# ---------------------------------------------------------------------------
# 模型目录布局（model_layout）
# ---------------------------------------------------------------------------

class ModelLayoutSchema(BaseModel):
    """模型目录布局（manifest.models.model_layout）。

    声明该引擎在 models/ 下的目录布局，ModelPlacementChecker 启动时强制校验。

    Attributes:
        base: 基座目录（必填，必须以 12 类顶层开头）。
            如 "image/checkpoints/ernie-image"。
        lora: LoRA 目录（选填，必须以 12 类顶层开头）。
            如 "image/loras/ernie-image"。
        controlnet: ControlNet 目录（选填）。
        embeddings: Textual Inversion 目录（选填）。
        ipadapter: IPAdapter 目录（选填）。
        upscale: 超分模型目录（选填）。
        vae: VAE 目录（v0.2.3 新增，选填，必须以 12 类顶层开头）。
            如 "image/vae/ernie-image"。
        scheduler: Scheduler 目录（v0.2.3 新增，选填）。
            如 "image/scheduler/ernie-image"。
        pe: Prompt Enhancer 目录（v0.2.3 新增，选填）。
            如 "image/pe/ernie-image"。
        text_encoder: Text Encoder 目录（v0.2.3 新增，选填，通常在 shared 下）。
            如 "shared/text_encoders/ernie-image"。
        tokenizer: Tokenizer 目录（v0.2.3 新增，选填，通常在 shared 下）。
            如 "shared/tokenizer/ernie-image"。
    """

    model_config = ConfigDict(extra="forbid")

    base: str = Field(..., min_length=1)
    lora: str | None = None
    controlnet: str | None = None
    embeddings: str | None = None
    ipadapter: str | None = None
    upscale: str | None = None
    # v0.2.3 新增字段（ERNIE-Image 改造）
    vae: str | None = None
    scheduler: str | None = None
    pe: str | None = None
    text_encoder: str | None = None
    tokenizer: str | None = None

    @field_validator("base")
    @classmethod
    def _validate_base(cls, v: str) -> str:
        return _validate_model_top_level_dir(v, "model_layout.base")

    @field_validator("lora")
    @classmethod
    def _validate_lora(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.lora")

    @field_validator("controlnet")
    @classmethod
    def _validate_controlnet(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.controlnet")

    @field_validator("embeddings")
    @classmethod
    def _validate_embeddings(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.embeddings")

    @field_validator("ipadapter")
    @classmethod
    def _validate_ipadapter(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.ipadapter")

    @field_validator("upscale")
    @classmethod
    def _validate_upscale(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.upscale")

    # v0.2.3 新增字段校验
    @field_validator("vae")
    @classmethod
    def _validate_vae(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.vae")

    @field_validator("scheduler")
    @classmethod
    def _validate_scheduler(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.scheduler")

    @field_validator("pe")
    @classmethod
    def _validate_pe(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.pe")

    @field_validator("text_encoder")
    @classmethod
    def _validate_text_encoder(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.text_encoder")

    @field_validator("tokenizer")
    @classmethod
    def _validate_tokenizer(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_model_top_level_dir(v, "model_layout.tokenizer")


# ---------------------------------------------------------------------------
# LoRA 扫描配置（loras）
# ---------------------------------------------------------------------------

class LoraScanConfigSchema(BaseModel):
    """LoRA 扫描配置（manifest.models.loras）。

    声明 LoRA 目录的扫描规则，引擎启动时自动扫描填充 available_loras。

    Attributes:
        scan_dir: 扫描目录（必填，必须以 12 类顶层开头）。
            如 "image/loras/ernie-image"。
        format: 文件格式（选填）。safetensors / pt / gguf。
        runtime_switch: 是否支持运行时切换（选填，默认 False）。
    """

    model_config = ConfigDict(extra="forbid")

    scan_dir: str = Field(..., min_length=1)
    format: str | None = None
    runtime_switch: bool = False

    @field_validator("scan_dir")
    @classmethod
    def _validate_scan_dir(cls, v: str) -> str:
        return _validate_model_top_level_dir(v, "loras.scan_dir")

    @field_validator("format")
    @classmethod
    def _validate_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            return None
        return v.strip()


# ---------------------------------------------------------------------------
# 下载器配置（downloader）
# ---------------------------------------------------------------------------

class DownloaderConfigSchema(BaseModel):
    """下载器配置（manifest.models.downloader）。

    管理 diffusers 配置组件（vae/text_encoder/scheduler/pe 等）的下载策略。
    GGUF 主模型下载仍走 model_files 字段（由 core 层 ModelDownloader 处理）。

    v0.2.4 变更：
        - 移除 shared_subdirs / partial_shared_subdirs / dedup_strategy 字段
          （SharedBlobs 去重机制已废弃）
        - 移除 auto_migrate_legacy_cache 字段（迁移脚本 scripts/migrate_ernie_image_models.py 已删除）

    Attributes:
        hf_endpoint: HF 镜像端点（选填，已弃用，由 base 层 DEFAULT_HF_MIRRORS 统一管理）。
        config_repos: diffusers 配置仓库映射（base_model -> repo_id）。如 {"ernie-image": "Baidu/ERNIE-Image"}。
        modelscope_repos: ModelScope 镜像仓库映射（base_model -> repo_id）。
        allow_patterns: 下载白名单（glob 模式列表）。
        ignore_patterns: 下载黑名单（glob 模式列表）。
        max_workers: 并发下载数（默认 1，单线程避免镜像限速）。
    """

    model_config = ConfigDict(extra="forbid")

    hf_endpoint: str | None = None
    config_repos: dict[str, str] = Field(default_factory=dict)
    modelscope_repos: dict[str, str] = Field(default_factory=dict)
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)
    max_workers: int = 1

    @field_validator("hf_endpoint")
    @classmethod
    def _validate_hf_endpoint(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"downloader.hf_endpoint {v!r} 必须以 http:// 或 https:// 开头"
            )
        return v.strip()

    @field_validator("max_workers")
    @classmethod
    def _validate_max_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"downloader.max_workers {v} 必须 >= 1"
            )
        return v


# ---------------------------------------------------------------------------
# 工作流模板配置（pipeline_template）
# ---------------------------------------------------------------------------

class PipelineTemplateConfigSchema(BaseModel):
    """工作流模板配置（manifest.models.pipeline_template）。

    把引擎抽象为 ComfyUI 式工作流：主模型槽 + 配套组件槽 + LoRA 槽。
    主模型可切换（variant 机制），配套组件跨 base_model 共享，LoRA 运行时切换。

    Attributes:
        main_model: 主模型槽配置（选填）。
            含 architecture / switchable / switch_via 字段。
        components: 配套组件列表（默认空）。
            每项含 name / shared / partial / shared_files / unique_files / source 字段。
        lora: LoRA 槽配置（选填）。
            含 scan_dir / format / runtime_switch 字段。
    """

    model_config = ConfigDict(extra="forbid")

    main_model: dict[str, Any] | None = None
    components: list[dict[str, Any]] = Field(default_factory=list)
    lora: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# manifest.models 块容器（v0.2.1）
# ---------------------------------------------------------------------------

class SchemeSchema(BaseModel):
    """引擎方案声明（manifest.models.schemes[]，任务 64 决策 2 新增）。

    平台级统一字段：声明引擎提供的运行方案（精度 / 显存档位 / 模型来源），
    供桌面 UI / 网关 / MCP 统一感知与切换；与 variants[].schemes 双向关联。
    引擎私有的方案实现细节（acceleration / inference / block_swap 等）留引擎侧
    schemes/*.yaml，不进 manifest。

    Attributes:
        id: 方案 ID（必填）。如 "A" / "T4" / "F16"。
        name: 方案名（必填）。如 "24G 显存流畅版"。
        description: 方案说明（选填）。
        vram_required: 显存需求（选填）。如 "24GB" / "48GB+"。
        precision: 精度（选填）。如 "sdnq_uint4" / "bf16"。
        model_repo: 模型来源仓库（选填）。如 "OzzyGT/MiniMax_H3_sdnq_dynamic_4bit"。
        model_subdir: 模型子目录（选填）。如 "sdnq_format"。
        extra: 引擎私有扩展槽。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str | None = None
    vram_required: str | None = None
    precision: str | None = None
    model_repo: str | None = None
    model_subdir: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelsBlockSchema(BaseModel):
    """manifest.yaml 的 models 块容器（v0.2.1 新增）。

    v0.2.1 起 manifest.yaml 必填 models 块（schema 层 Optional 兼容旧 manifest，
    EngineImporter 导入时强制必填）。

    详见 PR/architecture.md §16.4。

    Attributes:
        model_layout: 模型目录布局（必填，含 base / lora / controlnet 等）。
        variants: 模型变体列表（v0.2.1 可选，v0.3 强制必填）。
        model_files: 模型文件清单（v0.2.1 可选，v0.3 强制必填）。
        loras: LoRA 扫描配置（选填）。
        downloader: 下载器配置（选填，diffusers 配置组件下载策略）。
        pipeline_template: 工作流模板配置（选填，从 registry 片段迁移）。

    跨字段校验：
        - variants[].model_file_ids 引用的 id 必须在 model_files[].id 中存在
          （由 EngineImporter 校验，schema 层不强制避免循环依赖）
        - default_variant 引用必须在 variants[].name 中存在
          （v0.4.4+ 新增校验，避免指向不存在的 variant）
    """

    model_config = ConfigDict(extra="forbid")

    model_layout: ModelLayoutSchema
    # v0.4.4+：从 v0.3.0 漏掉的字段，补充进 schema 防止严格校验失败
    # （错误信息原本是"Extra inputs are not permitted"）
    default_variant: str | None = None
    variants: list[VariantSchema] = Field(default_factory=list)
    model_files: list[ModelFileSchema] = Field(default_factory=list)
    loras: LoraScanConfigSchema | None = None
    downloader: DownloaderConfigSchema | None = None
    pipeline_template: PipelineTemplateConfigSchema | None = None
    # ===== 任务 64（决策 2：全平台统一）：方案列表 =====
    # 由「引擎私有 / models.extra」提升为平台声明；与 variants[].schemes 双向关联。
    schemes: list[SchemeSchema] = Field(default_factory=list)
    # 引擎私有扩展槽
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_model_file_ids_unique(self) -> ModelsBlockSchema:
        """校验 model_files[].id 在块内唯一。"""
        seen: set[str] = set()
        for f in self.model_files:
            if f.id in seen:
                raise ValueError(
                    f"model_files[].id {f.id!r} 重复（id 必须在块内唯一）"
                )
            seen.add(f.id)
        return self

    @model_validator(mode="after")
    def _validate_default_variant_exists(self) -> ModelsBlockSchema:
        """v0.4.4+：校验 default_variant 引用必须指向一个真实 variant。"""
        if not self.default_variant:
            return self
        variant_names = {v.name for v in self.variants}
        if self.default_variant not in variant_names:
            raise ValueError(
                f"models.default_variant {self.default_variant!r} 未在 variants[].name 中找到。"
                f"现有 variants: {sorted(variant_names)}。"
                f"原因：BaseEngine.infer 会在用户未指定 variant 时按 default_variant 加载，"
                f"指向不存在的 name 会导致 silent fallback 或 启动失败。"
            )
        return self

    @model_validator(mode="after")
    def _validate_variant_names_unique(self) -> ModelsBlockSchema:
        """校验 variants[].name 在块内唯一。"""
        seen: set[str] = set()
        for v in self.variants:
            if v.name in seen:
                raise ValueError(
                    f"variants[].name {v.name!r} 重复（name 必须在块内唯一）"
                )
            seen.add(v.name)
        return self

    def get_model_file_by_id(self, file_id: str) -> ModelFileSchema | None:
        """按 id 查询 model_files 中的文件声明。

        Args:
            file_id: 文件 ID。

        Returns:
            ModelFileSchema 实例或 None（未找到时）。
        """
        for f in self.model_files:
            if f.id == file_id:
                return f
        return None

    def get_variant_by_name(self, name: str) -> VariantSchema | None:
        """按 name 查询 variants 中的变体声明。

        Args:
            name: 变体名。

        Returns:
            VariantSchema 实例或 None（未找到时）。
        """
        for v in self.variants:
            if v.name == name:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        """转换为普通 dict（不含 None 字段）。"""
        return self.model_dump(exclude_none=True)


# ===========================================================================
# manifest.yaml env_config 块（v0.2.5+ 引擎环境设置面板）
# ===========================================================================

class CpuOffloadLevelSchema(BaseModel):
    """单个 CPU offload 等级的声明（v0.4.x+ 引入）。

    详见 PR/development-guide.md §37.2.1 CpuOffloadSchema 设计。

    Attributes:
        name: 等级显示名（如 "全 GPU"）。
        description: 等级说明。
        min_vram_gb: 该等级所需的最小显存（GB），用于 auto 推断。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    min_vram_gb: int = 0


class CpuOffloadSchema(BaseModel):
    """manifest.env_config.cpu_offload 块的 schema（v0.4.x+ 引入）。

    实现"显存不够内存补"：GPU 显存不足时，将不活跃的模型层临时存放在
    系统内存（RAM），需要计算时再搬入 GPU。

    详见 PR/development-guide.md §37.2.1。

    Attributes:
        supported: 该引擎是否支持 CPU offload。
        default_level: 默认等级（auto / L0 / L1 / L2 / L3）。
        levels: 等级声明字典，key 必须是 L0 / L1 / L2 / L3 之一。
    """

    model_config = ConfigDict(extra="forbid")

    supported: bool = False
    default_level: str = "auto"
    levels: dict[str, CpuOffloadLevelSchema] = Field(default_factory=dict)


class EnvConfigSchema(BaseModel):
    """manifest.yaml 的 env_config 块（v0.2.5+ 新增）。

    用于声明引擎环境的依赖与配置，供 EngineEnvManager 读取以决定：
        1. 是否检测 torch 版本（requires_torch）
        2. 默认 CUDA 版本（default_cuda_version）
        3. 支持的 CUDA 版本列表（supported_cuda_versions）
        4. Python 版本约束（requires_python）
        5. torch 版本约束（torch_version_constraint）
        6. CPU offload 等级支持（cpu_offload，v0.4.x+ 新增）

    详见 engine-env-settings-analysis.md §3.2 manifest 扩展设计
        和 PR/development-guide.md §37.2.1。

    Attributes:
        requires_torch: 引擎是否依赖 torch。
            True：调用 TorchVersionManager.detect_current 检测当前 torch 版本。
            False：直接返回 not_required，跳过检测（如 Llama.cpp 是 C++ 二进制包装层）。
            默认 True（向后兼容 v0.1/v0.2 旧 manifest）。
        torch_version_constraint: torch 版本约束（PEP 440 格式，如 ">=2.3.0"）。
            用于在环境设置面板显示约束，并作为 uv sync 校验依据。
            None 时不显示约束。
        default_cuda_version: 默认 CUDA 版本标签（如 "cu128"）。
            供 TorchCudaInstaller 在用户未选择时使用。
            默认 "cu128"（PyTorch 官方推荐版本）。
        supported_cuda_versions: 支持的 CUDA 版本列表（前端下拉框选项）。
            默认 ["cu121", "cu124", "cu126", "cu128"]（PyTorch 官方支持版本）。
        requires_python: Python 版本约束（PEP 440 格式，如 ">=3.10,<3.13"）。
            用于在环境设置面板显示约束，并作为 uv sync 校验依据。
            None 时不显示约束。
        notes: 备注（在环境设置面板底部显示）。
            用于补充说明引擎环境的特殊要求（如 "Llama.cpp 是 C++ 二进制包装层，不依赖 torch"）。
        cpu_offload: v0.4.x+ 新增：CPU offload 等级支持配置。
            None 时等价于 supported=False（向后兼容旧 manifest）。
    """

    model_config = ConfigDict(extra="forbid")

    requires_torch: bool = True
    torch_version_constraint: str | None = None
    default_cuda_version: str = "cu128"
    supported_cuda_versions: list[str] = Field(
        default_factory=lambda: ["cu121", "cu124", "cu126", "cu128"]
    )
    requires_python: str | None = None
    notes: str = ""
    recommended_versions: RecommendedVersionsSchema | None = None
    # v0.4.x+ 新增：GPU+内存混合计算（CPU Offload）等级支持
    cpu_offload: CpuOffloadSchema | None = None
    # v0.3.6+ 新增：设备策略显存降级支持（方案 A）
    # True 时 core 显存预检失败可自动降级 gpu+cpu 混合模式启动
    # （llama-server -ngl auto --fit on 按实际显存自动分配 GPU/CPU 层数）。
    # 未声明（False）的引擎维持硬门禁（显存不足直接报错，避免 load 阶段 OOM）
    # 详见 ANALYSIS-TASKS/LLAMACPP_VRAM_STARTUP_FAILURE_ANALYSIS.md
    supports_device_policy: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为普通 dict（不含 None 字段）。"""
        return self.model_dump(exclude_none=True)


class RecommendedVersionsSchema(BaseModel):
    """manifest.yaml 的 env_config.recommended_versions 块（v0.2.12+ 新增）。

    用于声明该引擎测试过的 torch + torchvision + torchaudio 版本组合。
    平台切换 torch 时优先使用引擎推荐版本，并同步切换匹配的 torchvision/torchaudio，
    避免 torch 与 torchvision 版本错配导致 _C.pyd 加载失败。

    优先级：
        1. 引擎 manifest 推荐版本（本字段）
        2. 平台内置 torch→torchvision 对应表（torch_compat.py）
        3. PyTorch 官方源最新版本

    Attributes:
        torch: 推荐的 torch 版本号（如 "2.11.0"），不含 cuda 标签。
        torchvision: 推荐的 torchvision 版本号（如 "0.26.0"），不含 cuda 标签。
            引擎不使用 torchvision 时设为 None。
        torchaudio: 推荐的 torchaudio 版本号（如 "2.11.0"），不含 cuda 标签。
            引擎不使用 torchaudio 时设为 None。
        reason: 推荐理由（如 "FoleyCrafter 测试通过的稳定版本组合"）。
    """

    model_config = ConfigDict(extra="forbid")

    torch: str
    torchvision: str | None = None
    torchaudio: str | None = None
    reason: str = ""


# ===========================================================================
# manifest.yaml schema 顶层
# ===========================================================================

class PackageInfo(BaseModel):
    """引擎包完整性信息（可选，manifest_schema_version >= 0.4）。

    ``aireliable-engine pack`` 会自动写入 ``sha256``；``signature`` 预留给
    后续签名机制（当前不强制）。

    Attributes:
        sha256: 整包规范化内容哈希（64 位十六进制）。
        signature: 可选 Ed25519 签名（base64），未签名为 None。
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str | None = None
    signature: str | None = None

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("package.sha256 必须是 64 位十六进制字符串")
        return v


class ManifestSchema(BaseModel):
    """manifest.yaml schema 顶层模型。

    v0.1 必填字段：
        - engine_id: 本地引擎 ID
        - engine_name: 显示名
        - engine_version: 语义化版本
        - aireliable_compat: AIReliable 兼容版本
        - uv_env_name: uv 环境目录名
        - export_mode: 导出模式（source / bundle）

    v0.1 可选字段：
        - created_at: ISO8601 导出时间
        - description: 引擎描述
        - checksum: 整包 SHA256
        - exported_from_os: bundle 模式必填（windows / linux / darwin）

    v0.2 新增字段（全部可选，但推荐填写）：
        - engine_uid: 全局唯一 ID
        - paradigm: 引擎范式
        - sdk_version: SDK 版本约束
        - author: 作者信息块
        - source: 来源信息块
        - upgrade: 升级信息块
        - node_packages: 依赖节点包列表

    v0.2.1 新增字段（manifest.models 块，schema 层 Optional 兼容旧 manifest，
    EngineImporter 导入时强制必填）：
        - models: 模型分类规范化块（含 model_layout / variants / model_files 等）

    v0.2.5+ 新增字段（manifest.env_config 块，全部可选）：
        - env_config: 引擎环境配置块（含 requires_torch / default_cuda_version 等）
            供 EngineEnvManager 读取以决定是否检测 torch、默认 CUDA 版本等。
            缺失时按默认值处理（requires_torch=True / default_cuda_version="cu128"）。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # ===== v0.1 必填字段 =====
    engine_id: str = Field(..., min_length=1)
    engine_name: str = Field(..., min_length=1)
    engine_version: str = Field(..., min_length=1)
    aireliable_compat: str = Field(..., min_length=1)
    uv_env_name: str = Field(..., min_length=1)
    export_mode: str = "source"

    # ===== v0.1 可选字段 =====
    created_at: str | None = None
    description: str = ""
    checksum: str | None = None
    exported_from_os: str | None = None

    # ===== v0.2 新增字段 =====
    engine_uid: str | None = None
    # v0.2.5：paradigm 默认值 "workflow"，可选 "workflow" / "integrated"
    # 保留 Optional 是为了向后兼容旧 manifest（缺失时按 workflow 处理）
    paradigm: str = "workflow"
    sdk_version: str | None = None
    # ===== 任务 67（拆仓契约）：版本轴与开源元数据 =====
    # engine_api_version：引擎 HTTP 契约版本（破坏性变更才 +1），用于启动握手。
    engine_api_version: str = ENGINE_API_VERSION
    # manifest_schema_version：本 manifest 遵循的 schema 版本（MAJOR.MINOR）。
    manifest_schema_version: str = CURRENT_MANIFEST_SCHEMA_VERSION
    # license：SPDX 标识（顶层权威，author.license 保留兼容）。
    license: str | None = None
    # repository：本引擎源码仓地址（区别于 source 描述的上游来源）。
    repository: str | None = None
    # visibility：public 进公开 index；private 仅私有 index 分发。
    visibility: str = "public"
    # ===== manifest_schema_version 0.4 新增字段（全部可选）=====
    # entry：Python 模块路径，平台用 `python -m <entry> --port N` 启动。
    # 缺省时平台回退 `engines.<engine_id>.server`（仅兼容旧引擎）。
    entry: str | None = None
    # package：整包完整性与可选签名（pack 生成）。
    package: PackageInfo | None = None
    # v0.2.7：author 语义重新定义为"创建引擎的作者"（不再是第三方开源项目方）
    author: AuthorInfo | None = None
    source: SourceInfo | None = None
    upgrade: UpgradeInfo | None = None
    node_packages: list[NodePackageDep] = Field(default_factory=list)

    # ===== v0.2.7 新增字段（第三方开源引用）=====
    # 引擎引用的第三方开源模型/代码，由作者负责标注。
    # 缺失时按空列表处理（向后兼容旧 manifest）。
    third_party_references: list[ThirdPartyReference] = Field(default_factory=list)

    # ===== v0.2.1 新增字段 =====
    models: ModelsBlockSchema | None = None

    # ===== 任务 64（Schema 治理）：顶层引擎私有扩展槽 =====
    # 承载平台不感知的引擎私有顶层字段（如 ernie 的 custom_node_dependencies）；
    # 其余未知顶层键仍按 extra="forbid" 报错（拼写保护）。
    extra: dict[str, Any] = Field(default_factory=dict)

    # ===== v0.2.5+ 新增字段（引擎环境设置面板）=====
    # 声明引擎环境的依赖与配置，供 EngineEnvManager 读取。
    # 缺失时按 EnvConfigSchema 默认值处理（requires_torch=True 等）。
    env_config: EnvConfigSchema | None = None

    # ===== v0.1 字段校验 =====

    @field_validator("engine_id")
    @classmethod
    def _validate_engine_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("engine_id 不能为空")
        v = v.strip()
        if not _ENGINE_ID_PATTERN.match(v):
            raise ValueError(
                f"engine_id {v!r} 格式无效（必须以小写字母开头，"
                f"仅含小写字母 / 数字 / 下划线，长度 2-64）"
            )
        return v

    @field_validator("engine_name")
    @classmethod
    def _validate_engine_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("engine_name 不能为空")
        return v.strip()

    @field_validator("engine_version")
    @classmethod
    def _validate_engine_version(cls, v: str) -> str:
        if not v:
            raise ValueError("engine_version 不能为空")
        if not _SEMVER_PATTERN.match(v):
            raise ValueError(
                f"engine_version {v!r} 不是合法的语义化版本号（如 0.2.0）"
            )
        return v

    @field_validator("entry")
    @classmethod
    def _validate_entry(cls, v: str | None) -> str | None:
        """校验 entry 为可 `python -m` 执行的模块路径。"""
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("entry 不能为空字符串（不需要时请删除该字段）")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*", v):
            raise ValueError(
                f"entry {v!r} 不是合法模块路径（示例：my_engine.server）"
            )
        return v

    @field_validator("aireliable_compat")
    @classmethod
    def _validate_aireliable_compat(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("aireliable_compat 不能为空")
        # 任务 67：从"非空字符串"升级为可解析区间，避免 0.2.14.8 这类非法号段。
        try:
            parse_specifier(v)
        except SpecifierError as e:
            raise ValueError(f"aireliable_compat {v!r} 不是合法版本区间: {e}") from e
        return v.strip()

    @field_validator("engine_api_version")
    @classmethod
    def _validate_engine_api_version(cls, v: str) -> str:
        v = str(v or "").strip()
        if not re.fullmatch(r"[1-9]\d*", v):
            raise ValueError(
                f"engine_api_version {v!r} 无效（必须是正整数，如 '1'）"
            )
        return v

    @field_validator("manifest_schema_version")
    @classmethod
    def _validate_manifest_schema_version(cls, v: str) -> str:
        v = str(v or "").strip()
        if not re.fullmatch(r"\d+\.\d+", v):
            raise ValueError(
                f"manifest_schema_version {v!r} 无效（必须是 MAJOR.MINOR，如 '0.3'）"
            )
        return v

    @field_validator("license")
    @classmethod
    def _validate_license(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            raise ValueError("license 不能为空字符串（不声明时请省略该字段）")
        return v

    @field_validator("repository")
    @classmethod
    def _validate_repository(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            raise ValueError("repository 不能为空字符串（不声明时请省略该字段）")
        if not (v.startswith("http://") or v.startswith("https://") or v.startswith("git@")):
            raise ValueError(
                f"repository {v!r} 无效（必须以 http:// / https:// / git@ 开头）"
            )
        return v

    @field_validator("visibility")
    @classmethod
    def _validate_visibility(cls, v: str) -> str:
        v = str(v or "").strip()
        if v not in ("public", "private"):
            raise ValueError(f"visibility {v!r} 无效（必须是 public 或 private）")
        return v

    @field_validator("uv_env_name")
    @classmethod
    def _validate_uv_env_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("uv_env_name 不能为空")
        return v.strip()

    @field_validator("export_mode")
    @classmethod
    def _validate_export_mode(cls, v: str) -> str:
        if v not in ("source", "bundle"):
            raise ValueError(
                f"export_mode {v!r} 无效（必须是 'source' 或 'bundle'）"
            )
        return v

    @field_validator("exported_from_os")
    @classmethod
    def _validate_exported_from_os(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in ("windows", "linux", "darwin"):
            raise ValueError(
                f"exported_from_os {v!r} 无效（必须是 windows / linux / darwin）"
            )
        return v

    @field_validator("checksum")
    @classmethod
    def _validate_checksum(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # 格式：sha256:<hex>
        if not v.startswith("sha256:"):
            raise ValueError(f"checksum {v!r} 必须以 'sha256:' 开头")
        hex_part = v[len("sha256:"):]
        if not re.match(r"^[a-f0-9]{64}$", hex_part):
            raise ValueError(
                f"checksum {v!r} 的 SHA256 哈希值无效（必须是 64 位十六进制）"
            )
        return v

    # ===== v0.2 字段校验 =====

    @field_validator("engine_uid")
    @classmethod
    def _validate_engine_uid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _ENGINE_UID_PATTERN.match(v):
            raise ValueError(
                f"engine_uid {v!r} 格式无效（必须是 "
                f"aireliable://<author>/<family>/<name>，"
                f"其中 author / family / name 仅含小写字母 / 数字 / 下划线）"
            )
        return v

    @field_validator("paradigm")
    @classmethod
    def _validate_paradigm(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in _PARADIGMS:
            raise ValueError(
                f"paradigm {v!r} 无效（必须是 {sorted(_PARADIGMS)} 之一）"
            )
        return v

    @field_validator("sdk_version")
    @classmethod
    def _validate_sdk_version(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            raise ValueError("sdk_version 不能为空字符串")
        # 任务 67：与 aireliable_compat 同步升级为可解析区间
        try:
            parse_specifier(v)
        except SpecifierError as e:
            raise ValueError(f"sdk_version {v!r} 不是合法版本区间: {e}") from e
        return v.strip()

    # ===== 跨字段校验 =====

    @model_validator(mode="after")
    def _validate_bundle_mode_requires_os(self) -> ManifestSchema:
        """export_mode=bundle 时 exported_from_os 必填。"""
        if self.export_mode == "bundle" and not self.exported_from_os:
            raise ValueError(
                "export_mode=bundle 时 exported_from_os 必填"
                "（windows / linux / darwin）"
            )
        return self

    @model_validator(mode="after")
    def _validate_workflow_paradigm_requires_uid(self) -> ManifestSchema:
        """v0.2.5：显式声明 paradigm=workflow 时强制要求 engine_uid。

        v0.2.5+ 移除 integrated / custom 范式，所有引擎必须为 workflow 范式。
        若 manifest 显式声明 paradigm=workflow（v0.2+ 包），则必须有 engine_uid
        （用于节点包 entry points 注册）。
        v0.1 老包缺失 paradigm 字段时不触发此校验（向后兼容）。
        """
        # 仅当 manifest 显式声明 paradigm=workflow 时才校验
        # （v0.1 老包没有 paradigm 字段，default=workflow 但不强制 engine_uid）
        if (
            self.model_fields_set
            and "paradigm" in self.model_fields_set
            and self.paradigm == "workflow"
            and not self.engine_uid
        ):
            raise ValueError(
                "paradigm=workflow 时 engine_uid 必填（用于节点包 entry points 注册）"
            )
        return self

    # ===== 转换方法 =====

    def to_dict(self) -> dict[str, Any]:
        """转换为普通 dict（不含 None 字段）。"""
        return self.model_dump(exclude_none=True)

    def to_yaml(self) -> str:
        """序列化为 yaml 字符串。"""
        return yaml.safe_dump(
            self.to_dict(),
            allow_unicode=True,
            sort_keys=False,
        )

    def get_author_name(self) -> str:
        """获取作者名（无 author 块时返回 'unknown'）。"""
        return self.author.name if self.author else "unknown"

    def get_license(self) -> str:
        """获取协议（无 author 块时返回 'UNKNOWN'）。"""
        return self.author.license if self.author else "UNKNOWN"

    def parse_engine_uid(self) -> tuple[str, str, str] | None:
        """解析 engine_uid 为 (author, family, name) 三元组。

        Returns:
            (author, family, name) 或 None（engine_uid 未设置时）。
        """
        if not self.engine_uid:
            return None
        # 格式：aireliable://<author>/<family>/<name>
        prefix = "aireliable://"
        if not self.engine_uid.startswith(prefix):
            return None
        parts = self.engine_uid[len(prefix):].split("/")
        if len(parts) != 3:
            return None
        return (parts[0], parts[1], parts[2])

    # ===== v0.2.1 兼容性检查 =====

    def is_v021_manifest(self) -> bool:
        """判断 manifest 是否为 v0.2.1 格式（含 models 块）。

        Args:
            无。

        Returns:
            True 表示是 v0.2.1 格式（含 models 块），False 表示是 v0.1/v0.2 格式。
        """
        return self.models is not None

    def validate_v021_strict(self) -> None:
        """v0.2.1 严格校验（EngineImporter 导入时调用）。

        校验规则（v0.2.4 简化后）：
            1. models 块必须存在
            2. models.variants[].model_file_ids 引用的 id 必须在 model_files[].id 中存在
            3. models.model_files[].target_dir 必须以 model_layout.base 开头
               （若 model_files[].target_dir != model_layout.base，需在 target_dir 声明）

        v0.2.4 变更：
            - 移除「model_files[].sha256 必填」校验（sha256 字段已废弃）

        Raises:
            ManifestSchemaError: 严格校验失败（含所有错误列表）。
        """
        if self.models is None:
            raise ManifestSchemaError(
                "v0.2.1 严格校验失败：models 块必填（缺失 manifest.models）",
                field_path="models",
            )

        errors: list[str] = []

        # 校验：variants[].model_file_ids 引用必须存在
        valid_ids = {f.id for f in self.models.model_files}
        for i, v in enumerate(self.models.variants):
            for j, fid in enumerate(v.model_file_ids):
                if fid not in valid_ids:
                    errors.append(
                        f"models.variants[{i}].model_file_ids[{j}] 引用 "
                        f"id={fid!r} 在 model_files 中不存在"
                    )

        if errors:
            msg = (
                f"v0.2.1 严格校验失败（{len(errors)} 处错误）：\n"
                + "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(errors))
            )
            raise ManifestSchemaError(msg, field_path="models")


# ===========================================================================
# 校验入口
# ===========================================================================

def _format_validation_error(e: ValidationError) -> ManifestSchemaError:
    """将 Pydantic ValidationError 转换为 ManifestSchemaError（含清晰路径）。"""
    errors = e.errors()
    if not errors:
        return ManifestSchemaError(str(e))

    messages: list[str] = []
    first_path = ""
    for err in errors:
        loc = err.get("loc", ())
        path = ".".join(str(x) for x in loc if x != "")
        msg = err.get("msg", "")
        typ = err.get("type", "")
        if not first_path:
            first_path = path
        suffix = f"（类型: {typ}）" if typ and typ != "value_error" else ""
        prefix = f"[{path}] " if path else ""
        messages.append(f"{prefix}{msg}{suffix}")

    if len(messages) == 1:
        message = messages[0]
    else:
        message = (
            f"manifest 校验失败（{len(messages)} 处错误）：\n"
            + "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(messages))
        )
    return ManifestSchemaError(message, field_path=first_path)


def validate_manifest_dict(data: dict[str, Any]) -> ManifestSchema:
    """校验 manifest dict。

    Args:
        data: manifest dict（已从 yaml 解析）。

    Returns:
        ManifestSchema 实例。

    Raises:
        ManifestSchemaError: 校验失败。

    Example:
        >>> schema = validate_manifest_dict({
        ...     "engine_id": "ernie_image",
        ...     "engine_name": "ERNIE-Image",
        ...     "engine_version": "0.2.0",
        ...     "aireliable_compat": ">=0.2.0",
        ...     "uv_env_name": "image-ernie-image-gguf",
        ... })
        >>> schema.engine_id
        'ernie_image'
    """
    if data is None:
        raise ManifestSchemaError("manifest 数据为 None")
    if not isinstance(data, dict):
        raise ManifestSchemaError(
            f"manifest 必须是 dict，实际 {type(data).__name__}",
        )
    try:
        return ManifestSchema.model_validate(data)
    except ValidationError as e:
        raise _format_validation_error(e) from e


def validate_manifest_yaml(yaml_text: str) -> ManifestSchema:
    """校验 manifest yaml 字符串。

    Args:
        yaml_text: yaml 字符串。

    Returns:
        ManifestSchema 实例。

    Raises:
        ManifestSchemaError: yaml 解析失败或 schema 校验失败。
    """
    if not isinstance(yaml_text, str):
        raise ManifestSchemaError(
            f"yaml_text 必须是 str，实际 {type(yaml_text).__name__}"
        )
    if not yaml_text.strip():
        raise ManifestSchemaError("yaml_text 为空字符串")

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ManifestSchemaError(f"yaml 解析失败：{e}") from e

    return validate_manifest_dict(data)


def is_v02_manifest(schema: ManifestSchema) -> bool:
    """判断 manifest 是否为 v0.2 格式（含 engine_uid）。

    Args:
        schema: 已校验的 ManifestSchema 实例。

    Returns:
        True 表示是 v0.2 格式（含 engine_uid），False 表示是 v0.1 格式。
    """
    return schema.engine_uid is not None


def is_v021_manifest(schema: ManifestSchema) -> bool:
    """判断 manifest 是否为 v0.2.1 格式（含 models 块）。

    Args:
        schema: 已校验的 ManifestSchema 实例。

    Returns:
        True 表示是 v0.2.1 格式（含 models 块），False 表示是 v0.1/v0.2 格式。

    详见：
        - PR/architecture.md §16.4 manifest.models 块
        - PR/development-guide.md §5.4-§5.5 模型文件预检与位置检测
    """
    return schema.models is not None


def validate_v021_manifest_strict(schema: ManifestSchema) -> None:
    """v0.2.1 严格校验入口（EngineImporter 导入时调用）。

    Args:
        schema: 已校验的 ManifestSchema 实例。

    Raises:
        ManifestSchemaError: 严格校验失败（含所有错误列表）。

    详见：
        - PR/architecture.md §16.4 字段强制策略
        - PR/development-guide.md §5.4 模型文件预检规范
    """
    schema.validate_v021_strict()

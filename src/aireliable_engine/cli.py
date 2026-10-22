"""``aireliable-engine`` 命令行入口。

四个子命令构成第三方引擎的本地开发闭环::

    aireliable-engine init my-engine      # 生成最小可运行引擎
    aireliable-engine validate            # 本地校验（与平台导入同源规则）
    aireliable-engine run --port 7100     # 按 manifest.entry 起 HTTP 服务
    aireliable-engine pack                # 产出可导入客户端的标准包

退出码：
    0  成功
    2  校验失败 / 参数错误
    3  运行失败（子进程异常退出、端口占用等）

所有子命令支持 ``--json`` 输出机器可读结果，便于 CI 集成。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    ManifestSchema,
    ManifestSchemaError,
    validate_manifest_yaml,
)
from .package import (
    MAX_FILE_BYTES,
    compute_content_hash,
    model_weight_files,
    oversized_files,
    write_package_json,
)

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_RUNTIME = 3

# engine_id 允许的形态，用于判断"目录名是否像一个 engine_id"
_ENGINE_ID_RE = re.compile(r"[a-z][a-z0-9_]{1,63}")

_CAPABILITY_TYPES = {
    "image": "image",
    "video": "video",
    "tts": "voice",
    "voice": "voice",
    "chat": "llm",
    "llm": "llm",
    "threed": "threed",
}


# ===========================================================================
# 通用工具
# ===========================================================================

def _engine_type_for(capability: str) -> str:
    """按能力推断引擎类型（manifest.engine_type 的取值）。"""
    cap = capability.lower()
    for prefix, engine_type in _CAPABILITY_TYPES.items():
        if cap.startswith(prefix):
            return engine_type
    return "image"


def _module_name(engine_id: str) -> str:
    """engine_id 转 Python 包名（下划线合法，连字符替换）。"""
    return re.sub(r"[^0-9a-zA-Z_]", "_", engine_id)


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestSchemaError(f"{path.name} 顶层必须是 YAML 映射（dict）")
    return data


def _load_manifest(root: Path) -> ManifestSchema:
    path = root / "manifest.yaml"
    if not path.is_file():
        raise ManifestSchemaError(f"未找到 manifest.yaml（目录：{root}）")
    return validate_manifest_yaml(path.read_text(encoding="utf-8"))


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for err in payload.get("errors", []):
        print(f"ERROR  {err}", file=sys.stderr)
    for warn in payload.get("warnings", []):
        print(f"WARN   {warn}", file=sys.stderr)
    for info in payload.get("info", []):
        print(f"OK     {info}")
    if payload.get("ok") and not payload.get("errors"):
        print(f"\n校验通过：{payload.get('engine_id', '?')}")


# ===========================================================================
# validate
# ===========================================================================

def _collect_workflow_files(root: Path) -> list[Path]:
    for name in ("workflows", "workflow"):
        candidate = root / name
        if candidate.is_dir():
            return sorted(candidate.glob("*.y*ml"))
    return []


def _check_entry_import(root: Path, entry: str) -> tuple[object | None, str | None]:
    """尝试导入 entry 模块；返回 (module, error)。"""
    sys.path.insert(0, str(root))
    try:
        module = __import__(entry, fromlist=["*"])
        return module, None
    except Exception as exc:  # noqa: BLE001 - 需要把任意导入错误反馈给开发者
        return None, f"entry 模块 {entry!r} 无法导入：{type(exc).__name__}: {exc}"
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def collect_validation(root: Path, *, strict: bool = False) -> dict[str, Any]:
    """执行全部校验，返回结构化结果。

    Returns:
        ``{"ok": bool, "engine_id": str|None, "errors": [...], "warnings": [...],
        "info": [...]}``
    """
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    # 1. manifest 结构
    try:
        schema = _load_manifest(root)
        engine_id = schema.engine_id
        info.append(f"manifest 校验通过（schema {CURRENT_MANIFEST_SCHEMA_VERSION}）")
    except ManifestSchemaError as exc:
        return {
            "ok": False,
            "engine_id": None,
            "errors": [str(exc)],
            "warnings": [],
            "info": [],
        }

    # 2. 必需文件 / 目录名
    # 仅当目录名本身符合 engine_id 命名时才提示不一致；
    # 模板/源码仓根目录（如 aireliable-engine-template）命名不受此限制。
    if _ENGINE_ID_RE.fullmatch(root.name) and root.name != engine_id:
        warnings.append(
            f"目录名 {root.name!r} 与 engine_id {engine_id!r} 不一致（pack 后不受影响）"
        )

    # 3. models 块与引用
    if schema.models is None:
        warnings.append("缺少 models 块：平台导入时会强制校验（旧引擎兼容期除外）")
    else:
        ids = {f.id for f in schema.models.model_files}
        for i, variant in enumerate(schema.models.variants):
            for fid in variant.model_file_ids:
                if fid not in ids:
                    errors.append(
                        f"models.variants[{i}].model_file_ids 引用 {fid!r} 在 model_files 中不存在"
                    )
        if schema.models.default_variant:
            names = {v.name for v in schema.models.variants}
            if schema.models.default_variant not in names:
                errors.append(
                    f"models.default_variant {schema.models.default_variant!r} "
                    f"不在 variants 中（现有：{sorted(names)}）"
                )

    # 4. entry
    entry = schema.entry
    entry_module = None
    if entry:
        entry_module, entry_error = _check_entry_import(root, entry)
        if entry_error:
            errors.append(entry_error)
        else:
            info.append(f"entry 可导入：{entry}")
    else:
        warnings.append(
            "未声明 manifest.entry：平台将回退 python -m engines.<engine_id>.server"
            "（新引擎建议显式声明）"
        )

    # 5. 工作流
    workflow_files = _collect_workflow_files(root)
    if not workflow_files:
        warnings.append("未找到 workflows/*.yaml（引擎将无法执行工作流）")
    # 导入内置节点库以触发 @register_node 注册（否则 NodeRegistry 为空）
    from . import nodes as _builtin_nodes  # noqa: F401
    from .workflow_schema import validate_workflow_yaml

    for wf in workflow_files:
        try:
            wf_schema = validate_workflow_yaml(wf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{wf.relative_to(root).as_posix()} 校验失败：{exc}")
            continue
        info.append(f"工作流通过：{wf.relative_to(root).as_posix()}")
        if entry_module is None:
            continue
        from .node_registry import list_nodes

        known = set(list_nodes())
        for node in wf_schema.nodes:
            if node.type not in known:
                errors.append(
                    f"{wf.name} 节点 {node.id!r} 的类型 {node.type!r} 未注册"
                    f"（可用：{', '.join(sorted(known)[:10])} ...）"
                )

    # 6. 环境 / 元数据
    if not (root / "env" / "pyproject.toml").is_file():
        warnings.append("缺少 env/pyproject.toml（客户端导入必需）")
    if not (root / "registry.yaml").is_file():
        warnings.append("缺少 registry.yaml（pack 必需）")
    if not (root / "LICENSE").is_file():
        warnings.append("缺少 LICENSE 文件")
    if not schema.license:
        warnings.append("manifest.license 未声明")
    if not schema.repository:
        warnings.append("manifest.repository 未声明（分发时建议填写）")

    # 7. 大文件 / 模型权重
    for path in oversized_files(root):
        errors.append(
            f"文件过大（>{MAX_FILE_BYTES // (1024 * 1024)}MB）："
            f"{path.relative_to(root).as_posix()}（模型权重应放平台 models/ 目录）"
        )
    for path in model_weight_files(root):
        errors.append(
            f"包内含模型权重：{path.relative_to(root).as_posix()}（请移出包外）"
        )

    if strict:
        errors.extend(f"[strict] {w}" for w in warnings)
        warnings = []

    return {
        "ok": not errors,
        "engine_id": engine_id,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.path or ".").resolve()
    if not root.is_dir():
        _emit({"ok": False, "errors": [f"目录不存在：{root}"], "warnings": [], "info": []},
              args.json)
        return EXIT_VALIDATION
    result = collect_validation(root, strict=args.strict)
    _emit(result, args.json)
    return EXIT_OK if result["ok"] else EXIT_VALIDATION


# ===========================================================================
# init（内置最小模板，离线可用）
# ===========================================================================

_APACHE_LICENSE_NOTICE = """\
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright {{ENGINE_NAME}} Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

_TEMPLATE_FILES: dict[str, str] = {
    "LICENSE": _APACHE_LICENSE_NOTICE,
    "manifest.yaml": """\
manifest_schema_version: "0.4"
engine_id: {{ENGINE_ID}}
engine_name: {{ENGINE_NAME}}
engine_version: "0.1.0"
aireliable_compat: ">=0.3.0"
uv_env_name: "{{ENV_NAME}}"
export_mode: source

paradigm: workflow
engine_uid: "aireliable://your-name/misc/{{ENGINE_ID}}"
sdk_version: ">=0.3.0"
engine_api_version: "1"
entry: {{MODULE}}.server

license: "Apache-2.0"
repository: "https://github.com/your-name/{{ENGINE_ID}}"

author:
  name: "Your Name"

models:
  model_layout:
    base: "misc/{{ENGINE_ID}}/"
  default_variant: default
  variants:
    - name: default
      alias: "Default"
      precision: fp32
      path: "misc/{{ENGINE_ID}}/default"
      model_file_ids: []
  model_files: []
""",
    "{{MODULE}}/__init__.py": """\
\"\"\"{{ENGINE_NAME}} 引擎包。\"\"\"

from .engine import {{CLASS_NAME}}

__all__ = ["{{CLASS_NAME}}"]
""",
    "{{MODULE}}/engine.py": """\
\"\"\"引擎实现：替换 ``_build_workflow`` 与节点逻辑即可。\"\"\"

from __future__ import annotations

from pathlib import Path
from typing import Optional

from aireliable_engine import CapabilitiesResponse, VariantInfo, WorkflowEngine


class {{CLASS_NAME}}(WorkflowEngine):
    \"\"\"最小工作流引擎（零模型依赖，默认回显输入文本）。\"\"\"

    def __init__(self) -> None:
        super().__init__(
            engine_id="{{ENGINE_ID}}",
            engine_name="{{ENGINE_NAME}}",
            capability="{{CAPABILITY}}",
            engine_type="{{ENGINE_TYPE}}",
            workflows_dir=Path(__file__).resolve().parent.parent / "workflows",
        )

    async def _build_workflow(self, name: Optional[str] = None) -> str:
        workflow_name = name or self.current_workflow
        return (self.workflows_dir / f"{workflow_name}.yaml").read_text(encoding="utf-8")

    def _get_default_workflow_template(self) -> str:
        \"\"\"首次启动时用于创建 workflows/default.yaml 的最小模板。\"\"\"
        return \"\"\"name: default
description: 把输入文本原样返回
nodes:
  - id: text_input
    type: TextInput
    params:
      value: ""
outputs:
  text: ${text_input.text}
\"\"\"

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=[
                VariantInfo(
                    name="default",
                    base_model="{{ENGINE_NAME}}",
                    precision="fp32",
                    loader="builtin",
                    vram_required="0GB",
                )
            ],
        )
""",
    "{{MODULE}}/server.py": """\
\"\"\"HTTP 服务入口：平台执行 ``python -m {{MODULE}}.server --port N``。\"\"\"

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

import aireliable_engine.nodes  # noqa: F401  注册内置节点（TextInput / ImageSave 等）
from aireliable_engine import create_engine_app, setup_engine_logging

from .engine import {{CLASS_NAME}}

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def main() -> None:
    parser = argparse.ArgumentParser(description="{{ENGINE_NAME}} engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7100)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_engine_logging(engine_id="{{ENGINE_ID}}", log_level=args.log_level)
    app = create_engine_app(
        {{CLASS_NAME}}(),
        workflows_dir=WORKFLOWS_DIR,
        title="{{ENGINE_NAME}}",
        version="0.1.0",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
""",
    "workflows/default.yaml": """\
name: default
description: 最小示例：把输入文本原样返回
nodes:
  - id: text_input
    type: TextInput
    params:
      value: ""
outputs:
  text: ${text_input.text}
""",
    "registry.yaml": """\
# 部署片段：平台导入后写入 config/engines/<engine_id>.yaml
engine_id: {{ENGINE_ID}}
engine_name: {{ENGINE_NAME}}
engine_version: "0.1.0"
sdk_version: ">=0.3.0"
capability: {{CAPABILITY}}
capabilities:
  - {{CAPABILITY}}
type: {{ENGINE_TYPE}}
deployment:
  mode: local
  local_config:
    port: 7100
    uv_env: envs/{{ENV_NAME}}
    startup_cmd: python -m {{MODULE}}.server --port 7100
  remote_config:
    url: ""
    auth_token: ""
default_residency: warm
""",
    "env/pyproject.toml": """\
[project]
name = "{{ENGINE_ID}}-engine"
version = "0.1.0"
requires-python = ">=3.10,<3.12"
dependencies = [
    "aireliable-engine-sdk>=0.3.0",
]

[tool.uv]
package = false
""",
    "README.md": """\
# {{ENGINE_NAME}}

由 `aireliable-engine init` 生成的最小引擎。

## 本地运行

```bash
pip install -e ".[dev]" 2>/dev/null || true
aireliable-engine validate
aireliable-engine run --port 7100
```

## 下一步

1. 修改 `{{MODULE}}/engine.py` 实现业务逻辑；
2. 修改 `workflows/default.yaml` 编排节点；
3. `aireliable-engine validate` → `aireliable-engine pack`；
4. 在 AIReliable 客户端导入 `dist/*.airengine`。
""",
    ".gitignore": """\
__pycache__/
*.pyc
.venv/
dist/
models/
""",
}


def _render(text: str, mapping: dict[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir).resolve()
    if target.exists() and any(target.iterdir()):
        if not args.force:
            print(f"目录已存在且非空：{target}（使用 --force 覆盖）", file=sys.stderr)
            return EXIT_VALIDATION

    engine_id = (args.id or target.name).strip()
    module = _module_name(engine_id)
    class_base = "".join(part.capitalize() for part in module.split("_"))
    class_name = class_base if class_base.endswith("Engine") else class_base + "Engine"
    capability = args.capability
    mapping = {
        "ENGINE_ID": engine_id,
        "ENGINE_NAME": args.name or engine_id.replace("_", " ").title(),
        "CAPABILITY": capability,
        "ENGINE_TYPE": _engine_type_for(capability),
        "MODULE": module,
        "CLASS_NAME": class_name,
        "ENV_NAME": engine_id.replace("_", "-"),
    }

    if args.template:
        if not _init_from_template(args.template, target, mapping):
            return EXIT_VALIDATION
    else:
        for rel, content in _TEMPLATE_FILES.items():
            path = target / _render(rel, mapping)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(content, mapping), encoding="utf-8")

    print(f"已生成引擎：{target}")
    print(f"  engine_id : {engine_id}")
    print(f"  entry     : {module}.server")
    print("\n下一步：")
    print(f"  cd {target}")
    print("  aireliable-engine validate")
    print("  aireliable-engine run --port 7100")
    return EXIT_OK


_TEXT_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json"})

# 官方模板仓使用这三个具体标记；init --template 会整体替换为目标命名
_TEMPLATE_MARKERS = {
    "ExampleEngine": "CLASS_NAME",
    "Example Engine": "ENGINE_NAME",
    "example_engine": "ENGINE_ID",
    "text-to-image": "CAPABILITY",
}


def _init_from_template(template: str, target: Path, mapping: dict[str, str]) -> bool:
    """从本地目录或 git 仓库复制模板，并重命名到目标 engine_id。

    模板约定（官方 `aireliable-engine-template`）：
        - 模块目录固定为 ``example_engine/``
        - 文本中以 ``ExampleEngine`` / ``Example Engine`` / ``example_engine`` 作为占位标记

    也支持使用 ``{{ENGINE_ID}}`` 这类花括号占位符的自定义模板。
    """
    source = Path(template).resolve()
    if source.is_dir():
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv", "dist"),
        )
    else:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", template, str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"拉取模板失败：{result.stderr.strip()}", file=sys.stderr)
            return False
        shutil.rmtree(target / ".git", ignore_errors=True)

    old_module = target / "example_engine"
    new_module = target / mapping["MODULE"]
    if old_module.is_dir() and old_module != new_module:
        old_module.rename(new_module)

    for path in target.rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        text = _render(text, mapping)
        for marker, key in _TEMPLATE_MARKERS.items():
            text = text.replace(marker, mapping[key])
        if mapping["ENGINE_TYPE"] != "image":
            text = text.replace("type: image", f"type: {mapping['ENGINE_TYPE']}")
        path.write_text(text, encoding="utf-8")
    return True


# ===========================================================================
# run
# ===========================================================================

def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.path or ".").resolve()
    try:
        schema = _load_manifest(root)
    except ManifestSchemaError as exc:
        print(f"manifest 校验失败：{exc}", file=sys.stderr)
        return EXIT_VALIDATION

    entry = schema.entry or f"engines.{schema.engine_id}.server"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("AIRELIABLE_ENGINE_ID", schema.engine_id)
    env.setdefault("AIRELIABLE_ENGINE_ROOT", str(root))

    cmd = [
        sys.executable, "-m", entry,
        "--host", args.host,
        "--port", str(args.port),
    ]
    print(f"启动 {schema.engine_id}：http://{args.host}:{args.port}")
    print(f"健康检查：http://{args.host}:{args.port}/health（entry={entry}）")

    try:
        proc = subprocess.Popen(cmd, cwd=str(root), env=env)
    except OSError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return EXIT_RUNTIME

    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return EXIT_RUNTIME


# ===========================================================================
# pack
# ===========================================================================

def _registry_for(root: Path, schema: ManifestSchema) -> dict[str, Any]:
    """读取引擎自带 registry.yaml（pack 必需）。"""
    source = root / "registry.yaml"
    if not source.is_file():
        raise ManifestSchemaError(
            "缺少 registry.yaml（pack 必需的部署片段，可参考 aireliable-engine init 生成的模板）"
        )
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ManifestSchemaError("registry.yaml 顶层必须是 YAML 映射（dict）")
    if data.get("engine_id") != schema.engine_id:
        raise ManifestSchemaError(
            f"registry.yaml engine_id={data.get('engine_id')!r} "
            f"与 manifest engine_id={schema.engine_id!r} 不一致"
        )
    return data


def cmd_pack(args: argparse.Namespace) -> int:
    root = Path(args.path or ".").resolve()
    result = collect_validation(root)
    if not result["ok"]:
        _emit(result, args.json)
        return EXIT_VALIDATION

    schema = _load_manifest(root)
    output_dir = Path(args.output or (root / "dist")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{schema.engine_id}-{schema.engine_version}.airengine"

    staging = output_dir / f".staging-{schema.engine_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        shutil.copy2(root / "manifest.yaml", staging / "manifest.yaml")
        (staging / "registry.yaml").write_text(
            yaml.safe_dump(_registry_for(root, schema), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (staging / "env").mkdir(exist_ok=True)
        shutil.copy2(root / "env" / "pyproject.toml", staging / "env" / "pyproject.toml")
        if (root / "env" / "uv.lock").is_file():
            shutil.copy2(root / "env" / "uv.lock", staging / "env" / "uv.lock")

        # code/<engine_id>/ → 平台 engines/<engine_id>/（整个工程根，含包内模块与 workflows）
        code_dir = staging / "code" / schema.engine_id
        code_dir.parent.mkdir(parents=True)
        shutil.copytree(
            root,
            code_dir,
            ignore=shutil.ignore_patterns(
                "models", "dist", "env", "docs", ".github",
                "__pycache__", "*.pyc", ".venv", ".git", ".pytest_cache",
                ".mypy_cache", ".ruff_cache", "node_modules", ".aireliable",
            ),
            dirs_exist_ok=True,
        )
        docs = root / "docs"
        if docs.is_dir():
            shutil.copytree(docs, staging / "docs", dirs_exist_ok=True)

        sha = compute_content_hash(staging)

        # 把哈希写进包内 manifest.yaml
        manifest_data = _read_yaml(staging / "manifest.yaml")
        manifest_data.setdefault("package", {})["sha256"] = sha
        (staging / "manifest.yaml").write_text(
            yaml.safe_dump(manifest_data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        package_meta = write_package_json(
            staging,
            engine_id=schema.engine_id,
            version=schema.engine_version,
            sha256=sha,
        )

        if archive.exists():
            archive.unlink()
        # 平台导入器只接受 7z 容器（.airengine 扩展名），与平台导出保持一致
        import py7zr

        with py7zr.SevenZipFile(archive, "w") as zf:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.writestr(path.read_bytes(), path.relative_to(staging).as_posix())
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    payload = {
        "ok": True,
        "engine_id": schema.engine_id,
        "version": schema.engine_version,
        "archive": str(archive),
        "sha256": sha,
        "files": len(package_meta.get("files", [])),  # type: ignore[arg-type]
        "errors": [],
        "warnings": result["warnings"],
        "info": [f"已生成 {archive.name}"],
    }
    _emit(payload, args.json)
    return EXIT_OK


# ===========================================================================
# 入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aireliable-engine",
        description="AIReliable 引擎开发工具（SDK 0.3）",
    )
    parser.add_argument("--version", action="version", version=f"aireliable-engine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="生成最小可运行引擎")
    p_init.add_argument("dir", help="目标目录")
    p_init.add_argument("--name", help="引擎显示名")
    p_init.add_argument("--id", dest="id", help="engine_id（默认取目录名）")
    p_init.add_argument("--capability", default="text-to-image",
                        help="能力（如 text-to-image / tts / chat）")
    p_init.add_argument("--template", help="模板目录或 git 仓库地址")
    p_init.add_argument("--force", action="store_true", help="覆盖非空目录")
    p_init.add_argument("--json", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_validate = sub.add_parser("validate", help="校验 manifest / 工作流 / 包结构")
    p_validate.add_argument("path", nargs="?", default=".", help="引擎根目录")
    p_validate.add_argument("--strict", action="store_true", help="警告视为错误")
    p_validate.add_argument("--json", action="store_true")
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="按 manifest.entry 启动 HTTP 服务")
    p_run.add_argument("path", nargs="?", default=".", help="引擎根目录")
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=7100)
    p_run.set_defaults(func=cmd_run)

    p_pack = sub.add_parser("pack", help="产出可导入客户端的标准包")
    p_pack.add_argument("path", nargs="?", default=".", help="引擎根目录")
    p_pack.add_argument("-o", "--output", help="输出目录（默认 dist/）")
    p_pack.add_argument("--json", action="store_true")
    p_pack.set_defaults(func=cmd_pack)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ManifestSchemaError as exc:
        print(f"manifest 校验失败：{exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底，避免堆栈直接抛给用户
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME


if __name__ == "__main__":
    raise SystemExit(main())

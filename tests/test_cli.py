"""``aireliable-engine`` CLI 的单元测试（init / validate / run / pack）。"""

from __future__ import annotations

import json
from pathlib import Path

import py7zr
import pytest
import yaml

from aireliable_engine import cli
from aireliable_engine.manifest import validate_manifest_yaml
from aireliable_engine.package import compute_content_hash


def _init(tmp_path: Path, name: str = "demo_engine", capsys=None) -> Path:
    target = tmp_path / name
    assert cli.main(["init", str(target), "--name", "Demo Engine"]) == cli.EXIT_OK
    if capsys is not None:
        capsys.readouterr()  # 丢弃 init 的人类可读输出，避免污染 JSON 断言
    return target


def _manifest(path: Path) -> dict:
    return yaml.safe_load((path / "manifest.yaml").read_text(encoding="utf-8"))


def _write_manifest(path: Path, data: dict) -> None:
    (path / "manifest.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def test_init_creates_runnable_scaffold(tmp_path: Path) -> None:
    root = _init(tmp_path)
    for rel in ("manifest.yaml", "registry.yaml", "env/pyproject.toml", "workflows/default.yaml"):
        assert (root / rel).is_file(), rel
    assert (root / "demo_engine" / "server.py").is_file()

    schema = validate_manifest_yaml((root / "manifest.yaml").read_text(encoding="utf-8"))
    assert schema.engine_id == "demo_engine"
    assert schema.entry == "demo_engine.server"
    assert schema.package is None


def test_init_rejects_non_empty_dir_without_force(tmp_path: Path) -> None:
    root = tmp_path / "demo_engine"
    root.mkdir()
    (root / "keep.txt").write_text("x", encoding="utf-8")
    assert cli.main(["init", str(root)]) == cli.EXIT_VALIDATION
    assert (root / "keep.txt").is_file()
    assert cli.main(["init", str(root), "--force"]) == cli.EXIT_OK


def test_validate_passes_on_scaffold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _init(tmp_path, capsys=capsys)
    assert cli.main(["validate", str(root), "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_validate_catches_variant_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _init(tmp_path, capsys=capsys)
    data = _manifest(root)
    data["models"]["variants"][0]["model_file_ids"] = ["missing_file"]
    _write_manifest(root, data)

    assert cli.main(["validate", str(root), "--json"]) == cli.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert any("missing_file" in err for err in payload["errors"])


def test_validate_catches_unknown_node_type(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _init(tmp_path, capsys=capsys)
    (root / "workflows" / "default.yaml").write_text(
        "name: default\nnodes:\n  - id: n1\n    type: NoSuchNode\noutputs:\n  x: ${n1.text}\n",
        encoding="utf-8",
    )
    assert cli.main(["validate", str(root), "--json"]) == cli.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert any("NoSuchNode" in err for err in payload["errors"])


def test_validate_strict_promotes_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path, capsys=capsys)
    (root / "LICENSE").unlink(missing_ok=True)
    assert cli.main(["validate", str(root)]) == cli.EXIT_OK
    capsys.readouterr()
    assert cli.main(["validate", str(root), "--strict", "--json"]) == cli.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert any("[strict]" in err for err in payload["errors"])


def test_validate_catches_missing_entry_module(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path, capsys=capsys)
    data = _manifest(root)
    data["entry"] = "not_installed.server"
    _write_manifest(root, data)
    assert cli.main(["validate", str(root), "--json"]) == cli.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert any("not_installed.server" in err for err in payload["errors"])


def test_pack_layout_and_hash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _init(tmp_path, capsys=capsys)
    out = tmp_path / "dist"
    assert cli.main(["pack", str(root), "-o", str(out), "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    archive = Path(payload["archive"])
    assert archive.is_file()

    with py7zr.SevenZipFile(archive) as zf:
        names = set(zf.getnames())
    assert {"manifest.yaml", "registry.yaml", "env/pyproject.toml", "package.json"} <= names
    # code/<engine_id>/ 对应平台 engines/<engine_id>/，必须保留工程根布局
    assert "code/demo_engine/demo_engine/server.py" in names
    assert "code/demo_engine/workflows/default.yaml" in names
    assert "code/demo_engine/manifest.yaml" in names

    unpacked = tmp_path / "unpacked"
    with py7zr.SevenZipFile(archive) as zf:
        zf.extractall(path=unpacked)
    manifest = yaml.safe_load((unpacked / "manifest.yaml").read_text(encoding="utf-8"))
    meta = json.loads((unpacked / "package.json").read_text(encoding="utf-8"))
    assert manifest["package"]["sha256"] == payload["sha256"]
    assert meta["sha256"] == payload["sha256"]


def test_pack_blocked_by_validation_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _init(tmp_path, capsys=capsys)
    data = _manifest(root)
    data["models"]["default_variant"] = "nope"
    _write_manifest(root, data)
    assert cli.main(["pack", str(root)]) == cli.EXIT_VALIDATION
    capsys.readouterr()


def test_content_hash_ignores_mtime_and_manifest(tmp_path: Path) -> None:
    root = _init(tmp_path)
    first = compute_content_hash(root)
    # manifest.yaml 与 package.json 不参与哈希
    (root / "manifest.yaml").write_text("changed: true\n", encoding="utf-8")
    assert compute_content_hash(root) == first
    # 业务文件变化必须改变哈希
    (root / "demo_engine" / "engine.py").write_text("# tampered\n", encoding="utf-8")
    assert compute_content_hash(root) != first


def test_run_resolves_entry_and_propagates_exit_code(tmp_path: Path) -> None:
    root = _init(tmp_path)
    marker = tmp_path / "spawned.txt"
    (root / "demo_engine" / "fake_server.py").write_text(
        "import pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    data = _manifest(root)
    data["entry"] = "demo_engine.fake_server"
    _write_manifest(root, data)

    assert cli.main(["run", str(root), "--port", "7199"]) == cli.EXIT_OK
    assert marker.is_file()
    assert "--port 7199" in marker.read_text(encoding="utf-8")


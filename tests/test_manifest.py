"""manifest 结构模型单测（SDK 侧契约）。"""

from __future__ import annotations

import pytest

from aireliable_engine.manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    ManifestSchemaError,
    validate_manifest_dict,
    validate_manifest_yaml,
)

BASE = {
    "engine_id": "demo_engine",
    "engine_name": "Demo Engine",
    "engine_version": "1.0.0",
    "aireliable_compat": ">=0.3.0",
    "uv_env_name": "demo-engine",
}


def test_schema_version() -> None:
    assert CURRENT_MANIFEST_SCHEMA_VERSION == "0.4"


def test_minimal_manifest_ok() -> None:
    schema = validate_manifest_dict(dict(BASE))
    assert schema.entry is None
    assert schema.package is None


def test_entry_and_package_ok() -> None:
    schema = validate_manifest_dict(
        {
            **BASE,
            "entry": "demo_engine.server",
            "package": {"sha256": "a" * 64, "signature": None},
        }
    )
    assert schema.entry == "demo_engine.server"
    assert schema.package is not None
    assert schema.package.sha256 == "a" * 64


@pytest.mark.parametrize("entry", ["bad/module", "1bad.server", "", "bad-name"])
def test_entry_rejects_invalid(entry: str) -> None:
    with pytest.raises(ManifestSchemaError):
        validate_manifest_dict({**BASE, "entry": entry})


@pytest.mark.parametrize("sha", ["abc", "z" * 64, "a" * 63])
def test_package_sha256_rejects_invalid(sha: str) -> None:
    with pytest.raises(ManifestSchemaError):
        validate_manifest_dict({**BASE, "package": {"sha256": sha}})


def test_unknown_top_level_field_rejected() -> None:
    with pytest.raises(ManifestSchemaError):
        validate_manifest_dict({**BASE, "typo_field": 1})


def test_models_variant_reference_validated() -> None:
    data = {
        **BASE,
        "models": {
            "model_layout": {"base": "misc/demo/"},
            "variants": [
                {
                    "name": "default",
                    "precision": "fp16",
                    "path": "misc/demo/default",
                    "model_file_ids": ["missing"],
                }
            ],
            "model_files": [],
        },
    }
    schema = validate_manifest_dict(data)
    with pytest.raises(ManifestSchemaError):
        schema.validate_v021_strict()


def test_yaml_roundtrip() -> None:
    text = (
        "engine_id: demo_engine\n"
        "engine_name: Demo Engine\n"
        "engine_version: '1.0.0'\n"
        "aireliable_compat: '>=0.3.0'\n"
        "uv_env_name: demo-engine\n"
        "entry: demo_engine.server\n"
    )
    schema = validate_manifest_yaml(text)
    assert schema.to_dict()["entry"] == "demo_engine.server"

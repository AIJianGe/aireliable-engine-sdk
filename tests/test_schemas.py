"""8 端点 Pydantic 模型测试。

覆盖 schemas.py 中所有 BaseModel 子类的字段默认值、必填校验、序列化。
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from aireliable_engine.schemas import (
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    InferRequest,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraLoadResponse,
    LoraSpec,
    LoraUnloadRequest,
    LoraUnloadResponse,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
)


class TestHealthResponse:
    """HealthResponse 测试。"""

    def test_default_status_ok(self):
        """默认 status=ok。"""
        r = HealthResponse(engine_id="ernie", state="idle")
        assert r.status == "ok"
        assert r.engine_id == "ernie"
        assert r.state == "idle"
        assert r.uptime == 0.0

    def test_uptime_custom(self):
        """uptime 可自定义。"""
        r = HealthResponse(engine_id="e", state="ready", uptime=12.5)
        assert r.uptime == 12.5

    def test_missing_required_fields(self):
        """必填字段缺失抛 ValidationError。"""
        with pytest.raises(ValidationError):
            HealthResponse()  # 缺 engine_id 和 state


class TestCapabilitiesResponse:
    """CapabilitiesResponse 测试。"""

    def test_default_empty_lists(self):
        """默认 variants / loras / supported_sizes 为空列表。"""
        r = CapabilitiesResponse(
            engine_id="e", engine_name="E", capability="text-to-image", type="image"
        )
        assert r.variants == []
        assert r.loras == []
        assert r.supported_sizes == []

    def test_with_variants(self):
        """可附带 variants。"""
        v = VariantInfo(
            name="v1",
            base_model="ernie",
            precision="q4_k_m",
            loader="gguf",
            vram_required="8GB",
        )
        r = CapabilitiesResponse(
            engine_id="e",
            engine_name="E",
            capability="text-to-image",
            type="image",
            variants=[v],
            loras=["lora1"],
            supported_sizes=["1024x1024"],
        )
        assert len(r.variants) == 1
        assert r.variants[0].name == "v1"
        assert r.loras == ["lora1"]
        assert r.supported_sizes == ["1024x1024"]


class TestVariantInfo:
    """VariantInfo 测试。"""

    def test_required_fields(self):
        """必填字段校验。"""
        v = VariantInfo(
            name="ernie-image-q4",
            base_model="ernie-image",
            precision="q4_k_m",
            loader="gguf",
            vram_required="8GB",
        )
        assert v.name == "ernie-image-q4"
        assert v.base_model == "ernie-image"
        assert v.precision == "q4_k_m"
        assert v.loader == "gguf"
        assert v.vram_required == "8GB"
        # 默认值
        assert v.default_steps is None
        assert v.default_guidance is None
        assert v.supports_pe is False
        assert v.available is True

    def test_missing_required_field(self):
        """缺必填字段抛 ValidationError。"""
        with pytest.raises(ValidationError):
            VariantInfo(name="v", base_model="b", precision="p")  # 缺 loader / vram_required

    def test_optional_fields(self):
        """可选字段可设置。"""
        v = VariantInfo(
            name="v",
            base_model="b",
            precision="p",
            loader="l",
            vram_required="8GB",
            default_steps=30,
            default_guidance=7.5,
            supports_pe=True,
            available=False,
        )
        assert v.default_steps == 30
        assert v.default_guidance == 7.5
        assert v.supports_pe is True
        assert v.available is False


class TestLoadRequestResponse:
    """LoadRequest / LoadResponse 测试。"""

    def test_load_request(self):
        """LoadRequest 简单字段。"""
        r = LoadRequest(variant="ernie-image-q4")
        assert r.variant == "ernie-image-q4"

    def test_load_request_missing_variant(self):
        """LoadRequest 缺 variant 抛 ValidationError。"""
        with pytest.raises(ValidationError):
            LoadRequest()

    def test_load_response_basic(self):
        """LoadResponse 字段。"""
        r = LoadResponse(variant="v", state="ready", vram_used="7.2GB")
        assert r.variant == "v"
        assert r.state == "ready"
        assert r.vram_used == "7.2GB"
        assert r.load_time is None

    def test_load_response_with_load_time(self):
        """LoadResponse 可带 load_time。"""
        r = LoadResponse(variant="v", state="ready", vram_used="7.2GB", load_time=12.5)
        assert r.load_time == 12.5


class TestUnloadResponse:
    """UnloadResponse 测试。"""

    def test_default_state_idle(self):
        """默认 state=idle。"""
        r = UnloadResponse()
        assert r.state == "idle"
        assert r.vram_freed is None

    def test_with_vram_freed(self):
        """可设置 vram_freed。"""
        r = UnloadResponse(state="idle", vram_freed="7.2GB")
        assert r.vram_freed == "7.2GB"


class TestInferRequest:
    """InferRequest 测试。"""

    def test_default_loras_empty(self):
        """默认 loras=[]。"""
        r = InferRequest(variant="v", inputs={"prompt": "a cat"})
        assert r.variant == "v"
        assert r.inputs == {"prompt": "a cat"}
        assert r.loras == []
        assert r.task_id is None

    def test_with_loras_and_task_id(self):
        """可附带 loras 和 task_id。"""
        r = InferRequest(
            variant="v",
            inputs={"prompt": "a cat"},
            loras=[LoraSpec(name="l1", weight=0.5)],
            task_id="task-123",
        )
        assert len(r.loras) == 1
        assert r.loras[0].name == "l1"
        assert r.task_id == "task-123"

    def test_missing_required(self):
        """缺必填字段抛 ValidationError。"""
        with pytest.raises(ValidationError):
            InferRequest(variant="v")  # 缺 inputs


class TestLoraSpec:
    """LoraSpec 测试。"""

    def test_default_weight(self):
        """默认 weight=1.0。"""
        s = LoraSpec(name="l1")
        assert s.name == "l1"
        assert s.weight == 1.0

    def test_custom_weight(self):
        """可自定义 weight。"""
        s = LoraSpec(name="l1", weight=0.7)
        assert s.weight == 0.7

    def test_missing_name(self):
        """缺 name 抛 ValidationError。"""
        with pytest.raises(ValidationError):
            LoraSpec(weight=0.5)


class TestStatusResponse:
    """StatusResponse 测试。"""

    def test_defaults(self):
        """默认值正确。"""
        r = StatusResponse(state="idle")
        assert r.state == "idle"
        assert r.loaded_variant is None
        assert r.vram_used == "0GB"
        assert r.vram_total is None
        assert r.loaded_loras == []
        assert r.last_used_at is None
        assert r.uptime == 0.0
        assert r.extra == {}

    def test_with_all_fields(self):
        """所有字段都设置。"""
        r = StatusResponse(
            state="ready",
            loaded_variant="ernie-image-q4",
            vram_used="7.2GB",
            vram_total="12GB",
            loaded_loras=["l1", "l2"],
            last_used_at=1700000000.0,
            uptime=3600.5,
            extra={"key": "value"},
        )
        assert r.loaded_variant == "ernie-image-q4"
        assert r.vram_total == "12GB"
        assert r.loaded_loras == ["l1", "l2"]
        assert r.last_used_at == 1700000000.0
        assert r.uptime == 3600.5
        assert r.extra == {"key": "value"}


class TestLoraLoadRequestResponse:
    """LoraLoadRequest / LoraLoadResponse 测试。"""

    def test_lora_load_request_defaults(self):
        """LoraLoadRequest 默认 weight=1.0。"""
        r = LoraLoadRequest(name="l1")
        assert r.name == "l1"
        assert r.weight == 1.0

    def test_lora_load_request_custom_weight(self):
        """LoraLoadRequest 自定义 weight。"""
        r = LoraLoadRequest(name="l1", weight=0.6)
        assert r.weight == 0.6

    def test_lora_load_request_missing_name(self):
        """缺 name 抛 ValidationError。"""
        with pytest.raises(ValidationError):
            LoraLoadRequest()

    def test_lora_load_response(self):
        """LoraLoadResponse 字段。"""
        r = LoraLoadResponse(name="l1", weight=0.6, loaded_loras=["l1", "l2"])
        assert r.name == "l1"
        assert r.weight == 0.6
        assert r.loaded_loras == ["l1", "l2"]

    def test_lora_load_response_missing_required(self):
        """LoraLoadResponse 缺必填字段抛 ValidationError。"""
        with pytest.raises(ValidationError):
            LoraLoadResponse(name="l1")  # 缺 weight / loaded_loras


class TestLoraUnloadRequestResponse:
    """LoraUnloadRequest / LoraUnloadResponse 测试。"""

    def test_lora_unload_request(self):
        """LoraUnloadRequest 字段。"""
        r = LoraUnloadRequest(name="l1")
        assert r.name == "l1"

    def test_lora_unload_request_missing_name(self):
        """缺 name 抛 ValidationError。"""
        with pytest.raises(ValidationError):
            LoraUnloadRequest()

    def test_lora_unload_response(self):
        """LoraUnloadResponse 字段。"""
        r = LoraUnloadResponse(name="l1", loaded_loras=["l2"])
        assert r.name == "l1"
        assert r.loaded_loras == ["l2"]

    def test_lora_unload_response_missing_required(self):
        """LoraUnloadResponse 缺必填字段抛 ValidationError。"""
        with pytest.raises(ValidationError):
            LoraUnloadResponse(name="l1")  # 缺 loaded_loras


class TestErrorResponse:
    """ErrorResponse 测试。"""

    def test_error_response(self):
        """ErrorResponse 字段。"""
        r = ErrorResponse(error={"type": "engine_busy", "message": "忙", "code": 400})
        assert r.error["type"] == "engine_busy"
        assert r.error["message"] == "忙"
        assert r.error["code"] == 400

    def test_error_response_missing_error(self):
        """缺 error 抛 ValidationError。"""
        with pytest.raises(ValidationError):
            ErrorResponse()


class TestSerialization:
    """模型序列化为 dict / JSON 测试。"""

    def test_health_response_to_dict(self):
        """HealthResponse 序列化为 dict。"""
        r = HealthResponse(engine_id="e", state="idle", uptime=1.5)
        d = r.model_dump()
        assert d == {"status": "ok", "engine_id": "e", "state": "idle", "uptime": 1.5}

    def test_health_response_to_json(self):
        """HealthResponse 序列化为 JSON。"""
        r = HealthResponse(engine_id="e", state="idle")
        s = r.model_dump_json()
        d = json.loads(s)
        assert d["status"] == "ok"
        assert d["engine_id"] == "e"
        assert d["state"] == "idle"

    def test_infer_request_to_dict(self):
        """InferRequest 序列化为 dict。"""
        r = InferRequest(
            variant="v",
            inputs={"prompt": "cat"},
            loras=[LoraSpec(name="l1", weight=0.5)],
            task_id="t1",
        )
        d = r.model_dump()
        assert d["variant"] == "v"
        assert d["inputs"] == {"prompt": "cat"}
        assert d["loras"][0] == {"name": "l1", "weight": 0.5}
        assert d["task_id"] == "t1"

    def test_status_response_extra_serialization(self):
        """StatusResponse extra 字段序列化。"""
        r = StatusResponse(state="ready", extra={"foo": "bar", "num": 42})
        d = r.model_dump()
        assert d["extra"] == {"foo": "bar", "num": 42}

    def test_capabilities_response_nested_serialization(self):
        """CapabilitiesResponse 嵌套 variant 序列化。"""
        v = VariantInfo(
            name="v1",
            base_model="b",
            precision="p",
            loader="l",
            vram_required="8GB",
        )
        r = CapabilitiesResponse(
            engine_id="e",
            engine_name="E",
            capability="c",
            type="t",
            variants=[v],
        )
        d = r.model_dump()
        assert d["variants"][0]["name"] == "v1"
        assert d["variants"][0]["precision"] == "p"

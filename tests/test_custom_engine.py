"""CustomEngine 全自定义引擎基类测试（T-007）。

覆盖：
- 继承 CustomEngine 不触发 DeprecationWarning
- 未实现的 7 个 ``_xxx_impl`` 方法抛 NotImplementedError
- 实现完整子类后状态机正常工作（load / infer / unload）
- 与 BaseEngine 的关系（CustomEngine 是 BaseEngine 子类）
- 与直接继承 BaseEngine 的对比（前者无警告，后者有警告）
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import pytest

from aireliable_engine import (
    BaseEngine,
    CapabilitiesResponse,
    CustomEngine,
    EngineBusyError,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraLoadResponse,
    LoraUnloadRequest,
    LoraUnloadResponse,
    StatusResponse,
    UnloadResponse,
    make_final_frame,
    make_frame,
)
from aireliable_engine.schemas import InferRequest
from aireliable_engine.state_machine import RuntimeState


# ===== 测试用 mock 子类 =====


class _CompleteCustomEngine(CustomEngine):
    """完整实现 7 个 _xxx_impl 的 CustomEngine 子类，用于正向测试。"""

    def __init__(self, fail_infer: bool = False):
        super().__init__(
            engine_id="mock_custom",
            engine_name="Mock Custom Engine",
            capability="text-to-speech",
            engine_type="voice",
        )
        self.load_called = False
        self.unload_called = False
        self.infer_called = False
        self.fail_infer = fail_infer

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
        )

    async def _load_impl(self, variant: str) -> LoadResponse:
        self.load_called = True
        return LoadResponse(variant=variant, state="ready", vram_used="2GB")

    async def _unload_impl(self) -> UnloadResponse:
        self.unload_called = True
        return UnloadResponse(state="idle")

    async def _infer_impl(self, request: Any, stream) -> None:
        self.infer_called = True
        if self.fail_infer:
            raise RuntimeError("mock infer failed")
        await stream.put(make_frame(0.5, "tts.generating"))
        await stream.put(
            make_final_frame(
                result={"audio_url": "/output/audio/test.wav", "duration": 1.5},
                timing={"total_seconds": 1.5},
            )
        )

    async def _status_impl(self) -> StatusResponse:
        return StatusResponse(state=self.state_machine.state.value)

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        return LoraLoadResponse(name=name, weight=weight, loaded_loras=[name])

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        return LoraUnloadResponse(name=name, loaded_loras=[])


class _BareCustomEngine(CustomEngine):
    """未实现任何 _xxx_impl 的 CustomEngine 子类，用于测试默认 NotImplementedError。"""

    pass


# ===== 测试类 =====


class TestCustomEngineInheritance:
    """CustomEngine 继承关系测试。"""

    def test_custom_engine_is_base_engine_subclass(self):
        """CustomEngine 是 BaseEngine 的子类。"""
        assert issubclass(CustomEngine, BaseEngine)

    def test_inheriting_custom_engine_no_deprecation_warning(self):
        """继承 CustomEngine 不触发 DeprecationWarning。"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            class _MyEngine(CustomEngine):
                pass

            deprecation_warnings = [
                item for item in w
                if issubclass(item.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0, (
                "继承 CustomEngine 不应触发 DeprecationWarning，"
                f"实际收到: {[str(w.message) for w in deprecation_warnings]}"
            )

    def test_custom_engine_module_attribute(self):
        """CustomEngine 模块名正确。"""
        assert CustomEngine.__module__.startswith("aireliable_engine")

    def test_complete_subclass_can_instantiate(self):
        """完整实现的 CustomEngine 子类可正常实例化。"""
        eng = _CompleteCustomEngine()
        assert eng.engine_id == "mock_custom"
        assert eng.engine_name == "Mock Custom Engine"
        assert eng.capability == "text-to-speech"
        assert eng.engine_type == "voice"
        assert eng.state_machine.state == RuntimeState.IDLE


class TestCustomEngineDefaultNotImplemented:
    """CustomEngine 子类未实现 _xxx_impl 时抛 NotImplementedError。"""

    def test_bare_engine_capabilities_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")
        with pytest.raises(NotImplementedError):
            asyncio.run(eng._capabilities_impl())

    def test_bare_engine_load_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")
        with pytest.raises(NotImplementedError):
            asyncio.run(eng._load_impl("v1"))

    def test_bare_engine_unload_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")
        with pytest.raises(NotImplementedError):
            asyncio.run(eng._unload_impl())

    def test_bare_engine_infer_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")

        async def _run():
            await eng._infer_impl(None, None)

        with pytest.raises(NotImplementedError):
            asyncio.run(_run())

    def test_bare_engine_status_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")
        with pytest.raises(NotImplementedError):
            asyncio.run(eng._status_impl())

    def test_bare_engine_load_lora_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")
        with pytest.raises(NotImplementedError):
            asyncio.run(eng._load_lora_impl("l1", 0.5))

    def test_bare_engine_unload_lora_raises(self):
        eng = _BareCustomEngine("id", "name", "cap", "type")
        with pytest.raises(NotImplementedError):
            asyncio.run(eng._unload_lora_impl("l1"))


class TestCustomEngineStateMachine:
    """CustomEngine 状态机集成测试（与 BaseEngine 一致）。"""

    @pytest.mark.asyncio
    async def test_load_success_transitions_to_ready(self):
        eng = _CompleteCustomEngine()
        r = await eng.load(LoadRequest(variant="v1"))
        assert eng.load_called is True
        assert r.variant == "v1"
        assert r.state == "ready"
        assert eng.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_unload_from_ready_to_idle(self):
        eng = _CompleteCustomEngine()
        await eng.load(LoadRequest(variant="v1"))
        r = await eng.unload()
        assert eng.unload_called is True
        assert r.state == "idle"
        assert eng.state_machine.state == RuntimeState.IDLE

    @pytest.mark.asyncio
    async def test_infer_returns_sse_stream(self):
        """infer() 返回 SSE 流并触发 _infer_impl。"""
        eng = _CompleteCustomEngine()
        await eng.load(LoadRequest(variant="v1"))
        req = InferRequest(variant="v1", inputs={"text": "你好"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)
        assert len(frames) == 2  # 进度帧 + 最终帧
        assert eng.infer_called is True
        # 推理后回 READY
        assert eng.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_infer_auto_loads_from_idle(self):
        """infer() 在 IDLE 自动加载后推理。"""
        eng = _CompleteCustomEngine()
        req = InferRequest(variant="v1", inputs={"text": "你好"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)
        assert eng.load_called is True
        assert eng.infer_called is True
        assert eng.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_infer_failure_transitions_to_error(self):
        """_infer_impl 抛异常时状态变 ERROR。"""
        eng = _CompleteCustomEngine(fail_infer=True)
        await eng.load(LoadRequest(variant="v1"))

        req = InferRequest(variant="v1", inputs={"text": "你好"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)

        assert len(frames) == 1
        assert eng.state_machine.state == RuntimeState.ERROR

    @pytest.mark.asyncio
    async def test_concurrent_loads_are_serialized(self):
        """并发 load：op_lock 串行化，第二个在 READY 上先 unload 再 load，两者都成功。"""
        eng = _CompleteCustomEngine()

        original_load = eng._load_impl

        async def slow_load(variant):
            await asyncio.sleep(0.05)
            return await original_load(variant)

        eng._load_impl = slow_load  # type: ignore[assignment]

        async def try_load():
            return await eng.load(LoadRequest(variant="v1"))

        results = await asyncio.gather(
            try_load(), try_load(), return_exceptions=True
        )

        assert all(not isinstance(r, Exception) for r in results), results
        assert eng.state_machine.state == RuntimeState.READY


class TestCustomEngineLora:
    """CustomEngine 的 LoRA 接口。"""

    @pytest.mark.asyncio
    async def test_load_lora_calls_impl(self):
        eng = _CompleteCustomEngine()
        r = await eng.load_lora(LoraLoadRequest(name="l1", weight=0.5))
        assert r.name == "l1"
        assert r.loaded_loras == ["l1"]

    @pytest.mark.asyncio
    async def test_unload_lora_calls_impl(self):
        eng = _CompleteCustomEngine()
        r = await eng.unload_lora(LoraUnloadRequest(name="l1"))
        assert r.name == "l1"
        assert r.loaded_loras == []

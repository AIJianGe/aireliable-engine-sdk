"""BaseEngine 抽象基类测试。

用 _MockEngine 子类实例化 BaseEngine，覆盖：
- 抽象基类 / 子类实例化
- get_output_path 路径管理
- health / capabilities / load / unload / infer / status / load_lora / unload_lora
- 状态机集成（busy 防护、错误恢复）
- 并发（asyncio.gather 两个 load 第二个抛 EngineBusyError）
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aireliable_engine import (
    BaseEngine,
    CapabilitiesResponse,
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


class _MockEngine(BaseEngine):
    """最小可用的 BaseEngine 子类，用于测试。"""

    def __init__(self, fail_load: bool = False, fail_unload: bool = False, fail_infer: bool = False):
        super().__init__(
            engine_id="mock",
            engine_name="Mock Engine",
            capability="text-to-image",
            engine_type="image",
        )
        self.load_called = False
        self.unload_called = False
        self.infer_called = False
        self.load_lora_called = False
        self.unload_lora_called = False
        self.fail_load = fail_load
        self.fail_unload = fail_unload
        self.fail_infer = fail_infer
        self.last_infer_request: Any = None

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
        )

    async def _load_impl(self, variant: str) -> LoadResponse:
        self.load_called = True
        if self.fail_load:
            raise RuntimeError("mock load failed")
        return LoadResponse(variant=variant, state="ready", vram_used="1GB")

    async def _unload_impl(self) -> UnloadResponse:
        self.unload_called = True
        if self.fail_unload:
            raise RuntimeError("mock unload failed")
        return UnloadResponse(state="idle")

    async def _infer_impl(self, request: Any, stream) -> None:
        self.infer_called = True
        self.last_infer_request = request
        if self.fail_infer:
            raise RuntimeError("mock infer failed")
        await stream.put(make_frame(0.5, "denoising"))
        await stream.put(
            make_final_frame(
                result={"image_urls": ["http://example.com/a.png"]},
                timing={"total": 1.0},
            )
        )

    async def _status_impl(self) -> StatusResponse:
        return StatusResponse(state=self.state_machine.state.value)

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        self.load_lora_called = True
        return LoraLoadResponse(name=name, weight=weight, loaded_loras=[name])

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        self.unload_lora_called = True
        return LoraUnloadResponse(name=name, loaded_loras=[])


class _IncompleteEngine(BaseEngine):
    """故意只实现部分抽象方法，用于测试 TypeError。"""

    async def _capabilities_impl(self):
        return None


# ===== 测试类 =====


class TestBaseEngineAbstraction:
    """BaseEngine 抽象基类测试。"""

    def test_cannot_instantiate_base_directly(self):
        """BaseEngine 是 ABC，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseEngine("id", "name", "cap", "type")  # type: ignore[abstract]

    def test_incomplete_subclass_raises_typeerror(self):
        """子类必须实现所有抽象方法，缺一个就 TypeError。"""
        with pytest.raises(TypeError):
            _IncompleteEngine("id", "name", "cap", "type")  # type: ignore[abstract]

    def test_complete_subclass_can_instantiate(self):
        """实现完整子类可正常实例化。"""
        eng = _MockEngine()
        assert eng.engine_id == "mock"
        assert eng.engine_name == "Mock Engine"
        assert eng.capability == "text-to-image"
        assert eng.engine_type == "image"
        assert eng.state_machine.state == RuntimeState.IDLE
        assert eng.output_root is None


class TestGetOutputPath:
    """get_output_path 测试。"""

    def test_raises_when_output_root_unset(self):
        """未设置 output_root 时抛 AttributeError。"""
        eng = _MockEngine()
        with pytest.raises(AttributeError, match="output_root"):
            eng.get_output_path("image", "out.png")

    def test_returns_correct_path(self, tmp_path: Path):
        """设置 output_root 后返回正确路径。"""
        eng = _MockEngine()
        eng.output_root = tmp_path
        p = eng.get_output_path("image", "out.png")
        assert p == tmp_path / "image" / "out.png"

    def test_creates_directory(self, tmp_path: Path):
        """get_output_path 自动创建目录。"""
        eng = _MockEngine()
        eng.output_root = tmp_path
        # 目录还不存在
        assert not (tmp_path / "image").exists()
        p = eng.get_output_path("image", "out.png")
        # 调用后目录已创建
        assert p.parent.exists()
        assert p.parent == tmp_path / "image"

    def test_nested_category_path(self, tmp_path: Path):
        """多级 category 路径创建。"""
        eng = _MockEngine()
        eng.output_root = tmp_path
        # 多级 category（虽然实际不会用，但 mkdir(parents=True) 应能处理）
        p = eng.get_output_path("sub/dir", "out.png")
        assert p.parent.exists()


class TestHealthEndpoint:
    """health() 测试。"""

    @pytest.mark.asyncio
    async def test_health_returns_correct_fields(self):
        """health() 返回正确字段。"""
        eng = _MockEngine()
        r = await eng.health()
        assert r.status == "ok"
        assert r.engine_id == "mock"
        assert r.state == "idle"  # 初始 IDLE
        assert r.uptime >= 0.0


class TestCapabilitiesEndpoint:
    """capabilities() 测试。"""

    @pytest.mark.asyncio
    async def test_capabilities_calls_impl(self):
        """capabilities() 调用 _capabilities_impl。"""
        eng = _MockEngine()
        r = await eng.capabilities()
        assert r.engine_id == "mock"
        assert r.engine_name == "Mock Engine"
        assert r.capability == "text-to-image"
        assert r.type == "image"


class TestLoadEndpoint:
    """load() 测试。"""

    @pytest.mark.asyncio
    async def test_load_success_transitions_to_ready(self):
        """load() 在 IDLE 状态下成功 → 状态变为 READY。"""
        eng = _MockEngine()
        r = await eng.load(LoadRequest(variant="v1"))
        assert eng.load_called is True
        assert r.variant == "v1"
        assert r.state == "ready"
        assert eng.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_load_when_inferring_raises_busy(self):
        """load() 在 INFERRING 状态下抛 EngineBusyError。"""
        eng = _MockEngine()
        # 手动把状态机推到 INFERRING
        eng.state_machine.transition(RuntimeState.LOADING)
        eng.state_machine.transition(RuntimeState.READY)
        eng.state_machine.transition(RuntimeState.INFERRING)
        with pytest.raises(EngineBusyError):
            await eng.load(LoadRequest(variant="v1"))

    @pytest.mark.asyncio
    async def test_load_failure_transitions_to_error(self):
        """load() 在 _load_impl 抛异常时状态变 ERROR。"""
        eng = _MockEngine(fail_load=True)
        with pytest.raises(RuntimeError, match="mock load failed"):
            await eng.load(LoadRequest(variant="v1"))
        assert eng.state_machine.state == RuntimeState.ERROR


class TestUnloadEndpoint:
    """unload() 测试。"""

    @pytest.mark.asyncio
    async def test_unload_from_ready_to_idle(self):
        """unload() 在 READY 状态下成功 → 状态变 IDLE。"""
        eng = _MockEngine()
        # 先加载到 READY
        await eng.load(LoadRequest(variant="v1"))
        assert eng.state_machine.state == RuntimeState.READY
        # 卸载
        r = await eng.unload()
        assert eng.unload_called is True
        assert r.state == "idle"
        assert eng.state_machine.state == RuntimeState.IDLE

    @pytest.mark.asyncio
    async def test_unload_when_inferring_raises_busy(self):
        """unload() 在 INFERRING 状态下抛 EngineBusyError。"""
        eng = _MockEngine()
        eng.state_machine.transition(RuntimeState.LOADING)
        eng.state_machine.transition(RuntimeState.READY)
        eng.state_machine.transition(RuntimeState.INFERRING)
        with pytest.raises(EngineBusyError):
            await eng.unload()


class TestInferEndpoint:
    """infer() 测试。"""

    @pytest.mark.asyncio
    async def test_infer_returns_sse_stream(self):
        """infer() 返回 SSE 流（异步迭代器）。"""
        eng = _MockEngine()
        # 先加载
        await eng.load(LoadRequest(variant="v1"))
        req = InferRequest(variant="v1", inputs={"prompt": "a cat"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)
        assert len(frames) == 2  # 进度帧 + 最终帧
        assert eng.infer_called is True
        # 推理结束后状态回 READY
        assert eng.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_infer_auto_loads_from_idle(self):
        """infer() 在 IDLE 状态下自动加载。"""
        eng = _MockEngine()
        req = InferRequest(variant="v1", inputs={"prompt": "a cat"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)
        # 加载和推理都被调用
        assert eng.load_called is True
        assert eng.infer_called is True
        # 自动加载 → 推理 → 回 READY
        assert eng.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_infer_when_inferring_pushes_busy_error(self):
        """infer() 在 INFERRING 状态下推送 engine_busy 错误帧。"""
        eng = _MockEngine()
        # 手动推到 INFERRING
        eng.state_machine.transition(RuntimeState.LOADING)
        eng.state_machine.transition(RuntimeState.READY)
        eng.state_machine.transition(RuntimeState.INFERRING)

        req = InferRequest(variant="v1", inputs={"prompt": "a cat"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)

        # 应只收到一帧错误帧
        assert len(frames) == 1
        body = frames[0][len("data: "):-2]
        data = json.loads(body)
        assert data["stage"] == "error"
        assert data["error"]["type"] == "engine_busy"
        # infer_impl 不应被调用
        assert eng.infer_called is False

    @pytest.mark.asyncio
    async def test_infer_exception_pushes_error_frame(self):
        """infer() 异常时推送错误帧 + 状态变 ERROR。"""
        eng = _MockEngine(fail_infer=True)
        # 先加载
        await eng.load(LoadRequest(variant="v1"))

        req = InferRequest(variant="v1", inputs={"prompt": "a cat"})
        frames = []
        async for frame in eng.infer(req):
            frames.append(frame)

        # 应收到一帧错误帧
        assert len(frames) == 1
        body = frames[0][len("data: "):-2]
        data = json.loads(body)
        assert data["stage"] == "error"
        assert "mock infer failed" in data["error"]["message"]
        # 状态变 ERROR
        assert eng.state_machine.state == RuntimeState.ERROR


class TestLoraEndpoints:
    """load_lora / unload_lora 测试。"""

    @pytest.mark.asyncio
    async def test_load_lora_calls_impl(self):
        """load_lora() 调用 _load_lora_impl。"""
        eng = _MockEngine()
        r = await eng.load_lora(LoraLoadRequest(name="l1", weight=0.5))
        assert eng.load_lora_called is True
        assert r.name == "l1"
        assert r.weight == 0.5
        assert r.loaded_loras == ["l1"]

    @pytest.mark.asyncio
    async def test_unload_lora_calls_impl(self):
        """unload_lora() 调用 _unload_lora_impl。"""
        eng = _MockEngine()
        r = await eng.unload_lora(LoraUnloadRequest(name="l1"))
        assert eng.unload_lora_called is True
        assert r.name == "l1"
        assert r.loaded_loras == []


class TestConcurrency:
    """并发场景测试。"""

    @pytest.mark.asyncio
    async def test_concurrent_loads_one_succeeds_one_fails(self):
        """两个 load 同时调用：op_lock 串行化，第二个进入时 state 已 READY。

        由于 ``op_lock`` 强制串行化，第二个 load 调用进入时第一个已经完成
        （state=READY）。READY→LOADING 不在允许转换集合里（READY 只能到
        {INFERRING, IDLE, ERROR}），所以第二个会抛 ValueError（状态机层防护）。
        """
        eng = _MockEngine()

        # 让 _load_impl 慢一点，确保两个 load 真的尝试并发
        original_load = eng._load_impl

        async def slow_load(variant):
            await asyncio.sleep(0.05)
            return await original_load(variant)

        eng._load_impl = slow_load  # type: ignore[assignment]

        # 同时发起两个 load
        async def try_load():
            return await eng.load(LoadRequest(variant="v1"))

        results = await asyncio.gather(
            try_load(), try_load(), return_exceptions=True
        )

        # 一个成功，一个抛异常
        successes = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(successes) == 1
        assert len(errors) == 1
        # 第二个 load 进入时 state=READY，transition(LOADING) 抛 ValueError
        assert isinstance(errors[0], ValueError)

    @pytest.mark.asyncio
    async def test_load_when_loading_state_raises_busy(self):
        """state=LOADING 时调用 load 抛 EngineBusyError（is_busy 防护）。

        这模拟并发场景下第二个 load 在第一个还在 LOADING 时进入的情况：
        is_busy()=True → 抛 EngineBusyError。
        """
        eng = _MockEngine()
        # 手动推到 LOADING（模拟另一个 load 正在进行）
        eng.state_machine.transition(RuntimeState.LOADING)
        with pytest.raises(EngineBusyError):
            await eng.load(LoadRequest(variant="v1"))

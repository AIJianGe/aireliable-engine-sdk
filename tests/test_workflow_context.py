"""workflow_context.py 单元测试。

覆盖：
    - WorkflowContext：构造 / 字段访问 / 便捷方法 / 不可变更新
    - ProgressReporter 实现：Null / Console / SSE
    - VRamManager：SimpleVRamManager
    - 取消信号：is_cancelled / wait_for_cancel / raise_if_cancelled
    - 路径便捷方法
    - make_default_context 工厂
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aireliable_engine.workflow_context import (
    ConsoleProgressReporter,
    NullProgressReporter,
    ProgressReporter,
    SimpleVRamManager,
    SSEProgressReporter,
    VRamManager,
    WorkflowContext,
    make_default_context,
)


# ===========================================================================
# fixtures
# ===========================================================================

@pytest.fixture
def basic_ctx() -> WorkflowContext:
    """基础 WorkflowContext。"""
    return WorkflowContext(
        engine_id="ernie_image",
        variant="default",
        device="cuda:0",
        models_root=Path("/models/image"),
        output_root=Path("/data/output"),
        task_id="task_001",
        inputs={"prompt": "a cat", "size": "1024x1024"},
        loras=[{"name": "style_x", "weight": 1.0}],
        extra={"engine_specific": "data"},
    )


# ===========================================================================
# WorkflowContext 基础构造
# ===========================================================================

class TestWorkflowContextBasic:
    """WorkflowContext 基础功能测试。"""

    def test_default_construction(self) -> None:
        """默认构造（无参）。"""
        ctx = WorkflowContext()
        assert ctx.engine_id == ""
        assert ctx.variant == "default"
        assert ctx.device == "cpu"
        assert ctx.models_root is None
        assert ctx.output_root is None
        assert ctx.task_id is None
        assert ctx.cancel_event is None
        assert ctx.inputs == {}
        assert ctx.loras == []
        assert ctx.extra == {}
        # progress_reporter 默认 NullProgressReporter
        assert isinstance(ctx.progress_reporter, NullProgressReporter)
        # vram_manager 默认 None
        assert ctx.vram_manager is None

    def test_full_construction(self, basic_ctx: WorkflowContext) -> None:
        """完整构造。"""
        ctx = basic_ctx
        assert ctx.engine_id == "ernie_image"
        assert ctx.variant == "default"
        assert ctx.device == "cuda:0"
        assert ctx.models_root == Path("/models/image")
        assert ctx.output_root == Path("/data/output")
        assert ctx.task_id == "task_001"
        assert ctx.inputs["prompt"] == "a cat"
        assert ctx.loras == [{"name": "style_x", "weight": 1.0}]
        assert ctx.extra["engine_specific"] == "data"

    def test_repr(self, basic_ctx: WorkflowContext) -> None:
        """repr 输出。"""
        repr_str = repr(basic_ctx)
        assert "ernie_image" in repr_str
        assert "default" in repr_str
        assert "cuda:0" in repr_str
        assert "task_001" in repr_str

    def test_metadata_default_factory_independence(self) -> None:
        """默认值独立性（inputs / loras / extra 不共享）。"""
        ctx1 = WorkflowContext()
        ctx2 = WorkflowContext()
        ctx1.inputs["k"] = "v"
        ctx1.loras.append({"k": "v"})
        ctx1.extra["k"] = "v"
        assert "k" not in ctx2.inputs
        assert ctx2.loras == []
        assert "k" not in ctx2.extra


# ===========================================================================
# 输入便捷方法
# ===========================================================================

class TestInputAccessors:
    """输入便捷方法测试。"""

    def test_get_input(self, basic_ctx: WorkflowContext) -> None:
        """get_input 返回字段值。"""
        assert basic_ctx.get_input("prompt") == "a cat"
        assert basic_ctx.get_input("size") == "1024x1024"
        # 不存在字段返回默认值
        assert basic_ctx.get_input("nonexistent") is None
        assert basic_ctx.get_input("nonexistent", "default") == "default"

    def test_has_input(self, basic_ctx: WorkflowContext) -> None:
        """has_input 检查字段存在。"""
        assert basic_ctx.has_input("prompt")
        assert basic_ctx.has_input("size")
        assert not basic_ctx.has_input("nonexistent")


# ===========================================================================
# 路径便捷方法
# ===========================================================================

class TestPathAccessors:
    """路径便捷方法测试。"""

    def test_get_model_path(self, basic_ctx: WorkflowContext) -> None:
        """get_model_path 构造模型路径。"""
        path = basic_ctx.get_model_path("checkpoint.gguf")
        assert path == Path("/models/image/checkpoint.gguf")

    def test_get_model_path_multi_parts(self, basic_ctx: WorkflowContext) -> None:
        """get_model_path 多片段。"""
        path = basic_ctx.get_model_path("subdir", "checkpoint.gguf")
        assert path == Path("/models/image/subdir/checkpoint.gguf")

    def test_get_model_path_raises_when_no_root(self) -> None:
        """models_root 未设置时抛 ValueError。"""
        ctx = WorkflowContext()
        with pytest.raises(ValueError, match="models_root 未设置"):
            ctx.get_model_path("foo")

    def test_get_output_path(self, basic_ctx: WorkflowContext) -> None:
        """get_output_path 构造输出路径。"""
        path = basic_ctx.get_output_path("images", "task_001.png")
        assert path == Path("/data/output/images/task_001.png")

    def test_get_output_path_raises_when_no_root(self) -> None:
        """output_root 未设置时抛 ValueError。"""
        ctx = WorkflowContext()
        with pytest.raises(ValueError, match="output_root 未设置"):
            ctx.get_output_path("foo")


# ===========================================================================
# 取消信号测试
# ===========================================================================

class TestCancelSignal:
    """取消信号测试。"""

    def test_is_cancelled_no_event(self) -> None:
        """cancel_event 为 None 时永不取消。"""
        ctx = WorkflowContext()
        assert ctx.is_cancelled() is False

    def test_is_cancelled_not_set(self) -> None:
        """event 未 set 时返回 False。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)
        assert ctx.is_cancelled() is False

    def test_is_cancelled_set(self) -> None:
        """event set 后返回 True。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)
        event.set()
        assert ctx.is_cancelled() is True

    def test_raise_if_cancelled_not_set(self) -> None:
        """未取消时不抛异常。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)
        # 不抛异常
        ctx.raise_if_cancelled()

    def test_raise_if_cancelled_set(self) -> None:
        """已取消时抛 CancelledError。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)
        event.set()
        with pytest.raises(asyncio.CancelledError, match="已被取消"):
            ctx.raise_if_cancelled()

    def test_raise_if_cancelled_no_event(self) -> None:
        """无 event 时不抛异常。"""
        ctx = WorkflowContext()
        ctx.raise_if_cancelled()  # 无副作用

    @pytest.mark.asyncio
    async def test_wait_for_cancel_returns_immediately_when_set(self) -> None:
        """event set 时立即返回 True。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)
        event.set()
        result = await ctx.wait_for_cancel(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_cancel_timeout(self) -> None:
        """超时返回 False。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)
        result = await ctx.wait_for_cancel(timeout=0.05)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_cancel_no_event(self) -> None:
        """无 event 时返回 False。"""
        ctx = WorkflowContext()
        result = await ctx.wait_for_cancel(timeout=0.01)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_cancel_returns_when_set_during_wait(self) -> None:
        """等待期间 set event 立即返回。"""
        event = asyncio.Event()
        ctx = WorkflowContext(cancel_event=event)

        async def set_after_delay():
            await asyncio.sleep(0.05)
            event.set()

        asyncio.create_task(set_after_delay())
        result = await ctx.wait_for_cancel(timeout=1.0)
        assert result is True


# ===========================================================================
# 进度上报测试
# ===========================================================================

class TestProgressReporter:
    """ProgressReporter 实现测试。"""

    @pytest.mark.asyncio
    async def test_null_reporter_no_side_effects(self) -> None:
        """NullProgressReporter 不抛异常。"""
        reporter = NullProgressReporter()
        await reporter.report_progress(0.5, "loading")
        await reporter.report_error("msg", "err")
        await reporter.report_info("info")

    @pytest.mark.asyncio
    async def test_console_reporter_does_not_raise(self) -> None:
        """ConsoleProgressReporter 不抛异常（仅打印）。"""
        reporter = ConsoleProgressReporter()
        await reporter.report_progress(0.5, "loading", step=10, total=50)
        await reporter.report_error("error", "test_error")
        await reporter.report_info("info message")

    @pytest.mark.asyncio
    async def test_sse_reporter_puts_frame(self) -> None:
        """SSEProgressReporter 通过 put 推送帧。"""
        class MockStream:
            def __init__(self) -> None:
                self.frames = []

            async def put(self, frame) -> None:
                self.frames.append(frame)

            def put_threadsafe(self, frame) -> None:
                self.frames.append(frame)

        stream = MockStream()
        reporter = SSEProgressReporter(stream)  # type: ignore[arg-type]
        await reporter.report_progress(0.5, "loading", step=10)
        await reporter.report_error("error msg", "test_error")
        await reporter.report_info("info")

        assert len(stream.frames) == 3
        # 第一帧应包含 progress=0.5
        assert "0.5" in str(stream.frames[0]) or 0.5 in stream.frames[0].values()
        # 错误帧应包含 stage=error
        assert stream.frames[1].get("stage") == "error"

    @pytest.mark.asyncio
    async def test_ctx_report_progress_delegates(self) -> None:
        """ctx.report_progress 代理到 progress_reporter。"""
        class MockReporter(ProgressReporter):
            def __init__(self) -> None:
                self.calls = []

            async def report_progress(self, progress, stage="", **kwargs):
                self.calls.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                self.calls.append(("error", message, error_type, kwargs))

            async def report_info(self, message, **kwargs):
                self.calls.append(("info", message, kwargs))

        reporter = MockReporter()
        ctx = WorkflowContext(progress_reporter=reporter)

        await ctx.report_progress(0.5, "stage_x", step=5)
        await ctx.report_error("err", "err_type")
        await ctx.report_info("info msg")

        assert len(reporter.calls) == 3
        assert reporter.calls[0][0] == "progress"
        assert reporter.calls[0][1] == 0.5
        assert reporter.calls[0][2] == "stage_x"
        assert reporter.calls[0][3] == {"step": 5}
        assert reporter.calls[1][0] == "error"
        assert reporter.calls[2][0] == "info"


# ===========================================================================
# VRamManager 测试
# ===========================================================================

class TestVRamManager:
    """VRamManager 实现测试。"""

    @pytest.mark.asyncio
    async def test_simple_manager_load_to_gpu_calls_to(self) -> None:
        """SimpleVRamManager.load_to_gpu 调用 model.to。"""
        class MockModel:
            def __init__(self):
                self.device = "cpu"

            def to(self, device):
                self.device = device
                return self

        model = MockModel()
        manager = SimpleVRamManager(device="cuda:0")
        result = await manager.load_to_gpu(model, "cuda:0")
        assert result is model
        assert model.device == "cuda:0"

    @pytest.mark.asyncio
    async def test_simple_manager_load_to_gpu_no_to_method(self) -> None:
        """无 to() 方法的对象不抛异常。"""
        class NoToDevice:
            pass

        manager = SimpleVRamManager()
        obj = NoToDevice()
        result = await manager.load_to_gpu(obj, "cuda:0")
        assert result is obj

    @pytest.mark.asyncio
    async def test_simple_manager_offload_no_op(self) -> None:
        """SimpleVRamManager.offload_to_cpu 是 no-op。"""
        manager = SimpleVRamManager()
        model = object()
        result = await manager.offload_to_cpu(model)
        assert result is model

    @pytest.mark.asyncio
    async def test_simple_manager_get_free_vram(self) -> None:
        """SimpleVRamManager 返回大数（不限制）。"""
        manager = SimpleVRamManager()
        vram = await manager.get_free_vram_bytes()
        assert vram > 1 << 50  # 大于 1 PB


# ===========================================================================
# 不可变更新方法
# ===========================================================================

class TestImmutableUpdate:
    """不可变更新方法测试。"""

    def test_with_progress_reporter_returns_new(self, basic_ctx: WorkflowContext) -> None:
        """with_progress_reporter 返回新实例。"""
        new_reporter = NullProgressReporter()
        new_ctx = basic_ctx.with_progress_reporter(new_reporter)
        assert new_ctx is not basic_ctx
        assert new_ctx.progress_reporter is new_reporter
        # 其他字段引用相同
        assert new_ctx.engine_id == basic_ctx.engine_id
        assert new_ctx.models_root is basic_ctx.models_root

    def test_with_extra_returns_new(self, basic_ctx: WorkflowContext) -> None:
        """with_extra 返回新实例。"""
        new_ctx = basic_ctx.with_extra(new_key="new_value")
        assert new_ctx is not basic_ctx
        assert new_ctx.extra["new_key"] == "new_value"
        assert new_ctx.extra["engine_specific"] == "data"  # 原有字段保留
        # 原实例 extra 不变
        assert "new_key" not in basic_ctx.extra

    def test_with_extra_overrides_existing(self, basic_ctx: WorkflowContext) -> None:
        """with_extra 覆盖现有字段。"""
        new_ctx = basic_ctx.with_extra(engine_specific="new_data")
        assert new_ctx.extra["engine_specific"] == "new_data"


# ===========================================================================
# 工厂方法测试
# ===========================================================================

class TestMakeDefaultContext:
    """make_default_context 工厂测试。"""

    def test_default_factory_creates_simple_manager(self) -> None:
        """默认创建 SimpleVRamManager。"""
        ctx = make_default_context()
        assert isinstance(ctx.vram_manager, SimpleVRamManager)

    def test_default_factory_creates_null_reporter(self) -> None:
        """默认创建 NullProgressReporter。"""
        ctx = make_default_context()
        assert isinstance(ctx.progress_reporter, NullProgressReporter)

    def test_factory_with_custom_fields(self) -> None:
        """工厂接受自定义字段。"""
        ctx = make_default_context(
            engine_id="ernie_image",
            variant="fp16",
            device="cuda:0",
            models_root=Path("/models"),
            task_id="task_001",
        )
        assert ctx.engine_id == "ernie_image"
        assert ctx.variant == "fp16"
        assert ctx.device == "cuda:0"
        assert ctx.models_root == Path("/models")
        assert ctx.task_id == "task_001"

    def test_factory_respects_custom_manager(self) -> None:
        """工厂尊重自定义 vram_manager。"""
        class CustomManager(SimpleVRamManager):
            pass

        manager = CustomManager()
        ctx = make_default_context(vram_manager=manager)
        assert ctx.vram_manager is manager

    def test_factory_respects_custom_reporter(self) -> None:
        """工厂尊重自定义 progress_reporter。"""
        reporter = ConsoleProgressReporter()
        ctx = make_default_context(progress_reporter=reporter)
        assert ctx.progress_reporter is reporter

    def test_factory_passes_extra_kwargs(self) -> None:
        """工厂接受额外 kwargs。"""
        ctx = make_default_context(
            extra={"k": "v"},
            inputs={"prompt": "test"},
            task_id="task_001",
        )
        assert ctx.extra == {"k": "v"}
        assert ctx.inputs == {"prompt": "test"}
        assert ctx.task_id == "task_001"

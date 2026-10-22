"""WorkflowEngine 单元测试（v0.2 阶段 5 任务 5.2）。

覆盖任务 5.2 验收标准：
    - 加载工作流测试
    - 推理测试（mock 节点）
    - 热重载测试

参考：
    - v0.2-tasks.md 任务 5.2
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aireliable_engine import (
    BaseNode,
    CapabilitiesResponse,
    InferRequest,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraLoadResponse,
    LoraSpec,
    LoraUnloadRequest,
    LoraUnloadResponse,
    NodeRegistry,
    PortSpec,
    SSEStream,
    StatusResponse,
    UnloadResponse,
    VariantInfo,
    WorkflowEngine,
    WorkflowEngineError,
    WorkflowLoadError,
    WorkflowNotFoundError,
    register_node,
)
from aireliable_engine.sse import make_error_frame, make_final_frame, make_frame
from aireliable_engine.state_machine import RuntimeState


# ===========================================================================
# 测试用节点
# ===========================================================================

class _TestSourceNode(BaseNode):
    """源节点：输出固定值或来自 params 的值。"""

    node_type = "TestSource"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [PortSpec(name="value", type="any")]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._value = params.get("value", "default") if params else "default"

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        return {"value": self._value}


class _TestPassThroughNode(BaseNode):
    """传递节点：把输入直接输出。"""

    node_type = "TestPassThrough"
    input_specs: list[PortSpec] = [PortSpec(name="data", type="any")]
    output_specs: list[PortSpec] = [PortSpec(name="data", type="any")]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        return {"data": inputs.get("data")}


class _TestFailingNode(BaseNode):
    """总是失败的节点。"""

    node_type = "TestFailing"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [PortSpec(name="value", type="any")]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        raise RuntimeError("测试节点执行失败")


class _TestSlowNode(BaseNode):
    """慢节点：执行前等待一段时间（用于并发测试）。"""

    node_type = "TestSlow"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [PortSpec(name="value", type="any")]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._delay = params.get("delay", 0.3) if params else 0.3

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        await asyncio.sleep(self._delay)
        return {"value": "slow_result"}


# ===========================================================================
# 测试用 WorkflowEngine 子类
# ===========================================================================

class _TestWorkflowEngine(WorkflowEngine):
    """测试用 WorkflowEngine 子类。

    通过 workflows_dict 字典提供工作流 yaml，避免文件 I/O。
    """

    def __init__(
        self,
        workflows_dict: dict[str, str],
        *,
        engine_id: str = "test_engine",
        engine_name: str = "Test Engine",
        capability: str = "text-to-image",
        engine_type: str = "image",
        default_workflow: str = "default",
        workflows_dir: Optional[Path] = None,
        node_registry: Optional[NodeRegistry] = None,
    ) -> None:
        super().__init__(
            engine_id=engine_id,
            engine_name=engine_name,
            capability=capability,
            engine_type=engine_type,
            workflows_dir=workflows_dir,
            default_workflow=default_workflow,
            node_registry=node_registry,
        )
        # 用 dict 模拟工作流文件存储
        self._workflows_dict = workflows_dict

    async def _build_workflow(self, name: Optional[str] = None) -> str:
        workflow_name = name or self.current_workflow
        if workflow_name not in self._workflows_dict:
            raise WorkflowNotFoundError(workflow_name, available=list(self._workflows_dict.keys()))
        return self._workflows_dict[workflow_name]

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=[VariantInfo(
                name="default",
                base_model="test",
                precision="fp16",
                loader="diffusers",
                vram_required="8GB",
            )],
            loras=[],
            supported_sizes=[],
        )

    def _get_default_workflow_template(self) -> str:
        """测试用默认工作流模板（v0.2.5 L6 抽象方法实现）。"""
        return """# 测试用默认工作流模板
workflow:
  name: default
  description: Test default workflow
  nodes:
    - id: input_prompt
      type: TextInput
      pos: [100, 100]
      params:
        field: prompt
    - id: image_out
      type: ImageSave
      pos: [400, 100]
      inputs:
        images: input_prompt
"""


# ===========================================================================
# 工具函数
# ===========================================================================

async def _collect_sse_frames(stream: SSEStream) -> list[dict]:
    """收集 SSE 流的所有帧。"""
    frames: list[dict] = []
    async for frame_str in stream.events():
        # 帧格式：data: {json}\n\n
        assert frame_str.startswith("data: ")
        assert frame_str.endswith("\n\n")
        json_str = frame_str[len("data: "):-2]
        frames.append(json.loads(json_str))
    return frames


def _make_simple_workflow_yaml() -> str:
    """构造一个简单的工作流 yaml。"""
    return """
name: simple_test
description: 简单测试工作流
nodes:
  - id: source
    type: TestSource
    params:
      value: "hello workflow"
outputs:
  result: source.output.value
"""


def _make_pass_through_workflow_yaml() -> str:
    """构造一个传递工作流 yaml。"""
    return """
name: pass_through
nodes:
  - id: src
    type: TestSource
    params:
      value: "${inputs.text}"
  - id: pass
    type: TestPassThrough
    inputs:
      data: src.output.value
outputs:
  result: pass.output.data
"""


def _make_failing_workflow_yaml() -> str:
    """构造一个失败的工作流 yaml。"""
    return """
name: failing
nodes:
  - id: fail
    type: TestFailing
outputs:
  result: fail.output.value
"""


# ===========================================================================
# fixtures
# ===========================================================================

@pytest.fixture
def registry() -> NodeRegistry:
    """独立节点注册表（不污染全局）。"""
    reg = NodeRegistry()
    reg.register("TestSource", _TestSourceNode)
    reg.register("TestPassThrough", _TestPassThroughNode)
    reg.register("TestFailing", _TestFailingNode)
    reg.register("TestSlow", _TestSlowNode)
    return reg


@pytest.fixture
def simple_engine(registry) -> _TestWorkflowEngine:
    """带简单工作流的测试引擎。"""
    return _TestWorkflowEngine(
        workflows_dict={"default": _make_simple_workflow_yaml()},
        node_registry=registry,
    )


@pytest.fixture
def ctx():
    """pytest-asyncio 默认事件循环 fixture（供创建 asyncio.Event 用）。"""
    return None


# ===========================================================================
# 构造测试
# ===========================================================================

class TestConstruction:
    """WorkflowEngine 构造测试。"""

    def test_default_construction(self, registry):
        """默认构造。"""
        engine = _TestWorkflowEngine(
            workflows_dict={},
            node_registry=registry,
        )
        assert engine.engine_id == "test_engine"
        assert engine.engine_name == "Test Engine"
        assert engine.capability == "text-to-image"
        assert engine.engine_type == "image"
        assert engine.default_workflow == "default"
        assert engine.current_workflow == "default"
        assert engine.node_registry is registry
        assert engine.loaded_variant is None
        assert engine.loaded_loras == []
        assert engine.last_used_at is None
        assert engine._workflow_cache == {}

    def test_default_workflow_custom(self, registry):
        """自定义默认工作流。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"custom": "..."},
            default_workflow="custom",
            node_registry=registry,
        )
        assert engine.default_workflow == "custom"
        assert engine.current_workflow == "custom"

    def test_uses_global_registry_if_not_provided(self):
        """未提供 registry 时使用全局默认注册表。"""
        engine = _TestWorkflowEngine(workflows_dict={})
        # 应该是全局默认注册表实例
        assert engine.node_registry is NodeRegistry._get_default()

    def test_workflows_dir_can_be_none(self, registry):
        """workflows_dir 可以为 None（子类自行管理路径）。"""
        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=None,
            node_registry=registry,
        )
        assert engine.workflows_dir is None
        assert engine.list_workflows() == []


# ===========================================================================
# 8 端点：load / unload / status
# ===========================================================================

class TestLoadUnloadStatus:
    """load / unload / status 端点测试。"""

    @pytest.mark.asyncio
    async def test_load_records_variant(self, simple_engine):
        """load 记录 variant 名。"""
        response = await simple_engine.load(LoadRequest(variant="ernie-q4"))
        assert response.variant == "ernie-q4"
        assert response.state == "ready"
        assert simple_engine.loaded_variant == "ernie-q4"
        assert simple_engine.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_unload_clears_variant(self, simple_engine):
        """unload 清空 variant 记录。"""
        await simple_engine.load(LoadRequest(variant="ernie-q4"))
        response = await simple_engine.unload()
        assert response.state == "idle"
        assert simple_engine.loaded_variant is None
        assert simple_engine.state_machine.state == RuntimeState.IDLE

    @pytest.mark.asyncio
    async def test_unload_clears_loras(self, simple_engine):
        """unload 清空 LoRA 列表。"""
        await simple_engine.load(LoadRequest(variant="v1"))
        await simple_engine.load_lora(LoraLoadRequest(name="lora1", weight=0.5))
        assert len(simple_engine.loaded_loras) == 1
        await simple_engine.unload()
        assert simple_engine.loaded_loras == []

    @pytest.mark.asyncio
    async def test_status_returns_workflow_info(self, simple_engine):
        """status 返回工作流信息。"""
        await simple_engine.load(LoadRequest(variant="v1"))
        simple_engine.last_used_at = 1234567890.0
        status = await simple_engine.status()
        assert status.state == "ready"
        assert status.loaded_variant == "v1"
        assert status.last_used_at == 1234567890.0
        assert "current_workflow" in status.extra
        assert status.extra["current_workflow"] == "default"

    @pytest.mark.asyncio
    async def test_status_extra_has_available_workflows(self, registry):
        """status.extra 包含 available_workflows。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"default": "...", "alt": "..."},
            node_registry=registry,
        )
        # 由于 workflows_dir 是 None，list_workflows 返回 []
        status = await engine.status()
        assert "available_workflows" in status.extra
        assert status.extra["available_workflows"] == []


# ===========================================================================
# 8 端点：LoRA 加载/卸载
# ===========================================================================

class TestLoraLoadUnload:
    """LoRA 加载 / 卸载测试。"""

    @pytest.mark.asyncio
    async def test_load_lora_appends(self, simple_engine):
        """load_lora 添加到列表。"""
        response = await simple_engine.load_lora(LoraLoadRequest(name="lora1", weight=0.8))
        assert response.name == "lora1"
        assert response.weight == 0.8
        assert "lora1" in response.loaded_loras
        assert "lora1" in simple_engine.loaded_loras

    @pytest.mark.asyncio
    async def test_load_lora_dedup(self, simple_engine):
        """重复 load_lora 同名 LoRA 不重复添加。"""
        await simple_engine.load_lora(LoraLoadRequest(name="lora1", weight=0.8))
        await simple_engine.load_lora(LoraLoadRequest(name="lora1", weight=0.5))
        assert simple_engine.loaded_loras.count("lora1") == 1

    @pytest.mark.asyncio
    async def test_load_multiple_loras(self, simple_engine):
        """加载多个不同 LoRA。"""
        await simple_engine.load_lora(LoraLoadRequest(name="lora1"))
        await simple_engine.load_lora(LoraLoadRequest(name="lora2"))
        assert len(simple_engine.loaded_loras) == 2

    @pytest.mark.asyncio
    async def test_unload_lora_removes(self, simple_engine):
        """unload_lora 从列表移除。"""
        await simple_engine.load_lora(LoraLoadRequest(name="lora1"))
        await simple_engine.load_lora(LoraLoadRequest(name="lora2"))
        response = await simple_engine.unload_lora(LoraUnloadRequest(name="lora1"))
        assert "lora1" not in response.loaded_loras
        assert "lora2" in response.loaded_loras
        assert "lora1" not in simple_engine.loaded_loras

    @pytest.mark.asyncio
    async def test_unload_nonexistent_lora_ok(self, simple_engine):
        """unload 不存在的 LoRA 不报错。"""
        response = await simple_engine.unload_lora(LoraUnloadRequest(name="not_loaded"))
        assert response.name == "not_loaded"
        assert response.loaded_loras == []


# ===========================================================================
# 推理测试
# ===========================================================================

class TestInference:
    """推理端点测试。"""

    @pytest.mark.asyncio
    async def test_basic_inference(self, simple_engine):
        """基本推理：执行工作流并返回结果。"""
        request = InferRequest(
            variant="default",
            inputs={},
        )
        frames = []
        async for frame in simple_engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        # 首次 infer 会先自动加载，model_loading 帧必须在 workflow 帧之前
        stages = [f["stage"] for f in frames]
        assert stages.index("model_loading") < stages.index("workflow_start")

        business = [f for f in frames if f["stage"] != "model_loading"]
        assert len(business) >= 3
        # 工作流第一帧：workflow_start
        assert business[0]["stage"] == "workflow_start"
        assert business[0]["workflow"] == "default"
        # 最终帧：completed
        assert business[-1]["stage"] == "completed"
        assert business[-1]["progress"] == 1.0
        assert business[-1]["result"]["result"] == "hello workflow"

    @pytest.mark.asyncio
    async def test_inference_workflow_loaded_frame(self, simple_engine):
        """推理时推送 workflow_loaded 帧（含执行顺序）。"""
        request = InferRequest(variant="default", inputs={})
        frames = []
        async for frame in simple_engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        # 找到 workflow_loaded 帧
        loaded_frames = [f for f in frames if f["stage"] == "workflow_loaded"]
        assert len(loaded_frames) == 1
        assert loaded_frames[0]["workflow"] == "default"
        assert loaded_frames[0]["node_count"] == 1
        assert loaded_frames[0]["execution_order"] == ["source"]

    @pytest.mark.asyncio
    async def test_inference_with_input_injection(self, registry):
        """推理时 inputs 注入到工作流。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"default": _make_pass_through_workflow_yaml()},
            node_registry=registry,
        )
        request = InferRequest(
            variant="default",
            inputs={"text": "injected text"},
        )
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        final = frames[-1]
        assert final["stage"] == "completed"
        assert final["result"]["result"] == "injected text"

    @pytest.mark.asyncio
    async def test_inference_with_workflow_field(self, registry):
        """request.workflow 指定工作流。"""
        engine = _TestWorkflowEngine(
            workflows_dict={
                "default": _make_simple_workflow_yaml(),
                "alt": """
name: alt_workflow
nodes:
  - id: src
    type: TestSource
    params:
      value: "from alt workflow"
outputs:
  result: src.output.value
""",
            },
            node_registry=registry,
        )
        request = InferRequest(
            variant="default",
            inputs={},
            workflow="alt",
        )
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        business = [f for f in frames if f["stage"] != "model_loading"]
        assert business[0]["workflow"] == "alt"
        assert business[-1]["result"]["result"] == "from alt workflow"

    @pytest.mark.asyncio
    async def test_inference_workflow_not_found(self, registry):
        """工作流不存在时推送错误帧。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"default": _make_simple_workflow_yaml()},
            node_registry=registry,
        )
        request = InferRequest(
            variant="default",
            inputs={},
            workflow="nonexistent",
        )
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        # 首个业务帧是 workflow_start（其前可能有自动加载的 model_loading 帧）
        business = [f for f in frames if f["stage"] != "model_loading"]
        assert business[0]["stage"] == "workflow_start"
        # 第二帧应该是错误帧
        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) == 1
        assert "nonexistent" in error_frames[0]["error"]["message"]

    @pytest.mark.asyncio
    async def test_inference_node_failure(self, registry):
        """节点执行失败时推送错误帧。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"default": _make_failing_workflow_yaml()},
            node_registry=registry,
        )
        request = InferRequest(variant="default", inputs={})
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) == 1
        assert "fail" in error_frames[0]["error"]["message"].lower() or "测试" in error_frames[0]["error"]["message"]

    @pytest.mark.asyncio
    async def test_inference_records_last_used_at(self, simple_engine):
        """推理后更新 last_used_at。"""
        assert simple_engine.last_used_at is None
        request = InferRequest(variant="default", inputs={})
        # 消费完整个流
        async for _ in simple_engine.infer(request):
            pass
        assert simple_engine.last_used_at is not None
        assert simple_engine.last_used_at > 0

    @pytest.mark.asyncio
    async def test_inference_state_transitions(self, simple_engine):
        """推理过程中状态机转换。"""
        # 初始 IDLE
        assert simple_engine.state_machine.state == RuntimeState.IDLE
        request = InferRequest(variant="default", inputs={})
        # 推理过程中应是 INFERRING，但因为是异步的，我们只能验证最终
        async for _ in simple_engine.infer(request):
            pass
        # 推理完成后应回到 READY（因 variant 自动加载）
        assert simple_engine.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_inference_with_loras(self, registry):
        """推理带 LoRA。"""
        engine = _TestWorkflowEngine(
            workflows_dict={
                "default": """
name: lora_test
nodes:
  - id: src
    type: TestSource
    params:
      value: "${loras}"
outputs:
  result: src.output.value
""",
            },
            node_registry=registry,
        )
        request = InferRequest(
            variant="default",
            inputs={},
            loras=[LoraSpec(name="lora1", weight=0.8), LoraSpec(name="lora2", weight=0.5)],
        )
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        final = frames[-1]
        assert final["stage"] == "completed"
        # loras 应该是 dict 列表
        loras_result = final["result"]["result"]
        assert isinstance(loras_result, list)
        assert len(loras_result) == 2
        assert loras_result[0]["name"] == "lora1"


# ===========================================================================
# 工作流管理
# ===========================================================================

class TestWorkflowManagement:
    """工作流管理测试。"""

    @pytest.mark.asyncio
    async def test_build_workflow_default(self, simple_engine):
        """_build_workflow 不传 name 时使用 current_workflow。"""
        yaml_text = await simple_engine._build_workflow()
        assert "simple_test" in yaml_text

    @pytest.mark.asyncio
    async def test_build_workflow_explicit_name(self, registry):
        """_build_workflow 传 name 时加载指定工作流。"""
        engine = _TestWorkflowEngine(
            workflows_dict={
                "default": "name: default_yaml\nnodes: []\n",
                "alt": "name: alt_yaml\nnodes: []\n",
            },
            node_registry=registry,
        )
        yaml_text = await simple_engine._build_workflow() if False else await engine._build_workflow("alt")
        assert "alt_yaml" in yaml_text

    @pytest.mark.asyncio
    async def test_build_workflow_not_found(self, simple_engine):
        """_build_workflow 加载不存在的工作流抛 WorkflowNotFoundError。"""
        with pytest.raises(WorkflowNotFoundError) as exc_info:
            await simple_engine._build_workflow("nonexistent")
        assert exc_info.value.workflow_name == "nonexistent"
        assert "default" in exc_info.value.available  # 提示可用工作流

    def test_set_current_workflow(self, simple_engine):
        """set_current_workflow 切换当前工作流。"""
        assert simple_engine.current_workflow == "default"
        simple_engine.set_current_workflow("alt")
        assert simple_engine.current_workflow == "alt"

    def test_reload_workflow_single(self, registry, tmp_path):
        """reload_workflow 清空单个工作流缓存。"""
        # 创建实际文件以测试 _read_workflow_file
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        wf_file = workflows_dir / "default.yaml"
        wf_file.write_text("name: default\nnodes: []\n", encoding="utf-8")

        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        # 直接读取文件
        text1 = engine._read_workflow_file()
        assert "default" in text1
        # 缓存命中
        text2 = engine._read_workflow_file()
        assert text1 == text2
        # 修改文件
        time.sleep(0.01)  # 确保 mtime 变化
        wf_file.write_text("name: default_updated\nnodes: []\n", encoding="utf-8")
        # 未清缓存但 mtime 变了，应自动热重载
        text3 = engine._read_workflow_file()
        assert "default_updated" in text3

    def test_reload_workflow_all(self, registry, tmp_path):
        """reload_workflow() 清空所有缓存。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text("name: default\n", encoding="utf-8")
        (workflows_dir / "alt.yaml").write_text("name: alt\n", encoding="utf-8")

        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        # 读取两个文件建立缓存
        engine._read_workflow_file("default")
        engine._read_workflow_file("alt")
        assert len(engine._workflow_cache) == 2

        # 清空所有缓存
        engine.reload_workflow()
        assert engine._workflow_cache == {}

    def test_list_workflows_empty(self, registry):
        """list_workflows 在 workflows_dir 为 None 时返回空列表。"""
        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=None,
            node_registry=registry,
        )
        assert engine.list_workflows() == []

    def test_list_workflows_from_dir(self, registry, tmp_path):
        """list_workflows 扫描 workflows_dir 目录。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text("name: default\n", encoding="utf-8")
        (workflows_dir / "alt.yaml").write_text("name: alt\n", encoding="utf-8")
        (workflows_dir / "readme.txt").write_text("not a workflow", encoding="utf-8")

        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        result = engine.list_workflows()
        assert sorted(result) == ["alt", "default"]

    def test_read_workflow_file_not_found(self, registry, tmp_path):
        """_read_workflow_file 文件不存在抛 WorkflowNotFoundError。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )
        with pytest.raises(WorkflowNotFoundError):
            engine._read_workflow_file("nonexistent")

    def test_read_workflow_file_no_workflows_dir(self, registry):
        """_read_workflow_file 在 workflows_dir 为 None 时抛 WorkflowNotFoundError。"""
        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=None,
            node_registry=registry,
        )
        with pytest.raises(WorkflowNotFoundError):
            engine._read_workflow_file("any")


# ===========================================================================
# 热重载测试
# ===========================================================================

class TestHotReload:
    """工作流热重载测试。"""

    def test_hot_reload_detects_mtime_change(self, registry, tmp_path):
        """文件 mtime 变化时自动重新加载。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        wf_file = workflows_dir / "default.yaml"
        wf_file.write_text("name: v1\n", encoding="utf-8")

        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )

        # 第一次读取
        text1 = engine._read_workflow_file()
        assert "v1" in text1

        # 第二次读取（缓存命中）
        text2 = engine._read_workflow_file()
        assert text1 == text2

        # 修改文件
        time.sleep(0.05)  # 确保 mtime 变化（不同文件系统精度不同）
        wf_file.write_text("name: v2\n", encoding="utf-8")

        # 第三次读取（应自动热重载）
        text3 = engine._read_workflow_file()
        assert "v2" in text3
        assert "v1" not in text3

    def test_hot_reload_clear_cache_forces_reload(self, registry, tmp_path):
        """reload_workflow 强制重新加载（即使 mtime 未变）。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        wf_file = workflows_dir / "default.yaml"
        wf_file.write_text("name: v1\n", encoding="utf-8")

        engine = _TestWorkflowEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )

        # 第一次读取
        text1 = engine._read_workflow_file()
        assert "v1" in text1

        # 修改文件但 mtime 未变（同秒内修改）
        wf_file.write_text("name: v1_updated\n", encoding="utf-8")

        # 未清缓存：mtime 可能未变，可能仍命中缓存
        # 清空缓存
        engine.reload_workflow("default")

        # 再次读取应重新加载
        text3 = engine._read_workflow_file()
        assert "v1_updated" in text3

    @pytest.mark.asyncio
    async def test_inference_picks_up_reloaded_workflow(self, registry, tmp_path):
        """推理时使用热重载后的工作流。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        wf_file = workflows_dir / "default.yaml"
        wf_file.write_text("""
name: reloadable
nodes:
  - id: src
    type: TestSource
    params:
      value: "before_reload"
outputs:
  result: src.output.value
""", encoding="utf-8")

        # 重写 _build_workflow 使用文件读取
        class _FileEngine(_TestWorkflowEngine):
            async def _build_workflow(self, name=None):
                return self._read_workflow_file(name)

        engine = _FileEngine(
            workflows_dict={},
            workflows_dir=workflows_dir,
            node_registry=registry,
        )

        # 第一次推理
        request = InferRequest(variant="default", inputs={})
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))
        assert frames[-1]["result"]["result"] == "before_reload"

        # 修改工作流
        time.sleep(0.05)
        wf_file.write_text("""
name: reloadable
nodes:
  - id: src
    type: TestSource
    params:
      value: "after_reload"
outputs:
  result: src.output.value
""", encoding="utf-8")

        # 第二次推理（应自动热重载）
        request2 = InferRequest(variant="default", inputs={})
        frames2 = []
        async for frame in engine.infer(request2):
            frames2.append(json.loads(frame[len("data: "):-2]))
        assert frames2[-1]["result"]["result"] == "after_reload"


# ===========================================================================
# 异常处理
# ===========================================================================

class TestErrorHandling:
    """异常处理测试。"""

    @pytest.mark.asyncio
    async def test_workflow_not_found_error_attributes(self):
        """WorkflowNotFoundError 包含 workflow_name 和 available 属性。"""
        err = WorkflowNotFoundError("missing", available=["a", "b"])
        assert err.workflow_name == "missing"
        assert err.available == ["a", "b"]
        assert "missing" in str(err)
        assert "a" in str(err)

    @pytest.mark.asyncio
    async def test_workflow_load_error_attributes(self):
        """WorkflowLoadError 包含 workflow_name 和 cause 属性。"""
        cause = OSError("disk error")
        err = WorkflowLoadError("bad", cause)
        assert err.workflow_name == "bad"
        assert err.cause is cause
        assert "bad" in str(err)

    def test_workflow_engine_error_is_exception(self):
        """WorkflowEngineError 是 Exception 子类。"""
        err = WorkflowEngineError("test")
        assert isinstance(err, Exception)

    @pytest.mark.asyncio
    async def test_inference_with_malformed_yaml(self, registry):
        """工作流 yaml 格式错误时推送错误帧。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"default": "not: valid: yaml: [unclosed"},
            node_registry=registry,
        )
        request = InferRequest(variant="default", inputs={})
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) >= 1

    @pytest.mark.asyncio
    async def test_inference_with_node_not_registered(self, registry):
        """节点未注册时推送错误帧。"""
        engine = _TestWorkflowEngine(
            workflows_dict={"default": """
name: missing_node
nodes:
  - id: src
    type: NotRegisteredNode
outputs:
  result: src.output.value
"""},
            node_registry=registry,
        )
        request = InferRequest(variant="default", inputs={})
        frames = []
        async for frame in engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) >= 1


# ===========================================================================
# 兼容性 / 接口契约
# ===========================================================================

class TestBaseEngineCompatibility:
    """WorkflowEngine 与 BaseEngine 接口兼容性测试。"""

    def test_inherits_base_engine(self, simple_engine):
        """WorkflowEngine 是 BaseEngine 子类。"""
        from aireliable_engine.engine_base import BaseEngine
        assert isinstance(simple_engine, BaseEngine)

    def test_has_all_8_endpoints(self, simple_engine):
        """WorkflowEngine 实现 8 个端点方法。"""
        # 健康检查
        assert hasattr(simple_engine, "health")
        # 能力声明
        assert hasattr(simple_engine, "capabilities")
        # 加载
        assert hasattr(simple_engine, "load")
        # 卸载
        assert hasattr(simple_engine, "unload")
        # 推理
        assert hasattr(simple_engine, "infer")
        # 状态
        assert hasattr(simple_engine, "status")
        # LoRA 加载
        assert hasattr(simple_engine, "load_lora")
        # LoRA 卸载
        assert hasattr(simple_engine, "unload_lora")

    @pytest.mark.asyncio
    async def test_health_response(self, simple_engine):
        """health 端点响应。"""
        response = await simple_engine.health()
        assert response.status == "ok"
        assert response.engine_id == "test_engine"
        assert response.state == "idle"

    @pytest.mark.asyncio
    async def test_capabilities_response(self, simple_engine):
        """capabilities 端点响应。"""
        response = await simple_engine.capabilities()
        assert response.engine_id == "test_engine"
        assert response.capability == "text-to-image"
        assert response.type == "image"
        assert len(response.variants) == 1
        assert response.variants[0].name == "default"

    @pytest.mark.asyncio
    async def test_infer_returns_sse_stream(self, simple_engine):
        """infer 返回 SSE 流（异步迭代器）。"""
        request = InferRequest(variant="default", inputs={})
        # infer 返回 AsyncIterator[str]
        result = simple_engine.infer(request)
        # 应该是异步可迭代的
        assert hasattr(result, "__aiter__")
        # 消费完所有帧
        async for _ in result:
            pass


# ===========================================================================
# 并发防护
# ===========================================================================

class TestConcurrencyGuard:
    """并发防护测试（继承自 BaseEngine，仅做兼容性验证）。

    详细并发测试见 test_engine_base.py。
    """

    @pytest.mark.asyncio
    async def test_load_in_inferring_state_raises(self, simple_engine):
        """INFERRING 状态下 load 抛 EngineBusyError。"""
        from aireliable_engine.engine_base import EngineBusyError

        # 状态转换：IDLE → LOADING → READY → INFERRING
        simple_engine.state_machine.transition(RuntimeState.LOADING)
        simple_engine.state_machine.transition(RuntimeState.READY)
        simple_engine.state_machine.transition(RuntimeState.INFERRING)

        with pytest.raises(EngineBusyError):
            await simple_engine.load(LoadRequest(variant="other"))

    @pytest.mark.asyncio
    async def test_infer_in_inferring_state_rejects(self, simple_engine):
        """INFERRING 状态下再次 infer 应推送 engine_busy 错误帧。"""
        # 状态转换：IDLE → LOADING → READY → INFERRING
        simple_engine.state_machine.transition(RuntimeState.LOADING)
        simple_engine.state_machine.transition(RuntimeState.READY)
        simple_engine.state_machine.transition(RuntimeState.INFERRING)

        request = InferRequest(variant="default", inputs={})
        frames = []
        async for frame in simple_engine.infer(request):
            frames.append(json.loads(frame[len("data: "):-2]))

        # 应立即返回 engine_busy 错误帧
        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) >= 1


async def _collect_sse_frames_from_infer(engine, request):
    """从 infer 收集所有帧。"""
    frames = []
    async for frame in engine.infer(request):
        frames.append(frame)
    return frames

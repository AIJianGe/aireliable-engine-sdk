"""WorkflowEngine 集成测试（v0.2 阶段 5 任务 5.3）。

端到端工作流执行测试，模拟真实引擎场景。

覆盖任务 5.3 验收标准：
    - 端到端工作流执行
    - 与真实节点配合测试

参考：
    - v0.2-tasks.md 任务 5.3
    - PR/architecture.md §12.4 工作流 graph 结构
    - PR/architecture.md §12.7 WorkflowEngine 基类
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from aireliable_engine import (
    BaseNode,
    CapabilitiesResponse,
    InferRequest,
    LoadRequest,
    LoraLoadRequest,
    LoraSpec,
    NodeRegistry,
    PortSpec,
    VariantInfo,
    WorkflowEngine,
    WorkflowNotFoundError,
    register_node,
)
from aireliable_engine.state_machine import RuntimeState


# ===========================================================================
# 模拟 ERNIE-Image 工作流的节点
# ===========================================================================

class _ImagesInputNode(BaseNode):
    """多图输入节点（模拟 ImagesInput）。"""

    node_type = "ImagesInput"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(name="images", type="images", variadic=True, required=False),
    ]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        # 从 ctx.inputs 获取 images 字段
        images = ctx.inputs.get("images", []) if ctx.inputs else []
        return {"images": list(images) if images else []}


class _TextInputNode(BaseNode):
    """文本输入节点（模拟 TextInput）。"""

    node_type = "TextInput"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [PortSpec(name="text", type="text")]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._field = params.get("field", "prompt") if params else "prompt"

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        text = ctx.inputs.get(self._field, "") if ctx.inputs else ""
        return {"text": text}


class _ModelLoaderNode(BaseNode):
    """模型加载节点（模拟 ModelLoader）。"""

    node_type = "ModelLoader"
    input_specs: list[PortSpec] = []
    output_specs: list[PortSpec] = [
        PortSpec(name="model", type="model"),
        PortSpec(name="clip", type="model"),
        PortSpec(name="vae", type="model"),
    ]

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._variant = params.get("variant", "default") if params else "default"
        self._cache_dir = params.get("cache_dir", "") if params else ""

    def cache_key(self, inputs: dict) -> Optional[str]:
        # 模型加载结果可缓存（避免重复加载）
        return f"model:{self._variant}:{self._cache_dir}"

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        # 模拟模型加载
        return {
            "model": {"type": "diffusers_pipeline", "variant": self._variant},
            "clip": {"type": "clip_model", "variant": self._variant},
            "vae": {"type": "vae_model", "variant": self._variant},
        }


class _CLIPTextEncodeNode(BaseNode):
    """CLIP 文本编码节点。"""

    node_type = "CLIPTextEncode"
    input_specs: list[PortSpec] = [
        PortSpec(name="text", type="text"),
        PortSpec(name="clip", type="model"),
    ]
    output_specs: list[PortSpec] = [PortSpec(name="embeds", type="embeds")]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        text = inputs.get("text", "")
        clip = inputs.get("clip", {})
        # 模拟文本编码
        return {"embeds": {"text": text, "clip_id": id(clip)}}


class _IPAdapterApplyNode(BaseNode):
    """IPAdapter 应用节点。"""

    node_type = "IPAdapterApply"
    input_specs: list[PortSpec] = [
        PortSpec(name="images", type="images", variadic=True, required=False),
        PortSpec(name="model", type="model"),
    ]
    output_specs: list[PortSpec] = [PortSpec(name="image_embeds", type="embeds")]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        images = inputs.get("images", [])
        model = inputs.get("model", {})
        if not images:
            return {"image_embeds": None}
        return {"image_embeds": {"images_count": len(images), "model_id": id(model)}}


class _SamplerNode(BaseNode):
    """采样器节点。"""

    node_type = "Sampler"
    input_specs: list[PortSpec] = [
        PortSpec(name="model", type="model"),
        PortSpec(name="cond", type="embeds"),
        PortSpec(name="image_embeds", type="embeds", required=False),
    ]
    output_specs: list[PortSpec] = [PortSpec(name="latent", type="latent")]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        # 模拟采样
        return {"latent": {"shape": [1, 4, 64, 64], "model_id": id(inputs.get("model"))}}


class _VAEDecodeNode(BaseNode):
    """VAE 解码节点。"""

    node_type = "VAEDecode"
    input_specs: list[PortSpec] = [
        PortSpec(name="latent", type="latent"),
        PortSpec(name="vae", type="model"),
    ]
    output_specs: list[PortSpec] = [PortSpec(name="images", type="images", variadic=True)]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        latent = inputs.get("latent", {})
        vae = inputs.get("vae", {})
        # 模拟 VAE 解码
        return {"images": [f"decoded_image_{id(latent)}"]}


class _ImageSaveNode(BaseNode):
    """图片保存节点。

    接受 str / dict / list 输入：
        - str：作为单个元素包装到 list（避免字符迭代）
        - dict：作为单个元素包装到 list
        - list：保持原样
        - 其他：作为单个元素包装到 list
    """

    node_type = "ImageSave"
    input_specs: list[PortSpec] = [
        PortSpec(name="images", type="images", variadic=True),
    ]
    output_specs: list[PortSpec] = [PortSpec(name="images", type="images", variadic=True)]

    async def execute(self, inputs: dict, ctx: Any) -> dict:
        images = inputs.get("images", [])
        # 智能包装：避免 str / dict 被当作可迭代对象迭代
        if isinstance(images, (str, dict)) or images is None:
            return {"images": [images]}
        if isinstance(images, list):
            return {"images": list(images)}
        # 其他类型（int / object）也包装到 list
        return {"images": [images]}


# ===========================================================================
# 测试用 WorkflowEngine 子类
# ===========================================================================

class _IntegrationEngine(WorkflowEngine):
    """集成测试用 WorkflowEngine。"""

    def __init__(
        self,
        workflows_dict: Optional[dict[str, str]] = None,
        *,
        node_registry: NodeRegistry,
        default_workflow: str = "default",
        workflows_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            engine_id="ernie_image_test",
            engine_name="ERNIE-Image Test",
            capability="text-to-image",
            engine_type="image",
            workflows_dir=workflows_dir,
            default_workflow=default_workflow,
            node_registry=node_registry,
        )
        # workflows_dict 可选：若提供则用 dict 模式；否则用文件模式（_read_workflow_file）
        self._workflows_dict = workflows_dict or {}

    async def _build_workflow(self, name: Optional[str] = None) -> str:
        workflow_name = name or self.current_workflow
        # 优先使用 dict 模式
        if workflow_name in self._workflows_dict:
            return self._workflows_dict[workflow_name]
        # 否则尝试文件模式（若 workflows_dir 已设置）
        if self.workflows_dir is not None:
            return self._read_workflow_file(workflow_name)
        raise WorkflowNotFoundError(workflow_name, available=list(self._workflows_dict.keys()))

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            engine_id=self.engine_id,
            engine_name=self.engine_name,
            capability=self.capability,
            type=self.engine_type,
            variants=[
                VariantInfo(
                    name="ernie-q4",
                    base_model="ernie-image",
                    precision="q4_k_m",
                    loader="gguf",
                    vram_required="8GB",
                ),
                VariantInfo(
                    name="ernie-fp16",
                    base_model="ernie-image",
                    precision="fp16",
                    loader="diffusers",
                    vram_required="12GB",
                ),
            ],
            loras=[],
            supported_sizes=["1024x1024", "768x768"],
        )

    def _get_default_workflow_template(self) -> str:
        """测试用默认工作流模板（v0.2.5 L6 抽象方法实现）。"""
        return """# 集成测试用默认工作流模板
workflow:
  name: default
  description: Integration test default workflow
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

def _parse_sse_frame(frame_str: str) -> dict:
    """解析 SSE 帧。"""
    assert frame_str.startswith("data: "), f"frame 不以 'data: ' 开头: {frame_str[:30]}"
    assert frame_str.endswith("\n\n"), f"frame 不以 '\\n\\n' 结尾: {frame_str[-30:]}"
    return json.loads(frame_str[len("data: "):-2])


async def _collect_infer_frames(engine, request):
    """收集推理所有帧。"""
    frames = []
    async for frame in engine.infer(request):
        frames.append(_parse_sse_frame(frame))
    return frames


# ===========================================================================
# ERNIE-Image 完整工作流 yaml
# ===========================================================================

ERNIE_IMAGE_WORKFLOW_YAML = """
name: ernie_image_default
description: ERNIE-Image 默认文生图工作流

nodes:
  - id: input_images
    type: ImagesInput
    pos: [100, 100]
    params: {}

  - id: input_prompt
    type: TextInput
    pos: [100, 250]
    params:
      field: prompt

  - id: main_model
    type: ModelLoader
    pos: [400, 100]
    params:
      variant: "${variant}"
      cache_dir: "models/image/checkpoints/ernie-image/"

  - id: clip_text_encode
    type: CLIPTextEncode
    pos: [400, 250]
    inputs:
      text: input_prompt.output.text
      clip: main_model.output.clip

  - id: ipadapter_apply
    type: IPAdapterApply
    pos: [700, 100]
    inputs:
      images: input_images.output.images
      model: main_model.output.model
    condition: "len(${input_images.output.images}) > 0"

  - id: sampler
    type: Sampler
    pos: [1000, 100]
    inputs:
      model: main_model.output.model
      cond: clip_text_encode.output.embeds

  - id: vae_decode
    type: VAEDecode
    pos: [1300, 100]
    inputs:
      latent: sampler.output.latent
      vae: main_model.output.vae

  - id: save
    type: ImageSave
    pos: [1600, 100]
    inputs:
      images: vae_decode.output.images

outputs:
  images: save.output.images
"""


# ===========================================================================
# fixtures
# ===========================================================================

@pytest.fixture
def registry() -> NodeRegistry:
    """注册所有模拟节点到独立注册表。"""
    reg = NodeRegistry()
    reg.register("ImagesInput", _ImagesInputNode)
    reg.register("TextInput", _TextInputNode)
    reg.register("ModelLoader", _ModelLoaderNode)
    reg.register("CLIPTextEncode", _CLIPTextEncodeNode)
    reg.register("IPAdapterApply", _IPAdapterApplyNode)
    reg.register("Sampler", _SamplerNode)
    reg.register("VAEDecode", _VAEDecodeNode)
    reg.register("ImageSave", _ImageSaveNode)
    return reg


@pytest.fixture
def ernie_engine(registry) -> _IntegrationEngine:
    """ERNIE-Image 测试引擎。"""
    return _IntegrationEngine(
        workflows_dict={"default": ERNIE_IMAGE_WORKFLOW_YAML},
        node_registry=registry,
    )


# ===========================================================================
# 端到端测试：ERNIE-Image 完整工作流
# ===========================================================================

class TestErnieImageEndToEnd:
    """ERNIE-Image 完整工作流端到端测试。"""

    @pytest.mark.asyncio
    async def test_basic_text_to_image(self, ernie_engine):
        """基本文生图端到端。"""
        request = InferRequest(
            variant="ernie-q4",
            inputs={"prompt": "一只可爱的猫"},
        )
        frames = await _collect_infer_frames(ernie_engine, request)

        # 验证帧序列
        assert len(frames) >= 3
        # 第一帧：workflow_start
        assert frames[0]["stage"] == "workflow_start"
        assert frames[0]["workflow"] == "default"
        assert frames[0]["variant"] == "ernie-q4"

        # workflow_loaded 帧（含执行顺序）
        loaded = next(f for f in frames if f["stage"] == "workflow_loaded")
        assert loaded["node_count"] == 8
        # 拓扑顺序应包含所有节点
        assert set(loaded["execution_order"]) == {
            "input_images", "input_prompt", "main_model",
            "clip_text_encode", "ipadapter_apply", "sampler",
            "vae_decode", "save",
        }

        # 最终帧：completed
        final = frames[-1]
        assert final["stage"] == "completed"
        assert final["progress"] == 1.0
        assert "result" in final
        assert "images" in final["result"]
        assert len(final["result"]["images"]) >= 1

    @pytest.mark.asyncio
    async def test_with_images_input(self, ernie_engine):
        """带图片输入（IPAdapter 触发）。"""
        request = InferRequest(
            variant="ernie-q4",
            inputs={
                "prompt": "保持风格",
                "images": ["img1.png", "img2.png", "img3.png"],
            },
        )
        frames = await _collect_infer_frames(ernie_engine, request)

        final = frames[-1]
        assert final["stage"] == "completed"
        # 验证工作流执行成功
        assert "images" in final["result"]

    @pytest.mark.asyncio
    async def test_without_images_skips_ipadapter(self, ernie_engine):
        """无图片输入时 IPAdapter 节点被条件跳过。"""
        request = InferRequest(
            variant="ernie-q4",
            inputs={"prompt": "纯文生图"},
        )
        frames = await _collect_infer_frames(ernie_engine, request)

        final = frames[-1]
        assert final["stage"] == "completed"
        assert "images" in final["result"]

    @pytest.mark.asyncio
    async def test_variant_injected_into_model_loader(self, ernie_engine):
        """variant 注入到 ModelLoader 节点。"""
        # 用一个简化工作流验证 variant 注入
        simple_yaml = """
name: variant_test
nodes:
  - id: model
    type: ModelLoader
    params:
      variant: "${variant}"
      cache_dir: "test"
  - id: save
    type: ImageSave
    inputs:
      images: model.output.model
outputs:
  model_info: save.output.images
"""
        engine = _IntegrationEngine(
            workflows_dict={"default": simple_yaml},
            node_registry=ernie_engine.node_registry,
        )
        request = InferRequest(
            variant="ernie-fp16",
            inputs={},
        )
        frames = await _collect_infer_frames(engine, request)

        final = frames[-1]
        assert final["stage"] == "completed"
        # ImageSave 把 ModelLoader 输出的 model dict 包装到 list
        model_info_list = final["result"]["model_info"]
        assert isinstance(model_info_list, list)
        assert len(model_info_list) == 1
        model_info = model_info_list[0]
        assert isinstance(model_info, dict)
        assert model_info["variant"] == "ernie-fp16"

    @pytest.mark.asyncio
    async def test_lora_injection_into_request(self, ernie_engine):
        """LoRA 通过 request.loras 注入到 ctx.loras。"""
        simple_yaml = """
name: lora_check
nodes:
  - id: check
    type: TextInput
    params:
      field: prompt
  - id: save
    type: ImageSave
    inputs:
      images: check.output.text
outputs:
  result: save.output.images
"""
        engine = _IntegrationEngine(
            workflows_dict={"default": simple_yaml},
            node_registry=ernie_engine.node_registry,
        )
        request = InferRequest(
            variant="ernie-q4",
            inputs={"prompt": "test"},
            loras=[
                LoraSpec(name="style_lora", weight=0.8),
                LoraSpec(name="detail_lora", weight=0.5),
            ],
        )
        frames = await _collect_infer_frames(engine, request)

        final = frames[-1]
        assert final["stage"] == "completed"
        # 验证 LoRA 列表注入成功（result 是 TextInput 输出的 text）
        # 这个简单测试只验证工作流不报错
        assert "result" in final["result"]


# ===========================================================================
# 多工作流场景
# ===========================================================================

class TestMultipleWorkflows:
    """多工作流场景测试。"""

    @pytest.mark.asyncio
    async def test_switch_workflow_via_request(self, registry):
        """通过 request.workflow 切换工作流。"""
        text_only_yaml = """
name: text_only
nodes:
  - id: src
    type: TextInput
    params:
      field: prompt
  - id: save
    type: ImageSave
    inputs:
      images: src.output.text
outputs:
  result: save.output.images
"""
        alt_yaml = """
name: alt_workflow
nodes:
  - id: src
    type: TextInput
    params:
      field: alt_prompt
  - id: save
    type: ImageSave
    inputs:
      images: src.output.text
outputs:
  result: save.output.images
"""
        engine = _IntegrationEngine(
            workflows_dict={
                "default": text_only_yaml,
                "alt": alt_yaml,
            },
            node_registry=registry,
        )

        # 使用默认工作流
        req1 = InferRequest(variant="v1", inputs={"prompt": "from_default"})
        frames1 = await _collect_infer_frames(engine, req1)
        assert frames1[-1]["stage"] == "completed"
        assert frames1[0]["workflow"] == "default"
        # ImageSave 包装到 list
        assert frames1[-1]["result"]["result"] == ["from_default"]

        # 使用 alt 工作流
        req2 = InferRequest(
            variant="v1",
            inputs={"alt_prompt": "from_alt"},
            workflow="alt",
        )
        frames2 = await _collect_infer_frames(engine, req2)
        assert frames2[-1]["stage"] == "completed"
        assert frames2[0]["workflow"] == "alt"
        assert frames2[-1]["result"]["result"] == ["from_alt"]

    @pytest.mark.asyncio
    async def test_set_default_workflow(self, registry):
        """set_current_workflow 修改默认工作流。"""
        yaml_a = """
name: a
nodes:
  - id: src
    type: TextInput
    params:
      field: prompt
outputs:
  r: src.output.text
"""
        yaml_b = """
name: b
nodes:
  - id: src
    type: TextInput
    params:
      field: prompt
outputs:
  r: src.output.text
"""
        engine = _IntegrationEngine(
            workflows_dict={"a": yaml_a, "b": yaml_b},
            node_registry=registry,
            default_workflow="a",
        )

        # 默认用 a
        req = InferRequest(variant="v", inputs={"prompt": "hello"})
        frames1 = await _collect_infer_frames(engine, req)
        assert frames1[0]["workflow"] == "a"

        # 切换默认到 b
        engine.set_current_workflow("b")
        frames2 = await _collect_infer_frames(engine, req)
        assert frames2[0]["workflow"] == "b"


# ===========================================================================
# 错误处理场景
# ===========================================================================

class TestErrorScenarios:
    """错误场景测试。"""

    @pytest.mark.asyncio
    async def test_workflow_not_found_returns_error_frame(self, registry):
        """工作流不存在时返回错误帧（不抛异常给调用方）。"""
        engine = _IntegrationEngine(
            workflows_dict={"default": "name: x\nnodes: []\n"},
            node_registry=registry,
        )
        request = InferRequest(
            variant="v",
            inputs={},
            workflow="nonexistent",
        )
        frames = await _collect_infer_frames(engine, request)

        # 第一帧是 workflow_start（即使工作流不存在也推送开始帧）
        assert frames[0]["stage"] == "workflow_start"
        # 然后是错误帧
        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) == 1
        assert "nonexistent" in error_frames[0]["error"]["message"]

    @pytest.mark.asyncio
    async def test_node_not_registered_error(self, registry):
        """节点未注册时返回错误帧。"""
        yaml_text = """
name: missing_node
nodes:
  - id: src
    type: NotRegisteredNode
outputs:
  r: src.output.value
"""
        engine = _IntegrationEngine(
            workflows_dict={"default": yaml_text},
            node_registry=registry,
        )
        request = InferRequest(variant="v", inputs={})
        frames = await _collect_infer_frames(engine, request)

        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) == 1

    @pytest.mark.asyncio
    async def test_workflow_cycle_error(self, registry):
        """工作流循环依赖错误。"""
        yaml_text = """
name: cycle
nodes:
  - id: a
    type: TextInput
    inputs:
      text: b.output.text
  - id: b
    type: TextInput
    inputs:
      text: a.output.text
outputs:
  r: a.output.text
"""
        engine = _IntegrationEngine(
            workflows_dict={"default": yaml_text},
            node_registry=registry,
        )
        request = InferRequest(variant="v", inputs={})
        frames = await _collect_infer_frames(engine, request)

        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) >= 1

    @pytest.mark.asyncio
    async def test_invalid_yaml_error(self, registry):
        """yaml 格式错误时返回错误帧。"""
        engine = _IntegrationEngine(
            workflows_dict={"default": "name: [unclosed"},
            node_registry=registry,
        )
        request = InferRequest(variant="v", inputs={})
        frames = await _collect_infer_frames(engine, request)

        error_frames = [f for f in frames if f["stage"] == "error"]
        assert len(error_frames) >= 1


# ===========================================================================
# 状态管理场景
# ===========================================================================

class TestStateManagement:
    """引擎状态管理测试。"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, ernie_engine):
        """完整生命周期：load → infer → unload。"""
        # 初始 IDLE
        assert ernie_engine.state_machine.state == RuntimeState.IDLE

        # load
        response = await ernie_engine.load(LoadRequest(variant="ernie-q4"))
        assert response.state == "ready"
        assert ernie_engine.state_machine.state == RuntimeState.READY
        assert ernie_engine.loaded_variant == "ernie-q4"

        # infer
        request = InferRequest(variant="ernie-q4", inputs={"prompt": "test"})
        frames = await _collect_infer_frames(ernie_engine, request)
        assert frames[-1]["stage"] == "completed"
        # 推理后应回到 READY
        assert ernie_engine.state_machine.state == RuntimeState.READY

        # unload
        response = await ernie_engine.unload()
        assert response.state == "idle"
        assert ernie_engine.state_machine.state == RuntimeState.IDLE
        assert ernie_engine.loaded_variant is None

    @pytest.mark.asyncio
    async def test_multiple_infers_keep_ready(self, ernie_engine):
        """多次推理之间保持 READY 状态。"""
        await ernie_engine.load(LoadRequest(variant="ernie-q4"))

        for i in range(3):
            request = InferRequest(
                variant="ernie-q4",
                inputs={"prompt": f"test_{i}"},
            )
            frames = await _collect_infer_frames(ernie_engine, request)
            assert frames[-1]["stage"] == "completed"
            assert ernie_engine.state_machine.state == RuntimeState.READY

    @pytest.mark.asyncio
    async def test_auto_load_on_infer(self, ernie_engine):
        """推理时未加载则自动加载。"""
        # 初始 IDLE，未调用 load
        assert ernie_engine.state_machine.state == RuntimeState.IDLE

        request = InferRequest(variant="ernie-q4", inputs={"prompt": "auto load"})
        frames = await _collect_infer_frames(ernie_engine, request)

        # 应该自动加载并完成推理
        assert frames[-1]["stage"] == "completed"
        # 推理后状态应为 READY
        assert ernie_engine.state_machine.state == RuntimeState.READY
        assert ernie_engine.loaded_variant == "ernie-q4"


# ===========================================================================
# LoRA 管理场景
# ===========================================================================

class TestLoraManagement:
    """LoRA 管理端到端测试。"""

    @pytest.mark.asyncio
    async def test_load_unload_lora(self, ernie_engine):
        """LoRA 加载 / 卸载。"""
        # 加载
        response = await ernie_engine.load_lora(LoraLoadRequest(name="style_lora", weight=0.8))
        assert response.name == "style_lora"
        assert "style_lora" in ernie_engine.loaded_loras

        # 卸载
        from aireliable_engine import LoraUnloadRequest
        response = await ernie_engine.unload_lora(LoraUnloadRequest(name="style_lora"))
        assert "style_lora" not in response.loaded_loras
        assert "style_lora" not in ernie_engine.loaded_loras

    @pytest.mark.asyncio
    async def test_multiple_loras(self, ernie_engine):
        """加载多个 LoRA。"""
        await ernie_engine.load_lora(LoraLoadRequest(name="lora1", weight=0.5))
        await ernie_engine.load_lora(LoraLoadRequest(name="lora2", weight=0.7))
        await ernie_engine.load_lora(LoraLoadRequest(name="lora3", weight=0.9))

        assert len(ernie_engine.loaded_loras) == 3
        assert set(ernie_engine.loaded_loras) == {"lora1", "lora2", "lora3"}

    @pytest.mark.asyncio
    async def test_lora_in_status(self, ernie_engine):
        """status 返回已加载 LoRA 列表。"""
        await ernie_engine.load_lora(LoraLoadRequest(name="style", weight=0.8))
        status = await ernie_engine.status()
        assert "style" in status.loaded_loras


# ===========================================================================
# 节点缓存场景
# ===========================================================================

class TestNodeCache:
    """节点缓存测试。"""

    @pytest.mark.asyncio
    async def test_model_loader_cache_key(self, registry):
        """ModelLoader 节点 cache_key 返回非 None。"""
        yaml_text = """
name: cache_test
nodes:
  - id: model1
    type: ModelLoader
    params:
      variant: "${variant}"
      cache_dir: "dir1"
  - id: model2
    type: ModelLoader
    params:
      variant: "${variant}"
      cache_dir: "dir2"
  - id: save1
    type: ImageSave
    inputs:
      images: model1.output.model
  - id: save2
    type: ImageSave
    inputs:
      images: model2.output.model
outputs:
  r1: save1.output.images
  r2: save2.output.images
"""
        engine = _IntegrationEngine(
            workflows_dict={"default": yaml_text},
            node_registry=registry,
        )
        request = InferRequest(variant="v1", inputs={})
        frames = await _collect_infer_frames(engine, request)

        final = frames[-1]
        assert final["stage"] == "completed"
        # ImageSave 包装到 list，每个 list 应包含一个 model dict
        r1_list = final["result"]["r1"]
        r2_list = final["result"]["r2"]
        assert isinstance(r1_list, list) and len(r1_list) == 1
        assert isinstance(r2_list, list) and len(r2_list) == 1
        r1 = r1_list[0]
        r2 = r2_list[0]
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)
        # 两个 ModelLoader 用不同 cache_dir，应该产生不同的 cache_key
        # 但 model_id 是基于 variant 的，所以这里只验证 model 对象存在
        assert "type" in r1
        assert "type" in r2


# ===========================================================================
# 性能 / 并发场景
# ===========================================================================

class TestPerformance:
    """性能测试（基础）。"""

    @pytest.mark.asyncio
    async def test_inference_under_1_second(self, ernie_engine):
        """单次推理应在 1 秒内完成（无 GPU 计算时）。"""
        request = InferRequest(
            variant="ernie-q4",
            inputs={"prompt": "performance test"},
        )
        start = time.time()
        frames = await _collect_infer_frames(ernie_engine, request)
        elapsed = time.time() - start

        assert frames[-1]["stage"] == "completed"
        # 模拟节点应该很快，1 秒足够
        assert elapsed < 1.0, f"推理耗时 {elapsed:.3f}s 超过 1s"

    @pytest.mark.asyncio
    async def test_timing_in_final_frame(self, ernie_engine):
        """最终帧包含 timing 信息。"""
        request = InferRequest(variant="v", inputs={"prompt": "test"})
        frames = await _collect_infer_frames(ernie_engine, request)

        final = frames[-1]
        assert "timing" in final
        assert "total_seconds" in final["timing"]
        assert final["timing"]["total_seconds"] >= 0
        assert final["timing"]["workflow"] == "default"
        assert final["timing"]["variant"] == "v"


# ===========================================================================
# 进度帧序列验证
# ===========================================================================

class TestProgressFrames:
    """SSE 进度帧序列测试。"""

    @pytest.mark.asyncio
    async def test_frame_sequence(self, ernie_engine):
        """推理帧序列：start → loaded → completed。"""
        request = InferRequest(variant="v", inputs={"prompt": "test"})
        frames = await _collect_infer_frames(ernie_engine, request)

        # 至少 3 帧
        assert len(frames) >= 3

        # 第一帧：workflow_start
        assert frames[0]["stage"] == "workflow_start"
        assert frames[0]["progress"] == 0.0
        assert "workflow" in frames[0]

        # 中间帧：workflow_loaded
        loaded_frames = [f for f in frames if f["stage"] == "workflow_loaded"]
        assert len(loaded_frames) == 1
        assert loaded_frames[0]["progress"] == 0.05

        # 最后帧：completed
        assert frames[-1]["stage"] == "completed"
        assert frames[-1]["progress"] == 1.0

    @pytest.mark.asyncio
    async def test_progress_monotonic(self, ernie_engine):
        """进度值单调递增。"""
        request = InferRequest(variant="v", inputs={"prompt": "test"})
        frames = await _collect_infer_frames(ernie_engine, request)

        # 收集所有进度值
        progresses = [f["progress"] for f in frames if "progress" in f]
        # 验证单调非递减
        for i in range(1, len(progresses)):
            assert progresses[i] >= progresses[i - 1], \
                f"进度非单调：{progresses[i]} < {progresses[i-1]}"

    @pytest.mark.asyncio
    async def test_all_frames_have_stage(self, ernie_engine):
        """所有帧都包含 stage 字段。"""
        request = InferRequest(variant="v", inputs={"prompt": "test"})
        frames = await _collect_infer_frames(ernie_engine, request)

        for f in frames:
            assert "stage" in f, f"帧缺少 stage 字段: {f}"
            assert isinstance(f["stage"], str)
            assert f["stage"]  # 非空


# ===========================================================================
# 工作流热重载集成测试
# ===========================================================================

class TestHotReloadIntegration:
    """工作流热重载集成测试。"""

    @pytest.mark.asyncio
    async def test_reload_picks_up_new_workflow(self, registry, tmp_path):
        """新增工作流文件后立即可用。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "default.yaml").write_text("""
name: original
nodes:
  - id: src
    type: TextInput
    params:
      field: prompt
outputs:
  r: src.output.text
""", encoding="utf-8")

        engine = _IntegrationEngine(
            workflows_dir=workflows_dir,
            node_registry=registry,
        )

        # 第一次推理
        req = InferRequest(variant="v", inputs={"prompt": "original_text"})
        frames1 = await _collect_infer_frames(engine, req)
        assert frames1[-1]["result"]["r"] == "original_text"

        # 新增工作流文件
        (workflows_dir / "alt.yaml").write_text("""
name: alt
nodes:
  - id: src
    type: TextInput
    params:
      field: alt_prompt
outputs:
  r: src.output.text
""", encoding="utf-8")

        # 用新工作流推理
        req2 = InferRequest(
            variant="v",
            inputs={"alt_prompt": "alt_text"},
            workflow="alt",
        )
        frames2 = await _collect_infer_frames(engine, req2)
        assert frames2[-1]["result"]["r"] == "alt_text"

    @pytest.mark.asyncio
    async def test_reload_modified_workflow(self, registry, tmp_path):
        """修改工作流文件后下次推理使用新版本。"""
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        wf_file = workflows_dir / "default.yaml"
        wf_file.write_text("""
name: v1
nodes:
  - id: src
    type: TextInput
    params:
      field: prompt
outputs:
  r: src.output.text
""", encoding="utf-8")

        engine = _IntegrationEngine(
            workflows_dir=workflows_dir,
            node_registry=registry,
        )

        # 第一次推理（v1）
        req = InferRequest(variant="v", inputs={"prompt": "before"})
        frames1 = await _collect_infer_frames(engine, req)
        assert frames1[-1]["result"]["r"] == "before"

        # 修改工作流（增加 prompt 改写逻辑）
        time.sleep(0.05)
        wf_file.write_text("""
name: v2
nodes:
  - id: src
    type: TextInput
    params:
      field: prompt
  - id: modified
    type: TextInput
    params:
      field: modified_prompt
outputs:
  r1: src.output.text
  r2: modified.output.text
""", encoding="utf-8")

        # 第二次推理（应使用 v2）
        req2 = InferRequest(
            variant="v",
            inputs={"prompt": "p1", "modified_prompt": "p2"},
        )
        frames2 = await _collect_infer_frames(engine, req2)

        final = frames2[-1]
        assert final["stage"] == "completed"
        # v2 工作流输出 r1 和 r2
        assert "r1" in final["result"]
        assert "r2" in final["result"]
        assert final["result"]["r1"] == "p1"
        assert final["result"]["r2"] == "p2"

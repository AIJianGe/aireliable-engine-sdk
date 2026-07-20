"""VAEDecode 节点单元测试。

覆盖任务 3.7 验收标准：
    - 实现 VAEDecodeNode 类
    - 输入：latent + vae
    - 输出：images（Images 包装类）

参考：v0.2-tasks.md 任务 3.7
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.decode.vae_decode import VAEDecodeNode
from aireliable_engine.types import Image, Images, Latent


# ===== Mock 子类 =====


class _MockDecoder(VAEDecodeNode):
    """模拟 VAE 解码器。"""

    node_type = "MockDecoder"

    def __init__(
        self,
        params=None,
        return_value: Any = None,
        raise_exc: Exception = None,
    ):
        super().__init__(params)
        if return_value is None:
            # 默认返回 1 张 MagicMock 图片的 Images
            return_value = {"images": Images.from_single(MagicMock(spec=Image))}
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.decode_called_with_latent = None
        self.decode_called_with_vae = None
        self.decode_called_with_params = None

    async def _decode_impl(self, latent, vae, ctx, params) -> dict:
        self.decode_called_with_latent = latent
        self.decode_called_with_vae = vae
        self.decode_called_with_params = params
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _FailingDecoder(VAEDecodeNode):
    node_type = "FailingDecoder"

    async def _decode_impl(self, latent, vae, ctx, params) -> dict:
        raise RuntimeError("VAE 内部解码错误")


class _InvalidReturnDecoder(VAEDecodeNode):
    """返回 dict 但缺少 images 字段。"""

    node_type = "InvalidReturnDecoder"

    async def _decode_impl(self, latent, vae, ctx, params) -> dict:
        return {"wrong_field": MagicMock()}


class _NonDictDecoder(VAEDecodeNode):
    """返回非 dict。"""

    node_type = "NonDictDecoder"

    async def _decode_impl(self, latent, vae, ctx, params) -> dict:
        return "not a dict"


class _ListReturnDecoder(VAEDecodeNode):
    """返回 list[Image]（应被自动包装为 Images）。"""

    node_type = "ListReturnDecoder"

    def __init__(self, params=None, image_count: int = 3):
        super().__init__(params)
        self._image_count = image_count
        self.decode_called_with_params = None

    async def _decode_impl(self, latent, vae, ctx, params) -> dict:
        self.decode_called_with_params = params
        images = [MagicMock(spec=Image) for _ in range(self._image_count)]
        return {"images": images}


class _SingleImageReturnDecoder(VAEDecodeNode):
    """返回单个 Image（应被自动包装为 Images 长度=1）。"""

    node_type = "SingleImageReturnDecoder"

    async def _decode_impl(self, latent, vae, ctx, params) -> dict:
        return {"images": MagicMock(spec=Image)}


class _BareDecoder(VAEDecodeNode):
    """未重写 _decode_impl。"""

    node_type = "BareDecoder"


# ===== 辅助 fixtures =====


@pytest.fixture
def mock_latent():
    """构造一个 mock Latent 对象。"""
    return MagicMock(spec=Latent)


@pytest.fixture
def mock_vae():
    """构造一个 mock VAE 对象。"""
    return MagicMock(name="mock_vae")


@pytest.fixture
def ctx():
    """默认工作流上下文。"""
    return make_default_context()


# ===== 基本属性测试 =====


class TestVAEDecodeNodeBasic:
    """VAEDecodeNode 基本属性测试。"""

    def test_node_type_is_vaedecode(self):
        node = VAEDecodeNode()
        assert node.node_type == "VAEDecode"

    def test_input_specs_has_two_ports(self):
        """输入端口：latent / vae。"""
        node = VAEDecodeNode()
        assert len(node.input_specs) == 2
        port_names = [p.name for p in node.input_specs]
        assert "latent" in port_names
        assert "vae" in port_names

    def test_input_port_latent_is_required(self):
        node = VAEDecodeNode()
        latent_port = next(p for p in node.input_specs if p.name == "latent")
        assert latent_port.required is True
        assert latent_port.type == "latent"

    def test_input_port_vae_is_required(self):
        node = VAEDecodeNode()
        vae_port = next(p for p in node.input_specs if p.name == "vae")
        assert vae_port.required is True
        assert vae_port.type == "model"

    def test_output_specs_has_one_port(self):
        node = VAEDecodeNode()
        assert len(node.output_specs) == 1

    def test_output_port_is_images_type(self):
        """验收标准：输出为 Images 包装类。"""
        node = VAEDecodeNode()
        port = node.output_specs[0]
        assert port.name == "images"
        assert port.type == "images"
        assert port.variadic is True

    def test_default_dtype_is_uint8(self):
        node = VAEDecodeNode()
        assert node._dtype == "uint8"

    def test_default_output_mode_is_rgb(self):
        node = VAEDecodeNode()
        assert node._output_mode == "RGB"

    def test_default_clamp_is_true(self):
        node = VAEDecodeNode()
        assert node._clamp is True

    def test_default_scaling_factor_is_none(self):
        node = VAEDecodeNode()
        assert node._scaling_factor is None

    def test_default_shift_factor_is_none(self):
        node = VAEDecodeNode()
        assert node._shift_factor is None

    def test_default_batch_split_is_none(self):
        node = VAEDecodeNode()
        assert node._batch_split is None

    def test_default_call_timeout_is_none(self):
        node = VAEDecodeNode()
        assert node._call_timeout is None

    def test_default_vae_type_check_is_true(self):
        node = VAEDecodeNode()
        assert node._vae_type_check is True

    def test_init_with_custom_params(self):
        node = VAEDecodeNode({
            "dtype": "float32",
            "output_mode": "RGBA",
            "clamp": False,
            "scaling_factor": 0.18215,
            "shift_factor": 0.0,
            "device": "cuda:0",
            "batch_split": 2,
            "call_timeout": 60.0,
            "vae_type_check": False,
        })
        assert node._dtype == "float32"
        assert node._output_mode == "RGBA"
        assert node._clamp is False
        assert node._scaling_factor == 0.18215
        assert node._shift_factor == 0.0
        assert node._device_override == "cuda:0"
        assert node._batch_split == 2
        assert node._call_timeout == 60.0
        assert node._vae_type_check is False

    def test_repr_contains_class_name(self):
        node = VAEDecodeNode({"dtype": "float32"})
        repr_str = repr(node)
        assert "VAEDecodeNode" in repr_str
        assert "float32" in repr_str


# ===== 参数校验测试 =====


class TestVAEDecodeNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_default_params(self):
        node = VAEDecodeNode()
        assert node.validate() == []

    def test_validate_reports_invalid_dtype(self):
        node = VAEDecodeNode({"dtype": "int8"})
        errors = node.validate()
        assert any("dtype" in e for e in errors)

    def test_validate_accepts_uint8_dtype(self):
        node = VAEDecodeNode({"dtype": "uint8"})
        assert node.validate() == []

    def test_validate_accepts_float32_dtype(self):
        node = VAEDecodeNode({"dtype": "float32"})
        assert node.validate() == []

    def test_validate_reports_invalid_output_mode(self):
        node = VAEDecodeNode({"output_mode": "YCbCr"})
        errors = node.validate()
        assert any("output_mode" in e for e in errors)

    def test_validate_accepts_rgb_mode(self):
        node = VAEDecodeNode({"output_mode": "RGB"})
        assert node.validate() == []

    def test_validate_accepts_rgba_mode(self):
        node = VAEDecodeNode({"output_mode": "RGBA"})
        assert node.validate() == []

    def test_validate_accepts_l_mode(self):
        node = VAEDecodeNode({"output_mode": "L"})
        assert node.validate() == []

    def test_validate_reports_non_bool_clamp(self):
        """clamp 非布尔值应报错（避免 bool('yes')=True 绕过类型校验）。"""
        node = VAEDecodeNode({"clamp": "yes"})
        errors = node.validate()
        assert any("clamp" in e for e in errors)

    def test_validate_accepts_true_clamp(self):
        node = VAEDecodeNode({"clamp": True})
        assert node.validate() == []

    def test_validate_accepts_false_clamp(self):
        node = VAEDecodeNode({"clamp": False})
        assert node.validate() == []

    def test_validate_reports_non_positive_scaling_factor(self):
        node = VAEDecodeNode({"scaling_factor": -0.5})
        errors = node.validate()
        assert any("scaling_factor" in e for e in errors)

    def test_validate_reports_zero_scaling_factor(self):
        node = VAEDecodeNode({"scaling_factor": 0})
        errors = node.validate()
        assert any("scaling_factor" in e for e in errors)

    def test_validate_accepts_positive_scaling_factor(self):
        node = VAEDecodeNode({"scaling_factor": 0.18215})
        assert node.validate() == []

    def test_validate_accepts_none_scaling_factor(self):
        node = VAEDecodeNode({"scaling_factor": None})
        assert node.validate() == []

    def test_validate_reports_non_numeric_shift_factor(self):
        node = VAEDecodeNode({"shift_factor": "auto"})
        errors = node.validate()
        assert any("shift_factor" in e for e in errors)

    def test_validate_accepts_zero_shift_factor(self):
        """shift_factor 可为 0（Flux VAE 默认值）。"""
        node = VAEDecodeNode({"shift_factor": 0.0})
        assert node.validate() == []

    def test_validate_reports_non_positive_batch_split(self):
        node = VAEDecodeNode({"batch_split": 0})
        errors = node.validate()
        assert any("batch_split" in e for e in errors)

    def test_validate_reports_negative_batch_split(self):
        node = VAEDecodeNode({"batch_split": -1})
        errors = node.validate()
        assert any("batch_split" in e for e in errors)

    def test_validate_accepts_positive_batch_split(self):
        node = VAEDecodeNode({"batch_split": 4})
        assert node.validate() == []

    def test_validate_accepts_none_batch_split(self):
        node = VAEDecodeNode({"batch_split": None})
        assert node.validate() == []

    def test_validate_reports_non_positive_call_timeout(self):
        node = VAEDecodeNode({"call_timeout": 0})
        errors = node.validate()
        assert any("call_timeout" in e for e in errors)

    def test_validate_reports_negative_call_timeout(self):
        node = VAEDecodeNode({"call_timeout": -1.0})
        errors = node.validate()
        assert any("call_timeout" in e for e in errors)

    def test_validate_accepts_positive_call_timeout(self):
        node = VAEDecodeNode({"call_timeout": 30.0})
        assert node.validate() == []

    def test_validate_accepts_none_call_timeout(self):
        node = VAEDecodeNode({"call_timeout": None})
        assert node.validate() == []

    def test_validate_reports_non_bool_vae_type_check(self):
        node = VAEDecodeNode({"vae_type_check": "yes"})
        errors = node.validate()
        assert any("vae_type_check" in e for e in errors)

    def test_validate_accepts_true_vae_type_check(self):
        node = VAEDecodeNode({"vae_type_check": True})
        assert node.validate() == []

    def test_validate_accepts_false_vae_type_check(self):
        node = VAEDecodeNode({"vae_type_check": False})
        assert node.validate() == []


# ===== execute 流程测试 =====


class TestVAEDecodeNodeExecute:
    """execute 流程测试。"""

    @pytest.mark.asyncio
    async def test_execute_returns_images_dict(self, mock_latent, mock_vae, ctx):
        """验收标准：execute 返回 {"images": Images}。"""
        node = _MockDecoder()
        result = await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert isinstance(result, dict)
        assert "images" in result
        assert isinstance(result["images"], Images)

    @pytest.mark.asyncio
    async def test_execute_calls_decode_impl_with_inputs(
        self, mock_latent, mock_vae, ctx
    ):
        node = _MockDecoder()
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert node.decode_called_with_latent is mock_latent
        assert node.decode_called_with_vae is mock_vae

    @pytest.mark.asyncio
    async def test_execute_passes_params_dict_to_impl(
        self, mock_latent, mock_vae, ctx
    ):
        node = _MockDecoder({"dtype": "float32", "output_mode": "RGBA"})
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        params = node.decode_called_with_params
        assert params["dtype"] == "float32"
        assert params["output_mode"] == "RGBA"
        assert params["clamp"] is True
        assert params["scaling_factor"] is None
        assert params["shift_factor"] is None
        assert params["batch_split"] is None
        assert params["call_timeout"] is None

    @pytest.mark.asyncio
    async def test_execute_uses_ctx_device_by_default(self, mock_latent, mock_vae):
        ctx = make_default_context(device="cuda:0")
        node = _MockDecoder()
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert node.decode_called_with_params["device"] == "cuda:0"

    @pytest.mark.asyncio
    async def test_execute_device_override_takes_precedence(
        self, mock_latent, mock_vae
    ):
        ctx = make_default_context(device="cpu")
        node = _MockDecoder({"device": "cuda:1"})
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert node.decode_called_with_params["device"] == "cuda:1"

    @pytest.mark.asyncio
    async def test_execute_raises_on_missing_latent(self, mock_vae, ctx):
        node = _MockDecoder()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"vae": mock_vae}, ctx)
        assert "latent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_none_latent(self, mock_vae, ctx):
        node = _MockDecoder()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"latent": None, "vae": mock_vae}, ctx)
        assert "latent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_missing_vae(self, mock_latent, ctx):
        node = _MockDecoder()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"latent": mock_latent}, ctx)
        assert "vae" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_none_vae(self, mock_latent, ctx):
        node = _MockDecoder()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"latent": mock_latent, "vae": None}, ctx)
        assert "vae" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_invalid_params(self, mock_latent, mock_vae, ctx):
        """参数校验失败时抛 NodeValidationError。"""
        node = _MockDecoder({"dtype": "int8"})  # 非法 dtype
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert "dtype" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_not_implemented_for_bare_subclass(
        self, mock_latent, mock_vae, ctx
    ):
        """未实现 _decode_impl 的子类应抛 NotImplementedError。"""
        node = _BareDecoder()
        with pytest.raises(NotImplementedError) as exc_info:
            await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert "_decode_impl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_wraps_runtime_error_on_failure(
        self, mock_latent, mock_vae, ctx
    ):
        """子类抛异常时包装为 RuntimeError。"""
        node = _FailingDecoder()
        with pytest.raises(RuntimeError) as exc_info:
            await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert "VAE 解码失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_raises_on_dict_without_images_field(
        self, mock_latent, mock_vae, ctx
    ):
        """子类返回 dict 但缺 images 字段时抛 TypeError。"""
        node = _InvalidReturnDecoder()
        with pytest.raises(TypeError) as exc_info:
            await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert "images" in str(exc_info.value)


# ===== 返回值标准化测试（验收标准：输出为 Images）=====


class TestVAEDecodeNodeOutputNormalization:
    """_normalize_images_output 测试（验收标准：输出为 Images 包装类）。"""

    @pytest.mark.asyncio
    async def test_execute_returns_images_when_impl_returns_images(
        self, mock_latent, mock_vae, ctx
    ):
        """子类返回 Images 时直接返回。"""
        img = MagicMock(spec=Image)
        images = Images.from_single(img)
        node = _MockDecoder(return_value={"images": images})
        result = await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert result["images"] is images
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 1

    @pytest.mark.asyncio
    async def test_execute_wraps_list_of_images(self, mock_latent, mock_vae, ctx):
        """子类返回 list[Image] 时自动包装为 Images。"""
        node = _ListReturnDecoder(image_count=3)
        result = await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 3

    @pytest.mark.asyncio
    async def test_execute_wraps_single_image(self, mock_latent, mock_vae, ctx):
        """子类返回单个 Image 时自动包装为 Images 长度=1。"""
        node = _SingleImageReturnDecoder()
        result = await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 1

    @pytest.mark.asyncio
    async def test_execute_returns_empty_images_for_empty_list(
        self, mock_latent, mock_vae, ctx
    ):
        """子类返回空 list 时包装为空 Images。"""
        node = _MockDecoder(return_value={"images": []})
        result = await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 0

    @pytest.mark.asyncio
    async def test_execute_raises_on_non_dict_non_images_return(
        self, mock_latent, mock_vae, ctx
    ):
        """子类返回非 dict / 非 Images / 非 list / 非 Image 时抛 TypeError。"""
        node = _NonDictDecoder()
        with pytest.raises(TypeError) as exc_info:
            await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        assert "images" in str(exc_info.value) or "dict" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_normalize_raises_on_list_with_non_image_element(self):
        """list 元素不是 Image 时抛 TypeError。"""
        node = VAEDecodeNode()
        with pytest.raises(TypeError) as exc_info:
            node._normalize_images_output({"images": ["not_an_image", MagicMock(spec=Image)]})
        assert "Image" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_normalize_raises_on_invalid_type(self):
        """返回 int 类型时抛 TypeError。"""
        node = VAEDecodeNode()
        with pytest.raises(TypeError) as exc_info:
            node._normalize_images_output({"images": 12345})
        assert "Images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_normalize_raises_on_dict_without_images_key(self):
        """dict 缺 images 键时抛 TypeError。"""
        node = VAEDecodeNode()
        with pytest.raises(TypeError) as exc_info:
            node._normalize_images_output({"wrong_field": MagicMock(spec=Image)})
        assert "images" in str(exc_info.value)

    def test_normalize_accepts_images_instance(self):
        node = VAEDecodeNode()
        images = Images.from_single(MagicMock(spec=Image))
        result = node._normalize_images_output({"images": images})
        assert result is images

    def test_normalize_accepts_images_instance_directly(self):
        """非 dict 的 Images 实例直接接受。"""
        node = VAEDecodeNode()
        images = Images.from_single(MagicMock(spec=Image))
        result = node._normalize_images_output(images)
        assert result is images

    def test_normalize_accepts_single_image_directly(self):
        """非 dict 的单个 Image 实例包装为 Images 长度=1。"""
        node = VAEDecodeNode()
        img = MagicMock(spec=Image)
        result = node._normalize_images_output(img)
        assert isinstance(result, Images)
        assert len(result) == 1

    def test_normalize_accepts_list_directly(self):
        """非 dict 的 list[Image] 包装为 Images。"""
        node = VAEDecodeNode()
        images_list = [MagicMock(spec=Image) for _ in range(2)]
        result = node._normalize_images_output(images_list)
        assert isinstance(result, Images)
        assert len(result) == 2


# ===== 进度上报测试 =====


class TestVAEDecodeNodeProgress:
    """进度上报测试。"""

    @pytest.mark.asyncio
    async def test_execute_reports_start_progress(
        self, mock_latent, mock_vae
    ):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                events.append(("error", message, error_type, kwargs))

            async def report_info(self, message, **kwargs):
                events.append(("info", message, kwargs))

        ctx = make_default_context(progress_reporter=_Capture())
        node = _MockDecoder()
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        # 至少有 start 和 done
        progress_events = [e for e in events if e[0] == "progress"]
        assert any(e[2] == "vae_decode.start" for e in progress_events)
        assert any(e[2] == "vae_decode.done" for e in progress_events)
        # start 进度为 0.0
        start_events = [e for e in progress_events if e[2] == "vae_decode.start"]
        assert start_events[0][1] == 0.0
        # done 进度为 1.0
        done_events = [e for e in progress_events if e[2] == "vae_decode.done"]
        assert done_events[0][1] == 1.0

    @pytest.mark.asyncio
    async def test_execute_start_progress_contains_dtype(
        self, mock_latent, mock_vae
    ):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        ctx = make_default_context(progress_reporter=_Capture())
        node = _MockDecoder({"dtype": "float32"})
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        start_events = [e for e in events if e[2] == "vae_decode.start"]
        assert start_events[0][3]["dtype"] == "float32"

    @pytest.mark.asyncio
    async def test_execute_start_progress_contains_device(
        self, mock_latent, mock_vae
    ):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        ctx = make_default_context(device="cuda:0", progress_reporter=_Capture())
        node = _MockDecoder({"device": "cuda:1"})
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        start_events = [e for e in events if e[2] == "vae_decode.start"]
        assert start_events[0][3]["device"] == "cuda:1"

    @pytest.mark.asyncio
    async def test_execute_done_progress_contains_image_count(
        self, mock_latent, mock_vae
    ):
        events = []

        class _Capture:
            async def report_progress(self, progress, stage="", **kwargs):
                events.append(("progress", progress, stage, kwargs))

            async def report_error(self, message, error_type="node_error", **kwargs):
                pass

            async def report_info(self, message, **kwargs):
                pass

        ctx = make_default_context(progress_reporter=_Capture())
        # 返回 3 张图
        node = _ListReturnDecoder(image_count=3)
        await node.execute({"latent": mock_latent, "vae": mock_vae}, ctx)
        done_events = [e for e in events if e[2] == "vae_decode.done"]
        assert done_events[0][3]["image_count"] == 3


# ===== VAE 类型一致性检查测试 =====


class TestVAEDecodeNodeVAETypeCheck:
    """_check_vae_type_compatibility 测试。"""

    def test_check_returns_silently_for_matching_types(self):
        """latent.vae_type 与 vae.name 一致时不应报错。"""
        node = VAEDecodeNode({"vae_type_check": True})
        latent = Latent(tensor=MagicMock(), vae_type="sdxl-vae")
        vae = MagicMock()
        vae.name = "sdxl-vae"
        # 不应抛异常
        node._check_vae_type_compatibility(latent, vae)

    def test_check_logs_warning_for_mismatched_types(self, caplog):
        """latent.vae_type 与 vae.name 不一致时记录 warning。"""
        import logging

        node = VAEDecodeNode({"vae_type_check": True})
        latent = Latent(tensor=MagicMock(), vae_type="sdxl-vae")
        vae = MagicMock()
        vae.name = "flux-vae"
        with caplog.at_level(logging.WARNING, logger="aireliable_engine.nodes.decode.vae_decode"):
            node._check_vae_type_compatibility(latent, vae)
        assert any("不一致" in record.message for record in caplog.records)

    def test_check_skips_when_latent_vae_type_empty(self):
        """latent.vae_type 为空时跳过检查。"""
        node = VAEDecodeNode({"vae_type_check": True})
        latent = Latent(tensor=MagicMock(), vae_type="")
        vae = MagicMock(name="my_vae")
        # 不应抛异常
        node._check_vae_type_compatibility(latent, vae)

    def test_check_skips_when_latent_not_latent_instance(self):
        """latent 不是 Latent 实例时跳过检查。"""
        node = VAEDecodeNode({"vae_type_check": True})
        # 普通 MagicMock 不是 Latent 实例
        latent = MagicMock()
        vae = MagicMock()
        node._check_vae_type_compatibility(latent, vae)

    def test_check_skips_when_vae_has_no_type_identifier(self):
        """vae 无类型标识字段时跳过检查。"""
        node = VAEDecodeNode({"vae_type_check": True})
        latent = Latent(tensor=MagicMock(), vae_type="sdxl-vae")
        vae = MagicMock(spec=[])  # 无任何属性
        node._check_vae_type_compatibility(latent, vae)

    def test_check_swallows_unexpected_exception(self):
        """检查过程中的异常被吞掉（不影响主流程）。"""
        node = VAEDecodeNode({"vae_type_check": True})

        # 构造一个会在 getattr 时抛异常的对象
        class _BadVAE:
            @property
            def name(self):
                raise RuntimeError("oops")

        latent = Latent(tensor=MagicMock(), vae_type="sdxl-vae")
        vae = _BadVAE()
        # 不应抛异常
        node._check_vae_type_compatibility(latent, vae)

    @pytest.mark.asyncio
    async def test_execute_skips_type_check_when_disabled(
        self, mock_latent, mock_vae, ctx
    ):
        """vae_type_check=False 时不调用检查。"""
        node = _MockDecoder({"vae_type_check": False})
        # 使用真实 Latent 对象触发检查路径
        latent = Latent(tensor=MagicMock(), vae_type="sdxl-vae")
        vae = MagicMock()
        vae.name = "flux-vae"
        await node.execute({"latent": latent, "vae": vae}, ctx)
        # 不抛异常即通过

    @pytest.mark.asyncio
    async def test_execute_still_runs_when_type_mismatch(
        self, mock_vae, ctx
    ):
        """类型不一致时仅警告，工作流继续执行。"""
        node = _MockDecoder({"vae_type_check": True})
        latent = Latent(tensor=MagicMock(), vae_type="sdxl-vae")
        vae = MagicMock()
        vae.name = "flux-vae"
        # 不应抛异常，execute 正常完成
        result = await node.execute({"latent": latent, "vae": vae}, ctx)
        assert "images" in result


# ===== cache_key 测试 =====


class TestVAEDecodeNodeCacheKey:
    """cache_key 测试。"""

    def test_cache_key_returns_none_by_default(self):
        """默认不缓存（每次解码结果可能不同）。"""
        node = VAEDecodeNode()
        assert node.cache_key({}) is None

    def test_cache_key_returns_none_even_with_inputs(self):
        node = VAEDecodeNode({"dtype": "float32"})
        assert node.cache_key({"latent": MagicMock(), "vae": MagicMock()}) is None


# ===== 集成测试 =====


class TestVAEDecodeNodeIntegration:
    """端到端集成测试。"""

    @pytest.mark.asyncio
    async def test_full_pipeline_returns_correct_images_count(
        self, mock_vae
    ):
        """模拟完整流程：latent + vae → Images(N)。"""
        latent = Latent(tensor=MagicMock(), vae_type="ernie-image")
        node = _ListReturnDecoder(image_count=4)
        ctx = make_default_context(device="cuda:0")
        result = await node.execute({"latent": latent, "vae": mock_vae}, ctx)
        assert isinstance(result["images"], Images)
        assert len(result["images"]) == 4
        # 验证传入子类的参数
        assert node.decode_called_with_params["device"] == "cuda:0"

    @pytest.mark.asyncio
    async def test_full_pipeline_with_all_params(self, mock_vae):
        """所有参数都生效。"""
        latent = Latent(tensor=MagicMock(), vae_type="sdxl")
        node = _MockDecoder({
            "dtype": "float32",
            "output_mode": "RGBA",
            "clamp": False,
            "scaling_factor": 0.13025,
            "shift_factor": 0.0609,
            "device": "cuda:1",
            "batch_split": 2,
            "call_timeout": 120.0,
            "vae_type_check": False,
        })
        ctx = make_default_context(device="cpu")
        await node.execute({"latent": latent, "vae": mock_vae}, ctx)
        params = node.decode_called_with_params
        assert params["dtype"] == "float32"
        assert params["output_mode"] == "RGBA"
        assert params["clamp"] is False
        assert params["scaling_factor"] == 0.13025
        assert params["shift_factor"] == 0.0609
        assert params["device"] == "cuda:1"
        assert params["batch_split"] == 2
        assert params["call_timeout"] == 120.0

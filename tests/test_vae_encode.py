"""VAEEncode 节点单元测试（v0.2.5 新增）。

覆盖验收标准：
    - 实现 VAEEncodeNode 类
    - 输入：images + vae
    - 输出：latent（Latent 类型）
    - 支持单图编码（Images 长度=1）
    - 支持多图批量编码（Images 长度=N）
    - 参数校验（images 必填 / vae 必填）
    - 异常处理（_encode_impl 抛异常 → RuntimeError 包装）

参考：v0.2.5-vae-encode-tasks.md A5-1
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aireliable_engine import make_default_context
from aireliable_engine.node import NodeValidationError
from aireliable_engine.nodes.encode.vae_encode import VAEEncodeNode
from aireliable_engine.types import Image, Images, Latent


# ===== Mock 子类 =====


class _MockEncoder(VAEEncodeNode):
    """模拟 VAE 编码器。"""

    node_type = "MockEncoder"

    def __init__(
        self,
        params=None,
        return_value: Any = None,
        raise_exc: Exception = None,
    ):
        super().__init__(params)
        if return_value is None:
            # 默认返回 1 个 Latent 实例（构造真实 Latent 避免 mock 属性问题）
            return_value = {"latent": Latent(tensor=MagicMock())}
        self._return_value = return_value
        self._raise_exc = raise_exc
        self.encode_called_with_images = None
        self.encode_called_with_vae = None
        self.encode_called_with_params = None

    async def _encode_impl(self, images, vae, ctx, params) -> dict:
        self.encode_called_with_images = images
        self.encode_called_with_vae = vae
        self.encode_called_with_params = params
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._return_value


class _FailingEncoder(VAEEncodeNode):
    node_type = "FailingEncoder"

    async def _encode_impl(self, images, vae, ctx, params) -> dict:
        raise RuntimeError("VAE 内部编码错误")


class _InvalidReturnEncoder(VAEEncodeNode):
    """返回 dict 但缺少 latent 字段。"""

    node_type = "InvalidReturnEncoder"

    async def _encode_impl(self, images, vae, ctx, params) -> dict:
        return {"wrong_field": MagicMock()}


class _NonDictEncoder(VAEEncodeNode):
    """返回非 dict。"""

    node_type = "NonDictEncoder"

    async def _encode_impl(self, images, vae, ctx, params) -> dict:
        return "not a dict"


class _TensorReturnEncoder(VAEEncodeNode):
    """返回 torch.Tensor（应被自动包装为 Latent）。"""

    node_type = "TensorReturnEncoder"

    def __init__(self, params=None):
        super().__init__(params)
        self.encode_called_with_params = None

    async def _encode_impl(self, images, vae, ctx, params) -> dict:
        self.encode_called_with_params = params
        # 模拟 tensor 对象（有 dim 方法）
        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 4
        mock_tensor.shape = [1, 4, 32, 32]
        return {"latent": mock_tensor}


class _BareEncoder(VAEEncodeNode):
    """未重写 _encode_impl（使用默认实现，会因 torch 未安装而失败）。"""

    node_type = "BareEncoder"


# ===== 辅助 fixtures =====


@pytest.fixture
def mock_images():
    """构造一个 mock Images 对象（1 张图）。"""
    img = MagicMock(spec=Image)
    return Images.from_single(img)


@pytest.fixture
def mock_images_multi():
    """构造一个 mock Images 对象（3 张图）。"""
    items = [MagicMock(spec=Image) for _ in range(3)]
    return Images(items=items)


@pytest.fixture
def mock_vae():
    """构造一个 mock VAE 对象。"""
    return MagicMock(name="mock_vae")


@pytest.fixture
def ctx():
    """默认工作流上下文。"""
    return make_default_context()


# ===== 基本属性测试 =====


class TestVAEEncodeNodeBasic:
    """VAEEncodeNode 基本属性测试。"""

    def test_node_type_is_vaeencode(self):
        node = VAEEncodeNode()
        assert node.node_type == "VAEEncode"

    def test_category_is_encode(self):
        node = VAEEncodeNode()
        assert node.category == "encode"

    def test_subcategory_is_vae(self):
        node = VAEEncodeNode()
        assert node.subcategory == "vae"

    def test_input_specs_has_two_ports(self):
        """输入端口：images / vae。"""
        node = VAEEncodeNode()
        assert len(node.input_specs) == 2
        port_names = [p.name for p in node.input_specs]
        assert "images" in port_names
        assert "vae" in port_names

    def test_input_port_images_is_required(self):
        node = VAEEncodeNode()
        images_port = next(p for p in node.input_specs if p.name == "images")
        assert images_port.required is True
        assert images_port.type == "images"

    def test_input_port_vae_is_required(self):
        node = VAEEncodeNode()
        vae_port = next(p for p in node.input_specs if p.name == "vae")
        assert vae_port.required is True
        assert vae_port.type == "model"

    def test_output_specs_has_one_port(self):
        node = VAEEncodeNode()
        assert len(node.output_specs) == 1

    def test_output_port_is_latent_type(self):
        """验收标准：输出为 Latent 类型。"""
        node = VAEEncodeNode()
        port = node.output_specs[0]
        assert port.name == "latent"
        assert port.type == "latent"

    def test_default_scaling_factor_is_none(self):
        node = VAEEncodeNode()
        assert node._scaling_factor is None

    def test_default_shift_factor_is_none(self):
        node = VAEEncodeNode()
        assert node._shift_factor is None

    def test_default_batch_split_is_none(self):
        node = VAEEncodeNode()
        assert node._batch_split is None

    def test_params_schema_has_four_params(self):
        """params_schema 包含 4 个参数（全部 advanced）。"""
        node = VAEEncodeNode()
        assert "scaling_factor" in node.params_schema
        assert "shift_factor" in node.params_schema
        assert "device" in node.params_schema
        assert "batch_split" in node.params_schema
        # 全部 advanced
        for key, spec in node.params_schema.items():
            assert spec.get("advanced") is True, f"{key} 应为 advanced"


# ===== 参数校验测试 =====


class TestVAEEncodeNodeValidate:
    """参数校验测试。"""

    def test_validate_returns_empty_for_default_params(self):
        node = VAEEncodeNode()
        assert node.validate() == []

    def test_validate_reports_negative_scaling_factor(self):
        node = VAEEncodeNode({"scaling_factor": -0.1})
        errors = node.validate()
        assert any("scaling_factor" in e for e in errors)

    def test_validate_reports_zero_scaling_factor(self):
        node = VAEEncodeNode({"scaling_factor": 0})
        errors = node.validate()
        assert any("scaling_factor" in e for e in errors)

    def test_validate_reports_negative_batch_split(self):
        node = VAEEncodeNode({"batch_split": -1})
        errors = node.validate()
        assert any("batch_split" in e for e in errors)

    def test_validate_reports_zero_batch_split(self):
        node = VAEEncodeNode({"batch_split": 0})
        errors = node.validate()
        assert any("batch_split" in e for e in errors)

    def test_validate_accepts_valid_scaling_factor(self):
        node = VAEEncodeNode({"scaling_factor": 0.18215})
        assert node.validate() == []

    def test_validate_accepts_valid_batch_split(self):
        node = VAEEncodeNode({"batch_split": 4})
        assert node.validate() == []


# ===== execute 方法测试（Mock 子类）=====


class TestVAEEncodeNodeExecute:
    """execute 方法测试。"""

    @pytest.mark.asyncio
    async def test_execute_single_image(self, mock_images, mock_vae, ctx):
        """测试单图编码（Images 长度=1）。"""
        node = _MockEncoder()
        result = await node.execute(
            {"images": mock_images, "vae": mock_vae}, ctx
        )
        assert "latent" in result
        assert isinstance(result["latent"], Latent) or hasattr(result["latent"], "tensor")
        # 验证 _encode_impl 被调用
        assert node.encode_called_with_images is not None
        assert len(node.encode_called_with_images) == 1

    @pytest.mark.asyncio
    async def test_execute_multi_images(self, mock_images_multi, mock_vae, ctx):
        """测试多图批量编码（Images 长度=3）。"""
        node = _MockEncoder()
        result = await node.execute(
            {"images": mock_images_multi, "vae": mock_vae}, ctx
        )
        assert "latent" in result
        # 验证 _encode_impl 收到 3 张图
        assert node.encode_called_with_images is not None
        assert len(node.encode_called_with_images) == 3

    @pytest.mark.asyncio
    async def test_execute_missing_images_raises(self, mock_vae, ctx):
        """images 缺失时报错。"""
        node = _MockEncoder()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"vae": mock_vae}, ctx)
        assert "images" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_missing_vae_raises(self, mock_images, ctx):
        """vae 缺失时报错。"""
        node = _MockEncoder()
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute({"images": mock_images}, ctx)
        assert "vae" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_empty_images_raises(self, mock_vae, ctx):
        """images 为空 Images 时报错。"""
        node = _MockEncoder()
        empty_images = Images(items=[])
        with pytest.raises(NodeValidationError) as exc_info:
            await node.execute(
                {"images": empty_images, "vae": mock_vae}, ctx
            )
        assert "images" in str(exc_info.value) or "空" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_failing_impl_wraps_as_runtime_error(
        self, mock_images, mock_vae, ctx
    ):
        """_encode_impl 抛异常 → RuntimeError 包装。"""
        node = _FailingEncoder()
        with pytest.raises(RuntimeError) as exc_info:
            await node.execute(
                {"images": mock_images, "vae": mock_vae}, ctx
            )
        assert "VAE 编码失败" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_invalid_return_dict_raises(
        self, mock_images, mock_vae, ctx
    ):
        """_encode_impl 返回 dict 缺少 latent 字段 → TypeError。"""
        node = _InvalidReturnEncoder()
        with pytest.raises(TypeError) as exc_info:
            await node.execute(
                {"images": mock_images, "vae": mock_vae}, ctx
            )
        assert "latent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_non_dict_return_raises(
        self, mock_images, mock_vae, ctx
    ):
        """_encode_impl 返回非 dict → TypeError。"""
        node = _NonDictEncoder()
        with pytest.raises(TypeError) as exc_info:
            await node.execute(
                {"images": mock_images, "vae": mock_vae}, ctx
            )

    @pytest.mark.asyncio
    async def test_execute_tensor_return_wrapped_as_latent(
        self, mock_images, mock_vae, ctx
    ):
        """_encode_impl 返回 torch.Tensor → 自动包装为 Latent。"""
        node = _TensorReturnEncoder()
        result = await node.execute(
            {"images": mock_images, "vae": mock_vae}, ctx
        )
        assert "latent" in result
        assert isinstance(result["latent"], Latent)

    @pytest.mark.asyncio
    async def test_execute_passes_params_to_impl(
        self, mock_images, mock_vae, ctx
    ):
        """验证 execute 把参数传给 _encode_impl。"""
        node = _MockEncoder({"scaling_factor": 0.18215, "batch_split": 2})
        await node.execute(
            {"images": mock_images, "vae": mock_vae}, ctx
        )
        assert node.encode_called_with_params is not None
        assert node.encode_called_with_params["scaling_factor"] == 0.18215
        assert node.encode_called_with_params["batch_split"] == 2


# ===== 输入标准化测试 =====


class TestVAEEncodeNodeNormalizeInput:
    """_normalize_images_input 方法测试。"""

    def test_normalize_images_instance(self):
        """Images 实例直接返回。"""
        node = VAEEncodeNode()
        img = MagicMock(spec=Image)
        images = Images.from_single(img)
        result = node._normalize_images_input(images)
        assert isinstance(result, Images)
        assert len(result) == 1

    def test_normalize_single_image(self):
        """单个 Image 包装为 Images 长度=1。"""
        node = VAEEncodeNode()
        img = MagicMock(spec=Image)
        result = node._normalize_images_input(img)
        assert isinstance(result, Images)
        assert len(result) == 1

    def test_normalize_list_of_images(self):
        """list[Image] 包装为 Images。"""
        node = VAEEncodeNode()
        items = [MagicMock(spec=Image) for _ in range(3)]
        result = node._normalize_images_input(items)
        assert isinstance(result, Images)
        assert len(result) == 3

    def test_normalize_unsupported_type_raises(self):
        """不支持的类型报错。"""
        node = VAEEncodeNode()
        with pytest.raises(TypeError):
            node._normalize_images_input(12345)


# ===== Latent 输出标准化测试 =====


class TestVAEEncodeNodeNormalizeLatent:
    """_normalize_latent_output 方法测试。"""

    def test_normalize_latent_instance(self):
        """Latent 实例直接返回。"""
        node = VAEEncodeNode()
        latent = Latent(tensor=MagicMock())
        result = node._normalize_latent_output({"latent": latent})
        assert result is latent

    def test_normalize_tensor_wrapped_as_latent(self):
        """torch.Tensor 包装为 Latent。"""
        node = VAEEncodeNode()
        mock_tensor = MagicMock()
        mock_tensor.dim.return_value = 4
        result = node._normalize_latent_output({"latent": mock_tensor})
        assert isinstance(result, Latent)

    def test_normalize_dict_missing_latent_raises(self):
        """dict 缺少 latent 字段 → TypeError。"""
        node = VAEEncodeNode()
        with pytest.raises(TypeError):
            node._normalize_latent_output({"wrong": MagicMock()})

    def test_normalize_unsupported_type_raises(self):
        """不支持的类型报错。"""
        node = VAEEncodeNode()
        with pytest.raises(TypeError):
            node._normalize_latent_output({"latent": "not a tensor"})


# ===== scaling_factor 推断测试 =====


class TestVAEEncodeNodeInferFactors:
    """_infer_scaling_factor / _infer_shift_factor 方法测试。"""

    def test_infer_scaling_factor_from_config(self):
        """从 vae.config.scaling_factor 推断。"""
        node = VAEEncodeNode()
        vae = MagicMock()
        vae.config.scaling_factor = 0.18215
        result = node._infer_scaling_factor(vae)
        assert result == 0.18215

    def test_infer_scaling_factor_default(self):
        """无 config 时用默认值 0.18215。"""
        node = VAEEncodeNode()
        vae = MagicMock()
        del vae.config  # 移除 config 属性
        result = node._infer_scaling_factor(vae)
        assert result == 0.18215

    def test_infer_shift_factor_from_config(self):
        """从 vae.config.shift_factor 推断（Flux VAE）。"""
        node = VAEEncodeNode()
        vae = MagicMock()
        vae.config.shift_factor = 0.1
        result = node._infer_shift_factor(vae)
        assert result == 0.1

    def test_infer_shift_factor_none(self):
        """无 shift_factor 时返回 None。"""
        node = VAEEncodeNode()
        vae = MagicMock()
        del vae.config
        result = node._infer_shift_factor(vae)
        assert result is None

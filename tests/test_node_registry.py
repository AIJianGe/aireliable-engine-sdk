"""node_registry.py 节点注册表单元测试。

覆盖：
    - register / get / list / unregister / clear / contains
    - 重复注册策略（override）
    - 节点类校验（非 BaseNode 子类 / 缺 node_type / 不一致）
    - 装饰器 @register_node
    - discover_plugins（mock entry points）
    - 全局默认注册表 vs 实例化注册表（测试隔离）
    - 线程安全（基本测试）
"""

from __future__ import annotations

import threading
import pytest

from aireliable_engine.node import BaseNode, PortSpec
from aireliable_engine.node_registry import (
    InvalidNodeClassError,
    NodeAlreadyRegisteredError,
    NodeNotFoundError,
    NodeRegistry,
    get_node,
    list_nodes,
    register_node,
)


# ===========================================================================
# 测试 fixtures
# ===========================================================================

@pytest.fixture
def isolated_registry() -> NodeRegistry:
    """提供独立的 NodeRegistry 实例（不污染全局）。"""
    return NodeRegistry()


@pytest.fixture
def sample_node_class() -> type:
    """返回一个示例 BaseNode 子类。"""

    class _SampleNode(BaseNode):
        node_type = "SampleNode"
        input_specs: list[PortSpec] = [PortSpec(name="text", type="text")]
        output_specs: list[PortSpec] = [PortSpec(name="result", type="text")]

        async def execute(self, inputs, ctx):  # type: ignore[no-untyped-def]
            return {"result": inputs.get("text", "")}

    return _SampleNode


@pytest.fixture
def clean_global_registry():
    """测试前后清理全局注册表（确保测试隔离）。"""
    NodeRegistry.clear_default()
    yield
    NodeRegistry.clear_default()


# ===========================================================================
# NodeRegistry 实例方法测试
# ===========================================================================

class TestNodeRegistryBasic:
    """NodeRegistry 基础功能测试。"""

    def test_register_and_get(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """注册并查询节点。"""
        isolated_registry.register("SampleNode", sample_node_class)
        assert isolated_registry.get("SampleNode") is sample_node_class

    def test_get_not_found(self, isolated_registry: NodeRegistry) -> None:
        """查询未注册节点抛 NodeNotFoundError。"""
        with pytest.raises(NodeNotFoundError, match="未注册"):
            isolated_registry.get("NonExistent")

    def test_try_get_returns_none(self, isolated_registry: NodeRegistry) -> None:
        """try_get 未注册返回 None。"""
        assert isolated_registry.try_get("NonExistent") is None

    def test_try_get_returns_cls(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """try_get 已注册返回类。"""
        isolated_registry.register("SampleNode", sample_node_class)
        assert isolated_registry.try_get("SampleNode") is sample_node_class

    def test_list_returns_copy(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """list 返回字典副本，修改不影响注册表。"""
        isolated_registry.register("SampleNode", sample_node_class)
        items = isolated_registry.list()
        assert items == {"SampleNode": sample_node_class}
        # 修改副本
        items["foo"] = sample_node_class
        # 原注册表不受影响
        assert "foo" not in isolated_registry.list()

    def test_list_types(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """list_types 返回类型字符串列表。"""
        isolated_registry.register("SampleNode", sample_node_class)
        assert isolated_registry.list_types() == ["SampleNode"]

    def test_contains(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """contains 检查。"""
        assert not isolated_registry.contains("SampleNode")
        isolated_registry.register("SampleNode", sample_node_class)
        assert isolated_registry.contains("SampleNode")

    def test_in_operator(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """__contains__ 支持 in 操作符。"""
        assert "SampleNode" not in isolated_registry
        isolated_registry.register("SampleNode", sample_node_class)
        assert "SampleNode" in isolated_registry

    def test_len(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """__len__ 返回数量。"""
        assert len(isolated_registry) == 0
        isolated_registry.register("SampleNode", sample_node_class)
        assert len(isolated_registry) == 1

    def test_unregister(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """unregister 移除节点。"""
        isolated_registry.register("SampleNode", sample_node_class)
        removed = isolated_registry.unregister("SampleNode")
        assert removed is sample_node_class
        assert "SampleNode" not in isolated_registry

    def test_unregister_not_found(self, isolated_registry: NodeRegistry) -> None:
        """unregister 未注册返回 None。"""
        assert isolated_registry.unregister("NonExistent") is None

    def test_clear(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """clear 清空注册表。"""
        isolated_registry.register("SampleNode", sample_node_class)
        isolated_registry.clear()
        assert len(isolated_registry) == 0
        # _discovered 也应被重置
        assert isolated_registry._discovered is False

    def test_repr(self, isolated_registry: NodeRegistry) -> None:
        """repr 输出。"""
        repr_str = repr(isolated_registry)
        assert "NodeRegistry" in repr_str
        assert "registered=0" in repr_str


# ===========================================================================
# 注册校验测试
# ===========================================================================

class TestRegisterValidation:
    """register 参数校验测试。"""

    def test_register_non_basenode_class_raises(
        self, isolated_registry: NodeRegistry
    ) -> None:
        """注册非 BaseNode 子类抛 InvalidNodeClassError。"""
        class NotANode:
            node_type = "Foo"

        with pytest.raises(InvalidNodeClassError, match="必须是 BaseNode 的子类"):
            isolated_registry.register("Foo", NotANode)  # type: ignore[arg-type]

    def test_register_class_without_node_type_raises(
        self, isolated_registry: NodeRegistry
    ) -> None:
        """注册缺少 node_type 类属性的 BaseNode 子类。"""
        class NoTypeNode(BaseNode):
            node_type = ""  # 空 node_type
            async def execute(self, inputs, ctx):
                return {}

        with pytest.raises(InvalidNodeClassError, match="缺少 node_type"):
            isolated_registry.register("NoTypeNode", NoTypeNode)

    def test_register_mismatched_type_raises(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """node_type 参数与类属性不一致。"""
        with pytest.raises(InvalidNodeClassError, match="不一致"):
            isolated_registry.register("DifferentName", sample_node_class)

    def test_register_same_class_twice_no_error(
        self,
        isolated_registry: NodeRegistry,
        sample_node_class: type,
    ) -> None:
        """重复注册同一类不报错（幂等）。"""
        isolated_registry.register("SampleNode", sample_node_class)
        isolated_registry.register("SampleNode", sample_node_class)  # 同一个类
        assert len(isolated_registry) == 1


# ===========================================================================
# 重复注册策略测试
# ===========================================================================

class TestOverride:
    """重复注册策略测试。"""

    def test_register_different_class_same_type_raises(
        self,
        isolated_registry: NodeRegistry,
    ) -> None:
        """不同类用相同 node_type 注册（无 override）抛异常。"""
        class NodeA(BaseNode):
            node_type = "SameType"
            async def execute(self, inputs, ctx):
                return {}

        class NodeB(BaseNode):
            node_type = "SameType"
            async def execute(self, inputs, ctx):
                return {}

        isolated_registry.register("SameType", NodeA)
        with pytest.raises(NodeAlreadyRegisteredError, match="已注册"):
            isolated_registry.register("SameType", NodeB)

    def test_register_override_replaces(
        self,
        isolated_registry: NodeRegistry,
    ) -> None:
        """override=True 时覆盖已注册节点。"""
        class NodeA(BaseNode):
            node_type = "SameType"
            async def execute(self, inputs, ctx):
                return {}

        class NodeB(BaseNode):
            node_type = "SameType"
            async def execute(self, inputs, ctx):
                return {}

        isolated_registry.register("SameType", NodeA)
        isolated_registry.register("SameType", NodeB, override=True)
        assert isolated_registry.get("SameType") is NodeB


# ===========================================================================
# @register_node 装饰器测试
# ===========================================================================

class TestRegisterNodeDecorator:
    """@register_node 装饰器测试。"""

    def test_decorator_registers_to_global(self, clean_global_registry) -> None:
        """装饰器模式注册到全局注册表。"""
        @register_node
        class DecoratedNode(BaseNode):
            node_type = "DecoratedNode"
            async def execute(self, inputs, ctx):
                return {}

        # 装饰器应返回类本身
        assert DecoratedNode is not None
        # 全局注册表应包含该节点
        assert "DecoratedNode" in NodeRegistry.list_all()
        assert NodeRegistry.get_cls("DecoratedNode") is DecoratedNode

    def test_decorator_returns_class_unchanged(self, clean_global_registry) -> None:
        """装饰器应返回原始类（不修改）。"""
        @register_node
        class MyNode(BaseNode):
            node_type = "MyNode"
            async def execute(self, inputs, ctx):
                return {}

        # 验证类属性 / 方法保留
        assert MyNode.node_type == "MyNode"
        assert hasattr(MyNode, "execute")

    def test_decorator_non_basenode_raises(self, clean_global_registry) -> None:
        """装饰非 BaseNode 子类抛 InvalidNodeClassError。"""
        with pytest.raises(InvalidNodeClassError, match="必须是 BaseNode 子类"):
            @register_node
            class NotANode:
                node_type = "NotANode"

    def test_decorator_missing_node_type_raises(self, clean_global_registry) -> None:
        """装饰缺少 node_type 的类抛 InvalidNodeClassError。"""
        with pytest.raises(InvalidNodeClassError, match="缺少 node_type"):
            @register_node
            class NoTypeNode(BaseNode):
                node_type = ""
                async def execute(self, inputs, ctx):
                    return {}

    def test_function_style_register(
        self,
        clean_global_registry,
        sample_node_class: type,
    ) -> None:
        """函数式调用：register_node("Type", Class)。"""
        register_node("SampleNode", sample_node_class)
        assert "SampleNode" in NodeRegistry.list_all()

    def test_module_level_get_node(self, clean_global_registry) -> None:
        """模块级 get_node 便捷函数。"""
        @register_node
        class MyNode(BaseNode):
            node_type = "MyNode"
            async def execute(self, inputs, ctx):
                return {}

        assert get_node("MyNode") is MyNode

    def test_module_level_list_nodes(self, clean_global_registry) -> None:
        """模块级 list_nodes 便捷函数。"""
        @register_node
        class Node1(BaseNode):
            node_type = "Node1"
            async def execute(self, inputs, ctx):
                return {}

        @register_node
        class Node2(BaseNode):
            node_type = "Node2"
            async def execute(self, inputs, ctx):
                return {}

        nodes = list_nodes()
        assert "Node1" in nodes
        assert "Node2" in nodes


# ===========================================================================
# Entry points 自动发现测试
# ===========================================================================

class TestDiscoverPlugins:
    """discover_plugins 测试（用 mock entry points）。"""

    def test_discover_plugins_no_packages(self, isolated_registry: NodeRegistry) -> None:
        """无第三方节点包时不报错。"""
        result = isolated_registry.discover_plugins()
        # 当前测试环境没有 aireliable.nodes entry points
        assert result == {}

    def test_discover_plugins_idempotent(
        self, isolated_registry: NodeRegistry
    ) -> None:
        """默认只扫描一次，再次调用返回空。"""
        isolated_registry.discover_plugins()
        result = isolated_registry.discover_plugins()
        assert result == {}

    def test_discover_plugins_force_rescans(
        self, isolated_registry: NodeRegistry
    ) -> None:
        """force=True 可重新扫描。"""
        isolated_registry.discover_plugins()
        # 不应抛异常
        result = isolated_registry.discover_plugins(force=True)
        assert result == {}  # 测试环境无 entry points

    def test_discover_plugins_with_mock_entry_points(
        self,
        isolated_registry: NodeRegistry,
        monkeypatch,
    ) -> None:
        """mock entry_points 验证发现流程。"""
        class MockEntryPoint:
            def __init__(self, name, cls):
                self.name = name
                self.value = f"mock:{cls.__name__}"
                self._cls = cls

            def load(self):
                return self._cls

        class MockNode(BaseNode):
            node_type = "MockDiscoveredNode"
            async def execute(self, inputs, ctx):
                return {}

        def mock_entry_points(group=None):
            if group == "aireliable.nodes":
                return [MockEntryPoint("mock_node", MockNode)]
            return []

        monkeypatch.setattr(
            "importlib.metadata.entry_points", mock_entry_points
        )

        result = isolated_registry.discover_plugins()
        assert "MockDiscoveredNode" in result
        assert isolated_registry.contains("MockDiscoveredNode")

    def test_discover_plugins_skips_invalid_class(
        self,
        isolated_registry: NodeRegistry,
        monkeypatch,
    ) -> None:
        """加载的对象不是 BaseNode 子类时跳过。"""
        class NotANode:
            pass

        class MockEntryPoint:
            def __init__(self, name, obj):
                self.name = name
                self.value = "mock"
                self._obj = obj

            def load(self):
                return self._obj

        def mock_entry_points(group=None):
            if group == "aireliable.nodes":
                return [MockEntryPoint("bad", NotANode)]
            return []

        monkeypatch.setattr(
            "importlib.metadata.entry_points", mock_entry_points
        )

        result = isolated_registry.discover_plugins()
        assert result == {}
        assert len(isolated_registry) == 0

    def test_discover_plugins_skips_load_failure(
        self,
        isolated_registry: NodeRegistry,
        monkeypatch,
    ) -> None:
        """entry point load 抛异常时跳过。"""
        class MockEntryPoint:
            def __init__(self, name):
                self.name = name
                self.value = "mock"

            def load(self):
                raise ImportError("simulated load failure")

        def mock_entry_points(group=None):
            if group == "aireliable.nodes":
                return [MockEntryPoint("bad")]
            return []

        monkeypatch.setattr(
            "importlib.metadata.entry_points", mock_entry_points
        )

        result = isolated_registry.discover_plugins()
        assert result == {}
        assert len(isolated_registry) == 0

    def test_discover_plugins_skips_missing_node_type(
        self,
        isolated_registry: NodeRegistry,
        monkeypatch,
    ) -> None:
        """加载的节点类缺少 node_type 时跳过。"""
        class NoTypeNode(BaseNode):
            node_type = ""  # 空
            async def execute(self, inputs, ctx):
                return {}

        class MockEntryPoint:
            def __init__(self, name, obj):
                self.name = name
                self.value = "mock"
                self._obj = obj

            def load(self):
                return self._obj

        def mock_entry_points(group=None):
            if group == "aireliable.nodes":
                return [MockEntryPoint("no_type", NoTypeNode)]
            return []

        monkeypatch.setattr(
            "importlib.metadata.entry_points", mock_entry_points
        )

        result = isolated_registry.discover_plugins()
        assert result == {}

    def test_discover_plugins_custom_group(
        self, isolated_registry: NodeRegistry
    ) -> None:
        """自定义 entry points group。"""
        result = isolated_registry.discover_plugins(group="custom.group")
        assert result == {}


# ===========================================================================
# 线程安全测试
# ===========================================================================

class TestThreadSafety:
    """线程安全测试。"""

    def test_concurrent_register(self) -> None:
        """并发注册不丢失节点。"""
        registry = NodeRegistry()

        def make_node(i: int) -> type:
            class N(BaseNode):
                node_type = f"Node{i}"
                async def execute(self, inputs, ctx):
                    return {}
            return N

        def register_batch(start: int, count: int) -> None:
            for i in range(start, start + count):
                registry.register(f"Node{i}", make_node(i))

        threads = [
            threading.Thread(target=register_batch, args=(i * 10, 10))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 应有 40 个节点全部注册成功
        assert len(registry) == 40

    def test_concurrent_get(self, isolated_registry: NodeRegistry) -> None:
        """并发查询不报错。"""
        class N(BaseNode):
            node_type = "N"
            async def execute(self, inputs, ctx):
                return {}

        isolated_registry.register("N", N)

        results = []
        def query():
            for _ in range(100):
                results.append(isolated_registry.get("N"))

        threads = [threading.Thread(target=query) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is N for r in results)


# ===========================================================================
# v0.2.5 P0-J2：discover_local_dir 测试
# ===========================================================================

class TestDiscoverLocalDir:
    """discover_local_dir 本地目录节点发现测试。"""

    def test_normal_load(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """正常加载：有效节点文件被发现并注册。"""
        node_file = tmp_path / "my_node.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode, PortSpec\n"
            "\n"
            "class MyCustomNode(BaseNode):\n"
            "    node_type = 'MyCustomNode'\n"
            "    input_specs = []\n"
            "    output_specs = [PortSpec(name='out', type='text')]\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {'out': 'hello'}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path)
        assert "MyCustomNode" in result
        assert isolated_registry.contains("MyCustomNode")

    def test_load_failure_skipped(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """加载失败：语法错误文件被跳过，记录错误。"""
        bad_file = tmp_path / "bad_node.py"
        bad_file.write_text("this is not valid python !!!", encoding="utf-8")

        result = isolated_registry.discover_local_dir(tmp_path)
        assert result == {}
        assert len(isolated_registry) == 0
        # 错误被记录
        errors = isolated_registry.list_errors()
        assert len(errors) == 1
        assert "bad_node.py" in errors

    def test_empty_directory(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """空目录：返回空字典。"""
        result = isolated_registry.discover_local_dir(tmp_path)
        assert result == {}

    def test_nonexistent_directory(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """不存在的目录：返回空字典。"""
        result = isolated_registry.discover_local_dir(tmp_path / "nonexistent")
        assert result == {}

    def test_multiple_nodes_in_file(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """单文件多节点：嵌套类全部被发现。"""
        node_file = tmp_path / "multi.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode, PortSpec\n"
            "\n"
            "class NodeA(BaseNode):\n"
            "    node_type = 'NodeA'\n"
            "    input_specs = []\n"
            "    output_specs = []\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n"
            "\n"
            "class NodeB(BaseNode):\n"
            "    node_type = 'NodeB'\n"
            "    input_specs = []\n"
            "    output_specs = []\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path)
        assert "NodeA" in result
        assert "NodeB" in result
        assert len(isolated_registry) == 2

    def test_skip_underscore_files(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """下划线开头的文件被跳过（如 __init__.py、_template.py）。"""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("# package init\n", encoding="utf-8")
        template_file = tmp_path / "_template.py"
        template_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class TemplateNode(BaseNode):\n"
            "    node_type = 'Template'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )
        # 正常节点文件
        node_file = tmp_path / "good.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class GoodNode(BaseNode):\n"
            "    node_type = 'GoodNode'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path)
        assert "GoodNode" in result
        assert "Template" not in result
        assert len(isolated_registry) == 1

    def test_override_existing(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """override=True 时覆盖已注册节点。"""
        # 先注册一个
        class OriginalNode(BaseNode):
            node_type = "OverrideMe"
            async def execute(self, inputs, ctx):
                return {}

        isolated_registry.register("OverrideMe", OriginalNode)
        assert isolated_registry.get("OverrideMe") is OriginalNode

        # 用 override=True 覆盖
        node_file = tmp_path / "override.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class OverrideMe(BaseNode):\n"
            "    node_type = 'OverrideMe'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {'overridden': True}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path, override=True)
        assert "OverrideMe" in result
        assert isolated_registry.get("OverrideMe") is not OriginalNode

    def test_missing_node_type_recorded(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """缺少 node_type 属性的节点类被跳过并记录错误。"""
        node_file = tmp_path / "no_type.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class NoTypeNode(BaseNode):\n"
            "    node_type = ''\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path)
        assert result == {}
        errors = isolated_registry.list_errors()
        assert len(errors) >= 1

    def test_recursive_scan(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """递归扫描子目录。"""
        sub_dir = tmp_path / "sub" / "deep"
        sub_dir.mkdir(parents=True)
        node_file = sub_dir / "nested.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class NestedNode(BaseNode):\n"
            "    node_type = 'NestedNode'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path, recursive=True)
        assert "NestedNode" in result

    def test_non_recursive_scan(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """非递归扫描不进入子目录。"""
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        node_file = sub_dir / "nested.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class NestedNode(BaseNode):\n"
            "    node_type = 'NestedNode'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )
        top_file = tmp_path / "top.py"
        top_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "class TopNode(BaseNode):\n"
            "    node_type = 'TopNode'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path, recursive=False)
        assert "TopNode" in result
        assert "NestedNode" not in result

    def test_imported_baseclass_skipped(self, isolated_registry: NodeRegistry, tmp_path) -> None:
        """import 进来的 BaseNode 基类本身不被注册。"""
        node_file = tmp_path / "user_node.py"
        node_file.write_text(
            "from aireliable_engine.node import BaseNode\n"
            "\n"
            "class UserNode(BaseNode):\n"
            "    node_type = 'UserNode'\n"
            "    async def execute(self, inputs, ctx):\n"
            "        return {}\n",
            encoding="utf-8",
        )

        result = isolated_registry.discover_local_dir(tmp_path)
        assert "UserNode" in result
        # BaseNode 本身不应被注册
        assert "BaseNode" not in result

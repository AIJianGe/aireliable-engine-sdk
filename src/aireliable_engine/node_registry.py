"""节点注册表（v0.2 新增）。

提供节点类的注册、查询与自动发现机制。

设计参考：

核心特性：
    1. 显式注册：register_node("TypeName", NodeClass) 或 @register_node 装饰器
    2. Entry points 自动发现：扫描 aireliable.nodes group，pip install 后自动注册
    3. 全局默认注册表 + 实例化注册表（多引擎独立注册表 / 测试隔离）
    4. 线程安全（注册时加锁）

使用示例：
    # 方式 1：显式注册
    class MyNode(BaseNode):
        node_type = "MyNode"
        ...
    register_node("MyNode", MyNode)

    # 方式 2：装饰器注册（自动用 node_type 字段作为 key）
    @register_node
    class MyNode(BaseNode):
        node_type = "MyNode"
        ...

    # 查询节点
    NodeClass = NodeRegistry.get("MyNode")
    all_nodes = NodeRegistry.list()

    # 自动发现第三方节点包
    NodeRegistry.discover_plugins()
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from .node import BaseNode


logger = logging.getLogger(__name__)


# ===========================================================================
# 异常类
# ===========================================================================

class NodeNotFoundError(KeyError):
    """节点类型未注册异常。"""

    def __init__(self, node_type: str):
        self.node_type = node_type
        super().__init__(f"节点类型 {node_type!r} 未注册")


class NodeAlreadyRegisteredError(ValueError):
    """节点类型已注册异常（重复注册时）。"""

    def __init__(self, node_type: str, existing_cls: type):
        self.node_type = node_type
        self.existing_cls = existing_cls
        super().__init__(
            f"节点类型 {node_type!r} 已注册为 {existing_cls.__name__}，"
            f"无法重复注册（如需覆盖，传 override=True）"
        )


class InvalidNodeClassError(TypeError):
    """无效的节点类异常（不是 BaseNode 子类 / 缺少 node_type）。"""

    def __init__(self, message: str):
        super().__init__(message)


# ===========================================================================
# NodeRegistry：节点注册表
# ===========================================================================

class NodeRegistry:
    """节点注册表。

    管理节点类型字符串 → 节点类的映射。提供：
        - register(node_type, node_cls)：显式注册
        - get(node_type)：查询节点类
        - list()：列出所有已注册节点
        - unregister(node_type)：注销节点
        - discover_plugins()：通过 entry points 自动发现第三方节点包
        - clear()：清空注册表（仅测试用）

    设计：
        - 全局默认注册表通过类方法访问（NodeRegistry.register / get / list）
        - 实例化注册表用于测试隔离或多引擎独立注册表
        - 线程安全：所有写操作加锁
        - 重复注册策略：默认报错，传 override=True 可覆盖

    Example:
        # 全局注册表（推荐生产用法）
        NodeRegistry.register("MyNode", MyNode)
        NodeClass = NodeRegistry.get("MyNode")

        # 实例化注册表（测试隔离）
        registry = NodeRegistry()
        registry.register("MyNode", MyNode)
    """

    # 全局默认注册表实例（所有类方法代理到此实例）
    _default: "NodeRegistry"
    _default_initialized: bool = False

    def __init__(self) -> None:
        """初始化空注册表。"""
        self._nodes: dict[str, Type["BaseNode"]] = {}
        self._lock = threading.RLock()
        self._discovered: bool = False  # 是否已执行过 discover_plugins
        # v0.2.5 P0-H3：加载失败的节点记录（key=节点标识，value=错误信息 dict）
        # 供 GET /nodes?include_errors=true 返回，前端可显示红色标记 + 错误 tooltip
        self._load_errors: dict[str, dict] = {}

    # ===== 实例方法 =====

    def register(
        self,
        node_type: str,
        node_cls: Type["BaseNode"],
        *,
        override: bool = False,
    ) -> None:
        """注册节点类。

        Args:
            node_type: 节点类型字符串（如 "ModelLoader"）。需与 node_cls.node_type 一致。
            node_cls: BaseNode 子类。
            override: 是否覆盖已注册的同名节点。默认 False（重复注册报错）。

        Raises:
            InvalidNodeClassError: node_cls 不是 BaseNode 子类，或 node_type 与类属性不一致。
            NodeAlreadyRegisteredError: 节点类型已注册且 override=False。
        """
        # 延迟 import 避免循环依赖
        from .node import BaseNode

        # 校验 node_cls 是 BaseNode 子类
        if not (isinstance(node_cls, type) and issubclass(node_cls, BaseNode)):
            raise InvalidNodeClassError(
                f"node_cls 必须是 BaseNode 的子类，得到 {node_cls!r}"
            )

        # 校验 node_type 与类属性一致
        cls_node_type = getattr(node_cls, "node_type", "")
        if not cls_node_type:
            raise InvalidNodeClassError(
                f"节点类 {node_cls.__name__} 缺少 node_type 类属性"
            )
        if cls_node_type != node_type:
            raise InvalidNodeClassError(
                f"node_type 参数 {node_type!r} 与类属性 node_type={cls_node_type!r} 不一致"
            )

        with self._lock:
            existing = self._nodes.get(node_type)
            if existing is not None and existing is not node_cls:
                if not override:
                    raise NodeAlreadyRegisteredError(node_type, existing)
                logger.warning(
                    "覆盖已注册节点 %s: %s → %s",
                    node_type, existing.__name__, node_cls.__name__,
                )
            self._nodes[node_type] = node_cls
            logger.debug("注册节点 %s → %s", node_type, node_cls.__name__)

    def get(self, node_type: str) -> Type["BaseNode"]:
        """查询节点类。

        Args:
            node_type: 节点类型字符串。

        Returns:
            节点类。

        Raises:
            NodeNotFoundError: 节点类型未注册。
        """
        with self._lock:
            cls = self._nodes.get(node_type)
        if cls is None:
            raise NodeNotFoundError(node_type)
        return cls

    def try_get(self, node_type: str) -> Optional[Type["BaseNode"]]:
        """查询节点类（不抛异常）。

        Args:
            node_type: 节点类型字符串。

        Returns:
            节点类，未注册返回 None。
        """
        with self._lock:
            return self._nodes.get(node_type)

    def list(self) -> dict[str, Type["BaseNode"]]:
        """列出所有已注册节点。

        Returns:
            节点类型 → 节点类 的字典（副本，修改不影响注册表）。
        """
        with self._lock:
            return dict(self._nodes)

    def list_types(self) -> list[str]:
        """列出所有已注册节点类型字符串。

        Returns:
            节点类型字符串列表。
        """
        with self._lock:
            return list(self._nodes.keys())

    def unregister(self, node_type: str) -> Optional[Type["BaseNode"]]:
        """注销节点。

        Args:
            node_type: 节点类型字符串。

        Returns:
            被移除的节点类，未注册返回 None。
        """
        with self._lock:
            return self._nodes.pop(node_type, None)

    def clear(self) -> None:
        """清空注册表。

        主要用于测试隔离。生产环境慎用。
        """
        with self._lock:
            self._nodes.clear()
            self._discovered = False
            self._load_errors.clear()

    # ===== v0.2.5 P0-H3：加载错误跟踪 =====

    def record_load_error(
        self,
        node_identifier: str,
        error: str,
        *,
        source: str = "unknown",
        module: Optional[str] = None,
    ) -> None:
        """记录节点加载失败（供 GET /nodes?include_errors=true 返回）。

        Args:
            node_identifier: 节点标识（entry point name / 文件名 / 类名等）。
            error: 错误信息（异常字符串）。
            source: 错误来源（"entry_point" / "local_dir" / "register" 等）。
            module: 节点模块路径（如 "my_custom_nodes.style_transfer"）。
        """
        with self._lock:
            self._load_errors[node_identifier] = {
                "error": str(error),
                "source": source,
                "module": module,
            }
            logger.warning(
                "节点加载失败: %s (source=%s, module=%s): %s",
                node_identifier, source, module, error,
            )

    def list_errors(self) -> dict[str, dict]:
        """列出所有加载失败的节点记录。

        Returns:
            节点标识 → 错误信息 dict 的副本。
        """
        with self._lock:
            return dict(self._load_errors)

    def clear_errors(self) -> None:
        """清空加载错误记录。"""
        with self._lock:
            self._load_errors.clear()

    def contains(self, node_type: str) -> bool:
        """检查节点类型是否已注册。"""
        with self._lock:
            return node_type in self._nodes

    def __len__(self) -> int:
        """已注册节点数量。"""
        with self._lock:
            return len(self._nodes)

    def __contains__(self, node_type: str) -> bool:
        """支持 `in` 操作符。"""
        return self.contains(node_type)

    def __repr__(self) -> str:
        with self._lock:
            n = len(self._nodes)
        return f"NodeRegistry(registered={n})"

    # ===== Entry points 自动发现 =====

    def discover_plugins(
        self,
        *,
        group: str = "aireliable.nodes",
        override: bool = False,
        force: bool = False,
    ) -> dict[str, Type["BaseNode"]]:
        """通过 Python entry points 发现并注册第三方节点包。

        扫描指定 entry points group，加载所有声明的节点类并注册。

        Args:
            group: entry points group 名称。默认 "aireliable.nodes"。
            override: 是否覆盖已注册节点（同名冲突时）。
            force: 是否强制重新扫描（默认只扫描一次）。

        Returns:
            本次扫描发现的节点字典 {node_type: node_cls}。

        Note:
            - 默认只扫描一次（避免重复日志）。force=True 可重复扫描。
            - 单个节点加载失败不会中断整体扫描，仅记录 warning。
            - 发现的节点会立即注册到当前注册表。
        """
        with self._lock:
            if self._discovered and not force:
                logger.debug("discover_plugins 已执行过，跳过（force=True 可重新扫描）")
                return {}
            self._discovered = True

        discovered: dict[str, Type["BaseNode"]] = {}

        try:
            from importlib.metadata import entry_points
        except ImportError:  # Python < 3.10 不可能（requires-python >=3.10）
            logger.warning("importlib.metadata 不可用，跳过 entry points 发现")
            return discovered

        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python 3.9- 兼容（entry_points 不接受 group 参数）
            # SDK 已要求 Python >=3.10，此分支仅防御性
            eps = entry_points().get(group, [])
        except Exception as e:
            logger.warning("扫描 entry points 失败: %s", e)
            return discovered

        for ep in eps:
            try:
                node_cls = ep.load()
            except Exception as e:
                # v0.2.5 P0-H3：记录加载失败的 entry point
                self.record_load_error(
                    ep.name,
                    f"加载 entry point 失败: {e}",
                    source="entry_point",
                    module=ep.value,
                )
                continue

            # 校验是 BaseNode 子类
            from .node import BaseNode
            if not (isinstance(node_cls, type) and issubclass(node_cls, BaseNode)):
                self.record_load_error(
                    ep.name,
                    f"加载的对象不是 BaseNode 子类: {node_cls!r}",
                    source="entry_point",
                    module=ep.value,
                )
                continue

            # 用类属性 node_type 作为注册 key
            node_type = getattr(node_cls, "node_type", "")
            if not node_type:
                self.record_load_error(
                    ep.name,
                    f"节点类 {node_cls.__name__} 缺少 node_type 属性",
                    source="entry_point",
                    module=ep.value,
                )
                continue

            try:
                self.register(node_type, node_cls, override=override)
                discovered[node_type] = node_cls
                logger.info(
                    "发现第三方节点: %s → %s (entry point: %s)",
                    node_type, node_cls.__name__, ep.name,
                )
            except NodeAlreadyRegisteredError as e:
                if override:
                    # 不应到达（已传 override=True）
                    pass
                else:
                    logger.info(
                        "节点 %s 已注册（%s），跳过 entry point %s "
                        "（override=True 可覆盖）",
                        node_type, e.existing_cls.__name__, ep.name,
                    )
            except Exception as e:
                self.record_load_error(
                    ep.name,
                    f"注册失败: {e}",
                    source="entry_point",
                    module=ep.value,
                )

        if discovered:
            logger.info(
                "entry points 发现完成：共 %d 个节点（group=%s）",
                len(discovered), group,
            )
        else:
            logger.debug("entry points 未发现任何节点（group=%s）", group)

        return discovered

    # ===== v0.2.5 P0-J1：本地目录节点自动发现 =====

    def discover_local_dir(
        self,
        directory,
        *,
        override: bool = False,
        recursive: bool = True,
    ) -> dict[str, Type["BaseNode"]]:
        """扫描本地目录下的 .py 文件，发现并注册自定义节点。

        v0.2.5 P0-J1：支持从引擎 custom_nodes/ 目录加载用户自定义节点。

        扫描规则：
            1. 递归扫描目录下所有 .py 文件（recursive=True 时含子目录）
            2. 跳过以 _ 开头的文件（如 __init__.py、_template.py）
            3. 用 importlib.util.spec_from_file_location 动态加载每个文件
            4. 遍历模块中所有 BaseNode 子类并注册
            5. 单个文件加载失败仅 record_load_error，不中断扫描

        Args:
            directory: 要扫描的目录路径（str / Path）。
            override: 是否覆盖已注册节点（同名冲突时）。默认 False。
            recursive: 是否递归扫描子目录。默认 True。

        Returns:
            本次扫描发现的节点字典 {node_type: node_cls}。
        """
        from pathlib import Path
        from importlib.util import spec_from_file_location, module_from_spec

        directory = Path(directory) if not isinstance(directory, Path) else directory
        discovered: dict[str, Type["BaseNode"]] = {}

        if not directory.exists():
            logger.debug("discover_local_dir: 目录不存在 %s", directory)
            return discovered
        if not directory.is_dir():
            logger.warning("discover_local_dir: %s 不是目录", directory)
            return discovered

        # 收集所有 .py 文件
        if recursive:
            py_files = sorted(directory.rglob("*.py"))
        else:
            py_files = sorted(directory.glob("*.py"))

        # 过滤以 _ 开头的文件
        py_files = [f for f in py_files if not f.name.startswith("_")]

        if not py_files:
            logger.debug("discover_local_dir: %s 下无 .py 节点文件", directory)
            return discovered

        # 延迟 import BaseNode
        from .node import BaseNode

        for py_file in py_files:
            # 用唯一模块名避免冲突
            module_name = f"_custom_node_{py_file.stem}_{abs(hash(str(py_file)))}"
            try:
                spec = spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    self.record_load_error(
                        py_file.name,
                        "无法创建模块 spec（spec_from_file_location 返回 None）",
                        source="local_dir",
                        module=str(py_file),
                    )
                    continue

                module = module_from_spec(spec)
                spec.loader.exec_module(module)

            except Exception as e:
                self.record_load_error(
                    py_file.name,
                    f"加载文件失败: {e}",
                    source="local_dir",
                    module=str(py_file),
                )
                continue

            # 遍历模块中所有 BaseNode 子类
            found_in_file = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                # 必须是类 + BaseNode 子类 + 定义在当前模块中（非 import 进来的）
                if not (isinstance(attr, type) and issubclass(attr, BaseNode)):
                    continue
                if attr.__module__ != module_name:
                    continue  # 跳过 import 进来的基类
                if attr is BaseNode:
                    continue

                node_type = getattr(attr, "node_type", "")
                if not node_type:
                    self.record_load_error(
                        py_file.name,
                        f"节点类 {attr.__name__} 缺少 node_type 属性",
                        source="local_dir",
                        module=str(py_file),
                    )
                    continue

                try:
                    self.register(node_type, attr, override=override)
                    discovered[node_type] = attr
                    found_in_file += 1
                    logger.info(
                        "发现自定义节点: %s → %s (file: %s)",
                        node_type, attr.__name__, py_file.name,
                    )
                except NodeAlreadyRegisteredError as e:
                    if override:
                        pass
                    else:
                        logger.info(
                            "节点 %s 已注册（%s），跳过 %s "
                            "（override=True 可覆盖）",
                            node_type, e.existing_cls.__name__, py_file.name,
                        )
                except Exception as e:
                    self.record_load_error(
                        py_file.name,
                        f"注册失败: {e}",
                        source="local_dir",
                        module=str(py_file),
                    )

            if found_in_file == 0:
                logger.debug("discover_local_dir: %s 中未发现节点类", py_file.name)

        if discovered:
            logger.info(
                "本地目录发现完成：%s 下共 %d 个节点",
                directory, len(discovered),
            )
        else:
            logger.debug("本地目录未发现任何节点：%s", directory)

        return discovered

    # ===== 类方法：代理到全局默认注册表 =====

    @classmethod
    def _get_default(cls) -> "NodeRegistry":
        """获取（必要时创建）全局默认注册表实例。"""
        if not cls._default_initialized:
            with threading.Lock():
                if not cls._default_initialized:
                    cls._default = cls()
                    cls._default_initialized = True
        return cls._default

    @classmethod
    def register_cls(
        cls,
        node_type: str,
        node_cls: Type["BaseNode"],
        *,
        override: bool = False,
    ) -> None:
        """注册到全局默认注册表（类方法）。"""
        cls._get_default().register(node_type, node_cls, override=override)

    @classmethod
    def get_cls(cls, node_type: str) -> Type["BaseNode"]:
        """从全局默认注册表查询（类方法）。"""
        return cls._get_default().get(node_type)

    @classmethod
    def list_all(cls) -> dict[str, Type["BaseNode"]]:
        """列出全局默认注册表所有节点（类方法）。"""
        return cls._get_default().list()

    @classmethod
    def discover_all_plugins(
        cls,
        *,
        group: str = "aireliable.nodes",
        override: bool = False,
        force: bool = False,
    ) -> dict[str, Type["BaseNode"]]:
        """从全局默认注册表扫描 entry points（类方法）。"""
        return cls._get_default().discover_plugins(
            group=group, override=override, force=force,
        )

    @classmethod
    def clear_default(cls) -> None:
        """清空全局默认注册表（仅测试用）。"""
        if cls._default_initialized:
            cls._default.clear()


# ===========================================================================
# 模块级便捷函数
# ===========================================================================

def register_node(
    node_type_or_cls,
    node_cls: Optional[Type["BaseNode"]] = None,
    *,
    override: bool = False,
):
    """注册节点（模块级便捷函数，支持装饰器与函数两种调用方式）。

    用法 1：函数调用
        register_node("MyNode", MyNodeClass)

    用法 2：装饰器调用（自动用 node_type 类属性作为 key）
        @register_node
        class MyNode(BaseNode):
            node_type = "MyNode"
            ...

    Args:
        node_type_or_cls: 装饰器模式时为节点类；函数模式时为节点类型字符串。
        node_cls: 函数模式时为节点类；装饰器模式时为 None。
        override: 是否覆盖已注册节点。

    Returns:
        装饰器模式：返回节点类本身（无副作用）。
        函数模式：返回 None。
    """
    # 装饰器模式：@register_node 用在 class 上
    if node_cls is None and isinstance(node_type_or_cls, type):
        node_cls = node_type_or_cls
        # 延迟 import 避免循环依赖
        from .node import BaseNode
        if not issubclass(node_cls, BaseNode):
            raise InvalidNodeClassError(
                f"@register_node 装饰的类必须是 BaseNode 子类: {node_cls!r}"
            )
        node_type = getattr(node_cls, "node_type", "")
        if not node_type:
            raise InvalidNodeClassError(
                f"@register_node 装饰的类 {node_cls.__name__} 缺少 node_type 类属性"
            )
        NodeRegistry._get_default().register(node_type, node_cls, override=override)
        return node_cls  # 装饰器返回类本身

    # 函数模式：register_node("MyNode", MyNodeClass)
    if isinstance(node_type_or_cls, str) and node_cls is not None:
        NodeRegistry._get_default().register(
            node_type_or_cls, node_cls, override=override,
        )
        return None

    raise TypeError(
        f"register_node 调用方式错误：用法 register_node(type_str, cls) 或 "
        f"@register_node class。得到 register_node({node_type_or_cls!r}, {node_cls!r})"
    )


def get_node(node_type: str) -> Type["BaseNode"]:
    """从全局默认注册表查询节点类（便捷函数）。"""
    return NodeRegistry._get_default().get(node_type)


def list_nodes() -> dict[str, Type["BaseNode"]]:
    """列出全局默认注册表所有节点（便捷函数）。"""
    return NodeRegistry._get_default().list()


__all__ = [
    # 注册表
    "NodeRegistry",
    # 便捷函数
    "register_node",
    "get_node",
    "list_nodes",
    # 异常
    "NodeNotFoundError",
    "NodeAlreadyRegisteredError",
    "InvalidNodeClassError",
]

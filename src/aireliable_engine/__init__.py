"""AIReliable Engine SDK。

引擎开发工具包，提供：
    - BaseEngine：8 端点契约的引擎抽象基类（范式 A/C）
    - WorkflowEngine：工作流引擎基类（范式 B）
    - BaseNode + PortSpec：节点抽象（v0.2 新增）
    - NodeRegistry：节点注册表 + entry points 自动发现
    - WorkflowExecutor + workflow_dsl：工作流执行器 + 安全条件 DSL
    - schemas / SSEStream / StateMachine：8 端点契约配套
    - Images / Image / Audio / Video / Latent / Tensor：节点数据类型
    - 平台基础节点库（ImagesInput / TextInput / ModelLoader / ...）

安装：
    pip install aireliable-engine-sdk            # 最小核心
    pip install "aireliable-engine-sdk[image]"    # 图片引擎开发
    pip install "aireliable-engine-sdk[llm]"      # LLM 引擎开发

使用：
    from aireliable_engine import (
        BaseEngine, InferRequest, SSEStream,
    )

    class MyEngine(BaseEngine):
        ...
"""

from .engine_base import (
    BaseEngine,
    EngineBusyError,
    EngineInferError,
    EngineLoadError,
)
from .schemas import (
    AudioInput,
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    ImageInput,
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
    VideoInput,
)
from .sse import (
    SSEStream,
    make_error_frame,
    make_final_frame,
    make_frame,
)
from .state_machine import (
    EngineState,
    RuntimeState,
    StateMachine,
)
from .logging_config import EngineLogFormatter, setup_engine_logging
from .downloader import (
    BaseEngineDownloader,
    DownloadResult,
    DownloadSpec,
    HFSnapshotDownloader,
    ProgressCallback,
)
from .subprocess_utils import (
    decode_subprocess_output,
    find_uv_executable,
    make_engine_subprocess_flags,
    make_uv_run_command,
)

# v0.2 节点框架（阶段 2 完成）
from .node import (
    BaseNode,
    NodeExecutionError,
    NodeValidationError,
    PortSpec,
)
from .types import (
    Audio,
    Image,
    Images,
    Latent,
    Tensor,
    Video,
    get_port_type_class,
    is_valid_port_type,
)
from .node_registry import (
    InvalidNodeClassError,
    NodeAlreadyRegisteredError,
    NodeNotFoundError,
    NodeRegistry,
    get_node,
    list_nodes,
    register_node,
)
from .workflow_context import (
    ConsoleProgressReporter,
    NullProgressReporter,
    ProgressReporter,
    SimpleVRamManager,
    SSEProgressReporter,
    VRamManager,
    WorkflowContext,
    make_default_context,
)

# v0.2 工作流执行器 / 安全 DSL（阶段 4 完成）
# WorkflowEngine / 平台基础节点库
from .workflow_dsl import (
    DSLParseError,
    DSLEvalError,
    DSLError,
    Evaluator,
    Lexer,
    Parser,
    Token,
    Literal,
    VariableRef,
    FuncCall,
    MethodCall,
    BinaryOp,
    UnaryOp,
    evaluate_condition,
    validate_expression,
    parse_expression,
    eval_ast,
)
from .workflow_executor import (
    WorkflowCycleError,
    WorkflowExecutionError,
    WorkflowExecutor,
    WorkflowExecutorError,
    WorkflowReferenceError,
    WorkflowValidationError,
)
# v0.2 工作流 yaml schema（阶段 4 任务 4.5）
from .workflow_schema import (
    BranchDef,
    NodeDef,
    WorkflowSchema,
    WorkflowSchemaError,
    validate_workflow_dict,
    validate_workflow_yaml,
)
# v0.2 工作流引擎基类（阶段 5 任务 5.1）
from .workflow_engine import (
    WorkflowEngine,
    WorkflowEngineError,
    WorkflowLoadError,
    WorkflowNotFoundError,
)
# v0.2 工作流管理 API（阶段 12）
from .workflow_routes import (
    WorkflowRouter,
    create_workflow_router,
)
# v0.2 节点定义 API（阶段 12）
from .node_routes import (
    NodeRouter,
    create_node_router,
)
# v0.2 远程引擎协议（阶段 13）
from .remote_routes import (
    AuthMiddleware,
    RemoteEngineConfig,
    create_engine_app,
    mount_engine_routes,
)

__version__ = "0.2.0"

__all__ = [
    # ===== 引擎基类 =====
    "BaseEngine",
    "EngineBusyError",
    "EngineLoadError",
    "EngineInferError",
    # ===== 8 端点 schema =====
    "HealthResponse",
    "CapabilitiesResponse",
    "VariantInfo",
    "LoadRequest",
    "LoadResponse",
    "UnloadResponse",
    "InferRequest",
    "LoraSpec",
    "StatusResponse",
    "LoraLoadRequest",
    "LoraLoadResponse",
    "LoraUnloadRequest",
    "LoraUnloadResponse",
    "ErrorResponse",
    # ===== v0.2 多模态输入 schema =====
    "ImageInput",
    "AudioInput",
    "VideoInput",
    # ===== SSE =====
    "SSEStream",
    "make_frame",
    "make_final_frame",
    "make_error_frame",
    # ===== 状态机 =====
    "EngineState",
    "RuntimeState",
    "StateMachine",
    # ===== 日志 =====
    "EngineLogFormatter",
    "setup_engine_logging",
    # ===== 下载器 =====
    "BaseEngineDownloader",
    "DownloadSpec",
    "DownloadResult",
    "HFSnapshotDownloader",
    "ProgressCallback",
    # ===== 子进程 =====
    "make_engine_subprocess_flags",
    "make_uv_run_command",
    "decode_subprocess_output",
    "find_uv_executable",
    # ===== 节点框架（v0.2）=====
    "BaseNode",
    "PortSpec",
    "NodeValidationError",
    "NodeExecutionError",
    # ===== 节点数据类型（v0.2）=====
    "Image",
    "Images",
    "Audio",
    "Video",
    "Tensor",
    "Latent",
    "get_port_type_class",
    "is_valid_port_type",
    # ===== 节点注册表（v0.2）=====
    "NodeRegistry",
    "register_node",
    "get_node",
    "list_nodes",
    "NodeNotFoundError",
    "NodeAlreadyRegisteredError",
    "InvalidNodeClassError",
    # ===== 工作流上下文（v0.2）=====
    "WorkflowContext",
    "make_default_context",
    "ProgressReporter",
    "NullProgressReporter",
    "ConsoleProgressReporter",
    "SSEProgressReporter",
    "VRamManager",
    "SimpleVRamManager",
    # ===== 条件 DSL（v0.2 阶段 4）=====
    "DSLError",
    "DSLParseError",
    "DSLEvalError",
    "Token",
    "Lexer",
    "Parser",
    "Evaluator",
    "Literal",
    "VariableRef",
    "FuncCall",
    "MethodCall",
    "BinaryOp",
    "UnaryOp",
    "evaluate_condition",
    "validate_expression",
    "parse_expression",
    "eval_ast",
    # ===== 工作流执行器（v0.2 阶段 4）=====
    "WorkflowExecutor",
    "WorkflowExecutorError",
    "WorkflowValidationError",
    "WorkflowCycleError",
    "WorkflowExecutionError",
    "WorkflowReferenceError",
    # ===== 工作流 yaml schema（v0.2 阶段 4 任务 4.5）=====
    "WorkflowSchema",
    "NodeDef",
    "BranchDef",
    "WorkflowSchemaError",
    "validate_workflow_dict",
    "validate_workflow_yaml",
    # ===== 工作流引擎基类（v0.2 阶段 5 任务 5.1）=====
    "WorkflowEngine",
    "WorkflowEngineError",
    "WorkflowNotFoundError",
    "WorkflowLoadError",
    # ===== 工作流管理 API（v0.2 阶段 12）=====
    "WorkflowRouter",
    "create_workflow_router",
    # ===== 节点定义 API（v0.2 阶段 12）=====
    "NodeRouter",
    "create_node_router",
    # ===== 远程引擎协议（v0.2 阶段 13）=====
    "AuthMiddleware",
    "RemoteEngineConfig",
    "create_engine_app",
    "mount_engine_routes",
]

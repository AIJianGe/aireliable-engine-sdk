"""AIReliable Engine SDK。

引擎开发工具包，提供：
    - BaseEngine：8 端点契约的引擎抽象基类（**内部基类，引擎不应直接继承**）
    - WorkflowEngine：工作流引擎基类（**v0.2.5+ 唯一支持的引擎基类**）
    - BaseNode + PortSpec：节点抽象
    - NodeRegistry：节点注册表 + entry points 自动发现
    - WorkflowExecutor + workflow_dsl：工作流执行器 + 安全条件 DSL
    - schemas / SSEStream / StateMachine：8 端点契约配套
    - Images / Image / Audio / Video / Latent / Tensor：节点数据类型
    - 平台基础节点库（ImagesInput / TextInput / ModelLoader / ...）

v0.2.5 变更：
    - 移除 integrated / custom 范式，统一为 workflow 范式
    - 所有引擎必须继承 WorkflowEngine（直接继承 BaseEngine 触发 DeprecationWarning）
    - WorkflowEngine 子类只需实现 _build_workflow + _capabilities_impl 两个方法

v0.2.9 新增 engine_id 三同名契约（强制）：
    - engine_id 必须匹配正则 `^[a-z][a-z0-9_]{1,63}$`（小写字母+数字+下划线，禁止连字符）
    - 三同名契约：目录名 = engine_id = Python 模块名，三者必须字符完全相同
    - 详见 PR/development-guide.md §17.5 命名约定 + §四十 创建引擎前检测规范
    - 配套架构：PR/architecture.md §3.26 engine_id 三同名契约架构
    - 配套需求：PR/requirements.md §二十七 engine_id 命名规范需求

安装：
    pip install aireliable-engine-sdk            # 最小核心
    pip install "aireliable-engine-sdk[image]"    # 图片引擎开发
    pip install "aireliable-engine-sdk[llm]"      # LLM 引擎开发

使用：
    from aireliable_engine import WorkflowEngine, InferRequest, SSEStream

    class MyEngine(WorkflowEngine):
        async def _build_workflow(self, name=None) -> str:
            return open(self.workflows_dir / f"{name or 'default'}.yaml").read()

        async def _capabilities_impl(self) -> CapabilitiesResponse:
            return CapabilitiesResponse(...)
"""

from .engine_base import (
    BaseEngine,  # 内部基类，引擎应继承 WorkflowEngine 或 CustomEngine，不要直接继承 BaseEngine
    EngineBusyError,
    EngineInferError,
    EngineLoadError,
)
from .custom_engine import CustomEngine  # 全自定义引擎基类（integrated 范式）
from .schemas import (
    ENGINE_API_VERSION,
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
from .logging_config import (
    EngineLogFormatter,
    JSONFormatter,
    setup_engine_logging,
)
# v0.2.5+ T6.2：运行时日志级别动态调整
from .logging_config import (
    get_engine_log_level,
    register_log_level_endpoint,
    set_engine_log_level,
)
# v0.2.5+ T6.3：远程引擎日志回传
from .remote_log_reporter import (
    RemoteLogConfig,
    RemoteLogHandler,
    RemoteLogReporter,
    get_remote_reporter,
    stop_all_remote_reporters,
    stop_remote_reporter,
)
from .log_manager import (
    LogBuffer,
    LogEntry,
    LogManager,
    SSELogSubscriber,
    parse_engine_stdout_line,
)
# v0.2.5+ T6.1：日志链路追踪上下文
from .log_context import (
    LogContextSnapshot,
    clear_context,
    current_context,
    get_request_id,
    get_task_id,
    log_context,
    reset_request_id,
    reset_task_id,
    set_request_id,
    set_task_id,
)
from .downloader import (
    BaseEngineDownloader,
    DownloadResult,
    DownloadSpec,
    HFSnapshotDownloader,
    ProgressCallback,
)
# v0.2.7+ 通用下载器重构：BaseDownloader / HttpxDownloader / MirrorSelector
from .base_downloader import (
    BaseDownloader,
    DownloadOptions,
    DownloadProgress,
    DownloadResult as UniversalDownloadResult,
    ProgressCallback as UniversalProgressCallback,
)
from .httpx_downloader import HttpxDownloader
from .mirror_selector import (
    DEFAULT_HF_MIRRORS,
    DEFAULT_PYTORCH_MIRRORS,
    MirrorSelector,
    MirrorSource,
    get_hf_mirror_selector,
    get_pytorch_mirror_selector,
)
# v0.2.7+ 通用 HF 下载器：统一 core/sdk 的 huggingface_hub 下载重复代码
from .hf_hub_downloader import HfHubDownloader
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
# v0.3.3+ 跨引擎显存管理 SDK（引擎子进程 → 平台 core 状态上报）
from .vram_manager import (
    PlatformVramManager,
    close_platform_vram_manager,
    get_platform_vram_manager,
)
# v0.3.4+ 引擎任务上报 SDK（引擎子进程 → 平台 gateway 任务状态上报）
# 详见 TASK_CENTER_V034_ANALYSIS.md T5
from .task_reporter import (
    PlatformTaskReporter,
    close_platform_task_reporter,
    get_platform_task_reporter,
)
# v0.3.5+ 音色库客户端 SDK（引擎子进程 → gateway 音色库 API 调用）
# 详见 INDEXTTS2_SDK_VOICE_LIBRARY_ANALYSIS.md
# 用于 IndexTTS2 / VoxCPM 等 TTS 引擎 webUI 访问平台音色库 CRUD
from .voice_library_client import (
    VoiceLibraryClient,
    VoiceLibraryClientError,
    close_voice_library_client,
    get_voice_library_client,
)

__version__ = "0.3.0"

__all__ = [
    # ===== 引擎基类 =====
    "BaseEngine",
    "CustomEngine",
    "EngineBusyError",
    "EngineLoadError",
    "EngineInferError",
    # ===== 8 端点 schema =====
    "ENGINE_API_VERSION",
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
    "JSONFormatter",
    "setup_engine_logging",
    "LogEntry",
    "LogBuffer",
    "LogManager",
    "SSELogSubscriber",
    "parse_engine_stdout_line",
    # v0.2.5+ T6.2：运行时日志级别动态调整
    "set_engine_log_level",
    "get_engine_log_level",
    "register_log_level_endpoint",
    # v0.2.5+ T6.3：远程引擎日志回传
    "RemoteLogConfig",
    "RemoteLogHandler",
    "RemoteLogReporter",
    "get_remote_reporter",
    "stop_remote_reporter",
    "stop_all_remote_reporters",
    # v0.2.5+ T6.1：日志链路追踪上下文
    "LogContextSnapshot",
    "current_context",
    "get_task_id",
    "get_request_id",
    "log_context",
    "set_task_id",
    "reset_task_id",
    "set_request_id",
    "reset_request_id",
    "clear_context",
    # ===== 下载器 =====
    "BaseEngineDownloader",
    "DownloadSpec",
    "DownloadResult",
    "HFSnapshotDownloader",
    "ProgressCallback",
    # v0.2.7+ 通用下载器重构
    "BaseDownloader",
    "DownloadOptions",
    "DownloadProgress",
    "UniversalDownloadResult",
    "UniversalProgressCallback",
    "HttpxDownloader",
    "MirrorSelector",
    "MirrorSource",
    "DEFAULT_HF_MIRRORS",
    "DEFAULT_PYTORCH_MIRRORS",
    "get_hf_mirror_selector",
    "get_pytorch_mirror_selector",
    "HfHubDownloader",
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
    # ===== 跨引擎显存管理（v0.3.3+）=====
    "PlatformVramManager",
    "get_platform_vram_manager",
    "close_platform_vram_manager",
    # ===== 引擎任务上报（v0.3.4+）=====
    "PlatformTaskReporter",
    "get_platform_task_reporter",
    "close_platform_task_reporter",
    # ===== 音色库客户端（v0.3.5+）=====
    "VoiceLibraryClient",
    "VoiceLibraryClientError",
    "get_voice_library_client",
    "close_voice_library_client",
]

"""兼容层测试。

验证 ``from engines.base.xxx import ...`` 仍可用，且兼容层符号与 SDK 符号是同一对象。
"""

from __future__ import annotations


class TestCompatibilityLayerImports:
    """兼容层 re-export 测试。"""

    def test_engines_base_init_reexports(self):
        """``from engines.base import BaseEngine`` 可用。"""
        from engines.base import BaseEngine  # noqa: F401
        import aireliable_engine
        assert BaseEngine is aireliable_engine.BaseEngine

    def test_engine_base_reexports(self):
        """``from engines.base.engine_base import BaseEngine, EngineBusyError, ...`` 可用。"""
        from engines.base.engine_base import (
            BaseEngine,
            EngineBusyError,
            EngineInferError,
            EngineLoadError,
        )
        from aireliable_engine.engine_base import (
            BaseEngine as SdkBaseEngine,
            EngineBusyError as SdkEngineBusyError,
            EngineInferError as SdkEngineInferError,
            EngineLoadError as SdkEngineLoadError,
        )
        assert BaseEngine is SdkBaseEngine
        assert EngineBusyError is SdkEngineBusyError
        assert EngineInferError is SdkEngineInferError
        assert EngineLoadError is SdkEngineLoadError

    def test_schemas_reexports(self):
        """``from engines.base.schemas import ...`` 所有 schema 可用。"""
        from engines.base.schemas import (
            CapabilitiesResponse,
            ErrorResponse,
            HealthResponse,
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
        )
        from aireliable_engine.schemas import (
            CapabilitiesResponse as SdkCapabilitiesResponse,
            ErrorResponse as SdkErrorResponse,
            HealthResponse as SdkHealthResponse,
            InferRequest as SdkInferRequest,
            LoadRequest as SdkLoadRequest,
            LoadResponse as SdkLoadResponse,
            LoraLoadRequest as SdkLoraLoadRequest,
            LoraLoadResponse as SdkLoraLoadResponse,
            LoraSpec as SdkLoraSpec,
            LoraUnloadRequest as SdkLoraUnloadRequest,
            LoraUnloadResponse as SdkLoraUnloadResponse,
            StatusResponse as SdkStatusResponse,
            UnloadResponse as SdkUnloadResponse,
            VariantInfo as SdkVariantInfo,
        )
        assert CapabilitiesResponse is SdkCapabilitiesResponse
        assert ErrorResponse is SdkErrorResponse
        assert HealthResponse is SdkHealthResponse
        assert InferRequest is SdkInferRequest
        assert LoadRequest is SdkLoadRequest
        assert LoadResponse is SdkLoadResponse
        assert LoraLoadRequest is SdkLoraLoadRequest
        assert LoraLoadResponse is SdkLoraLoadResponse
        assert LoraSpec is SdkLoraSpec
        assert LoraUnloadRequest is SdkLoraUnloadRequest
        assert LoraUnloadResponse is SdkLoraUnloadResponse
        assert StatusResponse is SdkStatusResponse
        assert UnloadResponse is SdkUnloadResponse
        assert VariantInfo is SdkVariantInfo

    def test_sse_reexports(self):
        """``from engines.base.sse import SSEStream, make_frame, ...`` 可用。"""
        from engines.base.sse import (
            SSEStream,
            make_error_frame,
            make_final_frame,
            make_frame,
        )
        from aireliable_engine.sse import (
            SSEStream as SdkSSEStream,
            make_error_frame as SdkMakeErrorFrame,
            make_final_frame as SdkMakeFinalFrame,
            make_frame as SdkMakeFrame,
        )
        assert SSEStream is SdkSSEStream
        assert make_frame is SdkMakeFrame
        assert make_final_frame is SdkMakeFinalFrame
        assert make_error_frame is SdkMakeErrorFrame

    def test_state_machine_reexports(self):
        """``from engines.base.state_machine import EngineState, RuntimeState, StateMachine`` 可用。"""
        from engines.base.state_machine import (
            EngineState,
            RuntimeState,
            StateMachine,
        )
        from aireliable_engine.state_machine import (
            EngineState as SdkEngineState,
            RuntimeState as SdkRuntimeState,
            StateMachine as SdkStateMachine,
        )
        assert EngineState is SdkEngineState
        assert RuntimeState is SdkRuntimeState
        assert StateMachine is SdkStateMachine

    def test_logging_config_reexports(self):
        """``from engines.base.logging_config import EngineLogFormatter, setup_engine_logging`` 可用。"""
        from engines.base.logging_config import (
            EngineLogFormatter,
            setup_engine_logging,
        )
        from aireliable_engine.logging_config import (
            EngineLogFormatter as SdkEngineLogFormatter,
            setup_engine_logging as SdkSetupEngineLogging,
        )
        assert EngineLogFormatter is SdkEngineLogFormatter
        assert setup_engine_logging is SdkSetupEngineLogging

    def test_downloader_reexports(self):
        """``from engines.base.downloader import ...`` 可用。"""
        from engines.base.downloader import (
            BaseEngineDownloader,
            DownloadResult,
            DownloadSpec,
            HFSnapshotDownloader,
            ProgressCallback,
        )
        from aireliable_engine.downloader import (
            BaseEngineDownloader as SdkBaseEngineDownloader,
            DownloadResult as SdkDownloadResult,
            DownloadSpec as SdkDownloadSpec,
            HFSnapshotDownloader as SdkHFSnapshotDownloader,
            ProgressCallback as SdkProgressCallback,
        )
        assert BaseEngineDownloader is SdkBaseEngineDownloader
        assert DownloadResult is SdkDownloadResult
        assert DownloadSpec is SdkDownloadSpec
        assert HFSnapshotDownloader is SdkHFSnapshotDownloader
        assert ProgressCallback is SdkProgressCallback

    def test_subprocess_utils_reexports(self):
        """``from engines.base.subprocess_utils import ...`` 可用。"""
        from engines.base.subprocess_utils import (
            decode_subprocess_output,
            find_uv_executable,
            make_engine_subprocess_flags,
            make_uv_run_command,
        )
        from aireliable_engine.subprocess_utils import (
            decode_subprocess_output as SdkDecodeSubprocessOutput,
            find_uv_executable as SdkFindUvExecutable,
            make_engine_subprocess_flags as SdkMakeEngineSubprocessFlags,
            make_uv_run_command as SdkMakeUvRunCommand,
        )
        assert decode_subprocess_output is SdkDecodeSubprocessOutput
        assert find_uv_executable is SdkFindUvExecutable
        assert make_engine_subprocess_flags is SdkMakeEngineSubprocessFlags
        assert make_uv_run_command is SdkMakeUvRunCommand

    def test_engines_base_init_all_symbols(self):
        """``from engines.base import ...`` 所有公开符号可用且与 SDK 一致。"""
        # 一次性 import 所有兼容层 __all__ 中的符号
        import engines.base as compat
        import aireliable_engine as sdk

        # 关键符号 identity 校验
        for name in [
            "BaseEngine",
            "EngineBusyError",
            "EngineLoadError",
            "EngineInferError",
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
            "SSEStream",
            "make_frame",
            "make_final_frame",
            "make_error_frame",
            "EngineState",
            "RuntimeState",
            "StateMachine",
            "EngineLogFormatter",
            "setup_engine_logging",
            "BaseEngineDownloader",
            "DownloadSpec",
            "DownloadResult",
            "HFSnapshotDownloader",
            "ProgressCallback",
            "make_engine_subprocess_flags",
            "make_uv_run_command",
            "decode_subprocess_output",
            "find_uv_executable",
        ]:
            assert hasattr(compat, name), f"engines.base 缺少符号: {name}"
            assert hasattr(sdk, name), f"aireliable_engine 缺少符号: {name}"
            assert getattr(compat, name) is getattr(sdk, name), (
                f"符号 {name} 不一致：兼容层与 SDK 不是同一对象"
            )

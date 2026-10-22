"""引擎抽象基类。

所有引擎继承 BaseEngine，实现 8 个标准 HTTP 端点的业务逻辑。
server.py 的 FastAPI 路由直接调用本类的方法。

通用能力（状态管理、并发防护、SSE 流）由基类提供。

v0.2.5 变更：
    - 移除 integrated / custom 范式，统一为 workflow 范式
    - BaseEngine 的 7 个 ``_xxx_impl`` 方法从 ``@abstractmethod`` 改为默认
      ``raise NotImplementedError`` 实现
    - **不要直接继承 BaseEngine**，必须继承 WorkflowEngine
    - 直接继承 BaseEngine 会触发 ``DeprecationWarning``
    - WorkflowEngine 子类只需实现 ``_build_workflow`` + ``_capabilities_impl``
      两个方法，其余 6 个 ``_xxx_impl`` 继承 WorkflowEngine 默认实现
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from abc import ABC
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Optional

from .schemas import (
    ENGINE_API_VERSION,
    CapabilitiesResponse,
    ErrorResponse,
    HealthResponse,
    LoadRequest,
    LoadResponse,
    LoraLoadRequest,
    LoraLoadResponse,
    LoraUnloadRequest,
    LoraUnloadResponse,
    LoraSpec,
    StatusResponse,
    UnloadResponse,
)
from .sse import SSEStream, make_error_frame, make_frame
from .state_machine import RuntimeState, StateMachine

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """所有引擎的抽象基类。

    职责：
        1. 维护引擎运行态状态机
        2. 提供并发防护（同时只能加载/推理一个任务）
        3. 提供 SSE 流管理
        4. 定义 8 个端点的统一调用入口（子类实现具体逻辑）

    不负责：
        - HTTP 路由（由 server.py 处理）
        - 进程管理（由 EngineManager 处理）
        - UI 渲染（由 ui.py 处理）

    v0.2.5 警告：
        不要直接继承 BaseEngine，必须继承 WorkflowEngine。
        直接继承 BaseEngine 会触发 DeprecationWarning。
        7 个 ``_xxx_impl`` 方法有默认 ``raise NotImplementedError`` 实现，
        WorkflowEngine 已覆写其中 6 个（除 ``_capabilities_impl``）。
    """

    def __init_subclass__(cls, **kwargs):
        """子类创建时触发，检查是否直接继承 BaseEngine。

        v0.2.5+ 不允许直接继承 BaseEngine，必须继承 WorkflowEngine 或 CustomEngine。
        """
        super().__init_subclass__(**kwargs)
        # 排除 SDK 内部的 WorkflowEngine 和 CustomEngine
        # （它们直接继承 BaseEngine 是合法的）
        # 通过类名 + 模块名联合判断，避免误报
        if cls.__name__ in ("WorkflowEngine", "CustomEngine") and cls.__module__.startswith(
            "aireliable_engine"
        ):
            return
        # 检查父类链：若直接父类是 BaseEngine（而非 WorkflowEngine/CustomEngine），触发警告
        for base in cls.__bases__:
            if base is BaseEngine:
                warnings.warn(
                    f"v0.2.5+ 引擎 {cls.__name__} 直接继承 BaseEngine 已废弃，"
                    "必须继承 WorkflowEngine（工作流模式）或 CustomEngine（全自定义模式）。"
                    "直接继承 BaseEngine 会导致 7 个 _xxx_impl 方法抛 NotImplementedError。",
                    DeprecationWarning,
                    stacklevel=2,
                )
                break

    def __init__(self, engine_id: str, engine_name: str, capability: str, engine_type: str):
        self.engine_id: str = engine_id
        self.engine_name: str = engine_name
        self.capability: str = capability               # text-to-image / tts / chat ...
        self.engine_type: str = engine_type             # image / voice / video / threed / llm

        self.state_machine: StateMachine = StateMachine(initial=RuntimeState.IDLE)
        self._started_at: float = time.time()

        # 并发锁：load/unload/infer 互斥
        self._op_lock: asyncio.Lock = asyncio.Lock()

        # 输出根目录（子类在 __init__ 中设置，如 self.output_root = config.output_root）
        # 不输出文件的引擎（如纯流式 chat）可不设置，调用 get_output_path 时会抛错
        self.output_root: Optional[Path] = None

        # 默认 variant 名（由子类从 manifest.models.default_variant 读取后赋值）。
        # BaseEngine.infer 的 auto-load 优先级：
        #   request.variant > self.default_variant > 第一个 available variant
        # 引擎若不设置，则为 None，auto-load 回退到"第一个 available variant"。
        self.default_variant: Optional[str] = None

    # ============ default_variant 读取 ============

    def _read_default_variant_from_manifest(self, manifest_path: Path) -> Optional[str]:
        """从指定 manifest.yaml 读取 models.default_variant 字段（通用方法）。

        Args:
            manifest_path: manifest.yaml 的绝对路径（通常由子类的 __file__ 推导）

        Returns:
            default_variant 字符串，或 None（未声明 / 文件不存在 / 读取失败）

        子类在 __init__ 中调用：
            from pathlib import Path
            manifest_path = Path(__file__).parent / "manifest.yaml"
            self.default_variant = self._read_default_variant_from_manifest(manifest_path)
        """
        try:
            import yaml
            if not manifest_path.exists():
                return None
            with open(manifest_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            models_block = data.get("models", {}) or {}
            return models_block.get("default_variant")
        except Exception as e:
            logger.warning("读取 manifest default_variant 失败: %s", e)
            return None

    # ============ 输出文件管理 ============

    def get_output_path(self, category: str, filename: str) -> Path:
        """按类型构造输出文件路径。

        统一输出目录结构：``{output_root}/{category}/{filename}``

        Args:
            category: 输出类别（image / video / audio / voice / text / 3d）
            filename: 文件名（如 ``{task_id}.png``）

        Returns:
            完整路径（目录已自动创建）

        Raises:
            AttributeError: 子类未设置 ``output_root``

        规范详见 PR/architecture.md §3.7 输出文件存储规范。
        """
        if self.output_root is None:
            raise AttributeError(
                f"引擎 {self.engine_id} 未设置 output_root，"
                "请在子类 __init__ 中设置（如 self.output_root = config.output_root），"
                "或重写 get_output_path 方法"
            )
        p = self.output_root / category
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    # ============ 8 个 HTTP 端点的统一入口 ============

    async def health(self) -> HealthResponse:
        """GET /health。"""
        return HealthResponse(
            status="ok",
            engine_id=self.engine_id,
            state=self.state_machine.state.value,
            uptime=self.state_machine.uptime,
            engine_api_version=ENGINE_API_VERSION,
        )

    async def capabilities(self) -> CapabilitiesResponse:
        """GET /capabilities。"""
        return await self._capabilities_impl()

    async def load(self, request: LoadRequest) -> LoadResponse:
        """POST /load。

        包含并发防护：同一时间只允许一个 load/unload 操作。

        v0.3.7+ 修复：READY 状态时先 unload 再 load。
        场景：反代路由 auto-load 后状态机为 READY，gateway 再调 /load
        加载特定 variant，原来直接 transition(LOADING) 会因
        READY → LOADING 非法转换失败。
        """
        async with self._op_lock:
            if self.state_machine.is_busy():
                raise EngineBusyError(
                    f"引擎忙（state={self.state_machine.state.value}），请稍后再试"
                )
            if self.state_machine.state == RuntimeState.INFERRING:
                raise EngineBusyError("推理进行中，无法加载")

            # v0.3.7+ 修复：READY/ERROR 状态时先清理再 load
            current = self.state_machine.state
            if current == RuntimeState.READY:
                logger.info(
                    "load()：当前状态 READY，先 unload 释放资源再加载 variant=%s",
                    request.variant,
                )
                try:
                    await self._unload_impl()
                    self.state_machine.transition(RuntimeState.IDLE)
                except Exception as e:
                    logger.warning("load()：unload 失败，强制 reset 状态机: %s", e)
                    try:
                        self.state_machine.transition(RuntimeState.ERROR)
                        self.state_machine.reset()
                    except Exception:
                        pass
            elif current == RuntimeState.ERROR:
                logger.info("load()：当前状态 ERROR，reset 到 IDLE 再加载")
                try:
                    self.state_machine.reset()
                except Exception as e:
                    logger.warning("load()：reset 失败: %s", e)

            self.state_machine.transition(RuntimeState.LOADING)
            try:
                response = await self._load_impl(request.variant)
                self.state_machine.transition(RuntimeState.READY)
                return response
            except Exception as e:
                logger.exception("加载失败: variant=%s", request.variant)
                self.state_machine.transition(RuntimeState.ERROR)
                raise

    async def unload(self) -> UnloadResponse:
        """POST /unload。"""
        async with self._op_lock:
            if self.state_machine.state == RuntimeState.INFERRING:
                raise EngineBusyError("推理进行中，无法卸载")

            try:
                response = await self._unload_impl()
                # 从 READY/LOADING 回到 IDLE
                if self.state_machine.state in (RuntimeState.READY, RuntimeState.LOADING):
                    self.state_machine.transition(RuntimeState.IDLE)
                return response
            except Exception as e:
                logger.exception("卸载失败")
                self.state_machine.transition(RuntimeState.ERROR)
                raise

    async def infer(self, request: Any) -> AsyncIterator[str]:
        """POST /infer。

        返回 SSE 流（异步迭代器，FastAPI StreamingResponse 直接消费）。
        """
        # 创建 SSE 流（绑定当前事件循环）
        loop = asyncio.get_event_loop()
        stream = SSEStream(loop=loop)

        async def _run_infer():
            """后台执行推理，把进度帧推到 stream。"""
            async with self._op_lock:
                if self.state_machine.state == RuntimeState.INFERRING:
                    stream.put_threadsafe(
                        make_error_frame("引擎正在推理中，请稍后", "engine_busy")
                    )
                    stream.close()
                    return

                if self.state_machine.state == RuntimeState.LOADING:
                    stream.put_threadsafe(
                        make_error_frame("模型正在加载中，请稍候", "model_loading")
                    )
                    stream.close()
                    return

                try:
                    # ERROR 自愈（任务 50 实测踩中）：上一次失败会把状态机锁在 ERROR，
                    # 而 ERROR 只允许转 IDLE（ERROR→INFERRING/LOADING 均非法）→ 后续每次
                    # infer 都在 transition(INFERRING) 抛"非法状态转换"，引擎彻底无法恢复
                    # （用户观感："点生成没反应，只闪一下"）。这里先 reset 回 IDLE，
                    # 再走下面的 auto-load 分支。
                    if self.state_machine.state == RuntimeState.ERROR:
                        logger.warning("infer：当前状态 ERROR → reset 回 IDLE 后重试（自愈）")
                        self.state_machine.reset()

                    # 若模型未加载，先自动加载
                    if self.state_machine.state == RuntimeState.IDLE:
                        # auto-load 优先级：
                        #   request.variant > self.default_variant > None
                        # None 交给子类 _load_impl 决定（如 router mode 只启动子进程，
                        # 或子类自行回退到第一个 available variant）
                        variant_to_load = request.variant or self.default_variant
                        logger.info(
                            "auto-load 触发: request.variant=%s, default_variant=%s, 选用=%s",
                            request.variant, self.default_variant, variant_to_load,
                        )
                        # 推送加载提示帧：大模型冷加载可达数分钟（如 FLUX Q8 ~5min），
                        # 期间 GPU/显存可能无变化（T5 等组件在 CPU 反量化），
                        # 不推帧会让 WebUI 长时间停留在"连接引擎…"，用户以为卡死。
                        stream.put_threadsafe(make_frame(
                            0.0, "model_loading",
                            detail=(
                                f"模型首次加载中（{variant_to_load or '默认档位'}，"
                                "大模型约需 3-5 分钟），期间 GPU/显存可能暂无变化，请耐心等待"
                            ),
                        ))
                        # 在锁内调用 load（不通过公开 load()，避免重复加锁）
                        self.state_machine.transition(RuntimeState.LOADING)
                        await self._load_impl(variant_to_load)
                        self.state_machine.transition(RuntimeState.READY)
                        stream.put_threadsafe(make_frame(
                            0.02, "model_loading",
                            detail="模型加载完成，开始推理",
                        ))

                    # 进入推理态
                    self.state_machine.transition(RuntimeState.INFERRING)
                    try:
                        await self._infer_impl(request, stream)
                    finally:
                        # 推理结束回 READY（保留模型）
                        if self.state_machine.state == RuntimeState.INFERRING:
                            self.state_machine.transition(RuntimeState.READY)
                except Exception as e:
                    logger.exception("推理失败")
                    stream.put_threadsafe(make_error_frame(str(e)))
                    self.state_machine.transition(RuntimeState.ERROR)
                finally:
                    stream.close()

        # v0.4.x+（任务 58 走网络实测）修复：把 busy 判定前移到**加锁之前**（fail-fast）。
        # 根因：SDK 的 load/unload/infer 都是"先获取 _op_lock 再检查状态"，故第一个操作
        # 全程持锁 → 其后到达的并发请求只能**挂起排队**（实测第二 /infer 挂到客户端读超时、
        # 第二 /unload 挂 90s+），锁内的 busy 分支因永远在没人持锁时才执行而**不可达**（死代码）。
        # 这里在顶层（create_task 之前）先判一次状态，命中即立刻推错误帧并关流，第二个
        # 并发 /infer 立即拿到 engine_busy / model_loading 帧，不再排队等待。
        # 锁内保留同源判定作为 TOCTOU 第二道防线（两次判定之间状态仍可能变化）。
        early_state = self.state_machine.state
        if early_state == RuntimeState.INFERRING:
            stream.put_threadsafe(
                make_error_frame("引擎正在推理中，请稍后", "engine_busy")
            )
            stream.close()
        elif early_state == RuntimeState.LOADING:
            stream.put_threadsafe(
                make_error_frame("模型正在加载中，请稍候", "model_loading")
            )
            stream.close()
        else:
            # 启动后台推理任务
            asyncio.create_task(_run_infer())

        # 返回 SSE 流
        async for frame in stream.events():
            yield frame

    async def status(self) -> StatusResponse:
        """GET /status。"""
        return await self._status_impl()

    async def load_lora(self, request: LoraLoadRequest) -> LoraLoadResponse:
        """POST /lora/load。"""
        return await self._load_lora_impl(request.name, request.weight)

    async def unload_lora(self, request: LoraUnloadRequest) -> LoraUnloadResponse:
        """POST /lora/unload。"""
        return await self._unload_lora_impl(request.name)

    # ============ 子类应实现的业务逻辑（v0.2.5+ 改为默认 NotImplementedError）============
    #
    # v0.2.5+ 不再要求子类必须实现这 7 个方法：
    #   - WorkflowEngine 已覆写其中 6 个（_load_impl / _unload_impl / _infer_impl /
    #     _status_impl / _load_lora_impl / _unload_lora_impl）
    #   - 唯一仍需子类实现的是 _capabilities_impl（WorkflowEngine 也未覆写）
    # 直接继承 BaseEngine（非 WorkflowEngine）会触发 DeprecationWarning，
    # 且这 7 个方法的默认实现会抛 NotImplementedError。

    async def _capabilities_impl(self) -> CapabilitiesResponse:
        """构造能力声明。

        v0.2.5+：WorkflowEngine 也未覆写此方法，子类必须实现。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _capabilities_impl()。"
            "v0.2.5+ 必须继承 WorkflowEngine 并实现此方法。"
        )

    async def _load_impl(self, variant: Optional[str]) -> LoadResponse:
        """实际加载逻辑。

        Args:
            variant: 要加载的 variant 名。可为 None，表示由子类决定：
                - 优先用 manifest 声明的 default_variant
                - router mode 引擎可只启动子进程不预加载
                - 其他引擎可回退到第一个 available variant

        v0.2.5+：WorkflowEngine 已提供默认实现（基于工作流节点图加载）。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _load_impl(variant)。"
            "v0.2.5+ 必须继承 WorkflowEngine 获取默认实现。"
        )

    async def _unload_impl(self) -> UnloadResponse:
        """实际卸载逻辑。

        v0.2.5+：WorkflowEngine 已提供默认实现。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _unload_impl()。"
            "v0.2.5+ 必须继承 WorkflowEngine 获取默认实现。"
        )

    async def _infer_impl(self, request: Any, stream: SSEStream) -> None:
        """实际推理逻辑。

        通过 stream.put_threadsafe / await stream.put 推送进度帧。

        v0.2.5+：WorkflowEngine 已提供默认实现（基于 WorkflowExecutor 执行节点图）。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _infer_impl(request, stream)。"
            "v0.2.5+ 必须继承 WorkflowEngine 获取默认实现。"
        )

    async def _status_impl(self) -> StatusResponse:
        """构造状态响应。

        v0.2.5+：WorkflowEngine 已提供默认实现。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _status_impl()。"
            "v0.2.5+ 必须继承 WorkflowEngine 获取默认实现。"
        )

    async def _load_lora_impl(self, name: str, weight: float) -> LoraLoadResponse:
        """实际 LoRA 加载逻辑。

        v0.2.5+：WorkflowEngine 已提供默认实现（记录到 loaded_loras 列表）。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _load_lora_impl(name, weight)。"
            "v0.2.5+ 必须继承 WorkflowEngine 获取默认实现。"
        )

    async def _unload_lora_impl(self, name: str) -> LoraUnloadResponse:
        """实际 LoRA 卸载逻辑。

        v0.2.5+：WorkflowEngine 已提供默认实现。
        """
        raise NotImplementedError(
            f"引擎 {self.__class__.__name__} 未实现 _unload_lora_impl(name)。"
            "v0.2.5+ 必须继承 WorkflowEngine 获取默认实现。"
        )

    # ============ v0.2.11+ 方案 B：variant 检测 API ============

    @staticmethod
    def _get_platform_core_url() -> Optional[str]:
        """从环境变量读取平台 core 服务 URL（v0.2.11+ 方案 B 新增）。

        引擎子进程启动时，平台会注入 `AIRELIABLE_CORE_URL` 环境变量
        （如 "http://127.0.0.1:8001"）。引擎可调用平台 API 检测 variant 可用性。

        Returns:
            core URL 字符串（无尾斜杠）；环境变量未设置时返回 None
        """
        import os
        return os.environ.get("AIRELIABLE_CORE_URL")

    async def get_variants(self) -> Optional[list[dict]]:
        """调用平台 API 获取引擎 variant 可用性（v0.2.11+ 方案 B 新增）。

        引擎 model_scanner 可调用此方法替代自己扫描文件系统。
        平台从 manifest.models.variants[] 读取 variant 声明，并检测关联文件。

        v0.2.16+（方案 C1）有界重试：core 重启窗口期的 5xx / 超时 / 连接错误
        按退避间隔（2s / 5s / 10s）重试，共最多 3 次尝试；4xx 客户端错误
        （如 404 引擎未注册）不重试直接返回 None。

        Returns:
            variant 列表（每个元素含 name/alias/base_model/precision/loader/
            vram_required/available/model_file_ids/missing_file_ids）；
            平台不可达时返回 None（调用方应 fallback 到本地扫描）

        规范详见 PR/architecture.md §3.29 模型检测架构统一（方案 B）。
        """
        core_url = self._get_platform_core_url()
        if not core_url:
            return None

        import asyncio

        # v0.2.16+ 方案 C1：退避间隔（2s / 5s / 10s），最多 3 次尝试
        retry_backoffs = (2.0, 5.0, 10.0)

        url = f"{core_url.rstrip('/')}/engines/{self.engine_id}/variants"
        last_reason = ""
        for attempt in range(len(retry_backoffs) + 1):
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:  # 平台内部回环，不走系统代理
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("variants")
                    # 4xx 客户端错误：平台在线但请求本身有问题，重试无益
                    if 400 <= resp.status_code < 500:
                        logger.warning(
                            "get_variants 平台返回 %d（客户端错误，不重试）: %s",
                            resp.status_code, url,
                        )
                        return None
                    # 5xx：core 启动中间态 / 网关重启窗口，可重试
                    last_reason = f"HTTP {resp.status_code}"
            except Exception as e:
                # 超时 / 连接错误：core 尚未监听端口，可重试
                last_reason = f"{type(e).__name__}: {e}"

            if attempt < len(retry_backoffs):
                delay = retry_backoffs[attempt]
                logger.warning(
                    "get_variants 第 %d 次尝试失败（%s），%.0fs 后重试（%d/%d）: %s",
                    attempt + 1, last_reason, delay,
                    attempt + 1, len(retry_backoffs), url,
                )
                await asyncio.sleep(delay)

        logger.warning(
            "get_variants 全部 %d 次尝试失败（%s），fallback 到本地扫描: %s",
            len(retry_backoffs) + 1, last_reason, url,
        )
        return None


class EngineBusyError(Exception):
    """引擎忙异常。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.error_type = "engine_busy"
        self.message = message


class EngineLoadError(Exception):
    """引擎加载失败异常。"""

    def __init__(self, message: str, error_type: str = "load_failed"):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class EngineInferError(Exception):
    """推理失败异常。"""

    def __init__(self, message: str, error_type: str = "infer_failed"):
        super().__init__(message)
        self.error_type = error_type
        self.message = message

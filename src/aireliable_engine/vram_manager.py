"""跨引擎显存管理 SDK（v0.3.3+ 新增）。

本模块供**引擎子进程**与**平台 core** 通信，报告引擎内部模型加载/卸载状态，
让平台 core 维护全局显存视图，用于跨引擎 LRU 决策。

与 `workflow_context.VRamManager` 的区别：
    - `VRamManager`：进程内模型 offload（CPU ↔ GPU），单引擎内部使用
    - `PlatformVramManager`：跨引擎显存状态上报，引擎子进程 → 平台 core

通信方式：HTTP（引擎子进程 → 平台 core REST API）
    - POST /engines/{id}/track_loaded   — 报告加载了模型
    - POST /engines/{id}/track_unloaded — 报告卸载了模型
    - GET  /engines/{id}/vram_used      — 查询显存占用
    - GET  /system/info                 — 查询系统显存（GPU 总/可用）

环境变量依赖：
    - AIRELIABLE_CORE_URL：平台 core 地址（如 http://127.0.0.1:8001）
      平台启动引擎子进程时由 engine_manager._build_engine_subprocess_env 注入
    - AIRELIABLE_ENGINE_ID：当前引擎 ID
      （可选，若未注入则需在构造函数显式传入）

使用示例（llama_cpp 反代层）：

    from aireliable_engine import PlatformVramManager

    vram_mgr = PlatformVramManager()  # 自动从环境变量读 core_url 和 engine_id

    # 加载模型后通知平台
    await vram_mgr.report_loaded(
        variant="qwen3.6-35b-a3b-q6-k",
        vram_used_bytes=30 * 1024 ** 3,
    )

    # 卸载模型后通知平台
    await vram_mgr.report_unloaded()

    # 查询当前可用显存
    free_gb = await vram_mgr.get_free_vram_gb()
    if free_gb is not None and free_gb < required_gb:
        # 显存不足，请求平台 LRU 卸载其他引擎
        await vram_mgr.request_vram(required_bytes=int(required_gb * 1024 ** 3))

详见：VRAM_MANAGEMENT_V033_ANALYSIS.md T5
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class PlatformVramManager:
    """跨引擎显存管理器（v0.3.3+ 新增）。

    引擎子进程通过本类向平台 core 报告显存状态变化，让平台维护全局显存视图。

    设计原则：
        1. **容错**：平台 core 不可达时所有方法降级（返回 None / False），不阻断引擎
        2. **异步**：所有 HTTP 调用都是 async，不阻塞引擎主流程
        3. **幂等**：重复调用 report_loaded/report_unloaded 安全
        4. **轻量**：只在模型加载/卸载时调用，不频繁轮询

    Attributes:
        core_url: 平台 core 地址（如 http://127.0.0.1:8001）
        engine_id: 当前引擎 ID
        client: httpx 异步客户端（复用连接池）
    """

    def __init__(
        self,
        core_url: Optional[str] = None,
        engine_id: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        """初始化 PlatformVramManager。

        Args:
            core_url: 平台 core 地址；None 时从 AIRELIABLE_CORE_URL 环境变量读取
            engine_id: 引擎 ID；None 时从 AIRELIABLE_ENGINE_ID 环境变量读取
            timeout: HTTP 请求超时秒数（默认 5s，容错优先）
        """
        self.core_url: str = (
            core_url
            or os.environ.get("AIRELIABLE_CORE_URL", "")
            or "http://127.0.0.1:8001"
        ).rstrip("/")
        self.engine_id: str = engine_id or os.environ.get("AIRELIABLE_ENGINE_ID", "")
        self._timeout: float = timeout
        self._client: Optional[httpx.AsyncClient] = None

        if not self.engine_id:
            logger.warning(
                "PlatformVramManager: engine_id 未设置（AIRELIABLE_ENGINE_ID 环境变量未注入），"
                "report_loaded/report_unloaded 将无法调用"
            )
        if not self.core_url:
            logger.warning("PlatformVramManager: core_url 未设置")

        logger.debug(
            "PlatformVramManager 初始化: core_url=%s engine_id=%s",
            self.core_url, self.engine_id,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """获取复用的 httpx 客户端（惰性创建）。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.core_url,
                timeout=httpx.Timeout(self._timeout, connect=2.0),
                # trust_env=False：引擎→core 平台内部回环，不走系统代理
                trust_env=False,
            )
        return self._client

    async def close(self) -> None:
        """关闭客户端（引擎停止时调用）。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ===== 状态上报 =====

    async def report_loaded(
        self,
        variant: str,
        vram_used_bytes: Optional[int] = None,
        vram_used_gb: Optional[float] = None,
        ctx_size: Optional[int] = None,
    ) -> bool:
        """通知平台：引擎加载了模型（v0.3.3+ 新增）。

        v0.3.6+ 新增 ctx_size：上报模型实际加载的上下文长度（运行时 n_ctx），
        供前端基本信息 / 运行参数 Tab 显示。

        Args:
            variant: 加载的 variant 名（如 "qwen3.6-35b-a3b-q6-k"）
            vram_used_bytes: 实际显存占用字节（与 vram_used_gb 二选一）
            vram_used_gb: 实际显存占用 GB（与 vram_used_bytes 二选一）
            ctx_size: 实际加载的上下文长度（运行时 n_ctx，可选）

        Returns:
            bool: True 表示平台已记录；False 表示失败（core 不可达 / engine_id 未设置）
        """
        if not self.engine_id:
            logger.debug("report_loaded 跳过: engine_id 未设置")
            return False

        payload: dict[str, Any] = {"variant": variant}
        if vram_used_bytes is not None:
            payload["vram_used_bytes"] = int(vram_used_bytes)
        elif vram_used_gb is not None:
            payload["vram_used_gb"] = float(vram_used_gb)
        if ctx_size is not None and ctx_size > 0:
            payload["ctx_size"] = int(ctx_size)

        try:
            client = await self._get_client()
            resp = await client.post(
                f"/engines/{self.engine_id}/track_loaded",
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    "report_loaded 成功: engine=%s variant=%s vram_gb=%s",
                    self.engine_id, variant,
                    data.get("vram_used_gb", "?"),
                )
                return True
            logger.warning(
                "report_loaded 失败: status=%d body=%s",
                resp.status_code, resp.text[:200],
            )
            return False
        except Exception as e:
            logger.warning("report_loaded 异常（不阻断引擎）: %s", e)
            return False

    async def report_unloaded(self) -> bool:
        """通知平台：引擎卸载了模型（v0.3.3+ 新增）。

        Returns:
            bool: True 表示平台已记录；False 表示失败
        """
        if not self.engine_id:
            logger.debug("report_unloaded 跳过: engine_id 未设置")
            return False

        try:
            client = await self._get_client()
            resp = await client.post(
                f"/engines/{self.engine_id}/track_unloaded",
            )
            if resp.status_code == 200:
                logger.info("report_unloaded 成功: engine=%s", self.engine_id)
                return True
            logger.warning(
                "report_unloaded 失败: status=%d body=%s",
                resp.status_code, resp.text[:200],
            )
            return False
        except Exception as e:
            logger.warning("report_unloaded 异常（不阻断引擎）: %s", e)
            return False

    async def report_planned_ctx(self, planned_ctx: dict) -> bool:
        """通知平台：引擎启动后各 variant 的生效 ctx（v0.3.6+ 新增）。

        引擎生成 models-preset INI 后调用，上报每个 variant 的"计划 ctx"
        （override > 显存自适应 > manifest.recommended），供前端运行参数
        Tab 显示"引擎推荐（按显存自适应）"默认值。

        Args:
            planned_ctx: {variant_name: ctx_size, ...}（如 {"qwen3.6-35b-a3b-q6-k": 38441}）

        Returns:
            bool: True 表示平台已记录；False 表示失败（core 不可达 / engine_id 未设置）
        """
        if not self.engine_id:
            logger.debug("report_planned_ctx 跳过: engine_id 未设置")
            return False
        if not planned_ctx:
            return False

        try:
            client = await self._get_client()
            resp = await client.post(
                f"/engines/{self.engine_id}/runtime/planned_ctx",
                json={"planned_ctx": planned_ctx},
            )
            if resp.status_code == 200:
                logger.info(
                    "report_planned_ctx 成功: engine=%s variants=%d",
                    self.engine_id, len(planned_ctx),
                )
                return True
            logger.warning(
                "report_planned_ctx 失败: status=%d body=%s",
                resp.status_code, resp.text[:200],
            )
            return False
        except Exception as e:
            logger.warning("report_planned_ctx 异常（不阻断引擎）: %s", e)
            return False

    # ===== 显存查询 =====

    async def check_vram(self, required_bytes: int) -> bool:
        """查询当前可用显存是否足够（v0.3.3+ 新增）。

        Args:
            required_bytes: 需要的显存字节数

        Returns:
            bool: True 表示显存足够；False 表示不足或查询失败（保守返回 False）
        """
        free_bytes = await self.get_free_vram_bytes()
        if free_bytes is None:
            # 查询失败，保守返回 False（让调用方决策）
            logger.debug("check_vram: 无法查询可用显存，保守返回 False")
            return False
        sufficient = free_bytes >= required_bytes
        logger.debug(
            "check_vram: required=%.2fGB free=%.2fGB sufficient=%s",
            required_bytes / (1024 ** 3),
            free_bytes / (1024 ** 3),
            sufficient,
        )
        return sufficient

    async def get_free_vram_gb(self) -> Optional[float]:
        """查询 GPU 当前可用显存（GB）（v0.3.3+ 新增）。

        通过平台 core 的 /system/info 端点查询。

        Returns:
            float | None: 可用显存 GB；无 GPU 或查询失败返回 None
        """
        try:
            client = await self._get_client()
            resp = await client.get("/system/info")
            if resp.status_code != 200:
                logger.warning("get_free_vram_gb: /system/info 返回 %d", resp.status_code)
                return None
            data = resp.json()
            if not data.get("gpu_detected", False):
                return None
            free_gb = data.get("gpu_free_vram_gb", 0)
            return float(free_gb) if free_gb is not None else None
        except Exception as e:
            logger.warning("get_free_vram_gb 异常: %s", e)
            return None

    async def get_free_vram_bytes(self) -> Optional[int]:
        """查询 GPU 当前可用显存（字节）（v0.3.3+ 新增）。

        Returns:
            int | None: 可用显存字节；无 GPU 或查询失败返回 None
        """
        free_gb = await self.get_free_vram_gb()
        if free_gb is None:
            return None
        return int(free_gb * (1024 ** 3))

    async def get_total_vram_gb(self) -> Optional[float]:
        """查询 GPU 总显存（GB）（v0.3.3+ 新增）。

        Returns:
            float | None: 总显存 GB；无 GPU 或查询失败返回 None
        """
        try:
            client = await self._get_client()
            resp = await client.get("/system/info")
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data.get("gpu_detected", False):
                return None
            total_gb = data.get("gpu_vram_gb", 0)
            return float(total_gb) if total_gb is not None else None
        except Exception as e:
            logger.warning("get_total_vram_gb 异常: %s", e)
            return None

    async def get_current_variant(self) -> Optional[str]:
        """查询平台记录的当前引擎加载的 variant（v0.3.3+ 新增）。

        Returns:
            str | None: variant 名；未加载返回空字符串；查询失败返回 None
        """
        if not self.engine_id:
            return None
        try:
            client = await self._get_client()
            resp = await client.get(f"/engines/{self.engine_id}/vram_used")
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("variant", "")
        except Exception as e:
            logger.warning("get_current_variant 异常: %s", e)
            return None

    async def get_current_vram_used_gb(self) -> Optional[float]:
        """查询平台记录的当前引擎显存占用（GB）（v0.3.3+ 新增）。

        Returns:
            float | None: 显存占用 GB；查询失败返回 None
        """
        if not self.engine_id:
            return None
        try:
            client = await self._get_client()
            resp = await client.get(f"/engines/{self.engine_id}/vram_used")
            if resp.status_code != 200:
                return None
            data = resp.json()
            return float(data.get("vram_used_gb", 0))
        except Exception as e:
            logger.warning("get_current_vram_used_gb 异常: %s", e)
            return None

    # ===== 显存请求（触发平台 LRU） =====

    async def request_vram(self, required_bytes: int) -> bool:
        """请求平台释放显存（触发 LRU 卸载其他引擎）（v0.3.3+ 新增）。

        注意：本方法目前通过平台的 stop_engine 间接实现（平台 core 在 start_engine
        时自动 LRU）。未来可扩展专门的 /vram/request 端点。

        当前实现：检查显存是否足够，不够则返回 False（让调用方决策是否
        停止其他引擎或等待）。真正的 LRU 自动停止由平台 core 在
        _ensure_vram_capacity 中实现（T3）。

        Args:
            required_bytes: 需要的显存字节数

        Returns:
            bool: True 表示显存已足够；False 表示仍不足
        """
        free_bytes = await self.get_free_vram_bytes()
        if free_bytes is None:
            logger.warning("request_vram: 无法查询可用显存")
            return False
        if free_bytes >= required_bytes:
            return True
        # 显存不足，平台 core 会在 start_engine 时自动 LRU
        # 本方法暂不主动触发 LRU（避免引擎子进程越权停止其他引擎）
        logger.warning(
            "request_vram: 显存不足（free=%.2fGB < required=%.2fGB），"
            "请通过平台 UI 启动引擎触发自动 LRU",
            free_bytes / (1024 ** 3),
            required_bytes / (1024 ** 3),
        )
        return False


# 模块级单例（惰性创建，引擎子进程可直接 import 使用）
_singleton: Optional[PlatformVramManager] = None
_singleton_lock = asyncio.Lock()


async def get_platform_vram_manager() -> PlatformVramManager:
    """获取模块级单例 PlatformVramManager（v0.3.3+ 新增）。

    引擎子进程可直接调用：

        from aireliable_engine import get_platform_vram_manager
        vram_mgr = await get_platform_vram_manager()
        await vram_mgr.report_loaded(variant="xxx", vram_used_bytes=...)

    Returns:
        PlatformVramManager: 单例实例
    """
    global _singleton
    if _singleton is None:
        async with _singleton_lock:
            if _singleton is None:
                _singleton = PlatformVramManager()
    return _singleton


async def close_platform_vram_manager() -> None:
    """关闭模块级单例（引擎停止时调用）。"""
    global _singleton
    if _singleton is not None:
        await _singleton.close()
        _singleton = None

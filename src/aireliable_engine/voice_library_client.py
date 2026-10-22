"""音色库客户端 SDK（v0.3.5+ 新增）。

本模块供**引擎子进程**（如 IndexTTS2 / VoxCPM 的 Gradio UI）与**平台 gateway**
通信，访问平台音色库的 6 个 CRUD 端点。

与 `task_reporter.PlatformTaskReporter` 的区别：
    - `PlatformTaskReporter`：引擎 → gateway，上报任务状态（POST /v1/tasks/internal*）
    - `VoiceLibraryClient`：引擎 → gateway，访问音色库 CRUD（GET/POST/DELETE /v1/voices*）

通信方式：HTTP（引擎子进程 → gateway REST API）
    - GET    /v1/voices                       — 列出全部音色
    - GET    /v1/voices/{id}                   — 查询单个音色
    - GET    /v1/voices/{id}/audio             — 获取音色音频文件 URL
    - GET    /v1/voices/by-name/{name}         — 按名称查询音色
    - POST   /v1/voices                        — 创建/更新音色（multipart）
    - DELETE /v1/voices/{id}                    — 删除音色

环境变量依赖：
    - AIRELIABLE_GATEWAY_URL：平台 gateway 地址（如 http://127.0.0.1:8000）
      平台启动引擎子进程时由 engine_manager._build_subprocess_env 注入
      详见 project-rules.md §第三部分 §1

容错原则：
    - gateway 不可达时所有查询方法降级（返回空列表 / None / False），不阻断引擎
    - 写入方法（create / delete）失败时抛 VoiceLibraryClientError，UI 提示用户
    - 所有调用记录 DEBUG 日志（URL、响应码、耗时），失败时 WARNING 日志

使用示例（Gradio UI 同步调用）：

    from aireliable_engine import VoiceLibraryClient

    client = VoiceLibraryClient()  # 自动从 AIRELIABLE_GATEWAY_URL 环境变量读取
    voices = client.list_voices_sync()
    if voices:
        for v in voices:
            print(v["id"], v["name"])

    # 上传音色
    result = client.create_voice_sync(
        audio_path="/path/to/ref.wav",
        name="主播小王",
        description="女声 30 岁",
    )

    # 删除音色
    ok = client.delete_voice_sync("voice_a1b2c3d4")

使用示例（异步引擎推理代码）：

    from aireliable_engine import get_voice_library_client

    client = await get_voice_library_client()
    voices = await client.list_voices()

设计依据：
    - PR/architecture.md §3.18.8：UI 音色库浏览推荐用 SDK 调用
    - project-rules.md §第三部分 §1：环境变量注入规范
    - 参考 task_reporter.py 的单例 + 容错 + 异步模式

Card 数据结构（v0.4.3+ 富化）
================================

所有查询方法（``list_voices`` / ``get_voice`` / ``get_voice_by_name`` / ``create_voice``）
返回的音色卡片 dict 都包含以下两个路径字段，调用方按需取用：

    card = client.get_voice_sync(voice_id)
    local_path = card["path"]         # 本地绝对路径（D:\\...\\data\\voices\\xxx.wav）
    audio_url  = card["audio_url"]    # 网络 URL（http://127.0.0.1:8000/v1/voices/voice_xxx/audio）

    # 场景 1：Gradio UI 选音色（必须用 path）
    gr.Audio(type="filepath", value=local_path)

    # 场景 2：浏览器直连播放（必须用 audio_url）
    # <audio src={audio_url} />

    # 场景 3：引擎推理（用 path）
    # librosa.load(local_path)

字段来源：
    - ``path``：来自 gateway VoiceCardResponse（已有，SDK 透传）
    - ``audio_url``：SDK 内部根据 ``voice_id`` + ``gateway_url`` 构造（v0.4.3+ 新增）

⚠️ **不要**用 ``get_voice_audio_url()`` 返回的 URL 灌进 ``gr.Audio(type="filepath")``，
   Gradio Audio 控件只接受本地文件路径，会报 ``[Errno 22] Invalid argument``。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# ===== 异常 =====

class VoiceLibraryClientError(Exception):
    """音色库客户端异常（用于写入操作失败时抛出）。"""

    def __init__(self, message: str, status_code: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


# ===== 客户端 =====

class VoiceLibraryClient:
    """音色库客户端（封装 gateway /v1/voices* 调用）。

    设计原则：
        1. **容错**：gateway 不可达时查询方法降级返回空结果，不阻断引擎
        2. **双模**：同时提供 sync + async 接口（sync 供 Gradio UI，async 供引擎推理）
        3. **复用**：单 httpx.AsyncClient 复用连接池（async 路径），sync 用短连接
        4. **可观测**：每次调用记录 URL、响应码、耗时（DEBUG），失败记录 WARNING

    Attributes:
        gateway_url: 平台 gateway 地址（如 http://127.0.0.1:8000）
        timeout: HTTP 请求超时秒数（默认 10s）
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        """初始化 VoiceLibraryClient。

        Args:
            gateway_url: 平台 gateway 地址；None 时从 AIRELIABLE_GATEWAY_URL 环境变量读取，
                         仍未设置时默认 http://127.0.0.1:8000
            timeout: HTTP 请求超时秒数（默认 10s）
        """
        self.gateway_url: str = (
            gateway_url
            or os.environ.get("AIRELIABLE_GATEWAY_URL", "")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self._timeout: float = timeout
        self._async_client: Optional[httpx.AsyncClient] = None

        if not os.environ.get("AIRELIABLE_GATEWAY_URL"):
            logger.warning(
                "VoiceLibraryClient: AIRELIABLE_GATEWAY_URL 环境变量未设置，"
                "使用默认值 %s（可能因 gateway 未启动而失败）",
                self.gateway_url,
            )

        logger.debug(
            "VoiceLibraryClient 初始化: gateway_url=%s timeout=%s",
            self.gateway_url, self._timeout,
        )

    # ===================================================================
    # 私有：异步客户端管理
    # ===================================================================

    async def _get_async_client(self) -> httpx.AsyncClient:
        """获取复用的异步客户端（惰性创建）。"""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.gateway_url,
                timeout=httpx.Timeout(self._timeout, connect=2.0), trust_env=False,  # 平台内部回环，不走系统代理
            )
        return self._async_client

    async def close(self) -> None:
        """关闭异步客户端（引擎停止时调用）。"""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None
            logger.debug("VoiceLibraryClient 异步客户端已关闭")

    # ===================================================================
    # 异步接口（async）
    # ===================================================================

    async def list_voices(self) -> list[dict]:
        """GET /v1/voices — 列出全部音色。

        Returns:
            音色列表，每项含 id/name/sha256/path/description/size/duration。
            失败时返回空列表（不抛异常，容错降级）。
        """
        return await self._list_voices_impl(async_mode=True)

    async def get_voice(self, voice_id: str) -> Optional[dict]:
        """GET /v1/voices/{id} — 查询单个音色。

        Args:
            voice_id: 音色 ID（如 voice_a1b2c3d4）

        Returns:
            音色详情 dict，失败时返回 None（不抛异常）
        """
        return await self._get_voice_impl(voice_id, async_mode=True)

    async def get_voice_by_name(self, name: str) -> Optional[dict]:
        """GET /v1/voices/by-name/{name} — 按名称查询音色。

        Args:
            name: 音色名称（URL 编码后传入）

        Returns:
            匹配的第一个音色 dict，未找到或失败时返回 None
        """
        return await self._get_voice_by_name_impl(name, async_mode=True)

    async def create_voice(
        self,
        audio_path: str,
        name: str,
        description: str = "",
        voice_id: str = "",
        force_overwrite: bool = False,
    ) -> dict:
        """POST /v1/voices — 上传创建音色。

        Args:
            audio_path: 音频文件本地路径
            name: 音色名称（必填，全局唯一）
            description: 音色描述（可选）
            voice_id: 音色 ID（留空自动生成）
            force_overwrite: ID 已存在时是否覆盖

        Returns:
            创建结果 dict（含 id/name/path 等字段）

        Raises:
            VoiceLibraryClientError: 上传失败（含 HTTP 状态码与响应体）
        """
        return await self._create_voice_impl(
            audio_path, name, description, voice_id, force_overwrite, async_mode=True
        )

    async def delete_voice(self, voice_id: str) -> bool:
        """DELETE /v1/voices/{id} — 删除音色。

        Args:
            voice_id: 待删除的音色 ID

        Returns:
            True 表示删除成功，False 表示失败（不抛异常）
        """
        return await self._delete_voice_impl(voice_id, async_mode=True)

    def get_voice_audio_url(self, voice_id: str) -> str:
        """构造音色音频文件 URL（用于浏览器直连播放场景）。

        ⚠️ **不要**用于 Gradio ``gr.Audio(type="filepath")`` 的 value 填充。
        Gradio Audio 控件只接受本地文件路径，请改用 SDK 返回的 ``card["path"]`` 字段。

        适用场景：
            - 浏览器直连播放：``<audio src={url}>``
            - 跨机/远程访问：先下载到本地再使用

        不发 HTTP 请求，仅返回 URL 字符串。

        Args:
            voice_id: 音色 ID

        Returns:
            完整 URL，如 ``http://127.0.0.1:8000/v1/voices/voice_xxx/audio``
        """
        return f"{self.gateway_url}/v1/voices/{voice_id}/audio"

    # ===================================================================
    # 同步接口（sync，供 Gradio UI 回调用）
    # ===================================================================

    def list_voices_sync(self) -> list[dict]:
        """GET /v1/voices — 列出全部音色（同步版本）。"""
        return asyncio.get_event_loop().run_until_complete(
            self._list_voices_impl(async_mode=False)
        ) if False else self._list_voices_sync_impl()

    def get_voice_sync(self, voice_id: str) -> Optional[dict]:
        """GET /v1/voices/{id} — 查询单个音色（同步版本）。"""
        return self._get_voice_sync_impl(voice_id)

    def get_voice_by_name_sync(self, name: str) -> Optional[dict]:
        """GET /v1/voices/by-name/{name} — 按名称查询音色（同步版本）。"""
        return self._get_voice_by_name_sync_impl(name)

    def create_voice_sync(
        self,
        audio_path: str,
        name: str,
        description: str = "",
        voice_id: str = "",
        force_overwrite: bool = False,
    ) -> dict:
        """POST /v1/voices — 上传创建音色（同步版本）。"""
        return self._create_voice_sync_impl(
            audio_path, name, description, voice_id, force_overwrite
        )

    def delete_voice_sync(self, voice_id: str) -> bool:
        """DELETE /v1/voices/{id} — 删除音色（同步版本）。"""
        return self._delete_voice_sync_impl(voice_id)

    # ===================================================================
    # 异步实现
    # ===================================================================

    async def _list_voices_impl(self, *, async_mode: bool) -> list[dict]:
        if async_mode:
            client = await self._get_async_client()
            url = "/v1/voices"
            t0 = time.monotonic()
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.monotonic() - t0) * 1000
                voices = self._parse_voices_response(data)
                logger.debug(
                    "[voice_library_client] list_voices: total=%d, latency=%.0fms",
                    len(voices), latency_ms,
                )
                # v0.4.3+ 富化：每个 card 注入 audio_url（path 已在 gateway 返回）
                return [self._enrich_card(v) for v in voices]
            except Exception as e:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[voice_library_client] list_voices 失败 (latency=%.0fms): %s",
                    latency_ms, e,
                )
                return []
        return []  # 不会执行（sync 走另一条路径）

    async def _get_voice_impl(self, voice_id: str, *, async_mode: bool) -> Optional[dict]:
        if async_mode:
            client = await self._get_async_client()
            url = f"/v1/voices/{voice_id}"
            t0 = time.monotonic()
            try:
                resp = await client.get(url)
                if resp.status_code == 404:
                    logger.debug("[voice_library_client] get_voice %s 未找到", voice_id)
                    return None
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "[voice_library_client] get_voice %s: latency=%.0fms",
                    voice_id, latency_ms,
                )
                # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                return self._enrich_card(resp.json())
            except Exception as e:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[voice_library_client] get_voice %s 失败 (latency=%.0fms): %s",
                    voice_id, latency_ms, e,
                )
                return None
        return None

    async def _get_voice_by_name_impl(self, name: str, *, async_mode: bool) -> Optional[dict]:
        if async_mode:
            client = await self._get_async_client()
            url = f"/v1/voices/by-name/{name}"
            t0 = time.monotonic()
            try:
                resp = await client.get(url)
                if resp.status_code == 404:
                    logger.debug("[voice_library_client] get_voice_by_name %s 未找到", name)
                    return None
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                data = resp.json()
                logger.debug(
                    "[voice_library_client] get_voice_by_name %s: latency=%.0fms",
                    name, latency_ms,
                )
                # v1.1 修复：gateway VoiceListByNameResponse 字段名是 data（schemas.py#L537-L545）
                # 之前错查 voices 字段，永远返回 None
                if isinstance(data, dict) and "data" in data:
                    voices = data["data"]
                    # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                    return self._enrich_card(voices[0]) if voices else None
                # 兼容旧/外部实现
                if isinstance(data, dict) and "voices" in data:
                    voices = data["voices"]
                    # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                    return self._enrich_card(voices[0]) if voices else None
                # 兜底：返回原始 dict（可能是单个音色对象）
                return self._enrich_card(data)
            except Exception as e:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[voice_library_client] get_voice_by_name %s 失败 (latency=%.0fms): %s",
                    name, latency_ms, e,
                )
                return None
        return None

    async def _create_voice_impl(
        self, audio_path: str, name: str, description: str,
        voice_id: str, force_overwrite: bool, *, async_mode: bool,
    ) -> dict:
        if async_mode:
            client = await self._get_async_client()
            url = "/v1/voices"
            t0 = time.monotonic()
            try:
                p = Path(audio_path)
                if not p.exists():
                    raise VoiceLibraryClientError(
                        f"音频文件不存在: {audio_path}", status_code=None
                    )
                with open(p, "rb") as f:
                    files = {"file": (p.name, f, "audio/wav")}
                    data = {"name": name}
                    if description:
                        data["description"] = description
                    if voice_id:
                        data["voice_id"] = voice_id
                    if force_overwrite:
                        data["force_overwrite"] = "true"
                    resp = await client.post(url, files=files, data=data)
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "[voice_library_client] create_voice %s: id=%s, latency=%.0fms",
                    name, resp.json().get("id"), latency_ms,
                )
                # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                return self._enrich_card(resp.json())
            except httpx.HTTPStatusError as e:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[voice_library_client] create_voice %s 失败 %s (latency=%.0fms): %s",
                    name, e.response.status_code, latency_ms, e.response.text[:200],
                )
                raise VoiceLibraryClientError(
                    f"上传失败 HTTP {e.response.status_code}: {e.response.text[:200]}",
                    status_code=e.response.status_code,
                    body=e.response.text,
                ) from e
            except Exception as e:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[voice_library_client] create_voice %s 异常 (latency=%.0fms): %s",
                    name, latency_ms, e,
                )
                raise VoiceLibraryClientError(f"上传异常: {e}") from e
        return {}

    async def _delete_voice_impl(self, voice_id: str, *, async_mode: bool) -> bool:
        if async_mode:
            client = await self._get_async_client()
            url = f"/v1/voices/{voice_id}"
            t0 = time.monotonic()
            try:
                resp = await client.delete(url)
                if resp.status_code == 404:
                    logger.debug("[voice_library_client] delete_voice %s 未找到", voice_id)
                    return False
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "[voice_library_client] delete_voice %s: latency=%.0fms",
                    voice_id, latency_ms,
                )
                return True
            except Exception as e:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "[voice_library_client] delete_voice %s 失败 (latency=%.0fms): %s",
                    voice_id, latency_ms, e,
                )
                return False
        return False

    # ===================================================================
    # 同步实现（直接用 httpx.Client，不依赖 asyncio loop）
    # ===================================================================

    def _list_voices_sync_impl(self) -> list[dict]:
        url = f"{self.gateway_url}/v1/voices"
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                latency_ms = (time.monotonic() - t0) * 1000
                voices = self._parse_voices_response(data)
                logger.debug(
                    "[voice_library_client] list_voices_sync: total=%d, latency=%.0fms",
                    len(voices), latency_ms,
                )
                # v0.4.3+ 富化：每个 card 注入 audio_url（path 已在 gateway 返回）
                return [self._enrich_card(v) for v in voices]
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[voice_library_client] list_voices_sync 失败 (latency=%.0fms): %s",
                latency_ms, e,
            )
            return []

    def _get_voice_sync_impl(self, voice_id: str) -> Optional[dict]:
        url = f"{self.gateway_url}/v1/voices/{voice_id}"
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = client.get(url)
                if resp.status_code == 404:
                    logger.debug("[voice_library_client] get_voice_sync %s 未找到", voice_id)
                    return None
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                logger.debug(
                    "[voice_library_client] get_voice_sync %s: latency=%.0fms",
                    voice_id, latency_ms,
                )
                # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                return self._enrich_card(resp.json())
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[voice_library_client] get_voice_sync %s 失败 (latency=%.0fms): %s",
                voice_id, latency_ms, e,
            )
            return None

    def _get_voice_by_name_sync_impl(self, name: str) -> Optional[dict]:
        url = f"{self.gateway_url}/v1/voices/by-name/{name}"
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = client.get(url)
                if resp.status_code == 404:
                    logger.debug("[voice_library_client] get_voice_by_name_sync %s 未找到", name)
                    return None
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                data = resp.json()
                logger.debug(
                    "[voice_library_client] get_voice_by_name_sync %s: latency=%.0fms",
                    name, latency_ms,
                )
                # v1.1 修复：gateway VoiceListByNameResponse 字段名是 data（schemas.py#L537-L545）
                # 之前错查 voices 字段，永远返回 None
                if isinstance(data, dict) and "data" in data:
                    voices = data["data"]
                    # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                    return self._enrich_card(voices[0]) if voices else None
                # 兼容旧/外部实现
                if isinstance(data, dict) and "voices" in data:
                    voices = data["voices"]
                    # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
                    return self._enrich_card(voices[0]) if voices else None
                # 兜底：返回原始 dict（可能是单个音色对象）
                return self._enrich_card(data)
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[voice_library_client] get_voice_by_name_sync %s 失败 (latency=%.0fms): %s",
                name, latency_ms, e,
            )
            return None

    def _create_voice_sync_impl(
        self, audio_path: str, name: str, description: str,
        voice_id: str, force_overwrite: bool,
    ) -> dict:
        url = f"{self.gateway_url}/v1/voices"
        t0 = time.monotonic()
        try:
            p = Path(audio_path)
            if not p.exists():
                raise VoiceLibraryClientError(
                    f"音频文件不存在: {audio_path}", status_code=None
                )
            with open(p, "rb") as f:
                files = {"file": (p.name, f, "audio/wav")}
                data = {"name": name}
                if description:
                    data["description"] = description
                if voice_id:
                    data["voice_id"] = voice_id
                if force_overwrite:
                    data["force_overwrite"] = "true"
                with httpx.Client(timeout=60.0, trust_env=False) as client:  # 上传用更长超时（平台内部回环，不走系统代理）
                    resp = client.post(url, files=files, data=data)
            resp.raise_for_status()
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "[voice_library_client] create_voice_sync %s: id=%s, latency=%.0fms",
                name, resp.json().get("id"), latency_ms,
            )
            # v0.4.3+ 富化：注入 audio_url（path 已在 gateway 返回）
            return self._enrich_card(resp.json())
        except httpx.HTTPStatusError as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[voice_library_client] create_voice_sync %s 失败 %s (latency=%.0fms): %s",
                name, e.response.status_code, latency_ms, e.response.text[:200],
            )
            raise VoiceLibraryClientError(
                f"上传失败 HTTP {e.response.status_code}: {e.response.text[:200]}",
                status_code=e.response.status_code,
                body=e.response.text,
            ) from e
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[voice_library_client] create_voice_sync %s 异常 (latency=%.0fms): %s",
                name, latency_ms, e,
            )
            raise VoiceLibraryClientError(f"上传异常: {e}") from e

    def _delete_voice_sync_impl(self, voice_id: str) -> bool:
        url = f"{self.gateway_url}/v1/voices/{voice_id}"
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:  # 平台内部回环，不走系统代理
                resp = client.delete(url)
                if resp.status_code == 404:
                    logger.debug("[voice_library_client] delete_voice_sync %s 未找到", voice_id)
                    return False
                resp.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "[voice_library_client] delete_voice_sync %s: latency=%.0fms",
                    voice_id, latency_ms,
                )
                return True
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning(
                "[voice_library_client] delete_voice_sync %s 失败 (latency=%.0fms): %s",
                voice_id, latency_ms, e,
            )
            return False

    # ===================================================================
    # 工具方法
    # ===================================================================

    def _enrich_card(self, card: dict) -> dict:
        """富化 card dict：注入 audio_url 字段（path 已在 gateway 返回）。

        gateway 返回的 VoiceCardResponse 已有 path（本地），本方法补充 audio_url（网络）。
        调用方拿到 dict 后可按需取 path 或 audio_url。

        Args:
            card: gateway 返回的音色卡片 dict

        Returns:
            富化后的 dict（已含 audio_url 字段）；非 dict 原样返回
        """
        if not isinstance(card, dict):
            return card
        voice_id = card.get("id")
        if voice_id and "audio_url" not in card:
            card["audio_url"] = self.get_voice_audio_url(voice_id)
        return card

    @staticmethod
    def _parse_voices_response(data: Any) -> list[dict]:
        """解析 GET /v1/voices 响应。

        gateway VoiceListResponse 实际 schema（schemas.py#L518-L522）::

            class VoiceListResponse(BaseModel):
                data: list[VoiceCardResponse]
                total: int

        即响应 JSON 为 ``{"data": [...], "total": N}``，字段名是 ``data``。

        兼容三种格式：
            - ``{"data": [...], "total": N}``    ← gateway 标准（v1.1 修复：之前误以为 voices）
            - ``{"voices": [...], "total": N}``  ← 兼容旧/外部实现
            - ``[...]``                            ← 兼容裸列表

        v1.1 修复：之前 docstring 错写"VoiceListResponse 返回 {voices: [...], total: N}"，
        实际 schema 字段名是 ``data``。代码只查 ``voices`` / ``items`` 字段，永远返回空列表，
        导致 UI 音色下拉永远空（gateway 已 200 OK 但 SDK 解析失败）。

        证据：
            - gateway schemas.py#L518-L522 字段名是 ``data``
            - 前端 app.js#L13053 ``return data.data || []`` 正确读取 ``data.data``
            - project_memory.md L22 硬约束："前端音色库页面必须读取后端返回的 ``data.data`` 字段"
        """
        if isinstance(data, dict):
            # gateway 标准字段（v1.1 修复：优先查 data，与前端 app.js#L13053 对齐）
            if "data" in data:
                return data["data"]
            # 兼容其他可能的 key
            if "voices" in data:
                return data["voices"]
            if "items" in data:
                return data["items"]
            # 兜底：返回空
            return []
        if isinstance(data, list):
            return data
        return []


# ===== 全局单例（参考 task_reporter 的模式）=====

_global_voice_library_client: Optional[VoiceLibraryClient] = None
_voice_library_client_lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None


async def get_voice_library_client() -> VoiceLibraryClient:
    """获取全局 VoiceLibraryClient 单例（异步）。

    首次调用时创建实例，后续复用。
    引擎停止时调用 close_voice_library_client() 释放连接池。
    """
    global _global_voice_library_client
    if _global_voice_library_client is None:
        async with _voice_library_client_lock or asyncio.Lock():
            if _global_voice_library_client is None:
                _global_voice_library_client = VoiceLibraryClient()
                logger.debug("已创建全局 VoiceLibraryClient 单例")
    return _global_voice_library_client


async def close_voice_library_client() -> None:
    """关闭全局 VoiceLibraryClient 单例（引擎停止时调用）。"""
    global _global_voice_library_client
    if _global_voice_library_client is not None:
        await _global_voice_library_client.close()
        _global_voice_library_client = None
        logger.debug("全局 VoiceLibraryClient 单例已关闭")

"""通用 Pydantic 模型（所有引擎共享的请求/响应结构）。

8 个 HTTP 端点的标准契约在这里定义。
引擎自身的特殊字段放引擎的 schemas.py 里继承扩展。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ===== /health =====
class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"                       # ok | error
    engine_id: str
    state: str                               # idle | loading | ready | inferring | error
    uptime: float = 0.0                      # 秒


# ===== /capabilities =====
class VariantInfo(BaseModel):
    """单个 variant 的能力声明。"""

    name: str                                # variant_id（与 model 字段对应）
    base_model: str                          # 基座模型
    precision: str                           # 量化/精度（q4_k_m / bf16 / fp8 ...）
    loader: str                              # 加载方式（gguf / diffusers / bnb）
    vram_required: str                       # 显存需求描述（"8GB"）
    default_steps: Optional[int] = None
    default_guidance: Optional[float] = None
    supports_pe: bool = False                # 是否支持 Prompt Enhancer
    available: bool = True                   # 模型文件是否存在


class CapabilitiesResponse(BaseModel):
    """能力声明响应。"""

    engine_id: str
    engine_name: str
    capability: str                          # text-to-image / tts / chat ...
    type: str                                # image / voice / video / threed / llm
    variants: list[VariantInfo] = Field(default_factory=list)
    loras: list[str] = Field(default_factory=list)        # 可用 LoRA 列表
    supported_sizes: list[str] = Field(default_factory=list)  # ["1024x1024", ...]


# ===== /load =====
class LoadRequest(BaseModel):
    """加载 variant 请求。"""

    variant: str


class LoadResponse(BaseModel):
    """加载响应。"""

    variant: str
    state: str                               # ready | error
    vram_used: str                           # "7.2GB"
    load_time: Optional[float] = None        # 加载耗时（秒）


# ===== /unload =====
class UnloadResponse(BaseModel):
    """卸载响应。"""

    state: str = "idle"
    vram_freed: Optional[str] = None


# ===== /infer =====
class LoraSpec(BaseModel):
    """LoRA 加载规格。"""

    name: str
    weight: float = 1.0


class ImageInput(BaseModel):
    """图片输入（OpenAI + AIReliable 兼容）。

    三种来源字段：
        - url：HTTP/HTTPS 链接，引擎需下载
        - b64_json：base64 编码的图片数据
        - path：本地文件路径（网关已下载或用户本地）

    role 字段为 AIReliable 扩展，用于区分图片用途：
        - source：源图（图生图基础图）
        - mask：蒙版（编辑区域指示）
        - reference：参考图（IPAdapter / 风格迁移）
        - control：控制图（ControlNet 条件输入）
    """

    url: Optional[str] = None
    b64_json: Optional[str] = None
    path: Optional[str] = None
    role: Optional[str] = None


class AudioInput(BaseModel):
    """音频输入。

    role 字段示例：
        - speaker：参考音色（TTS 克隆）
        - reference：风格参考
        - target：处理目标
    """

    url: Optional[str] = None
    b64_json: Optional[str] = None
    path: Optional[str] = None
    role: Optional[str] = None


class VideoInput(BaseModel):
    """视频输入。"""

    url: Optional[str] = None
    path: Optional[str] = None
    role: Optional[str] = None


class InferRequest(BaseModel):
    """推理请求（内部契约，网关调引擎用）。

    注意：与 OpenAI 的 /v1/images/generations 不同，这是网关→引擎的内部协议。
    网关层负责把 OpenAI 请求转换成此格式。

    v0.2 兼容性：
        - v0.1 的 InferRequest 只有 variant + inputs + loras + task_id
        - v0.2 新增 images / audios / videos / workflow 字段（全部可选 + 默认空）
        - 老引擎不读取这些字段仍能正常工作（向后兼容）
    """

    variant: str
    inputs: dict[str, Any]                                      # 引擎特定输入（prompt/size/steps 等）
    loras: list[LoraSpec] = Field(default_factory=list)
    images: list[ImageInput] = Field(default_factory=list)      # v0.2 多图输入
    audios: list[AudioInput] = Field(default_factory=list)      # v0.2 多音频输入
    videos: list[VideoInput] = Field(default_factory=list)      # v0.2 多视频输入
    workflow: Optional[str] = None                              # v0.2 指定工作流（None 用默认）
    task_id: Optional[str] = None


# ===== /status =====
class StatusResponse(BaseModel):
    """引擎状态查询响应。"""

    state: str                               # idle | loading | ready | inferring | error
    loaded_variant: Optional[str] = None
    vram_used: str = "0GB"
    vram_total: Optional[str] = None
    loaded_loras: list[str] = Field(default_factory=list)
    last_used_at: Optional[float] = None     # Unix 时间戳
    uptime: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)  # 引擎特定字段


# ===== /lora/* =====
class LoraLoadRequest(BaseModel):
    """LoRA 加载请求。"""

    name: str
    weight: float = 1.0


class LoraLoadResponse(BaseModel):
    """LoRA 加载响应。"""

    name: str
    weight: float
    loaded_loras: list[str]


class LoraUnloadRequest(BaseModel):
    """LoRA 卸载请求。"""

    name: str


class LoraUnloadResponse(BaseModel):
    """LoRA 卸载响应。"""

    name: str
    loaded_loras: list[str]


# ===== 错误响应 =====
class ErrorResponse(BaseModel):
    """统一错误响应格式。"""

    error: dict[str, Any]                    # {"type": "...", "message": "...", "code": 400}

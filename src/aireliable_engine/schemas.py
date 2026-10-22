"""通用 Pydantic 模型（所有引擎共享的请求/响应结构）。

8 个 HTTP 端点的标准契约在这里定义。
引擎自身的特殊字段放引擎的 schemas.py 里继承扩展。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ===== HTTP 契约版本（任务 67）=====
# 破坏性变更才 +1；平台在启动握手时校验 /health 返回值是否在支持集合内。
ENGINE_API_VERSION = "1"


# ===== /health =====
class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = "ok"                       # ok | error
    engine_id: str
    state: str                               # idle | loading | ready | inferring | error
    uptime: float = 0.0                      # 秒
    # 引擎实现的 HTTP 契约版本（与 SDK 版本解耦，拆仓后用于双端握手）。
    engine_api_version: str = ENGINE_API_VERSION


# ===== /capabilities =====
class VariantInfo(BaseModel):
    """单个 variant 的能力声明。

    v0.2.5+：extra="allow" 允许子类（如 VideoVariantInfo）扩展字段
    在 CapabilitiesResponse.model_dump() 中保留，不被 Pydantic 丢弃。
    """

    model_config = ConfigDict(extra="allow")

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
    # 默认 variant 名（manifest.models.default_variant 声明）。
    # BaseEngine.infer 的 auto-load 优先级：request.variant > default_variant > 第一个 available variant。
    # 平台 UI 可用此字段自动选中下拉菜单的默认项。
    default_variant: Optional[str] = None
    # 引擎推理输入的 JSON Schema（v0.4.1+ MCP 化新增，可选字段向后兼容）。
    # 引擎在 _capabilities_impl 中导出：input_schema = XxxInputs.model_json_schema()
    # PLATFORM_MCP_INTEGRATION_ANALYSIS.md §3.4）；旧引擎不导出时为 None，
    # MCP 专属层回退到通用层 tool，平台其余组件不受影响。
    input_schema: Optional[dict[str, Any]] = None
    # v0.4.3+ 多能力引擎声明（可选字段）。
    # 单个引擎可以支持多个能力（如 LTX-2.5 同时支持 text-to-video / image-to-video /
    # first-last-frame-to-video / multi-image-to-video / multi-reference-to-video）。
    # 旧客户端只读 capability（单数），新客户端可优先读 capabilities（复数）。
    # 引擎从 manifest.yaml 的 capabilities 列表读取赋值。
    capabilities: list[str] = Field(default_factory=list)


# ===== /load =====
class LoadRequest(BaseModel):
    """加载 variant 请求。

    variant 可为 None：表示不指定具体 variant，由引擎根据 manifest 的
    default_variant 或第一个 available variant 决定（router mode 场景下
    也可表示"只启动子进程，不预加载"）。
    """

    variant: Optional[str] = None


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
        - first_frame / last_frame：视频首尾帧插值

    strength / crf 为视频生成专用扩展（v0.2.17+）：
        - strength (0.0~1.0)：条件约束强度，None 时引擎按 role 给默认值
            （first_frame/last_frame=1.0 完全替换，reference=0.8 强约束）
        - crf (int, 0~40)：H.264 CRF 重压缩参数，None 时模型用训练默认值
            越小质量越高，越大压缩越狠（= 细节越少、引导越"宽松"）
    """

    url: Optional[str] = None
    b64_json: Optional[str] = None
    path: Optional[str] = None
    role: Optional[str] = None
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # v0.2.17 视频专用
    crf: Optional[int] = Field(default=None, ge=0, le=40)            # v0.2.17 视频专用


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

    variant 可为 None：BaseEngine.infer 的 auto-load 优先级为
        request.variant > engine.default_variant > 第一个 available variant
    引擎若不实现 default_variant 机制，None 会触发回退到第一个 available variant。

    lora_explicit（v0.2.16 请求级 LoRA 语义标记，网关填写）：
        - True  = 请求方显式指定了 LoRA 集合（extra.loras 存在，含空数组），
                  引擎应把挂载集合收敛到 loras（空数组 = 清空全部）
        - False = 请求方未指定 LoRA（字段缺省），引擎保持当前挂载集合
                  （WebUI 会话语义：挂载状态由 /lora/load /lora/unload 独立管理）
    """

    variant: Optional[str] = None
    inputs: dict[str, Any]                                      # 引擎特定输入（prompt/size/steps 等）
    loras: list[LoraSpec] = Field(default_factory=list)
    lora_explicit: bool = False                                 # v0.2.16 请求级 LoRA 语义标记
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

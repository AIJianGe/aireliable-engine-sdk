# AIReliable Engine SDK

> 版本：0.2.5（v0.2.15+ 日志目录统一与 RotatingFileHandler 升级）
> 状态：开发中
> 配套文档：项目根目录 `v0.2-analysis.md` / `v0.2-tasks.md` / `PR/development-guide.md` / [统一日志平台架构](../../docs/docs/architecture/logging-platform.md)

AIReliable 引擎开发工具包。引擎开发者只需 `pip install aireliable-engine-sdk` 即可开发可插拔引擎，无需 clone 主仓。

> 仓库地址：https://github.com/AIJianGe/aireliable-engine-sdk

!!! warning "v0.2.5 工作流唯一化"
    v0.2.5 起，SDK 移除 `integrated` / `custom` 范式支持，全平台统一为 `workflow` 唯一范式：
    - `BaseEngine` 的 7 个 `@abstractmethod`（`_capabilities_impl` / `_load_impl` / `_unload_impl` / `_infer_impl` / `_status_impl` / `_load_lora_impl` / `_unload_lora_impl`）改为默认 `raise NotImplementedError`。
    - 通过 `__init_subclass__` 检测直接继承 `BaseEngine` 的子类并发出 `DeprecationWarning`。
    - **v0.2.5+ 所有引擎必须继承 `WorkflowEngine`**，仅实现 `_build_workflow` + `_capabilities_impl` 两个方法。

---

## 一、安装

### 1.1 当前过渡方式（未发布 PyPI）

SDK 当前还在开发阶段尚未发布到 PyPI，请用 git 依赖过渡：

```bash
# 直接安装（最新 main 分支）
pip install git+https://github.com/AIJianGe/aireliable-engine-sdk.git

# 或在项目 pyproject.toml 中声明
# dependencies = [
#     "aireliable-engine-sdk @ git+https://github.com/AIJianGe/aireliable-engine-sdk.git",
# ]
```

如使用 `uv`：

```toml
# uv.toml 或 pyproject.toml [tool.uv.sources]
aireliable-engine-sdk = { git = "https://github.com/AIJianGe/aireliable-engine-sdk.git", branch = "main" }
```

### 1.2 发布到 PyPI 后（规划中）

```bash
# 最小核心（仅契约 + SSE + 状态机 + 下载器）
pip install aireliable-engine-sdk

# 图片引擎开发（含 torch / diffusers / transformers）
pip install "aireliable-engine-sdk[image]"

# LLM 引擎开发
pip install "aireliable-engine-sdk[llm]"

# 视频引擎开发
pip install "aireliable-engine-sdk[video]"

# 语音引擎开发
pip install "aireliable-engine-sdk[voice]"

# 含 Gradio Web UI
pip install "aireliable-engine-sdk[gradio]"

# 开发测试
pip install "aireliable-engine-sdk[dev]"
```

---

## 二、快速开始

> v0.2.5 起，`workflow` 是唯一支持的引擎范式。本节展示如何用 `WorkflowEngine` 快速搭建引擎。

### 2.1 工作流引擎（v0.2.5+ 唯一支持范式）

声明式工作流，由平台节点库 + 自定义节点编排而成。

```yaml
# workflows/default.yaml
nodes:
  - id: images_input
    type: ImagesInput
  - id: text_input
    type: TextInput
    params:
      field: prompt
  - id: model_loader
    type: ModelLoader
    params:
      variant: ${variant}
  - id: sampler
    type: Sampler
    inputs:
      model: ${model_loader.model}
      prompt: ${text_input.text}
      images: ${images_input.images}
  - id: save
    type: ImageSave
    inputs:
      images: ${sampler.images}

output:
  image_urls: ${save.image_urls}
```

```python
# engine.py
from aireliable_engine import WorkflowEngine

class MyWorkflowEngine(WorkflowEngine):
    async def _build_workflow(self) -> str:
        # 加载工作流 yaml（可按 variant 选择不同 yaml）
        return (self.workflows_dir / "default.yaml").read_text()

    async def _capabilities_impl(self):
        return {
            "variants": ["fp16", "bf16", "q4_k_m"],
            "loras": [],
            "stream_modes": ["sse", "json", "async"],
        }
```

只需实现两个方法：`_build_workflow`（返回 yaml 文本）+ `_capabilities_impl`（返回能力声明）。

### 2.2 直接继承 BaseEngine（v0.2.5 已废弃）

!!! warning "Deprecated 自 v0.2.5 起"
    v0.2.5 起，直接继承 `BaseEngine` 会触发 `DeprecationWarning`，且 7 个原 `@abstractmethod` 改为默认 `raise NotImplementedError`。新引擎**必须**改用 `WorkflowEngine`。下列写法仅作历史参考，不再推荐：

```python
from aireliable_engine import BaseEngine, InferRequest, SSEStream, make_frame

class MyEngine(BaseEngine):  # ⚠ 触发 DeprecationWarning
    def __init__(self):
        super().__init__(
            engine_id="my_engine",
            engine_name="My Engine",
            capability="text-to-image",
            engine_type="image",
        )

    async def _capabilities_impl(self):
        ...

    async def _load_impl(self, variant):       # ⚠ 默认 raise NotImplementedError
        ...

    async def _unload_impl(self):              # ⚠ 默认 raise NotImplementedError
        ...

    async def _infer_impl(self, request: InferRequest, stream: SSEStream):
        for step in range(50):
            await stream.put(make_frame(progress=step/50, stage="denoising", step=step))
        await stream.put(make_frame(progress=1.0, stage="completed",
                                     result={"image_urls": ["http://..."]}))

    async def _status_impl(self):              # ⚠ 默认 raise NotImplementedError
        ...

    async def _load_lora_impl(self, name, weight):   # ⚠ 默认 raise NotImplementedError
        ...

    async def _unload_lora_impl(self, name):         # ⚠ 默认 raise NotImplementedError
        ...
```

迁移到 `WorkflowEngine` 的步骤详见 [v0.1 到 v0.2 迁移指南 - integrated → workflow 迁移指南](../../docs/docs/migration/v01-to-v02.md#integrated--workflow-迁移指南)。

### 2.3 自定义 FastAPI 路由（仍声明 workflow 范式）

若引擎需要在 8 端点契约之外暴露自定义路由（如 webhook、健康检查扩展等），可在 `server.py` 中追加 FastAPI 路由，但 `manifest.yaml` 的 `paradigm` 仍必须声明为 `workflow`，且 `engine.py` 必须继承 `WorkflowEngine`：

```python
# server.py
from fastapi import FastAPI
from aireliable_engine import create_engine_app
from .engine import MyWorkflowEngine

app = create_engine_app(MyWorkflowEngine())

# 追加自定义路由（不影响 workflow 范式声明）
@app.get("/custom/healthz")
async def custom_healthz():
    return {"status": "ok"}
```

---

## 三、SDK 模块导出

| 模块 | 主要导出 | 用途 |
|---|---|---|
| `engine_base` | `BaseEngine`, `EngineBusyError`, ... | 引擎抽象基类（v0.2.5 起 7 个 `_xxx_impl` 改为默认 `raise NotImplementedError`，直接继承触发 `DeprecationWarning`） |
| `schemas` | `InferRequest`, `LoadRequest`, ... | 8 端点 Pydantic 模型 |
| `sse` | `SSEStream`, `make_frame`, ... | SSE 流式响应 |
| `state_machine` | `RuntimeState`, `StateMachine` | 引擎 6 态状态机 |
| `logging_config` | `setup_engine_logging` | 统一日志格式 |
| `downloader` | `BaseEngineDownloader`, `HFSnapshotDownloader` | 通用下载器 |
| `subprocess_utils` | `make_uv_run_command`, ... | 子进程工具 |
| `node` | `BaseNode`, `PortSpec` | 节点抽象（v0.2） |
| `node_registry` | `NodeRegistry`, `register_node` | 节点注册表（v0.2） |
| `workflow_executor` | `WorkflowExecutor` | 工作流执行器（v0.2） |
| `workflow_dsl` | `evaluate_condition` | 安全条件 DSL（v0.2） |
| `workflow_engine` | `WorkflowEngine` | 工作流引擎基类（v0.2；**v0.2.5+ 为唯一推荐基类**） |
| `types` | `Image`, `Images`, `Audio`, `Video` | 节点数据类型（v0.2） |

---

## 四、目录结构

```
aireliable-engine-sdk/
├── pyproject.toml
├── README.md
├── src/
│   └── aireliable_engine/
│       ├── __init__.py              # 公开 API 导出
│       ├── engine_base.py          # BaseEngine（v0.2.5 起降级为不推荐直接继承）
│       ├── workflow_engine.py      # WorkflowEngine（v0.2.5+ 唯一推荐基类）
│       ├── node.py                 # BaseNode + PortSpec
│       ├── node_registry.py        # NodeRegistry
│       ├── workflow_executor.py    # WorkflowExecutor
│       ├── workflow_dsl.py          # 安全条件 DSL
│       ├── schemas.py              # 8 端点 Pydantic 模型
│       ├── types.py                # Image/Images/Audio/...
│       ├── sse.py                  # SSE 流
│       ├── state_machine.py        # 6 态状态机
│       ├── logging_config.py       # 日志
│       ├── downloader.py           # 下载器基类
│       ├── subprocess_utils.py     # 子进程工具
│       └── nodes/                  # 平台基础节点库
│           ├── input/
│           ├── loader/
│           ├── encode/
│           ├── sample/
│           ├── decode/
│           └── postprocess/
└── tests/
```

---

## 五、本地开发

```bash
# 克隆仓库后，本地安装可编辑模式
git clone https://github.com/AIJianGe/aireliable-engine-sdk.git
cd aireliable-engine-sdk
pip install -e .

# 运行测试
pytest tests/ -v

# 代码风格检查
ruff check src/ tests/
```

---

## 六、版本

| 版本 | 主要变更 |
|---|---|
| 0.2.5 | 工作流唯一化：移除 `integrated` / `custom` 范式支持；`BaseEngine` 的 7 个 `@abstractmethod` 改为默认 `raise NotImplementedError`；通过 `__init_subclass__` 检测直接继承 `BaseEngine` 的子类并发出 `DeprecationWarning`；所有引擎必须继承 `WorkflowEngine`。 |
| 0.2.5+ | 节点体系增强：`params_schema` 增加 `format` 字段（multiline / enum-from-dir / file-paths / info）；`NodeRegistry.discover_local_dir()` 本地目录发现；`_node_to_definition` 透传 format + title 回退；`WorkflowEngine._get_default_workflow_template()` 抽象方法。 |
| 0.2.0 | v0.2 初始版本：抽取自 `engines/base/`，新增 WorkflowEngine / BaseNode / NodeRegistry / WorkflowExecutor / 安全条件 DSL / 平台基础节点库 |

---

## 自定义节点开发（v0.2.5 增补）

v0.2.5 新增 `custom_nodes/` 目录自动发现机制，引擎启动时自动扫描 .py 文件中的 `BaseNode` 子类并注册。

**快速开始**：
1. 在引擎目录下创建 `custom_nodes/` 文件夹（首次运行自动创建 + 生成 README）
2. 创建 .py 文件（如 `my_node.py`），继承 `BaseNode` 并实现 `execute()` 方法
3. 重启引擎，节点自动出现在工作流编辑器节点库中

**模板文件**：参考 `src/aireliable_engine/_custom_node_template.py`

**API**：`NodeRegistry.discover_local_dir(directory, override=False, recursive=True)` 扫描目录并注册节点。

**详细规范**：见 [PR/development-guide.md §28.12](../../PR/development-guide.md)

---

## 七、设备选择规范（CPU/GPU 启动选项）

> 状态：分析阶段，待实施（**前置条件已就绪**：torch CUDA 版本切换功能 v0.2.5+ 已实现）
> 目标：为每个引擎提供启动时 CPU/GPU 选择能力，**默认 GPU**

### 7.0 前置条件：torch CUDA 版本切换（v0.2.5+ 已实现）

!!! warning "重要前置条件"
    本规范的实施依赖 **torch CUDA 版本切换功能**。在 torch 仍是 CPU 版时，
    `torch.cuda.is_available()` 返回 False，device 选择形同虚设。

    v0.2.5+ 已实现 torch CUDA 版本切换功能：
    - **架构文档**：[torch CUDA 版本切换架构](../docs/docs/architecture/torch-cuda-switch.md)
    - **用户指南**：[如何切换到 GPU](../docs/docs/how-to/torch-cuda.md)
    - **任务表**：[torch-cuda-switch-tasks.md](../torch-cuda-switch-tasks.md)

    **建议流程**：
    1. 用户通过 torch CUDA 切换功能将引擎 .venv 切换到 CUDA 版 torch
    2. 切换完成后，本规范的 device 参数透传才有实际效果

    **torch CUDA 切换功能提供的端点**（core 端，本规范可复用）：
    - `GET /engines/{id}/torch_status` - 查询 torch 版本和系统 CUDA 状态
    - `POST /engines/{id}/torch_switch` - 启动 CUDA 切换任务
    - `GET /engines/{id}/torch_switch/status` - 查询切换进度
    - `DELETE /engines/{id}/torch_switch` - 取消切换任务

### 7.1 需求背景

当前所有引擎启动时**无法由用户选择设备**：

- 前端启动按钮仅传 `engine_id`，全链路无 device 参数
- 各引擎在自身 `config.py` 中硬编码默认设备并自行检测：
  - ERNIE-Image：默认 `cuda`，`torch.cuda.is_available()` 降级
  - VoxCPM-TTS：默认 `auto`，`_resolve_device()` 探测（但解析结果未传给 `from_pretrained`，存在 bug）
  - LLaMA-cpp：无 device 字段，通过 `gpu_vram_gb` + `nvidia-smi` 检测，用 `-ngl` 控制 GPU offload
- SDK 中无统一设备检测工具函数，各引擎重复实现
- `WorkflowEngine` 基类硬编码 `device="cpu"`（`workflow_engine.py:527`），依赖子类覆写

### 7.2 设计方案

采用 **"启动参数透传 + 环境变量注入 + SDK 统一工具"** 三层方案：

```
[前端 UI] device 选择
    └─ [Tauri] start_engine(engine_id, device)
        └─ [Core] POST /engines/{id}/start  body={device}
            └─ [EngineManager] _build_start_cmd 追加 --device
                              _build_engine_subprocess_env 注入 CUDA_VISIBLE_DEVICES
                └─ [引擎子进程] config.parse_args() 读取 --device
                    └─ SDK resolve_device() 统一解析
```

**device 取值规范**：

| 值 | 含义 | 行为 |
|----|------|------|
| `auto` | 自动检测（默认） | 优先 `cuda:0`，不可用降级 `cpu` |
| `cpu` | 强制 CPU | 不加载到 GPU |
| `cuda` | GPU（默认 GPU 设备） | 等价 `cuda:0` |
| `cuda:0` / `cuda:1` | 指定 GPU 设备 | 注入 `CUDA_VISIBLE_DEVICES=<N>`，子进程内用 `cuda:0` |
| `mps` | Apple Silicon | 仅 ERNIE-Image 声明支持，其余引擎忽略 |

### 7.3 各层改动点

#### 7.3.1 前端 UI（src-tauri/ui/app.js）

| 位置 | 改动 |
|------|------|
| `app.js:653, 1794, 4459` 启动按钮 | 附近增加设备选择下拉框（默认 `cuda`，可选 `cpu`/`auto`/`cuda:0`） |
| `app.js:3020-3047 triggerAsyncStart` | 签名改为 `triggerAsyncStart(engineId, device)`，`callBackend('start_engine', { engineId, device })` |
| `app.js:2725` 下载完成自动启动 | 同步传递 device（用用户选择的或默认 `cuda`） |
| `app.js:3053 triggerAsyncRestart` | restart 同样传递 device |

**UI 交互**：
- 设备选择控件显示在启动按钮左侧或上方
- 引擎运行中隐藏/禁用设备选择（运行时不可切换）
- 未检测到 GPU 时，下拉框自动降级为 `cpu` 并提示
- **新增**：设备下拉框旁显示 torch 状态徽章（来自 torch CUDA 切换功能）
  - CPU 版 torch 时点击 `cuda` 选项会提示"请先切换到 GPU"并跳转到设备状态卡片

#### 7.3.2 Tauri 命令层（src-tauri/src/）

| 文件 | 位置 | 改动 |
|------|------|------|
| `commands.rs:62-104` | `start_engine` | 增加 `device: Option<String>` 参数，透传给 `core_client` |
| `commands.rs:148-185` | `restart_engine` | 同样增加 `device` 参数 |
| `core_client.rs:193-236` | `start_engine` | 签名改为 `start_engine(&self, engine_id, device: Option<&str>)`，构造 JSON body `{"device": device}` |
| `core_client.rs:251` | `restart_engine` | 同样改造 |
| `lib.rs:90-160` | 命令注册 | 无需改（参数自动透传） |

#### 7.3.3 Core HTTP API（core/src/core/server.py）

| 位置 | 改动 |
|------|------|
| `server.py:892-954` `start_engine` 端点 | 新增 `StartEngineRequest` Pydantic schema（`device: Optional[str] = None`），端点签名改为 `start_engine(engine_id: str, request: StartEngineRequest = Body(default=None))` |
| `server.py:964-970` `restart_engine` 端点 | 同样改造 |

**StartEngineRequest schema**：
```python
class StartEngineRequest(BaseModel):
    device: Optional[str] = None  # auto / cpu / cuda / cuda:N / mps；None=用引擎默认
```

#### 7.3.4 Core EngineManager（core/src/core/engine_manager.py）

| 位置 | 改动 |
|------|------|
| `engine_manager.py:391` `start_engine` | 签名改为 `start_engine(self, engine_id, device=None)`，透传到 `_start_engine_bg` |
| `engine_manager.py:560-588` `_build_start_cmd` | 若 device 非空且引擎支持 `--device` 参数，追加 `--device <device>` 到 cmd |
| `engine_manager.py:1104-1188` `_build_engine_subprocess_env` | 增加 device → `CUDA_VISIBLE_DEVICES` 映射：`cuda:N` → `CUDA_VISIBLE_DEVICES=N`（子进程内仍用 `cuda:0`） |
| `engine_manager.py:859-894` `restart_engine` | 同样改造 |
| `EngineInstance` 数据类（line 63-102） | 增加 `device: Optional[str] = None` 字段记录当前运行设备 |

**命令构造策略**：
- 优先追加 `--device <device>` 到 startup_cmd（若引擎声明支持）
- 同时注入 `AIRELIABLE_DEVICE` 环境变量作为 fallback（引擎 `parse_args` 优先读 CLI，其次读环境变量）
- `cuda:N` 场景：注入 `CUDA_VISIBLE_DEVICES=N`，命令行传 `--device cuda:0`（子进程视角）

#### 7.3.5 引擎 config.py（各引擎）

| 引擎 | 现状 | 改动 |
|------|------|------|
| ERNIE-Image（`config.py:353`） | 已有 `--device`（默认 `cuda`） | 增加环境变量 `AIRELIABLE_DEVICE` fallback；统一默认值为 `auto` |
| VoxCPM-TTS（`config.py:125`） | 已有 `--device`（默认 `auto`） | 增加环境变量 fallback；修复 `_resolve_device` 结果未传给 `from_pretrained` 的 bug |
| LLaMA-cpp（`config.py`） | 无 `--device`，仅 `--gpu-vram-gb` | 新增 `--device` 参数（cpu/auto/cuda）；`generate()` 中 `device=cpu` 时强制 `ngl=0`，`device=cuda` 时保持原有显存检测逻辑 |

**LLaMA-cpp 特殊处理**：
- `--device cpu` → `ngl=0`（纯 CPU）
- `--device cuda` / `--device auto` → 保持原有 `gpu_vram_gb` + `detect_gpu_vram()` 逻辑
- 不改变 `-ngl` 的细粒度 offload 能力

#### 7.3.6 引擎 SDK（sdk/aireliable-engine-sdk/）

**新增模块** `src/aireliable_engine/device_utils.py`：

```python
def resolve_device(device: str = "auto") -> str:
    """统一解析设备字符串，含可用性检测。

    - "auto" → 优先 cuda:0，不可用降级 cpu
    - "cuda" → "cuda:0"（若可用）
    - "cuda:N" → "cuda:N"（若可用）
    - "cpu" → "cpu"
    - "mps" → "mps"（若可用）
    """

def detect_gpu_vram_gb() -> Optional[int]:
    """统一 GPU 显存检测（torch.cuda + nvidia-smi fallback）。"""

def get_torch_device(device: str) -> "torch.device":
    """转为 torch.device 对象。"""

def map_device_to_cuda_visible_devices(device: str) -> Optional[str]:
    """cuda:N → "N"；cpu/auto → None。"""
```

**改造点**：

| 位置 | 改动 |
|------|------|
| `workflow_context.py:315` | `device` 字段默认值从 `"cpu"` 改为 `"auto"`，文档化语义 |
| `workflow_engine.py:527` | 基类不再硬编码 `device="cpu"`，改为从 `EngineConfig` 或环境变量读取 |
| `__init__.py` | 导出 `resolve_device` / `detect_gpu_vram_gb` 等工具函数 |

### 7.4 manifest.yaml 字段扩展（可选）

在 `models` 块新增 `device_support` 字段声明引擎支持的设备：

```yaml
models:
  device_support:
    supported_devices: [auto, cpu, cuda]   # 支持的设备列表
    default_device: auto                    # 默认设备
    gpu_required: false                     # 是否必须 GPU（true 时无 GPU 拒绝启动）
```

**作用**：
- 前端根据 `supported_devices` 渲染设备选择下拉框
- core 根据 `gpu_required` 在无 GPU 时拒绝启动
- `default_device` 覆盖全局默认值

### 7.5 实施步骤（建议顺序）

| 步骤 | 内容 | 风险 | 优先级 |
|------|------|------|--------|
| 0 | **前置**：torch CUDA 版本切换功能（v0.2.5+ 已实现） | - | - |
| 1 | SDK 新增 `device_utils.py` 工具函数 | 低 | 高 |
| 2 | 修复 VoxCPM `_resolve_device` 结果未应用的 bug | 低 | 高 |
| 3 | 各引擎 `config.py` 增加 `AIRELIABLE_DEVICE` 环境变量 fallback | 低 | 高 |
| 4 | Core `server.py` 新增 `StartEngineRequest` schema + 端点改造 | 中 | 高 |
| 5 | Core `engine_manager.py` 命令构造 + 环境变量注入 | 中 | 高 |
| 6 | Tauri `commands.rs` + `core_client.rs` 透传 device | 中 | 高 |
| 7 | 前端 `app.js` 增加设备选择 UI + 调用链改造 | 中 | 高 |
| 8 | LLaMA-cpp 新增 `--device` 参数 + ngl 映射 | 中 | 中 |
| 9 | manifest.yaml `device_support` 字段 + schema 校验 | 低 | 低 |
| 10 | 多 GPU 选择（CUDA_VISIBLE_DEVICES）端到端联调 | 高 | 低 |

### 7.6 风险与注意事项

1. **LLaMA-cpp 不依赖 torch**：无法用 `torch.cuda.is_available()` 统一检测，需保留 `nvidia-smi` 路径或引入新依赖
2. **运行时不可切换**：三个引擎都不支持运行时切换设备（需重新 load 或重启子进程），UI 应在运行中禁用设备选择
3. **VoxCPM 现有 bug**：`_resolve_device` 解析的 device 未传给 `VoxCPM.from_pretrained`，导致设备选择实际不生效，需在步骤 2 修复
4. **CUDA_VISIBLE_DEVICES 与 cuda:N 的映射**：注入 `CUDA_VISIBLE_DEVICES=N` 后，子进程内 torch 只看到 1 个 GPU，应使用 `cuda:0` 而非 `cuda:N`
5. **VRamManager 未充分实现**：SDK 中 `OffloadVRamManager` / `SmartVRamManager` 仅 docstring 提及，显存管理仍由各引擎自行处理
6. **节点级 device override**：ERNIE-Image 的节点支持 `params.device` 覆盖 `ctx.device`，统一方案需保留此机制
7. **MPS 支持**：仅 ERNIE-Image 声明支持，VoxCPM/LLaMA-cpp 不支持，UI 应根据 `supported_devices` 过滤可选项

### 7.7 关键文件清单

| 文件 | 改动类型 | 章节 |
|------|----------|------|
| `sdk/aireliable-engine-sdk/src/aireliable_engine/device_utils.py` | 新增 | 7.3.6 |
| `sdk/aireliable-engine-sdk/src/aireliable_engine/workflow_context.py` | 改 | 7.3.6 |
| `sdk/aireliable-engine-sdk/src/aireliable_engine/workflow_engine.py` | 改 | 7.3.6 |
| `sdk/aireliable-engine-sdk/src/aireliable_engine/__init__.py` | 改 | 7.3.6 |
| `core/src/core/server.py` | 改 | 7.3.3 |
| `core/src/core/engine_manager.py` | 改 | 7.3.4 |
| `src-tauri/src/commands.rs` | 改 | 7.3.2 |
| `src-tauri/src/core_client.rs` | 改 | 7.3.2 |
| `src-tauri/ui/app.js` | 改 | 7.3.1 |
| `engines/e_909469ed21ed/config.py` | 改 | 7.3.5 |
| `engines/e_c4a8f2e7b931/config.py` | 改 | 7.3.5 |
| `engines/e_c4a8f2e7b931/voxcpm_wrapper.py` | 改（修 bug） | 7.3.5 |
| `engines/e_9cdb76f8b54c/config.py` | 改 | 7.3.5 |
| `engines/e_9cdb76f8b54c/llama_config.py` | 改 | 7.3.5 |
| `engines/*/manifest.yaml` | 改（可选） | 7.4 |
| `core/src/core/manifest_schema.py` | 改（可选） | 7.4 |

### 7.8 与 torch CUDA 版本切换功能的协同

本规范与 [torch CUDA 版本切换功能](../docs/docs/architecture/torch-cuda-switch.md) 的协同点：

| 协同点 | 本规范 | torch CUDA 切换功能 |
|--------|--------|---------------------|
| 设备检测 | `device_utils.resolve_device()` | `TorchVersionManager.detect_current()` |
| GPU 可用性 | `torch.cuda.is_available()` | `torch.cuda.is_available()`（已检测） |
| 前端 UI | 启动按钮旁设备下拉框 | 引擎详情页设备状态卡片 |
| 触发时机 | 启动引擎时 | 引擎创建后、启动前 |
| 默认值 | `cuda`（GPU 优先） | CPU 版 torch（用户主动切换） |

**用户使用流程**：
1. 创建引擎（默认 CPU 版 torch）
2. 在引擎详情页点击"切换到 GPU"完成 torch CUDA 切换
3. 启动引擎时选择 `cuda` 设备（默认）
4. 引擎使用 GPU 加速推理

**实施建议**：本规范的步骤 7（前端 UI 改造）应复用 torch CUDA 切换功能的 `state.engineTorchStatus[engineId]` 数据，避免重复请求。当用户选择 `cuda` 启动但 `engineTorchStatus.is_cuda === false` 时，前端应弹出提示引导用户先进行 torch CUDA 切换。

---

## 八、相关文档

- 完整架构：`PR/architecture.md`
- 开发规范：`PR/development-guide.md`
- v0.2 分析：`v0.2-analysis.md`
- v0.2 任务：`v0.2-tasks.md`
- 统一日志平台架构：`docs/docs/architecture/logging-platform.md`
- 日志使用指南：`docs/docs/how-to/logging.md`
- **torch CUDA 版本切换架构**（v0.2.5+）：`docs/docs/architecture/torch-cuda-switch.md`
- **如何切换到 GPU 用户指南**（v0.2.5+）：`docs/docs/how-to/torch-cuda.md`

---

## 九、日志

> v0.2.5+ 引入 **统一日志平台**，SDK 提供日志配置工具与数据结构。完整架构详见
> [统一日志平台架构](../../docs/docs/architecture/logging-platform.md)；
> 面向开发者的操作指南见 [查看与排查日志](../../docs/docs/how-to/logging.md)；
> 开发规范见 [开发指南 §9.3 日志规范](../../PR/development-guide.md#93-日志规范)。

### 9.1 模块导出

SDK 从 `aireliable_engine` 顶层导出以下日志相关符号：

| 符号 | 模块 | 用途 |
|---|---|---|
| `setup_engine_logging` | `logging_config` | 引擎日志配置入口（必须调用） |
| `EngineLogFormatter` | `logging_config` | 人类可读格式 formatter（文件用） |
| `JSONFormatter` | `logging_config` | JSON 格式 formatter（stdout 用，v0.2.5+） |
| `LogEntry` | `log_manager` | 统一日志条目数据结构 |
| `LogBuffer` | `log_manager` | 单来源日志缓冲区 |
| `LogManager` | `log_manager` | 统一日志管理器（多来源） |
| `SSELogSubscriber` | `log_manager` | SSE 日志订阅器 |
| `parse_engine_stdout_line` | `log_manager` | 解析引擎 stdout 行为 LogEntry |

### 9.2 `setup_engine_logging`

引擎启动时必须调用一次，配置 stdout + 文件双轨制日志。

#### 9.2.1 签名

```python
def setup_engine_logging(
    engine_id: str,
    log_dir: Optional[Path] = None,   # v0.2.15+ 改为可选，默认 None
    level: int = logging.INFO,
    json_stdout: bool = True,
) -> logging.Logger
```

#### 9.2.2 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `engine_id` | str | 必填 | 引擎 ID（如 `e_c4a8f2e7b931`），写入每条日志；必须与 manifest.yaml 引擎 ID 一致，确保文件名 `{engine_id}.log` 正确 |
| `log_dir` | `Optional[Path]` | `None` | v0.2.15+ 改为可选。日志文件目录，推导优先级：显式传入 > `AIRELIABLE_LOG_DIR` 环境变量 > 不写文件（仅 stdout）。推荐不传，由 SDK 读平台注入的 `AIRELIABLE_LOG_DIR` 推导到 `{project_root}/logs/engines/` |
| `level` | int | `logging.INFO` | 日志级别（`logging.DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`） |
| `json_stdout` | bool | `True` | v0.2.5+ stdout 是否输出 JSON 格式（推荐 True，core 自动解析） |

#### 9.2.3 行为

- **stdout handler**：
  - `json_stdout=True`（默认）：使用 `JSONFormatter`，输出 `{"timestamp":...,"level":...,"engine_id":...,"message":...}`
  - `json_stdout=False`：使用 `EngineLogFormatter`，输出 `2026-07-24 15:30:01,234 - e_xxx - INFO - 消息`
- **file handler**（v0.2.15+ 条件添加）：当 `log_dir` 经推导有值时，`RotatingFileHandler` 写入 `log_dir/<engine_id>.log`
  - 单文件 50MB，保留 5 个备份（v0.2.15+ 升级，此前为无轮转 `FileHandler`）
  - 始终使用 `EngineLogFormatter`（人类可读，便于 `tail -f`）
  - `log_dir` 推导优先级：① 显式传入 > ② `AIRELIABLE_LOG_DIR` 环境变量（平台 `engine_manager._build_engine_subprocess_env` 注入）> ③ 不写文件
- **logger 名称**：返回 `logging.getLogger("aireliable_engine")`，root logger 也被配置

#### 9.2.4 使用示例

```python
from aireliable_engine import setup_engine_logging
import logging

# v0.2.15+ 推荐方式：依赖平台注入的 AIRELIABLE_LOG_DIR 环境变量
# SDK 自动推导到 {project_root}/logs/engines/<engine_id>.log
logger = setup_engine_logging(
    engine_id="e_c4a8f2e7b931",
    level=logging.INFO,
    json_stdout=True,
    # log_dir 不传，由 SDK 读 AIRELIABLE_LOG_DIR 推导
)

logger.info("VoxCPM 引擎已就绪，监听端口 %d", port)
logger.warning("显存不足，回退到 CPU")
logger.error("模型文件缺失: %s", path)
```

### 9.3 `EngineLogFormatter`（人类可读格式）

用于日志文件，格式：

```
2026-07-24 15:30:01,234 - e_c4a8f2e7b931 - INFO - VoxCPM 引擎已就绪
```

字段顺序：`<timestamp> - <engine_id> - <level> - <message>`

- `timestamp`：`%Y-%m-%d %H:%M:%S,sss`（逗号后为毫秒）
- `engine_id`：来自 `setup_engine_logging` 参数
- `level`：DEBUG / INFO / WARNING / ERROR / CRITICAL
- `message`：已格式化的日志消息

异常信息（`exc_info`）追加到 message 后，多行缩进。

### 9.4 `JSONFormatter`（JSON 格式，v0.2.5+）

用于 stdout，便于 core `parse_engine_stdout_line` 自动解析为 `LogEntry`。

#### 9.4.1 输出格式

```json
{"timestamp":"2026-07-24 15:30:01,234","level":"INFO","engine_id":"e_c4a8f2e7b931","message":"VoxCPM 引擎已就绪","logger":"aireliable_engine","module":"engine","line":42}
```

#### 9.4.2 字段规范

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `timestamp` | str | 是 | `%Y-%m-%d %H:%M:%S,sss` 格式 |
| `level` | str | 是 | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| `engine_id` | str | 是 | 引擎 ID |
| `message` | str | 是 | 日志消息（已格式化） |
| `logger` | str | 否 | logger 名称 |
| `module` | str | 否 | 模块名 |
| `line` | int | 否 | 行号 |

异常信息追加到 `message` 字段（多行用 `\n` 分隔）。

### 9.5 `LogEntry`（统一日志条目）

定义于 `src/aireliable_engine/log_manager.py`，所有来源日志在 core 层统一抽象为此结构。

#### 9.5.1 字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `seq` | int | 是 | 全局单调递增序列号（由 `LogManager` 分配） |
| `timestamp` | str | 是 | `%Y-%m-%d %H:%M:%S` 格式 |
| `level` | str | 是 | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| `source` | str | 是 | 来源标识（`core` / `gateway` / `engine:<id>` / `shell`） |
| `message` | str | 是 | 日志消息（已格式化） |
| `engine_id` | str | 否 | 引擎 ID（仅 engine 来源） |
| `task_id` | str | 否 | 任务 ID（v0.2.5+ 预留，用于跨引擎链路追踪） |
| `extra` | dict | 否 | 附加字段（如 `logger` / `module` / `line`） |

#### 9.5.2 `to_dict()` 方法

```python
entry = LogEntry(
    timestamp="2026-07-24 15:30:01",
    level="INFO",
    source="engine:e_xxx",
    message="引擎就绪",
    engine_id="e_xxx",
)
d = entry.to_dict()
# {"seq":0,"timestamp":"2026-07-24 15:30:01","level":"INFO","source":"engine:e_xxx","message":"引擎就绪","engine_id":"e_xxx"}
```

### 9.6 `LogManager`（统一日志管理器）

定义于 `src/aireliable_engine/log_manager.py`，core 层用于管理多来源日志缓冲区与 SSE 订阅。

#### 9.6.1 主要 API

| 方法 | 说明 |
|---|---|
| `append(source, level, message, engine_id=None, task_id=None, extra=None) -> LogEntry` | 追加一条日志，自动分配 seq，推送到 SSE 订阅者 |
| `get_or_create(source) -> LogBuffer` | 获取或创建来源缓冲区 |
| `get_since(source=None, since=0, limit=500) -> tuple[list[LogEntry], int, list[str]]` | 增量查询：返回 (日志列表, last_seq, sources) |
| `subscribe(source=None) -> tuple[SSELogSubscriber, list[LogEntry]]` | 订阅 SSE 流，返回 (订阅器, 历史日志) |
| `unsubscribe(subscriber, source=None)` | 取消订阅 |

#### 9.6.2 使用示例（core 层）

```python
from aireliable_engine import LogManager

log_manager = LogManager(max_buffer_size=2000)

# 追加日志
entry = log_manager.append(
    source="engine:e_xxx",
    level="INFO",
    message="引擎就绪",
    engine_id="e_xxx",
)

# 增量查询
logs, last_seq, sources = log_manager.get_since(since=0, limit=200)

# SSE 订阅
subscriber, initial_logs = log_manager.subscribe(source="engine:e_xxx")
async for frame in subscriber.events():
    # frame 是 SSE 帧（"data: {...}\n\n"）
    yield frame
```

### 9.7 `parse_engine_stdout_line`（引擎 stdout 解析）

定义于 `src/aireliable_engine/log_manager.py`，core 用于把引擎子进程 stdout 行解析为 `LogEntry`。

#### 9.7.1 解析策略（按顺序尝试）

1. **JSON 格式**（v0.2.5+）：`{"timestamp":...,"level":...,"engine_id":...,"message":...}`
   - 必须含 `level` 和 `message` 字段
   - `engine_id` 缺失时用参数传入的 engine_id 补全
2. **人类可读格式**（v0.2.4 及更早）：`2026-07-24 15:30:01,234 - e_xxx - INFO - 消息`
   - 正则匹配 `<timestamp> - <engine_id> - <level> - <message>`
3. **兜底**：无法解析的行，作为 `level=INFO` 的原始消息

#### 9.7.2 签名

```python
def parse_engine_stdout_line(
    line: str,
    engine_id: str,
) -> Optional[LogEntry]
```

返回 `None` 表示空行或解析失败且无有效内容。

### 9.8 日志级别使用规范

详见 [开发指南 §9.4 日志级别使用规范](../../PR/development-guide.md#94-日志级别使用规范)。

| 级别 | 使用场景 |
|---|---|
| DEBUG | 详细内部状态，仅在排查问题时开启；生产环境默认关闭 |
| INFO | 正常运行里程碑：启动 / 就绪 / 加载完成 / 停止 / 配置加载 |
| WARNING | 非致命异常：显存不足回退 / 配置项缺失使用默认值 / 重试 / 降级 |
| ERROR | 单次操作失败（推理失败 / 加载失败 / 下载失败），但引擎仍可服务其他请求 |
| CRITICAL | 引擎不可用：模型损坏 / 端口冲突无法解决 / 必需依赖缺失 / 启动失败 |

### 9.9 core 自动采集机制

引擎子进程 **无需主动上报日志**，core 通过 `_collect_logs` 异步任务自动：

1. 读取子进程 stdout（`process.stdout.readline()`）
2. 调用 `parse_engine_stdout_line` 解析为 `LogEntry`
3. 追加到 core 的 `LogManager`（source = `engine:<engine_id>`）
4. 推送到所有 SSE 订阅者（前端 Logs 页面 + 引擎详情页终端）

> **v0.2.15+ 架构变更**：文件写入由引擎 SDK 自身负责（`setup_engine_logging` 内的
> `RotatingFileHandler` 写入 `logs/engines/<engine_id>.log`，路径由 `AIRELIABLE_LOG_DIR`
> 环境变量推导）。core `_collect_logs` **不再写文件**，仅读 stdout 解析为 `LogEntry`
> 推送 SSE，避免 core 与 SDK 双写同一文件导致 `RotatingFileHandler` 轮转冲突。
> 平台 `engine_manager._build_engine_subprocess_env` 会注入 `AIRELIABLE_LOG_DIR`
> 环境变量（值 = `{project_root}/logs/engines`）。

引擎开发者只需调用 `setup_engine_logging` 配置日志，无需关心上报逻辑。

---

## 十、引擎环境配置规范（v0.2.5+ 引入）

> **单一权威来源**：[docs/docs/architecture/engine-env-settings.md](../../docs/docs/architecture/engine-env-settings.md)
> **用户指南**：[docs/docs/how-to/engine-env-management.md](../../docs/docs/how-to/engine-env-management.md)
> **开发规范**：[PR/development-guide.md §37](../../PR/development-guide.md)

### 10.1 背景

v0.2.5+ 引入 **引擎环境设置面板**，让用户在引擎详情页直接管理每个引擎的 Python 运行环境
（`.venv`），包括初始化、重置、更新依赖、切换 torch CUDA 版本等操作。

引擎开发者需要在 `manifest.yaml` 中通过 `env_config` 块声明引擎的环境需求，平台会根据
这些声明决定：
- 是否检测 torch（`requires_torch`）
- 默认 CUDA 版本（`default_cuda_version`）
- 支持的 CUDA 版本列表（`supported_cuda_versions`）
- Python 版本约束（`requires_python`）
- 显示在 UI 底部的备注信息（`notes`）

### 10.2 env_config 字段规范

```yaml
env_config:
  requires_torch: true              # 是否依赖 torch（默认 true）
  torch_version_constraint: ">=2.3.0"  # torch 版本约束（PEP 440，可选）
  default_cuda_version: "cu128"     # 默认 CUDA 版本
  supported_cuda_versions:          # 支持的 CUDA 版本列表
    - "cu121"
    - "cu124"
    - "cu126"
    - "cu128"
  requires_python: ">=3.10"         # Python 版本约束（PEP 440，可选）
  notes: "备注信息，显示在环境设置面板底部"
```

**字段说明**：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `requires_torch` | bool | true | 引擎是否依赖 torch。false 时跳过 torch 检测 |
| `torch_version_constraint` | str? | null | torch 版本约束（PEP 440） |
| `default_cuda_version` | str | "cu128" | 默认 CUDA 版本标识 |
| `supported_cuda_versions` | list[str] | ["cu121","cu124","cu126","cu128"] | 支持的 CUDA 版本 |
| `requires_python` | str? | null | Python 版本约束（PEP 440） |
| `notes` | str | "" | 备注信息，显示在 UI 底部 |

### 10.3 各引擎 env_config 配置示例

#### 10.3.1 依赖 torch 的引擎（如 ERNIE-Image / VoxCPM）

```yaml
env_config:
  requires_torch: true
  torch_version_constraint: ">=2.3.0"
  default_cuda_version: "cu128"
  notes: "建议使用 CUDA 版本以获得 GPU 加速"
```

#### 10.3.2 不依赖 torch 的引擎（如 Llama.cpp）

```yaml
env_config:
  requires_torch: false
  notes: "Llama.cpp 是 C++ 二进制包装层，不依赖 torch"
```

### 10.4 环境设置面板的 UI 行为

环境设置面板会根据 `env_config` 自动调整 UI：

| 字段 | UI 行为 |
|---|---|
| `requires_torch: false` | 隐藏 torch 检测区域，不显示"切换到 GPU"按钮 |
| `requires_torch: true` | 显示 torch 版本和"切换到 GPU"按钮 |
| `notes` 非空 | 在环境操作区域底部显示备注信息 |
| `requires_python` 非空 | 在 Python 运行时区域显示版本约束 |

### 10.5 与设备选择规范（§7）的协同

环境设置面板是 §7 设备选择规范的 UI 入口之一：

1. 用户在环境设置面板点击"切换到 GPU"（torch CUDA 切换）
2. torch 切换到 CUDA 版，`torch.cuda.is_available()` 返回 True
3. 用户启动引擎时选择 `cuda` 设备（§7 device 参数透传）
4. 引擎使用 GPU 加速推理

**关键约束**：
- `requires_torch: false` 的引擎，§7 的 `cuda` 设备选项无意义（引擎不使用 torch）
- `requires_torch: true` 的引擎，必须先完成 torch CUDA 切换，§7 的 `cuda` 设备才能生效

### 10.6 开发期 SDK 依赖配置

开发期引擎 `pyproject.toml` 中的 SDK 依赖必须设置为本地 editable 模式：

```toml
[tool.uv.sources]
aireliable-engine-sdk = { path = "../../sdk/aireliable-engine-sdk", editable = true }
```

**原因**：
- editable 模式确保 SDK 源码修改即时反映到引擎 `.venv`
- 避免 git 依赖覆盖本地修改
- 环境设置面板会显示 SDK 模式为 `editable (本地源码)`，便于开发者确认

**禁止**：开发期使用 git 依赖方式（`{ git = "..." }`），会导致本地修改被覆盖。

### 10.7 相关文档

- [引擎环境设置面板架构](../../docs/docs/architecture/engine-env-settings.md)
- [管理引擎环境用户指南](../../docs/docs/how-to/engine-env-management.md)
- [torch CUDA 版本切换架构](../../docs/docs/architecture/torch-cuda-switch.md)
- [切换到 GPU 用户指南](../../docs/docs/how-to/torch-cuda.md)
- [开发指南 §37 引擎环境设置面板开发规范](../../PR/development-guide.md)

---

## 十一、engine_id 命名规范（v0.2.9+ 引入）

> **单一权威来源**：[PR/development-guide.md §四十 创建引擎前检测规范](../../PR/development-guide.md)
> **配套架构**：[PR/architecture.md §3.26 engine_id 三同名契约架构](../../PR/architecture.md) + [§3.28 engine_id 命名规范升级（v0.2.11+）](../../PR/architecture.md)
> **配套需求**：[PR/requirements.md §二十七 engine_id 命名规范需求](../../PR/requirements.md)
> **分析文档**：[engine-naming-convention-analysis.md](../../engine-naming-convention-analysis.md) + [engine-id-rename-and-model-detection-analysis.md](../../engine-id-rename-and-model-detection-analysis.md)

> **v0.2.11+ 推荐格式升级**：新引擎推荐使用 `<engine_name_slug>_<hex8>` 格式（如 `ernie_image_909469ed` / `llama_cpp_9cdb76f8` / `voxcpm_tts_c4a8f2e7`），提升目录可读性。旧 `e_<hex12>` 格式仍合法，但不再推荐。详见 [§3.28](../../PR/architecture.md)。

### 11.1 三同名契约（强制）

**目录名 = engine_id = Python 模块名**，三者必须字符完全相同。

```
engines/<engine_id>/                 ← 目录名
├── manifest.yaml                    ← engine_id 字段
├── <engine_id>/                     ← Python 包名（必须与目录名相同）
│   ├── __init__.py
│   ├── server.py                    ← startup_cmd: python -m engines.<engine_id>.server
│   └── ...
└── pyproject.toml
```

### 11.2 字符集规范（强制）

engine_id 必须匹配正则：

```
^[a-z][a-z0-9_]{1,63}$
```

- 必须以小写字母开头
- 仅含小写字母 / 数字 / 下划线
- 长度 2-64
- **禁止连字符 `-`**（避免与 Python 模块名冲突）
- 必须不在保留字黑名单内（见 `core/engine_id_generator.RESERVED_WORDS`，含 38 个保留字如 `engines` / `config` / `envs` / `models` / `new` / `default` / `self` / `class` 等）

### 11.3 三层检测点架构

平台在三个入口强制检测 engine_id 合规性，违规时立即拦截：

| 层级 | 检测点 | 时机 | 行为 |
|------|--------|------|------|
| L1 | `EngineScaffolder` | 创建新引擎时 | 拒绝生成违规 engine_id |
| L2 | `EngineImporter` | 导入 .airengine 包时 | 拒绝导入违规 engine_id |
| L3 | `registry_loader` | 加载引擎注册表时 | 警告并跳过违规引擎 |

### 11.4 与 Python 模块导入的协同

由于 `startup_cmd` 使用 `python -m engines.<engine_id>.server` 形式启动引擎，engine_id 必须是合法的 Python 标识符：

- ✅ `foleycrafter_a3b2c1`（下划线，合法 Python 包名）
- ❌ `foleycrafter-a3b2c1`（连字符，非法 Python 包名，`import` 会失败）
- ❌ `1engine`（数字开头，非法 Python 包名）
- ❌ `engines`（保留字，与项目目录冲突）

### 11.5 引擎开发者自查清单

创建新引擎前，开发者必须确认：

- [ ] engine_id 匹配 `^[a-z][a-z0-9_]{1,63}$` 正则
- [ ] engine_id 不在保留字黑名单内
- [ ] 引擎目录名 `engines/<engine_id>/` 与 engine_id 完全相同
- [ ] Python 包名 `<engine_id>/` 与 engine_id 完全相同
- [ ] `manifest.yaml` 的 `engine_id` 字段与目录名一致
- [ ] `config/engines/<engine_id>.yaml` 文件名与 engine_id 一致
- [ ] `startup_cmd` 使用 `python -m engines.<engine_id>.server` 形式
- [ ] `pyproject.toml` 的 `name` 字段建议与 engine_id 一致（但允许 `-` 分隔，因为是包名而非模块名）

### 11.6 历史违规迁移案例（v0.2.9）

**FoleyCrafter 引擎迁移**（v0.2.6 → v0.2.9）：

- 迁移前：目录名 `foleycrafter_a3b2c1`，engine_id `foleycrafter-a3b2c1`（连字符违规）
- 迁移后：目录名 `foleycrafter_a3b2c1`，engine_id `foleycrafter_a3b2c1`（下划线合规）
- 影响：导致 registry_loader 找不到 manifest.yaml，前端显示"引擎未声明 model_files"
- 教训：双命名空间分裂（目录名 vs engine_id）会引发加载失败，必须严格遵循三同名契约

### 11.7 相关文档

- [架构文档 §3.26 engine_id 三同名契约架构](../../PR/architecture.md)
- [开发指南 §17.5 命名约定](../../PR/development-guide.md)
- [开发指南 §四十 创建引擎前检测规范](../../PR/development-guide.md)
- [需求文档 §二十七 engine_id 命名规范需求](../../PR/requirements.md)
- [引擎命名规范分析](../../engine-naming-convention-analysis.md)
- [引擎命名规范任务表](../../engine-naming-convention-tasks.md)

---

## 十二、variant 检测 API（v0.2.11+ 方案 B 引入）

> **配套架构**：[PR/architecture.md §3.29 模型检测架构统一（方案 B）](../../PR/architecture.md)
> **配套需求**：[PR/requirements.md §二十九 模型检测架构统一需求](../../PR/requirements.md)
> **配套规范**：[PR/development-guide.md §四十二 模型检测架构统一开发规范](../../PR/development-guide.md)
> **分析文档**：[engine-id-rename-and-model-detection-analysis.md](../../engine-id-rename-and-model-detection-analysis.md) §四

### 12.1 设计目标

引擎负责**注册**（manifest 声明 variant 元信息），平台负责**检索**（检测 variant 可用性）。引擎 `model_scanner` 不再自己扫描文件系统，而是调用平台 API 获取 variant 可用性。

### 12.2 BaseEngine.get_variants()

```python
async def get_variants(self) -> Optional[list[dict]]:
    """调用平台 API 获取引擎 variant 可用性。

    Returns:
        variant 列表；平台不可达时返回 None（调用方应 fallback 到本地扫描）
    """
```

**返回结构**（每个 dict）：

```python
{
    "name": "foleycrafter-v1",
    "alias": "FoleyCrafter-v1",
    "base_model": "auffusion-full-no-adapter",
    "precision": "fp16",
    "loader": "foleycrafter",
    "vram_required": "10GB",
    "available": True,                # 所有关联文件 ready 且位置正确
    "model_file_ids": ["fc-temporal-adapter", "fc-semantic-adapter", ...],
    "missing_file_ids": []            # available=False 时填充缺失文件 ID
}
```

### 12.3 平台 API

```
GET /engines/{engine_id}/variants
```

返回：

```json
{
    "engine_id": "foleycrafter_a3b2c1",
    "all_ready": true,
    "variants": [...]
}
```

### 12.4 引擎调用示例（声明式引擎）

```python
from aireliable_engine.engine_base import BaseEngine

class ModelScanner:
    def __init__(self, model_dir: Path, engine: BaseEngine):
        self.model_dir = model_dir
        self.engine = engine

    async def scan_variants(self) -> list[VariantInfo]:
        """方案 B：调用平台 API + 从 manifest 读引擎特有字段。"""
        # 1. 调用平台 API 获取 variant 可用性
        platform_variants = await self.engine.get_variants()

        # 2. 平台不可达 → fallback 到本地扫描（保留兼容性）
        if platform_variants is None:
            return self._scan_locally()

        # 3. 从 manifest 读取引擎特有字段（如 default_steps）
        manifest_variants = self._load_manifest_variants()

        # 4. 合并：平台判定 available + 引擎补充特有字段
        result = []
        for pv in platform_variants:
            mv = manifest_variants.get(pv["name"], {})
            result.append(VariantInfo(
                name=pv["name"],
                alias=pv["alias"],
                available=pv["available"],
                default_steps=mv.get("default_steps", 25),  # 引擎特有字段
                # ...
            ))
        return result
```

### 12.5 适用范围

| 引擎类型 | 是否适合方案 B | 说明 |
|---------|--------------|------|
| 声明式（FoleyCrafter / VoxCPM-TTS） | ✅ 适合 | variant 列表固定，可声明 |
| 混合式（Llama.cpp） | ⚠️ 部分适合 | 声明模式可改造，fallback 扫描保留 |
| 动态发现（ERNIE-Image） | ❌ 不适合 | variant 由 .gguf 文件名动态解析 |

### 12.6 环境变量

平台启动引擎子进程时注入 `AIRELIABLE_CORE_URL`（如 `http://127.0.0.1:8001`）。`BaseEngine._get_platform_core_url()` 读取此变量。

### 12.7 相关文档

- [架构文档 §3.29 模型检测架构统一（方案 B）](../../PR/architecture.md)
- [开发指南 §四十二 模型检测架构统一开发规范](../../PR/development-guide.md)
- [需求文档 §二十九 模型检测架构统一需求](../../PR/requirements.md)

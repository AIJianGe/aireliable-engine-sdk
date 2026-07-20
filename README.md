# AIReliable Engine SDK

> 版本：0.2.0
> 状态：开发中
> 配套文档：项目根目录 `v0.2-analysis.md` / `v0.2-tasks.md` / `PR/development-guide.md`

AIReliable 引擎开发工具包。引擎开发者只需 `pip install aireliable-engine-sdk` 即可开发可插拔引擎，无需 clone 主仓。

> 仓库地址：https://github.com/AIJianGe/aireliable-engine-sdk

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

### 2.1 范式 A：一体化引擎

最简单的引擎，`engine.py` 直接实现 8 端点业务逻辑。

```python
from aireliable_engine import BaseEngine, InferRequest, SSEStream, make_frame

class MyEngine(BaseEngine):
    def __init__(self):
        super().__init__(
            engine_id="my_engine",
            engine_name="My Engine",
            capability="text-to-image",
            engine_type="image",
        )

    async def _capabilities_impl(self):
        ...

    async def _load_impl(self, variant):
        ...

    async def _unload_impl(self):
        ...

    async def _infer_impl(self, request: InferRequest, stream: SSEStream):
        for step in range(50):
            await stream.put(make_frame(progress=step/50, stage="denoising", step=step))
        await stream.put(make_frame(progress=1.0, stage="completed",
                                     result={"image_urls": ["http://..."]}))

    async def _status_impl(self):
        ...

    async def _load_lora_impl(self, name, weight):
        ...

    async def _unload_lora_impl(self, name):
        ...
```

### 2.2 范式 B：工作流引擎（v0.2 完成后可用）

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
    def _build_workflow(self, variant):
        return load_yaml("workflows/default.yaml")
```

### 2.3 范式 C：自定义引擎

完全自由实现，仅遵守 8 端点 HTTP 契约（继承 `BaseEngine` 或直接用 FastAPI）。

---

## 三、SDK 模块导出

| 模块 | 主要导出 | 用途 |
|---|---|---|
| `engine_base` | `BaseEngine`, `EngineBusyError`, ... | 引擎抽象基类 |
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
| `workflow_engine` | `WorkflowEngine` | 工作流引擎基类（v0.2） |
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
│       ├── engine_base.py          # BaseEngine（范式 A）
│       ├── workflow_engine.py      # WorkflowEngine（范式 B）
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
| 0.2.0 | v0.2 初始版本：抽取自 `engines/base/`，新增 WorkflowEngine / BaseNode / NodeRegistry / WorkflowExecutor / 安全条件 DSL / 平台基础节点库 |

---

## 七、相关文档

- 完整架构：`PR/architecture.md`
- 开发规范：`PR/development-guide.md`
- v0.2 分析：`v0.2-analysis.md`
- v0.2 任务：`v0.2-tasks.md`

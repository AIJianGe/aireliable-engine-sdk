# AIReliable Engine SDK

> Build pluggable AIReliable engines with a pip-installable SDK — no platform source code required.
> 用 pip 安装的 SDK 开发 AIReliable 可插拔引擎，无需平台源码。

> 版本：0.3.0 ｜ Python：3.10 / 3.11 ｜ 许可证：Apache-2.0

---

## 一、许可证边界（先读这一段）

| 对象 | 许可证 | 说明 |
|---|---|---|
| **本 SDK**（`aireliable-engine-sdk`） | **Apache-2.0** | 可自由用于开发、分发引擎，包括商业引擎 |
| **平台客户端**（AIReliable） | 闭源 | 由官方分发，不提供源码 |
| **各官方引擎** | source-available | 源码可见，商用需授权；不作为 SDK 的一部分授权 |

SDK 只是开发工具，不包含平台实现，也不授予平台客户端的分发权。

---

## 二、安装

```bash
# 最小核心（契约 + SSE + 状态机 + 工作流 + 下载器）
pip install aireliable-engine-sdk

# 按引擎类型选装
pip install "aireliable-engine-sdk[image]"    # torch / diffusers / transformers / safetensors
pip install "aireliable-engine-sdk[llm]"
pip install "aireliable-engine-sdk[video]"
pip install "aireliable-engine-sdk[voice]"
pip install "aireliable-engine-sdk[gradio]"   # 需要自带 Web UI 时

# 开发测试
pip install "aireliable-engine-sdk[dev]"
```

尚未发布到 PyPI，当前用 GitHub Release 安装：

```bash
# 直接装 Release 里的 wheel
pip install https://github.com/AIJianGe/aireliable-engine-sdk/releases/download/v0.3.0/aireliable_engine_sdk-0.3.0-py3-none-any.whl

# 或从标签源码安装
pip install git+https://github.com/AIJianGe/aireliable-engine-sdk.git@v0.3.0
```

---

## 三、SDK 提供什么

| 能力 | 入口 | 用途 |
|---|---|---|
| 引擎基类 | `WorkflowEngine` | **推荐**：工作流声明式引擎，只需实现 2 个方法 |
| 引擎基类 | `CustomEngine` | 固定流程/封装式引擎（TTS、Chat 等），完全自控推理 |
| 内部基类 | `BaseEngine` | 8 端点契约实现，**引擎不应直接继承** |
| HTTP 契约 | `create_engine_app` | 一行挂载 `/health` `/load` `/infer` 等 8 端点 + 工作流/节点路由 |
| 数据结构 | `InferRequest` / `HealthResponse` / `Image` / `Audio` / `Video` … | 8 端点与节点类型 |
| 流式输出 | `SSEStream` / `make_frame` | 进度与结果推送 |
| 状态机 | `RuntimeState` / `StateMachine` | 6 态运行状态 |
| 节点体系 | `BaseNode` / `NodeRegistry` / `register_node` | 可复用节点与自定义节点 |
| 工作流 | `WorkflowExecutor` / `workflow_schema` / `workflow_dsl` | yaml 执行、校验、安全条件表达式 |
| 日志 | `setup_engine_logging` | stdout JSON + 滚动文件，平台自动采集 |
| 工具 | 下载器 / 镜像选择 / 显存管理 / 任务上报 / 音色库客户端 | 通用能力 |

完整导出清单见 `src/aireliable_engine/__init__.py` 的 `__all__`。

---

## 四、最小工作流引擎

### 4.1 `manifest.yaml`

```yaml
manifest_schema_version: "0.4"
engine_id: my_engine
engine_name: My Engine
engine_version: "1.0.0"
aireliable_compat: ">=0.3.0"
uv_env_name: "my-engine"
export_mode: source

paradigm: workflow
engine_uid: "aireliable://your-name/misc/my_engine"
sdk_version: ">=0.3.0"
engine_api_version: "1"

entry: my_engine.server          # 平台用 python -m my_engine.server --port N 启动

license: "Apache-2.0"
repository: "https://github.com/your-name/my_engine"

models:
  model_layout:
    base: "misc/my_engine/"
  default_variant: default
  variants:
    - name: default
      alias: "Default"
      precision: fp32
      path: "misc/my_engine/default"
      model_file_ids: []
  model_files: []
```

> `entry` 为可选字段；缺省时平台回退 `engines.<engine_id>.server`（仅为兼容旧引擎）。

### 4.2 `engine.py`

```python
from pathlib import Path

from aireliable_engine import WorkflowEngine


class MyEngine(WorkflowEngine):
    def __init__(self) -> None:
        super().__init__(
            engine_id="my_engine",
            engine_name="My Engine",
            capability="text-to-image",
            engine_type="image",
            workflows_dir=Path(__file__).parent / "workflows",
            models_root=Path("models"),
        )

    async def _build_workflow(self, name=None) -> str:
        return self._read_workflow_file(name)

    def _get_default_workflow_template(self) -> str:
        return "name: default\nnodes: []\n"

    async def _capabilities_impl(self):
        return {
            "variants": ["default"],
            "loras": [],
            "stream_modes": ["sse", "json", "async"],
        }
```

### 4.3 `server.py`

```python
import argparse
from pathlib import Path

import uvicorn

import aireliable_engine.nodes  # noqa: F401  注册内置节点
from aireliable_engine import create_engine_app, setup_engine_logging

from .engine import MyEngine

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7100)
    args = parser.parse_args()

    setup_engine_logging(engine_id="my_engine")
    app = create_engine_app(MyEngine(), workflows_dir=WORKFLOWS_DIR)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

### 4.4 本地起服务

```bash
python -m my_engine.server --port 7100

curl http://127.0.0.1:7100/health
curl -X POST http://127.0.0.1:7100/load -H "Content-Type: application/json" -d '{"variant":"default"}'
curl -X POST http://127.0.0.1:7100/infer -H "Content-Type: application/json" -d '{"variant":"default","inputs":{"value":"hello"}}'
```

---

## 五、约定（必读）

1. **engine_id**：`^[a-z][a-z0-9_]{1,63}$`，小写字母开头，仅小写字母/数字/下划线，禁止连字符。
2. **manifest**：`manifest.yaml` 放在引擎根目录，字段变更需同步 `manifest_schema_version`。
3. **入口**：`entry` 指向可 `python -m` 执行的模块，模块内必须解析 `--port`。
4. **日志**：启动时调用 `setup_engine_logging(engine_id=...)`，平台通过 stdout 采集。
5. **不依赖平台源码**：只 `import aireliable_engine`，不要 import 平台 `core` / `gateway`。

---

## 六、文档

完整文档集在 [docs/README.md](docs/README.md)，按需查阅：

| 文档 | 内容 |
|---|---|
| [manifest.yaml](docs/manifest.md) | 全字段、校验规则、models 块 |
| [HTTP 契约](docs/http-contract.md) | 8 端点、SSE 帧、错误格式、版本握手 |
| [工作流](docs/workflow.md) | YAML 结构、引用语法、条件 DSL、自定义节点 |
| [打包与分发](docs/package.md) | 包布局、`pack`、内容哈希、导入与来源记录 |
| [常见问题](docs/faq.md) | 启动、校验、推理、兼容性报错 |

## 七、开发与测试

```bash
git clone https://github.com/AIJianGe/aireliable-engine-sdk.git
cd aireliable-engine-sdk && pip install -e ".[dev]"
pytest tests/ -v && ruff check src/ tests/
```

---

## 八、版本与兼容

| SDK 版本 | 主要变更 |
|---|---|
| 0.3.0 | 独立仓库发布；manifest 契约与 CLI 逐步下沉（详见发布说明） |
| 0.2.5 | 节点体系增强（`params_schema.format`、本地目录发现、默认工作流模板） |
| 0.2.0 | 工作流范式：`WorkflowEngine` / `BaseNode` / `NodeRegistry` / `WorkflowExecutor` / 安全 DSL / 基础节点库 |

- SDK 版本走 semver；`engine_api_version`（当前 `1`，见 `/health`）是独立契约版本，两者不混用。
- 平台保证：同一 `engine_api_version` 内，8 端点契约向后兼容。

---

## 九、反馈

- 问题与需求：GitHub Issues（`AIJianGe/aireliable-engine-sdk`）
- 引擎开发文档：`docs/`（manifest / HTTP 契约 / 工作流 / 打包 / FAQ）

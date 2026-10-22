# AIReliable Engine SDK 文档

面向第三方引擎开发者的自包含文档集。**只依赖本 SDK，不需要 AIReliable 平台源码。**

> 平台客户端闭源分发；SDK 与模板仓为 Apache-2.0。引擎通过 HTTP 契约与平台进程隔离运行。

## 阅读路径

| 你是 | 建议顺序 |
|---|---|
| 第一次接触 | 本仓 `README.md` → [manifest.md](manifest.md) → [workflow.md](workflow.md) → [http-contract.md](http-contract.md) |
| 要改已有引擎 | [manifest.md](manifest.md) → [workflow.md](workflow.md) → [faq.md](faq.md) |
| 要发布/分发引擎 | [package.md](package.md) → [manifest.md](manifest.md) 的 `package` 段 |
| 联调报错 | [faq.md](faq.md) → [http-contract.md](http-contract.md) |

## 文档清单与行数（实测）

| 文档 | 内容 | 行数 |
|---|---|---|
| `README.md`（仓根） | 安装、最小引擎、CLI 四命令 | 155 |
| [manifest.md](manifest.md) | `manifest.yaml` 全字段与校验规则 | 140 |
| [http-contract.md](http-contract.md) | 8 端点契约、SSE、错误格式、版本握手 | 158 |
| [workflow.md](workflow.md) | 工作流 YAML、节点、条件 DSL、自定义节点 | 115 |
| [package.md](package.md) | 包布局、`pack`、哈希、导入与来源记录 | 86 |
| [faq.md](faq.md) | 常见报错与排查 | 59 |

合计 **713 行**（非空行口径），满足 5+1 篇 ≤ 800 行的预算。

## 约定

- 所有命令在引擎仓根目录执行。
- `engine_id` 必须匹配 `^[a-z][a-z0-9_]{1,63}$`，且**目录名 = engine_id = Python 模块名**（三同名契约）。
- 平台以独立子进程启动引擎：`python -m <manifest.entry> --port <N>`；未声明 `entry` 时回退 `engines.<engine_id>.server`（仅兼容旧引擎）。
- 文档中的 JSON 均为 UTF-8 明文；二进制输入用 `b64_json` 或本地 `path`，不建议把大文件塞进 base64。

## 版本

- 文档对应 SDK `0.3.0`、`manifest_schema_version: "0.4"`、`engine_api_version: "1"`。
- 契约破坏性变更才会提升 `engine_api_version`；平台启动握手会校验 `/health.engine_api_version`。

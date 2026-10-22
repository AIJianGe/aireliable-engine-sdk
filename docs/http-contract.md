# HTTP 契约（engine_api_version = "1"）

引擎是独立进程。平台用 `python -m <entry> --port <N>` 启动它，然后通过
`http://127.0.0.1:<N>` 调用下列端点。**网络边界只有这条本地回环，不需要公网。**

- 编码：UTF-8 JSON；`Content-Type: application/json`
- 流式：`POST /infer` 返回 `text/event-stream`（SSE）
- 鉴权：本地模式默认无鉴权；远程模式下配置 `auth_token`，除 `/health`、`/capabilities`
  外均需 `Authorization: <auth_token>`（例如 `Bearer xxx`）

## 版本握手

平台启动引擎后先请求 `GET /health`，读取 `engine_api_version`：

- 在平台支持集合内（当前 `"1"`）→ 接管
- 不在集合内 → 拒绝接管并在启动日志中给出明确错误

破坏性变更才提升该版本；新增可选字段不提升。

## 1. GET /health

```json
{"status":"ok","engine_id":"my_engine","state":"idle","uptime":12.3,"engine_api_version":"1"}
```

| 字段 | 说明 |
|---|---|
| `status` | `ok` / `error` |
| `engine_id` | 必须与 manifest 一致 |
| `state` | `idle` / `loading` / `ready` / `inferring` / `error` |
| `uptime` | 进程运行秒数 |
| `engine_api_version` | HTTP 契约版本，默认 `"1"` |

## 2. GET /capabilities

```json
{
  "engine_id": "my_engine",
  "engine_name": "My Engine",
  "capability": "text-to-image",
  "type": "image",
  "variants": [
    {"name":"q4","base_model":"my-base","precision":"q4_k_m","loader":"gguf",
     "vram_required":"8GB","default_steps":20,"default_guidance":7.0,
     "supports_pe":false,"available":true}
  ],
  "loras": [],
  "supported_sizes": ["1024x1024"],
  "default_variant": "q4",
  "capabilities": ["text-to-image"],
  "input_schema": null
}
```

- `capability`：主能力（单数，旧客户端读这个）
- `capabilities`：多能力列表（新客户端优先读）
- `type`：`image` / `voice` / `video` / `threed` / `llm` 等
- `available`：模型文件是否就绪；`false` 时平台会提示缺少模型
- `input_schema`：可选，引擎推理输入的 JSON Schema（供 MCP/表单生成）

## 3. POST /load

请求：

```json
{"variant": "q4"}
```

响应：

```json
{"variant":"q4","state":"ready","vram_used":"7.2GB","load_time":18.6}
```

- `variant` 可为 `null`：按 `manifest.models.default_variant` 或第一个可用变体决定。
- 幂等：同变体已就绪直接返回（`load_time: 0`）；不同变体先卸载再加载。
- 模型文件缺失返回 `404`；参数非法返回 `400`；其它失败 `500`。

## 4. POST /unload

```json
{"state":"idle","vram_freed":"7.2GB"}
```

无模型加载时幂等返回 `{"state":"idle","vram_freed":"0GB"}`。

## 5. POST /infer（SSE）

请求（网关→引擎内部协议，非 OpenAI 格式）：

```json
{
  "variant": "q4",
  "inputs": {"prompt": "a cat", "steps": 20, "size": "1024x1024"},
  "loras": [{"name":"style","weight":0.8}],
  "lora_explicit": false,
  "images": [{"path":"/abs/in.png","role":"source"}],
  "audios": [{"url":"https://.../ref.wav","role":"speaker"}],
  "videos": [{"path":"/abs/in.mp4","role":"target"}],
  "workflow": "default",
  "task_id": "task-123"
}
```

| 字段 | 说明 |
|---|---|
| `inputs` | 引擎自定义参数；无统一 schema，由引擎文档约定 |
| `images/audios/videos` | 多模态输入，元素支持 `url` / `b64_json` / `path`；`role` 区分用途（`source` / `mask` / `reference` / `control` / `first_frame` / `last_frame` / `speaker` / `target`） |
| `workflow` | 指定工作流名；`null` 用默认 |
| `lora_explicit` | `true` 表示请求方显式指定 LoRA 集合（空数组 = 清空）；`false` 保持已挂载集合 |
| `task_id` | 任务追踪 ID，会写入日志上下文 |

响应为 SSE，每帧 `data: {json}\n\n`：

```text
data: {"progress":0.1,"stage":"loading_model","message":"加载 q4"}
data: {"progress":0.45,"stage":"denoising","step":9,"total":20}
data: {"progress":1.0,"stage":"completed","result":{"images":["/abs/out.png"]},"timing":{"total":6.2}}
```

错误帧（HTTP 仍为 200，错误在流内）：

```text
data: {"progress":0.0,"stage":"error","error":{"type":"engine_busy","message":"引擎正在推理中，请稍后"}}
```

- 引擎忙 / 加载中：`stage=error`，`type=engine_busy` / `model_loading`
- 推理异常：`type=engine_error`
- 客户端断开时引擎应停止后续计算（SDK 默认处理）

## 6. GET /status

```json
{
  "state":"ready",
  "loaded_variant":"q4",
  "vram_used":"7.2GB",
  "vram_total":"24GB",
  "loaded_loras":["style"],
  "last_used_at":1740000000.0,
  "uptime":300.5,
  "extra":{}
}
```

`extra` 用于引擎私有状态（平台原样透传）。

## 7. POST /lora/load

```json
{"name":"style","weight":0.8}
```

响应：

```json
{"name":"style","weight":0.8,"loaded_loras":["style"]}
```

## 8. POST /lora/unload

```json
{"name":"style"}
```

响应：

```json
{"name":"style","loaded_loras":[]}
```

## 错误格式

非 SSE 端点的错误遵循 FastAPI 默认：

```json
{"detail":"模型文件不存在: ..."}
```

引擎内部统一错误对象（`ErrorResponse`）用于需要结构化错误的场景：

```json
{"error":{"type":"engine_error","message":"...","code":500}}
```

## 可选管理端点

`create_engine_app` 还会挂载以下只读/管理接口，平台 WebUI 使用：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/workflows` | 列出工作流 |
| GET/PUT/DELETE | `/workflows/{name}` | 读取 / 保存 / 删除工作流 |
| POST | `/workflows/{name}/reload` | 清缓存重载 |
| GET | `/workflows/{name}/validate` | 校验工作流合法性 |
| GET | `/nodes` | 列出已注册节点 |
| GET | `/nodes/definitions` | 节点定义（端口/参数 schema） |
| GET | `/nodes/{node_type}` | 单个节点定义 |
| GET/POST | `/log_level` | 查询 / 动态设置日志级别 |

## 最小联调脚本

```bash
curl http://127.0.0.1:7100/health
curl http://127.0.0.1:7100/capabilities
curl -X POST http://127.0.0.1:7100/load -H "Content-Type: application/json" -d '{"variant":"default"}'
curl -N -X POST http://127.0.0.1:7100/infer -H "Content-Type: application/json" \
  -d '{"variant":"default","inputs":{"value":"hello"}}'
```

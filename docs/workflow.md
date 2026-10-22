# 工作流（workflow YAML）

`paradigm: workflow` 的引擎用 YAML 声明推理流程，SDK 负责调度、进度上报、取消与输出收集。
工作流放在引擎仓 `workflows/<name>.yaml`，默认工作流建议命名为 `default.yaml`。

## 最小工作流

```yaml
name: default
description: 把输入文本原样返回

nodes:
  - id: text
    type: TextInput
    params:
      value: ""
  - id: save
    type: TextOutput          # 引擎自定义节点（见下文）
    inputs:
      text: text.text

outputs:
  text: text.text
```

- `name`：工作流唯一名，`POST /infer` 的 `workflow` 字段引用它
- `nodes[].id`：工作流内唯一，必须以字母/下划线开头，仅含字母数字下划线
- `nodes[].type`：节点注册表的 key（内置节点见下表）
- `nodes[].params`：节点参数，直接写在节点上（画布可编辑）
- `nodes[].inputs`：端口连接；值可以是引用字符串或字面量
- `outputs`：工作流最终输出，键名即 `/infer` 结果字段名

## 引用语法

| 写法 | 含义 |
|---|---|
| `text.text` | 引用上游节点 `text` 的输出端口 `text` |
| `"${variant}"` | 注入当前加载的变体名 |
| `"${inputs.prompt}"` | 注入 `InferRequest.inputs["prompt"]` |
| `"${loras}"` | 注入 LoRA 列表 |
| `"${steps}"` | 注入顶层请求字段（如 `steps`） |
| `1024` / `true` / `[1,2]` | 非字符串字面量原样传递 |

> 引用失败（端口不存在、类型不匹配）会在执行前构图阶段报错，并出现在 `/infer` 错误帧中。

## 条件执行与分支

节点级条件：

```yaml
  - id: upscale
    type: Upscale
    condition: '${inputs.enable_upscale} == true and len(${loras}) > 0'
```

分支（schema 已支持，执行语义按引擎实现）：

```yaml
branches:
  - name: fast_path
    condition: '${variant}.contains("turbo")'
    nodes:
      - id: quick
        type: Sampler
        params: {steps: 8}
    outputs:
      images: quick.images
```

条件 DSL 白名单：

- 比较：`== != > >= < <=`
- 逻辑：`and or not`
- 变量：`${path.to.value}`
- 函数：`len / exists / not_exists / contains / startswith / endswith / str / int`
- 方法：`.startswith() / .endswith() / .contains()`

不允许任意 Python 代码、属性访问或副作用调用。

## 内置节点

导入 `aireliable_engine.nodes` 即自动注册：

| 节点 | 分类 | 主要参数 | 输出 |
|---|---|---|---|
| `TextInput` | input | `value`（画布内文本；API 用 `inputs.value` 注入） | `text` |
| `ImagesInput` | input | `image_paths`（画布调试用；API 用 `inputs.images`） | `images`（variadic） |
| `ModelLoader` | loader | `architecture` / `variant` / `precision` / `device` | `model` / `vae` / `clip` / `scheduler`（后三者可选） |
| `VAELoader` | loader | `vae_name` / `source` / `dtype` | `vae` |
| `UNetLoader` | loader | `unet_name` / `dtype` | `unet` |
| `CLIPLoader` | loader | `clip_name` / `type` / `dtype` | `clip` |
| `CLIPTextEncode` | encode | 输入 `text` / `negative_text` / `clip`；参数 `max_length` | `embeds` / `negative_embeds` |
| `VAEEncode` | encode | 输入 `images` / `vae` | `latent` |
| `Sampler` | sample | 输入 `model` / `embeds` / `latent`；参数 `steps` / `guidance` / `size` / `seed` | `latent` |
| `EmptyLatentImage` | sample | `width` / `height` / `batch_size` | `latent` |
| `ModelSamplingAuraFlow` | sample | `shift` | `model` |
| `VAEDecode` | decode | 输入 `latent` / `vae` | `images` |
| `LoRALoader` | lora | `lora_name` / `strength` / `strength_clip` | `model` |
| `LoRALoaderAdvanced` | lora | `lora_name` / `strength_model` / `strength_clip` | `model` / `clip` |
| `ImageSave` | postprocess | 输入 `images`；参数 `format` / `filename_template` / `output_dir` / `quality` | `image_urls` |

## 自定义节点

```python
from aireliable_engine.node import BaseNode, PortSpec
from aireliable_engine.node_registry import register_node
from aireliable_engine.workflow_context import WorkflowContext


@register_node
class MyTextOutputNode(BaseNode):
    node_type = "MyTextOutput"
    title = "文本输出"
    category = "postprocess"
    params_schema = {
        "prefix": {"type": "string", "required": False, "default": ""},
    }
    input_specs = [PortSpec(name="text", type="text")]
    output_specs = [PortSpec(name="text", type="text")]

    async def execute(self, inputs: dict, ctx: WorkflowContext) -> dict:
        value = f"{self.get_param('prefix', '')}{inputs.get('text', '')}"
        await ctx.report_progress(1.0, stage="done", message="完成")
        return {"text": value}
```

约定：

- `node_type` 全局唯一；与内置节点同名会覆盖内置实现（谨慎）
- `execute` 必须是 `async def`，返回 `{端口名: 值}`
- 长计算请周期性 `ctx.raise_if_cancelled()`，并用 `ctx.report_progress()` 上报进度
- 节点模块必须在引擎启动时被导入（通常在 `server.py` 里 `import my_engine.nodes`）

## 校验与调试

```bash
aireliable-engine validate            # 校验 workflows/*.yaml 语法、节点是否注册、引用是否指向已有输出
aireliable-engine run --port 7100     # 起服务后用 /workflows/{name}/validate 复验
```

`validate` 会检查：

1. YAML 可解析且符合 schema（未知字段报错）
2. 每个 `nodes[].type` 已注册
3. `inputs` 引用形如 `<node_id>.<port>` 时，上游节点与端口存在
4. `outputs` 引用有效

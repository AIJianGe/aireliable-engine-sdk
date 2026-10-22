# manifest.yaml 字段说明

`manifest.yaml` 是引擎的**唯一身份与契约声明**。平台导入、启动、模型校验都以它为准。
SDK 与平台共用同一份 Pydantic 模型（`aireliable_engine.manifest.ManifestSchema`），
本地 `validate` 与平台导入的校验规则完全一致。

```yaml
manifest_schema_version: "0.4"
engine_id: my_engine
engine_name: My Engine
engine_version: "0.1.0"
aireliable_compat: ">=0.3.0"
uv_env_name: "misc-my-engine"
export_mode: source
paradigm: workflow
engine_uid: "aireliable://your-name/misc/my_engine"
engine_api_version: "1"
entry: my_engine.server
license: "Apache-2.0"
repository: "https://github.com/your-name/my_engine"
author:
  name: "Your Name"
models:
  model_layout:
    base: "misc/my_engine/"
  default_variant: default
  variants:
    - name: default
      precision: fp32
      path: "misc/my_engine/default"
      model_file_ids: []
  model_files: []
```

## 一、必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `engine_id` | str | 本地唯一 ID，必须匹配 `^[a-z][a-z0-9_]{1,63}$`（小写字母开头，仅小写字母/数字/下划线，2–64 字符） |
| `engine_name` | str | 显示名（UI 与日志使用） |
| `engine_version` | str | 语义化版本，如 `0.1.0` |
| `aireliable_compat` | str | 兼容的平台版本约束，如 `>=0.3.0`、`>=0.3.0,<0.5` |
| `uv_env_name` | str | 运行环境目录名（平台放在 `envs/<uv_env_name>/`），同时用于镜像名 |
| `export_mode` | str | `source`（分发源码+pyproject）或 `bundle`（分发预建 .venv）。默认 `source` |

> 三同名契约：**包内代码目录名 = `engine_id` = Python 模块名**。
> 违反时 `validate` 报 `engine_id 与目录名不一致`。

## 二、契约与版本字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `manifest_schema_version` | `"0.4"` | 本文件遵循的 schema 版本（`MAJOR.MINOR`） |
| `engine_api_version` | `"1"` | HTTP 契约版本。平台启动后读取 `/health.engine_api_version` 握手，不在支持集合内会拒绝接管 |
| `sdk_version` | 无 | 构建该引擎使用的 SDK 版本约束，如 `>=0.3.0` |
| `paradigm` | `workflow` | `workflow`（声明式 YAML 编排）或 `integrated`（全自定义推理流程） |
| `engine_uid` | 无 | 全局唯一 ID，格式 `aireliable://<author>/<family>/<name>`。**显式声明 `paradigm: workflow` 时必填** |
| `entry` | 无 | 启动模块，如 `my_engine.server`。平台执行 `python -m <entry> --port N`；缺省回退 `engines.<engine_id>.server`（仅兼容旧引擎） |
| `visibility` | `public` | `public` 进公开索引；`private` 仅私有分发 |

## 三、发布与来源字段

| 字段 | 说明 |
|---|---|
| `license` | SPDX 标识，如 `Apache-2.0`、`MIT` |
| `repository` | 引擎**自身**源码仓地址 |
| `author` | `name`（必填）、`email`、`github`、`website`、`license` |
| `source` | 上游来源：`type`（必填，如 `github` / `huggingface` / `local` / `third_party`）、`github_repo`、`download_url`、`homepage` |
| `third_party_references` | 引用的第三方开源模型/代码：`name`、`author`、`license`（均必填）、`github`、`website`、`description` |
| `package` | 打包元信息（`pack` 自动写入）：`sha256`（64 位十六进制）、`signature`（可选） |
| `upgrade` | 升级提示信息（可选）：版本与下载地址 |
| `node_packages` | 依赖的节点包列表 |

## 四、运行环境字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `env_config.requires_torch` | `true` | 是否需要 torch。纯 Python 引擎设为 `false` 可跳过 CUDA/torch 检测 |
| `env_config.default_cuda_version` | `cu128` | 默认 CUDA 轮子版本 |

## 五、models 块（模型引擎必填）

```yaml
models:
  model_layout:            # 模型在 models/ 下的目录布局
    base: "image/checkpoints/my-engine"   # 必填
    lora: "image/loras/my-engine"         # 选填，以下同
    vae: "image/vae/my-engine"
    controlnet: "image/controlnet/my-engine"
    embeddings: "image/embeddings/my-engine"
    ipadapter: "image/ipadapter/my-engine"
    upscale: "image/upscale/my-engine"
    scheduler: "image/scheduler/my-engine"
    pe: "image/pe/my-engine"
    text_encoder: "shared/text_encoders/my-engine"
    tokenizer: "shared/tokenizer/my-engine"
  default_variant: q4        # 默认变体名
  variants:
    - name: q4               # 必填，唯一
      alias: ["my-q4"]       # 选填，UI/API model 字段别名；也支持单个字符串
      base_model: my-base    # 选填
      precision: q4_k_m      # 必填
      loader: gguf           # 选填：gguf / diffusers / transformers
      vram_required: "8GB"   # 选填
      path: "image/checkpoints/my-engine/q4.gguf"  # 必填，相对 models_root
      model_file_ids: [main-q4]                    # 关联 model_files[].id
      default_steps: 20
      default_guidance: 7.0
      supports_pe: false
      supports_mmproj: false
      description: "低显存档"
      ram_required_gb: 16
      disk_required_gb: 8.5
      low_vram_note: "6GB 显存需开启 CPU offload"
      schemes: [fast]
      extra: {}              # 引擎私有扩展槽
  model_files:
    - id: main-q4            # 必填，唯一
      filename: "q4.gguf"    # 必填
      role: main             # main / mmproj / vae / clip / t5 / clip_vision / lora /
                             # controlnet / ipadapter / embedding / upscale / config /
                             # tokenizer / pe / scheduler / text_encoder（旧值 weights = main）
      size_bytes: 4939212390
      precision: q4_k_m      # 选填
      format: gguf           # safetensors / gguf / pt / bin / marker
      optional: false        # 缺失不阻断启动
      exclusive_group: my-engine-main
      required_for: [q4]     # 该文件服务的 variant；省略表示所有 variant 共享
      target_dir: "image/checkpoints/my-engine"
      source: {type: huggingface, repo_id: "author/model", filename: "q4.gguf"}
      description: "主权重"
```

> 完整性以 `size_bytes` 为准；`model_files` 不再使用 `sha256`。整包哈希见
> [package.md](package.md) 的 `package.sha256`。

约束：

- `model_layout` 的每个路径必须以 12 类顶层目录之一开头：
  `shared` / `image` / `video` / `voice` / `audio` / `llm` / `multimodal` / `embedding` /
  `reranker` / `detection` / `segmentation` / `restoration` / `misc`。
- `variants[].model_file_ids` 必须能在 `model_files[].id` 中找到，否则 `validate` 报错。
- `default_variant` 必须存在于 `variants[].name`。
- 零模型引擎（如纯文本处理）可省略整个 `models` 块，但平台会跳过模型校验。

## 六、扩展槽与严格模式

- 未知顶层字段会报错（`extra="forbid"`），防止拼写错误。
- 引擎私有扩展必须放进显式扩展槽：
  - 顶层：`extra: {}`
  - `variants[].extra`、`third_party_references[].extra`
- 禁止把平台字段塞进 `extra`；平台级字段应推动上游 schema 声明。

## 七、校验命令

```bash
aireliable-engine validate            # 结构 / 目录名 / 模型引用 / entry 可导入 / 工作流语法
aireliable-engine validate --strict   # 警告也视为错误（CI 建议）
aireliable-engine validate --json     # 机器可读输出，退出码 0/2/3
```

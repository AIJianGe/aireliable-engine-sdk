# 打包与分发

## 标准包布局

`aireliable-engine pack` 把引擎整理成平台可导入的标准包（`.airengine`，本质是 7z）：

```text
my-engine-0.1.0.airengine
├── manifest.yaml
├── registry.yaml
├── package.json
├── code/
│   └── my_engine/              # 整个工程根，导入后原样铺到 engines/my_engine/
│       ├── my_engine/          # manifest.entry 指向的包（如 my_engine.server）
│       │   ├── __init__.py
│       │   └── server.py
│       ├── workflows/          # 工作流随代码一起分发
│       │   └── default.yaml
│       ├── LICENSE
│       └── ...
├── env/
│   ├── pyproject.toml
│   └── uv.lock                 # 可选
└── docs/                       # 可选（工程根 docs/ 会平移到包根）
```

- `code/` 下的目录名必须等于 `engine_id`，内容为**工程根**而不是单个 Python 包；
  平台把 `code/<engine_id>/` 原样复制到 `engines/<engine_id>/`，并在 PYTHONPATH 中
  加入该目录，因此 `manifest.entry`（如 `my_engine.server`）可直接 `python -m` 启动
- 不打包 `models/`、`env/`（env 单独放包根）、`docs/`、`.venv`、`.git`、`.github`、
  `__pycache__`、`dist`
- 不打包 `manifest.yaml` / `package.json` 进内容哈希（见下文）

## 打包命令

```bash
aireliable-engine validate --strict        # 先确保零错误零警告
aireliable-engine pack -o dist            # 产物 dist/<engine_id>-<version>.airengine
aireliable-engine pack -o dist --json     # 机器可读：archive / sha256 / files
```

`pack` 会：

1. 跑一次完整校验，有错误直接退出（退出码 2）
2. 组装 staging 目录
3. 计算**规范化内容哈希**写入 `manifest.package.sha256`
4. 写 `package.json`（含 `sha256` 与逐文件清单）
5. 压缩为 `.airengine`

## 内容哈希规则

- 文件按**相对路径排序**，逐个写入 `路径 + 文件字节`
- 排除：`manifest.yaml`、`package.json`、`.aireliable/`、`.git/`、`.venv/`、
  `__pycache__/`、`dist/`、`models/`、`.pytest_cache/`
- 与 mtime、大小写无关；相同内容重复打包哈希一致

平台导入时用**同一份算法**（`aireliable_engine.package.compute_content_hash`）复算：

- `manifest.package.sha256` 存在且匹配 → 校验通过
- 存在但不匹配 → **拒绝导入**（提示包被篡改或损坏）
- 不存在（旧包）→ 仅告警，不阻断
- 目录导入 → 跳过声明哈希校验（开发目录文件集与包布局不同），仍写来源记录

## 导入后的来源记录

平台在 `engines/<engine_id>/.aireliable/source.json` 写入：

```json
{
  "schema": 1,
  "engine_id": "my_engine",
  "engine_uid": "aireliable://your-name/misc/my_engine",
  "engine_version": "0.1.0",
  "import_mode": "archive",
  "package_sha256": "…64 位…",
  "content_sha256": "…安装后代码树哈希…",
  "hash_verified": true,
  "signed": false,
  "signature": "",
  "source": {"type": "github", "github_repo": "your-name/my_engine"},
  "imported_at": 1740000000.0
}
```

用途：来源审计、重复导入一致性判断、覆盖升级时提示来源变化。
`.aireliable/` 不参与内容哈希与打包。

## 签名

- MVP 阶段**不强制签名**，平台不因未签名拒绝导入，但会在导入弹窗明确提示
  「第三方引擎以当前用户权限运行 / 未签名引擎不做来源担保」
- 如需签名，在 `manifest.package.signature` 写入签名值；平台会记录 `signed: true`
- 签名验证策略未来随 SDK 版本引入，届时通过 `engine_api_version` 区分

## 体积建议

- 代码 + 文档建议 < 20 MB；单文件超过 20 MB 时 `validate` 会提示
- 模型权重一律走 `models/` 外置下载，**不要打进包**
- `export_mode: bundle` 会连同 `.venv` 一起分发，体积大；默认用 `source`

## 分发渠道

| 渠道 | 说明 |
|---|---|
| GitHub Releases | 上传 `.airengine` 资产，导入弹窗填资产 URL |
| 自建 HTTP | 任意可下载 URL；确保 `Content-Length` 正确 |
| 目录 | 直接指向标准包目录或已安装目录，适合本地开发 |

平台导入时以**当前用户权限**解压并运行，请只从可信来源获取引擎包。

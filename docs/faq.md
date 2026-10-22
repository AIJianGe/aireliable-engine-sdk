# 常见问题

## 启动与导入

**Q：`ModuleNotFoundError: No module named 'my_engine'`**

- 检查 `manifest.entry`，例如 `entry: my_engine.server`
- 检查包内目录名是否等于 `engine_id`（三同名契约）
- 平台会把引擎根目录及其父目录注入 `PYTHONPATH`；手工运行时：

  ```bash
  # engines/<engine_id>/ 已被摊平为包目录时
  PYTHONPATH=/path/to/engines python -m my_engine.server --port 7100
  ```

**Q：平台一直提示引擎启动失败**

1. `python -m <entry> --port 7100` 手工启动，看第一条 Traceback
2. 确认 `GET /health` 可访问（绑定 `127.0.0.1` 或 `0.0.0.0`，不要只绑 IPv6）
3. 确认 `/health.engine_api_version` 是 `"1"`
4. 平台日志会打印退出码与最近日志行，按日志定位

**Q：导入时报「包内容哈希不匹配」**

包被改动过或传输损坏。用源码重新 `pack`，不要手工改 `.airengine` 内的文件。
如果确实需要改代码，请改源码后重新打包并在 `manifest.engine_version` 升版本。

**Q：导入弹窗的确认框必须勾选**

平台明确提示第三方引擎以当前用户权限运行；不勾选无法导入（可取消）。

**Q：旧包导入后提示「未声明 package.sha256」**

旧版打包产物没有整包哈希，平台放行但记录 `hash_verified: false`。
用当前 SDK 重新 `pack` 即可消除。

**Q：覆盖导入时提示「引擎来源信息与已安装版本不同」**

新旧包的 `engine_uid` 或 `source` 不一致。确认来源可信再继续；否则取消并先卸载旧引擎。

## 校验

**Q：`engine_uid 必填`**

显式声明了 `paradigm: workflow` 时必须提供：
`engine_uid: "aireliable://<author>/<family>/<name>"`。

**Q：`engine_id 与目录名不一致`**

把代码目录改名为 `engine_id`，或把 `engine_id` 改成目录名。

**Q：`未知字段 xxx`**

schema 使用 `extra="forbid"`。私有扩展请放进 `extra: {}`，
平台级字段请先确认 schema 是否已声明。

**Q：`variants[].model_file_ids 引用了不存在的 model_file id`**

`model_files[].id` 必须包含变体引用的所有 ID；`default_variant` 必须在 `variants` 中。

## 运行与推理

**Q：`/infer` 返回 `stage=error, type=engine_busy`**

同一引擎同一时刻只允许一个推理任务。平台侧对生成任务串行调度；
手工并发调用时请等待上一任务结束。

**Q：`/infer` 没有输出**

- 检查 `workflows/default.yaml` 的 `outputs` 键（不是 `output`）
- `TextInput` 的 API 注入键是 `inputs.value`
- 引擎自定义节点必须在启动时导入并完成注册

**Q：`节点类型未注册: TextInput`**

在 `server.py` 顶部加 `import aireliable_engine.nodes`（导入即注册内置节点）。

**Q：`/load` 返回 404**

模型文件缺失。看 `/capabilities` 里 `available: false` 的变体，
按 `manifest.models.model_files[].source` 下载到 `models/<target_dir>`。

**Q：`uv run` 在离线机器上失败**

`aireliable-engine run` 与平台启动都带 `--no-sync`，不会重新解析依赖。
请先把环境准备好（`uv sync` 或随包分发 `.venv`），再离线运行。

## 版本与兼容

**Q：平台不接受我的 `engine_api_version`**

平台只支持声明过的 HTTP 契约版本。降级 SDK 或升级平台；
不要私自改 `/health` 返回值来"骗过"握手。

**Q：`aireliable_compat` 怎么写**

写引擎实际验证过的平台版本范围，如 `>=0.3.0`。
不要写成 `*`；平台会按此字段提示兼容性。

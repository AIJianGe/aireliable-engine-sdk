"""统一日志配置（v0.2.5+ 双格式输出 / v0.2.15+ 轮转 + 环境变量推导）。

输出双轨制：
    - stdout：JSON 格式（一行一条 JSON，便于 core 解析与采集到 LogBuffer）
    - 文件：人类可读格式（`[timestamp] [level] [engine_id] message`）

JSON 格式示例：
    {"timestamp":"2026-07-24 12:34:56","level":"INFO","engine_id":"ernie_image","message":"加载模型..."}

被 `setup_engine_logging()` 配置到 root logger。
core 通过解析 stdout JSON 行，把日志写入 `LogManager` 并推送到 SSE 订阅者。

v0.2.15+ 改进：
    - 文件 handler 从 ``FileHandler`` 升级为 ``RotatingFileHandler``（50MB × 5），
      与 core / gateway 端轮转策略一致（见 project-rules.md §6.4）
    - ``log_dir=None`` 时自动读 ``AIRELIABLE_LOG_DIR`` 环境变量推导默认目录，
      平台通过 ``engine_manager._build_subprocess_env`` 注入此变量（§第三部分 1）
    - 引擎代码可显式传入 ``log_dir`` 覆盖环境变量值
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# v0.2.5+ T6.3：用于 setup_engine_logging 内部日志（避免与 root logger 冲突）
logger = logging.getLogger(__name__)


class EngineLogFormatter(logging.Formatter):
    """人类可读日志格式化器（写文件用）。

    格式：[2026-07-18 12:34:56] [INFO] [ernie_image] 这是一条日志
    """

    def __init__(self, engine_id: str):
        super().__init__()
        self.engine_id = engine_id

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        # v0.3.5+ 修复 Bug-B：输出异常堆栈（与 JSONFormatter.format L89-L92 对齐）
        # 之前 EngineLogFormatter 不处理 record.exc_info，
        # 导致 logger.exception() 在文件日志中只输出消息不输出 Traceback，
        # 违反 project-rules.md §6.5「加载失败必记完整 Traceback」，
        # 无法定位 500 错误真正根因。
        if record.exc_info:
            import traceback
            exc_text = "".join(traceback.format_exception(*record.exc_info))
            message = f"{message}\n{exc_text}"

        return f"[{timestamp}] [{level}] [{self.engine_id}] {message}"


class JSONFormatter(logging.Formatter):
    """JSON 日志格式化器（写 stdout 用，供 core 解析采集）。

    输出格式（一行一条 JSON，无缩进，ensure_ascii=False）：
        {"timestamp":"...","level":"INFO","engine_id":"...","message":"...","logger":"..."}

    设计原则：
        - 字段精简：只保留 core 采集必需字段
        - 单行：避免多行 JSON 导致 stdout 解析复杂化
        - UTF-8：保留中文等非 ASCII 字符
        - 异常信息：通过 record.exc_info 折叠到 message 字段

    解析方：`aireliable_engine.log_manager.parse_engine_stdout_line`
    """

    # 不输出的 record 属性前缀（避免泄露内部属性）
    _EXCLUDED_KEYS = {
        "args", "asctime", "created", "exc_info", "exc_text",
        "filename", "funcName", "id", "levelname", "levelno",
        "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated",
        "stack_info", "thread", "threadName", "engine_id",
        # v0.2.5+ T6.1：task_id / request_id 由 contextvars 注入，
        # 不从 record.__dict__ 透传（避免重复）
        "task_id", "request_id",
    }

    def __init__(self, engine_id: str):
        super().__init__()
        self.engine_id = engine_id

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # 构造 message（含异常堆栈，若有）
        message = record.getMessage()
        if record.exc_info:
            import traceback
            exc_text = "".join(traceback.format_exception(*record.exc_info))
            message = f"{message}\n{exc_text}"

        payload = {
            "timestamp": timestamp,
            "level": level,
            "engine_id": self.engine_id,
            "message": message,
            "logger": record.name,
        }

        # v0.2.5+ T6.1：从 contextvars 注入 task_id / request_id（链路追踪）
        # 延迟导入避免循环依赖
        try:
            from .log_context import current_context

            ctx = current_context()
            if ctx.task_id is not None:
                payload["task_id"] = ctx.task_id
            if ctx.request_id is not None:
                payload["request_id"] = ctx.request_id
        except Exception:
            # contextvars 读取失败不应影响日志输出
            pass

        # 透传 record.__dict__ 中的 extra 字段（非标准属性）
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._EXCLUDED_KEYS:
                continue
            try:
                json.dumps(value)  # 仅保留可序列化字段
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False)


def setup_engine_logging(
    engine_id: str,
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_stdout: bool = True,
    remote_config: Optional["RemoteLogConfig"] = None,
) -> logging.Logger:
    """配置引擎日志（v0.2.5+ 双格式输出 / v0.2.15+ 轮转 + 环境变量推导）。

    Args:
        engine_id: 引擎标识（如 ernie_image）
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_dir: 日志目录，若提供则同时写文件（人类可读格式）。
                 v0.2.15+ 若为 ``None``，自动读 ``AIRELIABLE_LOG_DIR`` 环境变量；
                 环境变量也未设置则不写文件（仅 stdout）。
                 引擎代码可显式传入覆盖环境变量值。
        json_stdout: 是否用 JSON 格式输出 stdout（默认 True）
                     - True：stdout 输出 JSON（供 core 解析采集）
                     - False：stdout 输出人类可读格式（向后兼容）
        remote_config: 远程日志上报配置（v0.2.5+ T6.3）。
                     - None：不启用远程上报（本地引擎默认走 stdout 采集）
                     - RemoteLogConfig：远程引擎通过 HTTP 批量上报到 core /logs/ingest
                     通常仅远程部署的引擎需要配置（core 不可通过 stdout 采集时）

    Returns:
        logging.Logger: 配置好的根 logger

    输出布局：
        - stdout：JSONFormatter（json_stdout=True）或 EngineLogFormatter（False）
        - 文件（log_dir 提供 / 推导）：RotatingFileHandler（50MB × 5，v0.2.15+）
        - 远程上报（remote_config 提供）：RemoteLogHandler 推送到 core /logs/ingest

    注意：
        core 的 _collect_logs 会按行读取 stdout：
        - 若行以 `{` 开头且为合法 JSON，按 JSON 解析为 LogEntry
        - 否则按 `[ts] [level] [src] msg` 人类可读格式解析
        - 都失败则整行作为 INFO 消息

        远程引擎（core 不可达 stdout）应同时配置 remote_config，
        这样 core 仍能在统一日志平台看到远程引擎的日志。

    v0.2.15+ 变更：
        - ``log_dir=None`` 时自动读 ``AIRELIABLE_LOG_DIR`` 环境变量（平台注入）
        - 文件 handler 升级为 ``RotatingFileHandler(maxBytes=50MB, backupCount=5)``
          与 core / gateway 端轮转策略一致（见 project-rules.md §6.4）
    """
    level_num = getattr(logging, log_level.upper(), logging.INFO)

    # v0.4.x 修复 emoji 前缀乱码（📊 TOKEN/S 显示为 ?? TOKEN/S）：
    # Windows 下引擎若未带 PYTHONUTF8=1 启动（手动调试 / 部分启动路径），
    # stdout 管道默认用 cp936 编码，json.dumps(ensure_ascii=False) 输出的
    # 非 ASCII 字符（📊/中文）被替换为 '?'。core _collect_logs 固定按 UTF-8
    # 解码，这里强制 stdout/stderr 切换 UTF-8，保证两端编码一致。
    # （PYTHONUTF8=1 已注入时 reconfigure 幂等无害）
    try:
        if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # stdout 非 TextIOWrapper（重定向到特殊对象）等场景忽略，不影响启动
        pass

    # 配置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level_num)

    # 清除既有 handler（避免重复）
    root_logger.handlers.clear()

    # stdout handler
    stdout_formatter = JSONFormatter(engine_id) if json_stdout else EngineLogFormatter(engine_id)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level_num)
    stdout_handler.setFormatter(stdout_formatter)
    root_logger.addHandler(stdout_handler)

    # v0.2.15+ log_dir 推导：显式传入 > AIRELIABLE_LOG_DIR 环境变量 > 不写文件
    if log_dir is None:
        env_log_dir = os.environ.get("AIRELIABLE_LOG_DIR")
        if env_log_dir:
            log_dir = Path(env_log_dir)

    # 文件 handler（RotatingFileHandler，50MB × 5，v0.2.15+）
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{engine_id}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level_num)
        file_handler.setFormatter(EngineLogFormatter(engine_id))
        root_logger.addHandler(file_handler)

    # v0.2.5+ T6.3：远程日志上报 handler（仅 remote_config 提供时挂载）
    if remote_config is not None:
        from .remote_log_reporter import RemoteLogHandler, RemoteLogReporter
        reporter = RemoteLogReporter(engine_id=engine_id, config=remote_config)
        reporter.start()  # 启动后台 flush 任务（幂等，无事件循环时安全跳过）
        remote_handler = RemoteLogHandler(
            engine_id=engine_id,
            reporter=reporter,
            level=level_num,
        )
        root_logger.addHandler(remote_handler)
        # 注册到全局表，供引擎 lifespan 关闭时 stop
        from .remote_log_reporter import _global_reporters
        _global_reporters[engine_id] = reporter
        logger.info(
            "远程日志上报已挂载: engine=%s, core_url=%s",
            engine_id, remote_config.core_url,
        )

    # 配置常用库的日志级别
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return root_logger


# ===========================================================================
# v0.2.5+ T6.2：运行时日志级别动态调整
# ====================================================================================


def set_engine_log_level(level: str) -> dict:
    """动态调整引擎日志级别（v0.2.5+ T6.2）。

    同时调整 root logger 及所有 handler 的级别，便于运行时排查问题。
    通常由引擎 `/log_level` 端点调用（由 core `POST /engines/{id}/log_level` 转发）。

    Args:
        level: 日志级别字符串（DEBUG / INFO / WARNING / ERROR / CRITICAL）
            大小写不敏感

    Returns:
        dict: 调整结果，含 before / after / logger_count 字段
        示例：{"before": "INFO", "after": "DEBUG", "logger_count": 5}

    Raises:
        ValueError: level 不是合法日志级别
    """
    level_upper = level.upper()
    level_num = getattr(logging, level_upper, None)
    if level_num is None or not isinstance(level_num, int):
        raise ValueError(
            f"非法日志级别: {level!r}，合法值：DEBUG / INFO / WARNING / ERROR / CRITICAL"
        )

    root_logger = logging.getLogger()
    before = logging.getLevelName(root_logger.level)

    # 1. 调整 root logger 级别
    root_logger.setLevel(level_num)

    # 2. 调整所有 handler 级别（stdout + file）
    handler_count = 0
    for handler in root_logger.handlers:
        handler.setLevel(level_num)
        handler_count += 1

    # 3. 调整常用库 logger 级别（保持 WARNING 作为下限，避免日志爆炸）
    # 但允许 DEBUG 时全部放开
    lib_loggers = ["uvicorn", "uvicorn.access", "diffusers", "transformers"]
    if level_num <= logging.DEBUG:
        # DEBUG 模式下放开常用库
        for name in lib_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)
    else:
        # 非 DEBUG 恢复 WARNING
        for name in lib_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    return {
        "before": before,
        "after": level_upper,
        "logger_count": handler_count,
    }


def get_engine_log_level() -> str:
    """获取当前引擎 root logger 的日志级别。

    Returns:
        级别字符串（DEBUG / INFO / WARNING / ERROR / CRITICAL）
    """
    return logging.getLevelName(logging.getLogger().level)


# ===========================================================================
# v0.2.5+ T6.2：FastAPI 端点注册（/log_level）
# ====================================================================================
#
# 提供一个可复用函数，把 GET/POST /log_level 端点挂载到引擎的 FastAPI app。
# 端点协议：
#     GET  /log_level           → {"level": "INFO"}
#     POST /log_level           → body: {"level": "DEBUG"}
#                                 → {"before": "INFO", "after": "DEBUG", "logger_count": N}
#
# core 的 POST /engines/{id}/log_level 会转发到引擎的 POST /log_level。
# 也可直接 curl 调试：
#     curl http://127.0.0.1:7001/log_level
#     curl -X POST http://127.0.0.1:7001/log_level -H 'Content-Type: application/json' \
#          -d '{"level":"DEBUG"}'


def register_log_level_endpoint(app) -> None:
    """为引擎 FastAPI app 注册 /log_level 端点（v0.2.5+ T6.2）。

    注册的端点：
        - GET  /log_level：查询当前日志级别
        - POST /log_level：动态调整日志级别（body: {"level": "DEBUG"}）

    用法：
        from aireliable_engine import setup_engine_logging, register_log_level_endpoint
        setup_engine_logging(engine_id="my_engine", log_level="INFO")
        register_log_level_endpoint(app)  # 在 _register_endpoints 中调用

    端点行为：
        - GET 返回当前 root logger 级别
        - POST 调用 set_engine_log_level，同时调整 root logger + 所有 handler 级别
        - 非法 level 返回 400

    Args:
        app: FastAPI 应用实例
    """
    from fastapi import HTTPException  # 延迟导入避免强依赖

    @app.get("/log_level", tags=["engine"])
    async def _get_log_level() -> dict:
        """查询当前引擎日志级别。"""
        return {"level": get_engine_log_level()}

    @app.post("/log_level", tags=["engine"])
    async def _set_log_level(body: dict = None) -> dict:
        """动态调整引擎日志级别。

        请求体 JSON:
            {"level": "DEBUG"}   # DEBUG / INFO / WARNING / ERROR / CRITICAL
        """
        body = body or {}
        level = body.get("level")
        if not level or not isinstance(level, str):
            raise HTTPException(
                status_code=400,
                detail="请求体必须包含 level 字段（字符串）",
            )
        try:
            return set_engine_log_level(level)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

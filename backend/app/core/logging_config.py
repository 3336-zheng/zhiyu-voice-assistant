"""结构化日志、请求上下文注入和本地轮转。"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from .errors import normalize_error_code, sanitize_text
from .observability import get_request_id

_configured = False
_EVENT_FIELDS = (
    "component",
    "operation",
    "error_code",
    "duration_ms",
    "retryable",
    "exception_type",
    "method",
    "path",
    "status_code",
    "task_id",
    "run_id",
)


class RequestContextFilter(logging.Filter):
    """自动为当前协程内的所有日志补充 Request ID。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = get_request_id() or "-"
        return True


class BelowErrorFilter(logging.Filter):
    """只允许普通日志进入标准输出，避免终端把 INFO 当成错误显示。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR


class JsonFormatter(logging.Formatter):
    """输出便于检索和采集的单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-") or "-",
            "message": sanitize_text(record.getMessage()),
        }
        for field in _EVENT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = sanitize_text(
                "".join(traceback.format_exception(*record.exc_info)),
                max_length=12_000,
            )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class SafeConsoleFormatter(logging.Formatter):
    """终端保持可读，同时执行与文件日志相同的脱敏。"""

    def format(self, record: logging.LogRecord) -> str:
        message = sanitize_text(record.getMessage())
        request_id = getattr(record, "request_id", "-") or "-"
        component = getattr(record, "component", None)
        error_code = getattr(record, "error_code", None)
        details = [f"request_id={request_id}"]
        if component:
            details.append(f"component={component}")
        if error_code:
            details.append(f"error_code={error_code}")
        rendered = (
            f"{self.formatTime(record, self.datefmt)} {record.levelname} "
            f"{record.name} {' '.join(details)}: {message}"
        )
        if record.exc_info:
            rendered += "\n" + sanitize_text(
                "".join(traceback.format_exception(*record.exc_info)),
                max_length=12_000,
            )
        return rendered


def _add_context_filter(handler: logging.Handler) -> logging.Handler:
    handler.addFilter(RequestContextFilter())
    return handler


def _rotating_handler(
    path: str,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    return _add_context_filter(handler)


def configure_logging(settings: Optional[Any] = None, *, force: bool = False) -> None:
    """初始化终端和文件日志；文件不可写时保留终端日志。"""
    global _configured
    if _configured and not force:
        return
    if settings is None:
        from .config import settings as app_settings

        settings = app_settings

    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    # 普通日志走 stdout，错误日志单独走 stderr。这样终端只会把 ERROR 及以上显示为错误色。
    console = _add_context_filter(logging.StreamHandler(sys.stdout))
    console.addFilter(BelowErrorFilter())
    console.setLevel(level)
    console.setFormatter(SafeConsoleFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(console)

    error_console = _add_context_filter(logging.StreamHandler(sys.stderr))
    error_console.setLevel(logging.ERROR)
    error_console.setFormatter(SafeConsoleFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(error_console)

    try:
        root.addHandler(
            _rotating_handler(
                settings.log_file,
                level=level,
                max_bytes=settings.log_max_bytes,
                backup_count=settings.log_backup_count,
            )
        )
        root.addHandler(
            _rotating_handler(
                settings.log_error_file,
                level=logging.ERROR,
                max_bytes=settings.log_max_bytes,
                backup_count=settings.log_backup_count,
            )
        )
    except OSError as exc:
        logging.getLogger(__name__).warning("日志文件不可写，仅使用终端日志: %s", exc)
    _configured = True


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    error_code: Optional[object] = None,
    duration_ms: Optional[float] = None,
    retryable: Optional[bool] = None,
    exception: Optional[BaseException] = None,
    **fields: Any,
) -> None:
    """以统一字段记录业务事件，避免各服务拼接不稳定文本。"""
    extra = {
        "component": component,
        "operation": operation,
        "error_code": normalize_error_code(error_code),
        "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
        "retryable": retryable,
        "exception_type": type(exception).__name__ if exception else None,
        **{key: value for key, value in fields.items() if key in _EVENT_FIELDS},
    }
    logger.log(
        level,
        sanitize_text(message),
        extra={key: value for key, value in extra.items() if value is not None},
        exc_info=(type(exception), exception, exception.__traceback__) if exception else None,
    )

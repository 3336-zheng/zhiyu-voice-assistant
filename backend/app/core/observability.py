"""请求追踪和阶段耗时记录。"""

import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar, Token
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .errors import ErrorCode, classify_stage_error, normalize_error_code, safe_error_message

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_stage_timings: ContextVar[Optional[Dict[str, Dict[str, float]]]] = ContextVar(
    "stage_timings",
    default=None,
)
_execution_timeline: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "execution_timeline",
    default=None,
)
_model_usage: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "model_usage",
    default=None,
)
_recent_traces: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_recent_traces_lock = threading.RLock()
_recent_trace_limit = 500


def start_request(request_id: Optional[str] = None) -> Tuple[str, Token, Token, Token, Token]:
    """初始化请求上下文，并返回可用于恢复上下文的 Token。"""
    normalized = (request_id or "").strip()
    if not REQUEST_ID_PATTERN.fullmatch(normalized):
        normalized = str(uuid.uuid4())
    request_token = _request_id.set(normalized)
    timings_token = _stage_timings.set({})
    timeline_token = _execution_timeline.set([])
    usage_token = _model_usage.set([])
    return normalized, request_token, timings_token, timeline_token, usage_token


def reset_request(
    request_token: Token,
    timings_token: Token,
    timeline_token: Token,
    usage_token: Token,
) -> None:
    """恢复进入请求前的上下文。"""
    _stage_timings.reset(timings_token)
    _execution_timeline.reset(timeline_token)
    _model_usage.reset(usage_token)
    _request_id.reset(request_token)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def record_timing(stage: str, duration_ms: float, status: str = "completed") -> None:
    """累计一个阶段的调用次数、总耗时和最大耗时。"""
    timings = _stage_timings.get()
    if timings is None:
        return
    item = timings.setdefault(stage, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
    item["count"] += 1
    item["total_ms"] += round(duration_ms, 3)
    item["max_ms"] = max(item["max_ms"], round(duration_ms, 3))
    timeline = _execution_timeline.get()
    if timeline is not None:
        timeline.append(
            {
                "stage": stage,
                "status": status,
                "duration_ms": round(duration_ms, 3),
            }
        )


def get_stage_timings() -> Dict[str, Dict[str, float]]:
    """返回当前请求的阶段耗时快照。"""
    timings = _stage_timings.get() or {}
    return {name: dict(values) for name, values in timings.items()}


def get_execution_timeline() -> List[Dict[str, Any]]:
    """返回按完成顺序排列的阶段快照。"""
    return [dict(item) for item in (_execution_timeline.get() or [])]


def record_error(
    *,
    stage: str,
    component: str,
    operation: str,
    error_code: ErrorCode | str,
    retryable: bool = False,
    exception: Optional[BaseException] = None,
    message: Optional[str] = None,
) -> None:
    """记录不含正文和凭证的请求级错误事件。"""
    timeline = _execution_timeline.get()
    if timeline is None:
        return
    timeline.append(
        {
            "stage": stage,
            "status": "failed",
            "component": component,
            "operation": operation,
            "error_code": normalize_error_code(error_code),
            "retryable": bool(retryable),
            "exception_type": type(exception).__name__ if exception else None,
            "message": safe_error_message(message or exception or "执行失败"),
        }
    )


def record_model_usage(
    *,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost: float = 0.0,
    duration_ms: float = 0.0,
    fallback_used: bool = False,
    success: bool = True,
    error_type: Optional[str] = None,
    operation: Optional[str] = None,
    finish_reason: Optional[str] = None,
    truncated: bool = False,
) -> None:
    """记录一次模型尝试，不包含提示词、答案或密钥。"""
    usage = _model_usage.get()
    if usage is None:
        return
    usage.append(
        {
            "provider": provider,
            "model": model,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "estimated_cost": round(float(estimated_cost or 0.0), 8),
            "duration_ms": round(float(duration_ms or 0.0), 3),
            "fallback_used": bool(fallback_used),
            "success": bool(success),
            "error_type": error_type,
            "operation": operation,
            "finish_reason": finish_reason,
            "truncated": bool(truncated),
        }
    )


def get_model_usage() -> Dict[str, Any]:
    """聚合当前请求中的模型调用与 Token/成本。"""
    calls = [dict(item) for item in (_model_usage.get() or [])]
    return {
        "calls": calls,
        "call_count": len(calls),
        "prompt_tokens": sum(item["prompt_tokens"] for item in calls),
        "completion_tokens": sum(item["completion_tokens"] for item in calls),
        "total_tokens": sum(item["total_tokens"] for item in calls),
        "estimated_cost": round(sum(item["estimated_cost"] for item in calls), 8),
        "fallback_used": any(item["fallback_used"] for item in calls),
        "truncated": any(item.get("truncated", False) for item in calls),
    }


def store_current_trace(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    total_ms: float,
    status: Optional[str] = None,
    trace_type: str = "http",
) -> Dict[str, Any]:
    """保存近期请求的安全快照，供本地诊断页查询。"""
    timeline = get_execution_timeline()
    errors = [item for item in timeline if item.get("error_code")]
    snapshot = {
        "request_id": request_id,
        "trace_type": trace_type,
        "method": method,
        "path": path,
        "status": status or ("completed" if status_code < 400 else "failed"),
        "status_code": int(status_code),
        "total_ms": round(float(total_ms), 3),
        "stage_timings": get_stage_timings(),
        "timeline": timeline,
        "model_usage": get_model_usage(),
        "error": deepcopy(errors[-1]) if errors else None,
        "completed_at": time.time(),
    }
    with _recent_traces_lock:
        _recent_traces[request_id] = deepcopy(snapshot)
        _recent_traces.move_to_end(request_id)
        while len(_recent_traces) > _recent_trace_limit:
            _recent_traces.popitem(last=False)
    return snapshot


def get_recent_trace(request_id: str) -> Optional[Dict[str, Any]]:
    with _recent_traces_lock:
        snapshot = _recent_traces.get(request_id)
        return deepcopy(snapshot) if snapshot else None


def list_recent_traces(limit: int = 50) -> List[Dict[str, Any]]:
    """按完成时间倒序返回摘要，不包含模型输入输出。"""
    normalized_limit = max(1, min(int(limit), 100))
    with _recent_traces_lock:
        snapshots = list(reversed(_recent_traces.values()))[:normalized_limit]
    return [
        {
            "request_id": item["request_id"],
            "trace_type": item["trace_type"],
            "method": item["method"],
            "path": item["path"],
            "status": item["status"],
            "status_code": item["status_code"],
            "total_ms": item["total_ms"],
            "completed_at": item["completed_at"],
            "error_code": (item.get("error") or {}).get("error_code"),
        }
        for item in snapshots
    ]


@contextmanager
def timed_stage(stage: str) -> Iterator[None]:
    """记录同步代码块耗时；异常不会吞掉。"""
    from .telemetry import trace_span

    started = time.perf_counter()
    status = "completed"
    try:
        with trace_span(stage):
            yield
    except Exception as exc:
        status = "failed"
        error_code, component, operation, retryable = classify_stage_error(stage)
        record_error(
            stage=stage,
            component=component,
            operation=operation,
            error_code=error_code,
            retryable=retryable,
            exception=exc,
        )
        raise
    finally:
        record_timing(stage, (time.perf_counter() - started) * 1000, status=status)


def build_server_timing(total_ms: float) -> str:
    """生成浏览器可读取的 Server-Timing 响应头。"""
    entries = [f"total;dur={total_ms:.3f}"]
    for stage, values in get_stage_timings().items():
        metric = re.sub(r"[^A-Za-z0-9_-]", "_", stage)
        entries.append(f"{metric};dur={values['total_ms']:.3f}")
    return ", ".join(entries)

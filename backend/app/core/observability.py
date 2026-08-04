"""请求追踪和阶段耗时记录。"""

import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Dict, Iterator, Optional, Tuple

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_stage_timings: ContextVar[Optional[Dict[str, Dict[str, float]]]] = ContextVar(
    "stage_timings",
    default=None,
)


def start_request(request_id: Optional[str] = None) -> Tuple[str, Token, Token]:
    """初始化请求上下文，并返回可用于恢复上下文的 Token。"""
    normalized = (request_id or "").strip()
    if not REQUEST_ID_PATTERN.fullmatch(normalized):
        normalized = str(uuid.uuid4())
    request_token = _request_id.set(normalized)
    timings_token = _stage_timings.set({})
    return normalized, request_token, timings_token


def reset_request(request_token: Token, timings_token: Token) -> None:
    """恢复进入请求前的上下文。"""
    _stage_timings.reset(timings_token)
    _request_id.reset(request_token)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def record_timing(stage: str, duration_ms: float) -> None:
    """累计一个阶段的调用次数、总耗时和最大耗时。"""
    timings = _stage_timings.get()
    if timings is None:
        return
    item = timings.setdefault(stage, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
    item["count"] += 1
    item["total_ms"] += round(duration_ms, 3)
    item["max_ms"] = max(item["max_ms"], round(duration_ms, 3))


def get_stage_timings() -> Dict[str, Dict[str, float]]:
    """返回当前请求的阶段耗时快照。"""
    timings = _stage_timings.get() or {}
    return {name: dict(values) for name, values in timings.items()}


@contextmanager
def timed_stage(stage: str) -> Iterator[None]:
    """记录同步代码块耗时；异常不会吞掉。"""
    started = time.perf_counter()
    try:
        yield
    finally:
        record_timing(stage, (time.perf_counter() - started) * 1000)


def build_server_timing(total_ms: float) -> str:
    """生成浏览器可读取的 Server-Timing 响应头。"""
    entries = [f"total;dur={total_ms:.3f}"]
    for stage, values in get_stage_timings().items():
        metric = re.sub(r"[^A-Za-z0-9_-]", "_", stage)
        entries.append(f"{metric};dur={values['total_ms']:.3f}")
    return ", ".join(entries)

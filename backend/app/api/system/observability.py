"""本地运行追踪查询接口。"""

from __future__ import annotations

import ipaddress
from typing import Any, Dict, List
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.errors import ErrorCode, safe_error_message
from backend.app.core.observability import (
    REQUEST_ID_PATTERN,
    get_recent_trace,
    list_recent_traces,
)
from backend.app.models.observability import AgentRun

router = APIRouter()


def _is_loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _require_local_access(request: Request) -> None:
    """追踪内容默认仅允许本机读取，避免运行信息暴露到公网。"""
    if not settings.observability_enabled or not settings.observability_trace_api_enabled:
        raise HTTPException(status_code=404, detail="运行追踪未启用")
    if settings.observability_trace_allow_remote:
        return
    host = request.client.host if request.client else ""
    if not _is_loopback_host(host):
        raise HTTPException(status_code=403, detail="运行追踪仅允许本机访问")
    origin = request.headers.get("origin")
    if origin:
        origin_host = urlparse(origin).hostname or ""
        if not _is_loopback_host(origin_host):
            raise HTTPException(status_code=403, detail="运行追踪拒绝非本机页面访问")


@router.get("/requests")
async def recent_requests(
    request: Request,
    limit: int = Query(default=30, ge=1, le=100),
) -> Dict[str, List[Dict[str, Any]]]:
    _require_local_access(request)
    return {"requests": list_recent_traces(limit)}


@router.get("/requests/{request_id}")
async def request_trace(
    request_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_local_access(request)
    if not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise HTTPException(status_code=422, detail="Request ID 格式不正确")

    snapshot = get_recent_trace(request_id)
    if snapshot is not None:
        run = db.get(AgentRun, request_id)
        if run is not None and run.retrieval_stats:
            snapshot["retrieval_stats"] = run.retrieval_stats
        return snapshot

    run = db.get(AgentRun, request_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到对应的运行记录")

    error = None
    if run.error:
        error = {
            "stage": "agent.run",
            "status": "failed",
            "component": "agent",
            "operation": "run",
            "error_code": ErrorCode.AGENT_EXECUTION_ERROR.value,
            "retryable": False,
            "message": safe_error_message(run.error),
        }
    return {
        "request_id": run.request_id,
        "trace_type": "agent",
        "method": "AGENT",
        "path": "agent.run",
        "status": run.status,
        "status_code": 200 if run.status == "completed" else 500,
        "total_ms": run.execution_time_ms or 0,
        "stage_timings": {},
        "timeline": run.timeline or [],
        "retrieval_stats": run.retrieval_stats or {},
        "model_usage": run.model_usage or {},
        "error": error,
        "completed_at": run.completed_at.timestamp() if run.completed_at else None,
        "source": "agent_run",
    }

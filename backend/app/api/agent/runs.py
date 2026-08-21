"""Agent 对话、运行状态和 SSE 事件路由。"""

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.agent.agent import get_agent
from backend.app.core.database import get_db
from backend.app.services.runtime.agent_runtime_service import (
    AgentRunConflict,
    AgentRunNotFound,
    get_agent_runtime_service,
)

from .presenters import present_agent_response
from .schemas import (
    AgentActionRequest,
    AgentChatRequest,
    AgentChatResponse,
    AgentRunStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse_payload(event) -> str:
    payload = event.model_dump(mode="json")
    return (
        f"id: {event.sequence}\n"
        f"event: {event.type.value}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


@router.post("/chat/", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest, db: Session = Depends(get_db)):
    """执行兼容的非流式 Agent 对话。"""
    try:
        response = await get_agent().run(
            request.query.strip(),
            session_id=request.session_id,
            db=db,
            allow_external_research=request.allow_external_research,
        )
        return present_agent_response(response)
    except Exception as exc:
        logger.error("Agent 对话失败: %s", exc, exc_info=True)
        return AgentChatResponse(
            query=request.query,
            response=f"处理请求时出现错误: {exc}",
            success=False,
        )


@router.post("/chat/stream/")
async def agent_chat_stream(request: AgentChatRequest):
    """启动与 HTTP 连接解耦的 Agent Run 并订阅事件。"""
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:16]}"
    runtime = get_agent_runtime_service()
    try:
        run = await runtime.start(
            request.query.strip(),
            session_id,
            allow_external_research=request.allow_external_research,
        )
    except AgentRunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def generate_stream():
        async for event in runtime.iter_events(run.run_id, session_id):
            yield ": heartbeat\n\n" if event is None else _sse_payload(event)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Agent-Run-ID": run.run_id,
        },
    )


@router.get("/runs/{run_id}", response_model=AgentRunStatusResponse)
async def get_agent_run(run_id: str, session_id: str):
    try:
        return await get_agent_runtime_service().get(run_id, session_id)
    except AgentRunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/events")
async def resume_agent_run_events(
    run_id: str,
    session_id: str,
    after_sequence: int = Query(default=0, ge=0),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    runtime = get_agent_runtime_service()
    try:
        await runtime.get(run_id, session_id)
    except AgentRunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cursor = after_sequence
    if last_event_id:
        try:
            cursor = max(cursor, int(last_event_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Last-Event-ID 必须是整数") from exc

    async def generate_stream():
        async for event in runtime.iter_events(run_id, session_id, cursor):
            yield ": heartbeat\n\n" if event is None else _sse_payload(event)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel", response_model=AgentRunStatusResponse)
async def cancel_agent_run(run_id: str, request: AgentActionRequest):
    try:
        return await get_agent_runtime_service().cancel(run_id, request.session_id)
    except AgentRunNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

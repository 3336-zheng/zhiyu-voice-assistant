"""Agent 会话管理路由。"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.agent.agent import get_agent
from backend.app.core.database import get_db
from backend.app.services.memory_service import get_memory_service

from .agent_schemas import SessionListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/sessions/", response_model=SessionListResponse)
async def list_sessions(
    query: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    try:
        sessions = get_memory_service().list_sessions(db, limit=limit, query=query)
        return SessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as exc:
        logger.error("列出会话失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    try:
        result = get_memory_service().get_session_messages(session_id, db)
        if result is None:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("读取会话历史失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str, db: Session = Depends(get_db)):
    try:
        if get_agent().clear_session(session_id, db):
            return {"message": f"会话 {session_id} 已清除", "success": True}
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或清除失败")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("清除会话失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sessions/cleanup")
async def cleanup_expired_sessions(db: Session = Depends(get_db)):
    try:
        result = get_memory_service().cleanup_expired_sessions(db)
        return {
            "message": (
                f"清理完成: {result.get('cleaned_sessions', 0)} 个会话, "
                f"{result.get('cleaned_messages', 0)} 条消息"
            ),
            "success": True,
            "detail": result,
        }
    except Exception as exc:
        logger.error("手动清理失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions/stats")
async def get_session_stats(db: Session = Depends(get_db)):
    try:
        return get_memory_service().get_session_stats(db)
    except Exception as exc:
        logger.error("获取统计信息失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

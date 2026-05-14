"""
Agent 对话 API 接口
支持多轮对话记忆和会话管理
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from backend.app.core.database import get_db
from backend.app.agent.agent import get_agent
from backend.app.agent.models import AgentResponse
from backend.app.services.memory_service import get_memory_service

logger = logging.getLogger(__name__)

router = APIRouter()


class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    query: str
    session_id: Optional[str] = None  # 会话 ID（可选）


class AgentChatResponse(BaseModel):
    """Agent 对话响应"""
    query: str
    response: str
    session_id: Optional[str] = None  # 返回会话 ID
    intent: Optional[str] = None
    plan_summary: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    execution_time_ms: Optional[int] = None
    success: bool = True


class HybridSearchRequest(BaseModel):
    """混合检索请求"""
    query: str
    top_k: int = 5
    bm25_top_k: Optional[int] = None
    embedding_top_k: Optional[int] = None


class HybridSearchResponse(BaseModel):
    """混合检索响应"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    execution_time_ms: Optional[int] = None


class CompareSearchRequest(BaseModel):
    """对比检索请求"""
    query: str
    top_k: int = 5


class CompareSearchResponse(BaseModel):
    """对比检索响应"""
    query: str
    bm25_results: List[Dict[str, Any]]
    embedding_results: List[Dict[str, Any]]
    hybrid_results: List[Dict[str, Any]]


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: List[Dict[str, Any]]
    total: int


@router.post("/chat/", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    db: Session = Depends(get_db)
):
    """
    Agent 对话接口

    支持：
    - 知识库检索（如"查找关于AI的笔记"）
    - 笔记管理（如"创建笔记标题是XXX"）
    - 时间查询（如"现在几点"）
    - 摘要总结（如"总结关于会议的内容"）
    - 多轮对话记忆（通过 session_id）
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    try:
        agent = get_agent()
        response: AgentResponse = await agent.run(
            request.query.strip(),
            session_id=request.session_id,
            db=db
        )

        plan_summary = None
        if response.plan:
            plan_summary = f"意图: {response.plan.intent.value}, 步骤: {len(response.plan.steps)}"

        return AgentChatResponse(
            query=request.query,
            response=response.response,
            session_id=response.session_id,
            intent=response.plan.intent.value if response.plan else None,
            plan_summary=plan_summary,
            sources=response.sources,
            execution_time_ms=response.execution_time_ms,
            success=True
        )

    except Exception as e:
        logger.error(f"Agent 对话失败: {e}", exc_info=True)
        return AgentChatResponse(
            query=request.query,
            response=f"处理请求时出现错误: {str(e)}",
            success=False
        )


@router.get("/sessions/", response_model=SessionListResponse)
async def list_sessions(db: Session = Depends(get_db)):
    """
    列出所有会话

    Returns:
        SessionListResponse: 会话列表
    """
    try:
        agent = get_agent()
        sessions = agent.list_sessions(db)
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions)
        )
    except Exception as e:
        logger.error(f"列出会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    清除指定会话的历史记录

    Args:
        session_id: 会话 ID

    Returns:
        Dict: 操作结果
    """
    try:
        agent = get_agent()
        success = agent.clear_session(session_id, db)
        if success:
            return {"message": f"会话 {session_id} 已清除", "success": True}
        else:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或清除失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清除会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/cleanup")
async def cleanup_expired_sessions(db: Session = Depends(get_db)):
    """
    手动触发过期会话清理

    Returns:
        Dict: 清理结果
    """
    try:
        memory_service = get_memory_service()
        result = memory_service.cleanup_expired_sessions(db)
        return {
            "message": f"清理完成: {result.get('cleaned_sessions', 0)} 个会话, {result.get('cleaned_messages', 0)} 条消息",
            "success": True,
            "detail": result
        }
    except Exception as e:
        logger.error(f"手动清理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/stats")
async def get_session_stats(db: Session = Depends(get_db)):
    """
    获取会话统计信息

    Returns:
        Dict: 会话统计数据
    """
    try:
        memory_service = get_memory_service()
        stats = memory_service.get_session_stats(db)
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    db: Session = Depends(get_db)
):
    """
    混合检索接口 (BM25 + Embedding + RRF + Reranker)
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    try:
        from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service
        import time

        start_time = time.time()
        hybrid_service = get_hybrid_retrieval_service()

        results = hybrid_service.search_hybrid(
            query=request.query.strip(),
            top_k=request.top_k,
            bm25_top_k=request.bm25_top_k,
            embedding_top_k=request.embedding_top_k,
            db=db
        )

        execution_time = int((time.time() - start_time) * 1000)

        return HybridSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"混合检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare/", response_model=CompareSearchResponse)
async def compare_search(
    request: CompareSearchRequest,
    db: Session = Depends(get_db)
):
    """
    对比三种检索方式 (BM25 / Embedding / Hybrid)
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    try:
        from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service

        hybrid_service = get_hybrid_retrieval_service()
        comparison = hybrid_service.compare_search_methods(
            query=request.query.strip(),
            top_k=request.top_k,
            db=db
        )

        return CompareSearchResponse(
            query=request.query,
            bm25_results=comparison["bm25"],
            embedding_results=comparison["embedding"],
            hybrid_results=comparison["hybrid"]
        )

    except Exception as e:
        logger.error(f"对比检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

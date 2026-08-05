"""
Agent 对话 API 接口
支持多轮对话记忆和会话管理，支持流式输出
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import json

from backend.app.core.database import get_db
from backend.app.agent.agent import AgentActionError, get_agent
from backend.app.agent.models import AgentResponse
from backend.app.services.memory_service import get_memory_service
from backend.app.services.llm_service import get_llm_service
from backend.app.services.external_research_service import (
    ExternalResearchConflict,
    ExternalResearchError,
    ExternalResearchNotFound,
    ExternalResearchUnavailable,
    get_external_research_service,
)

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
    confirmation_required: bool = False
    pending_action_id: Optional[str] = None
    action_preview: Optional[List[Dict[str, Any]]] = None
    evidence_status: str = "not_applicable"
    evidence_score: Optional[float] = None
    evidence_source_count: int = 0
    evidence_reason: Optional[str] = None
    external_research_available: bool = False
    execution_time_ms: Optional[int] = None
    success: bool = True


class AgentActionRequest(BaseModel):
    """确认或取消待处理 Agent 操作。"""

    session_id: str


class ExternalResearchRequest(BaseModel):
    """显式触发外部研究。"""

    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=64)


class ExternalResearchSaveRequest(BaseModel):
    """将研究草稿转换为待确认 Wiki 写入。"""

    session_id: str = Field(min_length=1, max_length=64)
    notebook: Optional[str] = Field(default=None, max_length=255)


class ExternalResearchSourceResponse(BaseModel):
    id: str
    title: str
    url: str
    snippet: Optional[str] = None
    provider: str
    retrieved_at: datetime


class ExternalResearchResponse(BaseModel):
    run_id: str
    session_id: str
    query: str
    status: str
    search_queries: List[str]
    answer: Optional[str] = None
    draft_title: Optional[str] = None
    draft_content: Optional[str] = None
    page_id: Optional[str] = None
    error: Optional[str] = None
    sources: List[ExternalResearchSourceResponse]
    created_at: datetime
    completed_at: Optional[datetime] = None


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
            confirmation_required=response.confirmation_required,
            pending_action_id=response.pending_action_id,
            action_preview=response.action_preview,
            evidence_status=response.evidence_status,
            evidence_score=response.evidence_score,
            evidence_source_count=response.evidence_source_count,
            evidence_reason=response.evidence_reason,
            external_research_available=response.external_research_available,
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


@router.post("/chat/stream/")
async def agent_chat_stream(
    request: AgentChatRequest,
    db: Session = Depends(get_db)
):
    """
    Agent 流式对话接口（SSE）

    支持：
    - 流式返回 Agent 思考过程和答案
    - 打字机效果
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    async def generate_stream():
        """生成 SSE 流"""
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'query': request.query})}\n\n"

            # 发送思考中事件
            yield f"data: {json.dumps({'type': 'thinking', 'message': '正在分析您的问题...'})}\n\n"

            # 获取 Agent 响应（非流式）
            agent = get_agent()
            response: AgentResponse = await agent.run(
                request.query.strip(),
                session_id=request.session_id,
                db=db
            )

            # 发送意图识别结果
            if response.plan:
                yield f"data: {json.dumps({'type': 'intent', 'intent': response.plan.intent.value})}\n\n"

            # 发送检索结果数量
            if response.sources:
                yield f"data: {json.dumps({'type': 'sources', 'count': len(response.sources), 'items': response.sources, 'evidence_status': response.evidence_status}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'type': 'evidence', 'status': response.evidence_status, 'score': response.evidence_score, 'source_count': response.evidence_source_count, 'reason': response.evidence_reason, 'external_research_available': response.external_research_available}, ensure_ascii=False)}\n\n"

            if response.confirmation_required:
                yield f"data: {json.dumps({'type': 'confirmation', 'pending_action_id': response.pending_action_id, 'preview': response.action_preview, 'message': response.response}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': response.session_id, 'execution_time_ms': response.execution_time_ms})}\n\n"
                return

            if response.evidence_status == "insufficient":
                yield f"data: {json.dumps({'type': 'token', 'content': response.response}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': response.session_id, 'execution_time_ms': response.execution_time_ms, 'evidence_status': response.evidence_status}, ensure_ascii=False)}\n\n"
                return

            # 流式返回答案
            llm = get_llm_service()
            messages = [
                {"role": "system", "content": "请直接回答以下问题，不要添加任何前缀。"},
                {"role": "user", "content": f"问题：{request.query}\n\n参考资料：{response.response}"}
            ]

            full_response = ""
            for chunk in llm.stream_chat(messages=messages, temperature=0.3):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            # 发送完成事件
            yield f"data: {json.dumps({'type': 'done', 'session_id': response.session_id, 'execution_time_ms': response.execution_time_ms})}\n\n"

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/actions/{action_id}/confirm", response_model=AgentChatResponse)
async def confirm_agent_action(
    action_id: str,
    request: AgentActionRequest,
    db: Session = Depends(get_db),
):
    """确认并执行首次请求中持久化的写入计划。"""
    try:
        response = await get_agent().confirm_action(action_id, request.session_id, db)
        plan_summary = None
        if response.plan:
            plan_summary = f"意图: {response.plan.intent.value}, 步骤: {len(response.plan.steps)}"
        return AgentChatResponse(
            query=response.query,
            response=response.response,
            session_id=response.session_id,
            intent=response.plan.intent.value if response.plan else None,
            plan_summary=plan_summary,
            sources=response.sources,
            evidence_status=response.evidence_status,
            evidence_score=response.evidence_score,
            evidence_source_count=response.evidence_source_count,
            evidence_reason=response.evidence_reason,
            external_research_available=response.external_research_available,
            execution_time_ms=response.execution_time_ms,
            success=response.confidence >= 1.0,
        )
    except AgentActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/research/", response_model=ExternalResearchResponse)
async def run_external_research(
    request: ExternalResearchRequest,
    db: Session = Depends(get_db),
):
    """通过配置的只读 MCP 工具研究公开资料，不自动写入 Wiki。"""
    try:
        return await get_external_research_service().research(
            request.query,
            request.session_id,
            db,
        )
    except ExternalResearchUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ExternalResearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/research/{run_id}", response_model=ExternalResearchResponse)
async def get_external_research(
    run_id: str,
    session_id: str,
    db: Session = Depends(get_db),
):
    """读取当前会话中的研究状态和可追溯来源。"""
    try:
        return get_external_research_service().get_run(run_id, session_id, db)
    except ExternalResearchNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/research/{run_id}/prepare-save",
    response_model=AgentChatResponse,
)
async def prepare_external_research_save(
    run_id: str,
    request: ExternalResearchSaveRequest,
    db: Session = Depends(get_db),
):
    """生成 Wiki 页面预览；页面仍需通过现有确认接口写入。"""
    try:
        response = get_external_research_service().prepare_save(
            run_id,
            request.session_id,
            db,
            get_agent(),
            request.notebook,
        )
        return AgentChatResponse(
            query=response.query,
            response=response.response,
            session_id=response.session_id,
            intent=response.plan.intent.value if response.plan else None,
            plan_summary=(
                f"意图: {response.plan.intent.value}, 步骤: {len(response.plan.steps)}"
                if response.plan
                else None
            ),
            sources=response.sources,
            confirmation_required=response.confirmation_required,
            pending_action_id=response.pending_action_id,
            action_preview=response.action_preview,
            success=True,
        )
    except ExternalResearchNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExternalResearchConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/actions/{action_id}/cancel")
async def cancel_agent_action(
    action_id: str,
    request: AgentActionRequest,
    db: Session = Depends(get_db),
):
    """取消尚未执行的写入计划。"""
    try:
        return get_agent().cancel_action(action_id, request.session_id, db)
    except AgentActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/sessions/", response_model=SessionListResponse)
async def list_sessions(db: Session = Depends(get_db)):
    """
    列出所有会话

    Returns:
        SessionListResponse: 会话列表
    """
    try:
        memory_service = get_memory_service()
        sessions = memory_service.list_sessions(db)
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

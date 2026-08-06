"""Agent 检索诊断路由。"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.services.hybrid_retrieval_service import get_hybrid_retrieval_service

from .agent_schemas import (
    CompareSearchRequest,
    CompareSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search/", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest, db: Session = Depends(get_db)):
    try:
        started = time.time()
        results = get_hybrid_retrieval_service().search_hybrid(
            query=request.query.strip(),
            top_k=request.top_k,
            bm25_top_k=request.bm25_top_k,
            embedding_top_k=request.embedding_top_k,
            db=db,
        )
        return HybridSearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            execution_time_ms=int((time.time() - started) * 1000),
        )
    except Exception as exc:
        logger.error("混合检索失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/compare/", response_model=CompareSearchResponse)
async def compare_search(request: CompareSearchRequest, db: Session = Depends(get_db)):
    try:
        comparison = get_hybrid_retrieval_service().compare_search_methods(
            query=request.query.strip(),
            top_k=request.top_k,
            db=db,
        )
        return CompareSearchResponse(
            query=request.query,
            bm25_results=comparison["bm25"],
            embedding_results=comparison["embedding"],
            hybrid_results=comparison["hybrid"],
        )
    except Exception as exc:
        logger.error("对比检索失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

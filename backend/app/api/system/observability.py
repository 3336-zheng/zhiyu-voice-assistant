"""本地运行追踪查询接口。"""

from __future__ import annotations

import ipaddress
from typing import Any, Dict, List
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.errors import ErrorCode, safe_error_message
from backend.app.core.observability import (
    REQUEST_ID_PATTERN,
    get_recent_trace,
)
from backend.app.models.observability import AgentRun
from backend.app.models.conversation import Conversation

router = APIRouter()


def _default_model_budget(operation: str | None) -> int:
    """为旧记录按阶段配置推导 max_tokens，避免历史预算显示为 0。"""
    name = str(operation or "")
    if name in {"agent.plan", "agent.replan"}:
        return settings.llm_max_tokens
    if name.endswith(".hyde"):
        return 200
    if name == "agent.query_rewrite" or name.startswith("agent.query_rewrite."):
        return 300
    if name == "agent.crag_grade":
        return 700
    if name.startswith("agent.crag_refine"):
        return 400
    if name == "agent.generation" or name.startswith("agent.generation."):
        return settings.llm_response_max_tokens
    return settings.llm_max_tokens


def _normalized_model_calls(usage: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """补齐旧 model_usage 中缺失的预算字段，不修改数据库原始数据。"""
    normalized = []
    for raw in list((usage or {}).get("calls") or []):
        call = dict(raw)
        budget = call.get("token_budget")
        if budget is None:
            budget = _default_model_budget(call.get("operation"))
        budget = max(0, int(budget or 0))
        completion = int(call.get("completion_tokens", 0) or 0)
        prompt = int(call.get("prompt_tokens", 0) or 0)
        input_budget = call.get("input_budget")
        if input_budget is None:
            input_budget = max(0, settings.llm_context_window_tokens - budget)
        call["token_budget"] = budget
        call["token_remaining"] = max(0, budget - completion)
        call["context_window_tokens"] = int(
            call.get("context_window_tokens") or settings.llm_context_window_tokens
        )
        call["input_budget"] = max(0, int(input_budget or 0))
        call["input_remaining"] = max(0, call["input_budget"] - prompt)
        normalized.append(call)
    return normalized


def _quality_for_run(run: AgentRun) -> Dict[str, Any]:
    """从终态快照读取查询可信度，兼容旧运行记录。"""
    snapshot = run.runtime_snapshot or {}
    quality = snapshot.get("answer_quality")
    if isinstance(quality, dict):
        return dict(quality)
    stats = run.retrieval_stats or {}
    return {
        "confidence": None,
        "evidence_status": "not_applicable",
        "evidence_score": stats.get("evidence_score"),
        "evidence_source_count": stats.get("evidence_source_count", 0),
        "evidence_reason": None,
    }


def _token_budget_for_run(run: AgentRun) -> Dict[str, Any]:
    """读取请求级 Token 预算；兼容尚未保存预算快照的旧记录。"""
    snapshot = run.runtime_snapshot or {}
    stored = snapshot.get("token_budget")
    usage = run.model_usage or {}
    stats = run.retrieval_stats or {}
    calls = _normalized_model_calls(usage)
    output_budget = sum(int(item.get("token_budget", 0) or 0) for item in calls)
    output_used = int(usage.get("completion_tokens", 0) or 0)
    input_budget = sum(int(item.get("input_budget", 0) or 0) for item in calls)
    input_used = int(usage.get("prompt_tokens", 0) or 0)
    rag_budget = stats.get("token_budget")
    rag_used = stats.get("context_tokens")
    model_contexts = {
        str(item.get("operation") or f"llm.{index + 1}"): {
            "total_budget": int(item.get("context_window_tokens", 0) or 0),
            "input_budget": int(item.get("input_budget", 0) or 0),
            "output_reserved_tokens": int(item.get("token_budget", 0) or 0),
            "used_tokens": int(item.get("prompt_tokens", 0) or 0),
            "remaining": int(item.get("input_remaining", 0) or 0),
            "system_tokens": None,
            "summary_tokens": None,
            "recent_tokens": None,
            "current_tokens": None,
            "dropped_recent_messages": None,
            "truncated": bool(item.get("truncated", False)),
            "legacy_estimate": True,
        }
        for index, item in enumerate(calls)
    }
    derived = {
        "definition": "模型输入/输出预算分别累计；RAG 和工具上下文预算独立统计。旧记录仅显示已保存数据。",
        "output": {
            "budget": output_budget,
            "used": output_used,
            "remaining": max(0, output_budget - output_used),
        },
        "input": {
            "budget": input_budget,
            "used": input_used,
            "remaining": max(0, input_budget - input_used),
        },
        "model": {
            "total_used": int(usage.get("total_tokens", 0) or 0),
            "call_count": len(calls),
            "context_window": sum(
                int(item.get("context_window_tokens", 0) or 0) for item in calls
            ),
        },
        "contexts": {
            "rag": {
                "budget": int(rag_budget) if rag_budget is not None else None,
                "used": int(rag_used) if rag_used is not None else None,
                "remaining": (
                    max(0, int(rag_budget) - int(rag_used))
                    if rag_budget is not None and rag_used is not None
                    else None
                ),
                "truncated": bool(stats.get("context_truncated", False)),
                "selected_results": int(stats.get("selected_results", 0) or 0),
            },
            "tool": {"budget": 0, "used": 0, "remaining": 0, "steps": []},
            "model": model_contexts,
            "response_summary": None,
            "memory": {
                "history_budget": settings.memory_context_token_budget,
                "summary_budget": settings.memory_summary_token_budget,
                "summary_trigger": settings.memory_summary_trigger_tokens,
                "summary_input_budget": settings.memory_summary_input_token_budget,
            },
        },
        "calls": [
            {
                "stage": item.get("operation") or "llm",
                "budget": int(item.get("token_budget", 0) or 0),
                "used": int(item.get("completion_tokens", 0) or 0),
                "total_used": int(item.get("total_tokens", 0) or 0),
                "input_tokens": int(item.get("prompt_tokens", 0) or 0),
                "output_tokens": int(item.get("completion_tokens", 0) or 0),
                "remaining": int(item.get("token_remaining", 0) or 0),
            }
            for item in calls
        ],
    }
    if not isinstance(stored, dict) or not ("output" in stored or "contexts" in stored):
        return derived

    merged = dict(stored)
    stored_output = dict(merged.get("output") or {})
    if int(stored_output.get("budget", 0) or 0) <= 0 and output_budget > 0:
        merged["output"] = derived["output"]
        merged["input"] = derived["input"]
        merged["model"] = derived["model"]
        merged["calls"] = derived["calls"]
    else:
        merged.setdefault("input", derived["input"])
        merged.setdefault("model", derived["model"])
        merged.setdefault("calls", derived["calls"])

    stored_contexts = dict(merged.get("contexts") or {})
    derived_contexts = derived["contexts"]
    stored_contexts.setdefault("rag", derived_contexts["rag"])
    stored_contexts.setdefault("tool", derived_contexts["tool"])
    if not stored_contexts.get("model"):
        stored_contexts["model"] = derived_contexts["model"]
    stored_contexts.setdefault("response_summary", None)
    stored_contexts.setdefault("memory", derived_contexts["memory"])
    merged["contexts"] = stored_contexts
    merged.setdefault("definition", derived["definition"])
    return merged


def _run_summary(run: AgentRun, conversation_title: str | None = None) -> Dict[str, Any]:
    quality = _quality_for_run(run)
    completed_at = run.completed_at or run.updated_at or run.created_at
    return {
        "request_id": run.request_id,
        "trace_type": "agent",
        "conversation_title": conversation_title or run.query or "新对话",
        "query": run.query,
        "intent": run.intent,
        "status": run.status,
        "total_ms": run.execution_time_ms or 0,
        "completed_at": completed_at.timestamp() if completed_at else None,
        "error_code": "AGENT_EXECUTION_ERROR" if run.error else None,
        **quality,
    }


def _legacy_stage_result(
    stage: str,
    run: AgentRun,
    stats: Dict[str, Any],
    quality: Dict[str, Any],
    model_call: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """用已持久化的聚合数据补全旧时间线，不推测未保存的内容。"""
    crag = stats.get("crag") or {}
    if stage == "agent.plan":
        return {
            "intent": run.intent,
            "recording_note": "旧记录未保存计划步骤",
        }
    if stage == "agent.fast_path":
        return {"mode": "fast_path", "planner_skipped": True}
    if stage == "agent.query_rewrite":
        result = {
            key: stats[key]
            for key in ("retrieval_query", "rewritten_queries", "query_count")
            if key in stats
        }
        if "rewritten_queries" not in result:
            result["recording_note"] = "旧记录未保存改写文本"
        return result
    if stage == "retrieval.embedding":
        return {"hits": stats["embedding_hits"]} if "embedding_hits" in stats else {}
    if stage == "retrieval.bm25":
        return {"hits": stats["bm25_hits"]} if "bm25_hits" in stats else {}
    if stage in {"retrieval.recall", "agent.retrieve"}:
        return {
            key: stats[key]
            for key in (
                "query_count",
                "bm25_hits",
                "embedding_hits",
                "fused_candidates",
                "reranked_candidates",
                "selected_results",
                "context_tokens",
                "token_budget",
            )
            if key in stats
        }
    if stage == "retrieval.fusion":
        return {"fused_candidates": stats["fused_candidates"]} if "fused_candidates" in stats else {}
    if stage == "retrieval.fetch_chunks":
        return {"fetched_candidates": stats["reranked_candidates"]} if "reranked_candidates" in stats else {}
    if stage == "retrieval.rerank":
        result = {}
        if "reranked_candidates" in stats:
            result["candidates"] = stats["reranked_candidates"]
        selected = stats.get("quality_filtered_candidates", stats.get("selected_results"))
        if selected is not None:
            result["selected"] = selected
            result["recording_note"] = "旧记录未保存保留文档明细"
        return result
    if stage == "agent.crag_grade":
        return dict(crag)
    if stage == "agent.evidence":
        if quality.get("evidence_status") != "not_applicable" or quality.get("evidence_score") is not None:
            return dict(quality)
        return {"recording_note": "旧记录未保存证据门禁结果"}
    if stage == "agent.generation":
        persisted_quality = (run.runtime_snapshot or {}).get("answer_quality")
        source_count = (
            quality.get("evidence_source_count")
            if isinstance(persisted_quality, dict) or "evidence_source_count" in stats
            else None
        )
        return {
            "answer_generated": bool(run.response) if run.response is not None else None,
            "source_count": source_count,
            "recording_note": "旧记录未保存引用明细",
        }
    if stage.startswith("llm.") and model_call:
        return {
            key: model_call[key]
            for key in (
                "operation",
                "model",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "duration_ms",
                "success",
                "finish_reason",
                "token_budget",
                "token_remaining",
                "input_budget",
                "input_remaining",
            )
            if model_call.get(key) is not None
        }
    if stage == "agent.total":
        return {"total_ms": run.execution_time_ms or 0}
    return {}


def _timeline_for_run(
    run: AgentRun,
    timeline: List[Dict[str, Any]],
    model_usage: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """保留新记录的原始结果，并为旧记录补全可恢复字段。"""
    stats = run.retrieval_stats or {}
    quality = _quality_for_run(run)
    calls = _normalized_model_calls(model_usage or run.model_usage)
    model_index = 0
    enriched = []
    for raw_item in timeline:
        item = dict(raw_item)
        stage = str(item.get("stage") or "")
        model_call = None
        if stage.startswith("llm."):
            if model_index < len(calls):
                model_call = calls[model_index]
            model_index += 1

        current = item.get("result")
        has_result = isinstance(current, dict) and any(key != "status" for key in current)
        if not has_result:
            result = _legacy_stage_result(stage, run, stats, quality, model_call)
            item["result"] = result or {"recording_note": "旧记录仅保存了阶段耗时"}
        enriched.append(item)
    return enriched


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
    db: Session = Depends(get_db),
    conversation: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=30, ge=1, le=100),
) -> Dict[str, List[Dict[str, Any]]]:
    _require_local_access(request)
    runs_query = db.query(AgentRun, Conversation.title).outerjoin(
        Conversation,
        Conversation.session_id == AgentRun.session_id,
    )
    normalized = (conversation or "").strip()
    if normalized:
        pattern = f"%{normalized}%"
        runs_query = runs_query.filter(
            or_(
                Conversation.title.ilike(pattern),
                AgentRun.query.ilike(pattern),
            )
        )
    rows = (
        runs_query
        .order_by(AgentRun.updated_at.desc(), AgentRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return {"requests": [_run_summary(run, title) for run, title in rows]}


@router.get("/requests/{request_id}")
async def request_trace(
    request_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_local_access(request)
    if not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise HTTPException(status_code=422, detail="Request ID 格式不正确")

    run = db.get(AgentRun, request_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到对应的运行记录")

    conversation_title = (
        db.query(Conversation.title)
        .filter(Conversation.session_id == run.session_id)
        .scalar()
    )
    snapshot = get_recent_trace(request_id)
    if snapshot is not None and snapshot.get("trace_type") == "agent":
        snapshot["conversation_title"] = conversation_title or run.query or "新对话"
        snapshot["query"] = run.query
        snapshot["intent"] = run.intent
        snapshot["answer_quality"] = _quality_for_run(run)
        snapshot["token_budget"] = _token_budget_for_run(run)
        snapshot["retrieval_stats"] = run.retrieval_stats or {}
        snapshot["timeline"] = _timeline_for_run(
            run,
            snapshot.get("timeline") or [],
            snapshot.get("model_usage") or {},
        )
        return snapshot

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
        "conversation_title": conversation_title or run.query or "新对话",
        "query": run.query,
        "intent": run.intent,
        "status": run.status,
        "status_code": 200 if run.status == "completed" else 500,
        "total_ms": run.execution_time_ms or 0,
        "stage_timings": {},
        "timeline": _timeline_for_run(run, run.timeline or []),
        "retrieval_stats": run.retrieval_stats or {},
        "answer_quality": _quality_for_run(run),
        "model_usage": run.model_usage or {},
        "token_budget": _token_budget_for_run(run),
        "error": error,
        "completed_at": run.completed_at.timestamp() if run.completed_at else None,
        "source": "agent_run",
    }

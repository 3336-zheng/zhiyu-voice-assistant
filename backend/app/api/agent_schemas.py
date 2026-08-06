"""Agent API 共享请求与响应模型。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=64)


class AgentChatResponse(BaseModel):
    query: str
    response: str
    session_id: Optional[str] = None
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
    request_id: Optional[str] = None
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_stats: Optional[Dict[str, Any]] = None
    model_usage: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    success: bool = True


class AgentActionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


class AgentRunStatusResponse(BaseModel):
    run_id: str
    session_id: str
    status: str
    last_sequence: int
    response: Optional[Any] = None
    error: Optional[str] = None


class ExternalResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=64)


class ExternalResearchSaveRequest(BaseModel):
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


class MCPStatusResponse(BaseModel):
    enabled: bool
    available: bool
    server_label: str
    tools: Dict[str, str]
    limits: Dict[str, Any]
    configuration_source: str
    status: Optional[str] = None
    error: Optional[str] = None


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100)
    bm25_top_k: Optional[int] = Field(default=None, ge=1, le=200)
    embedding_top_k: Optional[int] = Field(default=None, ge=1, le=200)


class HybridSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    execution_time_ms: Optional[int] = None


class CompareSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100)


class CompareSearchResponse(BaseModel):
    query: str
    bm25_results: List[Dict[str, Any]]
    embedding_results: List[Dict[str, Any]]
    hybrid_results: List[Dict[str, Any]]


class SessionListResponse(BaseModel):
    sessions: List[Dict[str, Any]]
    total: int

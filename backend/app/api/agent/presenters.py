"""Agent 领域响应到 HTTP 响应模型的转换。"""

from backend.app.agent.models import AgentResponse

from .schemas import AgentChatResponse


def present_agent_response(
    response: AgentResponse,
    *,
    success: bool = True,
) -> AgentChatResponse:
    """集中维护 Agent 响应字段映射，避免各路由重复拼装。"""
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
        confirmation_required=response.confirmation_required,
        pending_action_id=response.pending_action_id,
        action_preview=response.action_preview,
        evidence_status=response.evidence_status,
        evidence_score=response.evidence_score,
        evidence_source_count=response.evidence_source_count,
        evidence_reason=response.evidence_reason,
        external_research_available=response.external_research_available,
        request_id=response.request_id,
        timeline=response.timeline,
        retrieval_stats=response.retrieval_stats,
        model_usage=response.model_usage,
        token_budget=response.token_budget,
        execution_time_ms=response.execution_time_ms,
        success=success,
    )

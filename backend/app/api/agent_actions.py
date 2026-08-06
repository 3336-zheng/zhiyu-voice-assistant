"""Agent 知识变更确认路由。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.agent.agent import AgentActionError, get_agent
from backend.app.core.database import get_db

from .agent_presenters import present_agent_response
from .agent_schemas import AgentActionRequest, AgentChatResponse

router = APIRouter()


@router.post("/actions/{action_id}/confirm", response_model=AgentChatResponse)
async def confirm_agent_action(
    action_id: str,
    request: AgentActionRequest,
    db: Session = Depends(get_db),
):
    try:
        response = await get_agent().confirm_action(action_id, request.session_id, db)
        return present_agent_response(response, success=response.confidence >= 1.0)
    except AgentActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/actions/{action_id}/cancel")
async def cancel_agent_action(
    action_id: str,
    request: AgentActionRequest,
    db: Session = Depends(get_db),
):
    try:
        return get_agent().cancel_action(action_id, request.session_id, db)
    except AgentActionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

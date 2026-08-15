"""受控 MCP 外部研究路由。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.agent.agent import get_agent
from backend.app.core.database import get_db
from backend.app.services.research.external_research_service import (
    ExternalResearchConflict,
    ExternalResearchError,
    ExternalResearchNotFound,
    ExternalResearchUnavailable,
    get_external_research_service,
)
from backend.app.services.research.mcp_client_service import get_mcp_client_service

from .presenters import present_agent_response
from .schemas import (
    AgentChatResponse,
    ExternalResearchRequest,
    ExternalResearchResponse,
    ExternalResearchSaveRequest,
    MCPStatusResponse,
)

router = APIRouter()


@router.post("/research/", response_model=ExternalResearchResponse)
async def run_external_research(
    request: ExternalResearchRequest,
    db: Session = Depends(get_db),
):
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


@router.get("/mcp/status", response_model=MCPStatusResponse)
async def get_mcp_status():
    return get_mcp_client_service().describe()


@router.get("/mcp/health", response_model=MCPStatusResponse)
async def check_mcp_health():
    return await get_mcp_client_service().check_health()


@router.get("/research/{run_id}", response_model=ExternalResearchResponse)
async def get_external_research(
    run_id: str,
    session_id: str,
    db: Session = Depends(get_db),
):
    try:
        return get_external_research_service().get_run(run_id, session_id, db)
    except ExternalResearchNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/research/{run_id}/prepare-save", response_model=AgentChatResponse)
async def prepare_external_research_save(
    run_id: str,
    request: ExternalResearchSaveRequest,
    db: Session = Depends(get_db),
):
    try:
        response = get_external_research_service().prepare_save(
            run_id,
            request.session_id,
            db,
            get_agent(),
            request.notebook,
        )
        return present_agent_response(response)
    except ExternalResearchNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExternalResearchConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

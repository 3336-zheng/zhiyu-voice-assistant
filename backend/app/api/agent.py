"""Agent API 聚合入口。"""

from fastapi import APIRouter

from backend.app.agent.agent import get_agent

from .agent_actions import router as actions_router
from .agent_feedback import router as feedback_router
from .agent_research import router as research_router
from .agent_runs import router as runs_router
from .agent_schemas import *  # noqa: F403
from .agent_search import router as search_router
from .agent_sessions import router as sessions_router

router = APIRouter()
router.include_router(runs_router)
router.include_router(actions_router)
router.include_router(feedback_router)
router.include_router(research_router)
router.include_router(sessions_router)
router.include_router(search_router)

__all__ = ["router", "get_agent"]

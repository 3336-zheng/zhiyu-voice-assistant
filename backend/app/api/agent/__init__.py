"""Agent API 聚合入口。"""

from fastapi import APIRouter

from backend.app.agent.agent import get_agent

from .actions import router as actions_router
from .feedback import router as feedback_router
from .research import router as research_router
from .runs import router as runs_router
from .schemas import *  # noqa: F403
from .search import router as search_router
from .sessions import router as sessions_router

router = APIRouter()
router.include_router(runs_router)
router.include_router(actions_router)
router.include_router(feedback_router)
router.include_router(research_router)
router.include_router(sessions_router)
router.include_router(search_router)

__all__ = ["router", "get_agent"]

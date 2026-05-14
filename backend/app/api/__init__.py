"""
API路由模块
"""
from .audio import router as audio_router
from .notes import router as notes_router
from .health import router as health_router
from .agent import router as agent_router
from .docs import router as docs_router

__all__ = ["audio_router", "notes_router", "health_router", "agent_router", "docs_router"]
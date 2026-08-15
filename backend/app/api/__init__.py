"""
API路由模块
"""
from .agent import router as agent_router
from .ingestion.audio import router as audio_router
from .ingestion.summary import router as summary_router
from .system.health import router as health_router
from .system.observability import router as observability_router
from .wiki.documents import router as docs_router
from .wiki.notes import router as notes_router
from .wiki.pages import router as pages_router

__all__ = [
    "audio_router",
    "notes_router",
    "health_router",
    "agent_router",
    "docs_router",
    "summary_router",
    "pages_router",
    "observability_router",
]

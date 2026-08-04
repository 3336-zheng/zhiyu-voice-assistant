"""
智语端侧智能语音笔记助手后端应用
"""
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .api import audio_router, notes_router, health_router, agent_router, pages_router
from .api.docs import router as docs_router
from .api.summary import router as summary_router
from .core.config import settings
from .core.lifecycle import lifespan
from .core.observability import (
    build_server_timing,
    get_stage_timings,
    record_timing,
    reset_request,
    start_request,
)

logger = logging.getLogger(__name__)


# 请求追踪和异常边界中间件
class RequestContextMiddleware:
    """覆盖完整响应生命周期，流式接口也能记录最终阶段耗时。"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id, request_token, timings_token = start_request(
            headers.get("X-Request-ID")
        )
        started = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_context(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers["X-Request-ID"] = request_id
                elapsed_ms = (time.perf_counter() - started) * 1000
                response_headers["Server-Timing"] = build_server_timing(elapsed_ms)
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        except Exception as exc:
            logger.error(
                "请求失败 request_id=%s method=%s path=%s error=%s",
                request_id,
                scope.get("method"),
                scope.get("path"),
                type(exc).__name__,
                exc_info=True,
            )
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content={"detail": "服务器内部错误", "request_id": request_id},
            )
            await response(scope, receive, send_with_context)
        finally:
            total_ms = (time.perf_counter() - started) * 1000
            record_timing("http.total", total_ms)
            timings = get_stage_timings()
            logger.info(
                "请求完成 request_id=%s method=%s path=%s status=%s total_ms=%.3f timings=%s",
                request_id,
                scope.get("method"),
                scope.get("path"),
                status_code,
                total_ms,
                json.dumps(timings, ensure_ascii=False, sort_keys=True),
            )
            reset_request(request_token, timings_token)


# 创建应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="智语端侧智能语音笔记助手API",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# 添加请求日志中间件（最先添加，最后执行）
app.add_middleware(RequestContextMiddleware)

# 添加CORS中间件
# 开源部署时建议将 allow_origins 限制为实际前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(audio_router, prefix="/audio", tags=["音频管理"])
app.include_router(notes_router, prefix="/notes", tags=["兼容 API：笔记"])
app.include_router(agent_router, prefix="/agent", tags=["智能助手"])
app.include_router(health_router, prefix="/health", tags=["健康检查"])
app.include_router(docs_router, prefix="/api/documents", tags=["兼容 API：文档"])
app.include_router(summary_router, prefix="/summary", tags=["纪要总结"])
app.include_router(pages_router, prefix="/api/pages", tags=["Wiki 页面"])

# 挂载前端静态文件（必须放在最后）。构建后优先使用 React，未构建时回退旧页面。
react_frontend = Path("frontend/dist")
frontend_directory = react_frontend if react_frontend.exists() else Path("frontend")
app.mount("/", StaticFiles(directory=str(frontend_directory), html=True), name="frontend")

"""
智语端侧智能语音笔记助手后端应用
"""
import os
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from .api import audio_router, notes_router, health_router, agent_router
from .api.docs import router as docs_router
from .api.summary import router as summary_router
from .core.config import settings

logger = logging.getLogger(__name__)


# 请求日志中间件
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.debug(f"{request.method} {request.url.path}")
        try:
            response = await call_next(request)
            logger.debug(f"{request.method} {request.url.path} -> {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"{request.method} {request.url.path} -> {type(e).__name__}: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": f"服务器内部错误: {type(e).__name__}: {str(e)}"}
            )


def migrate_relative_paths():
    """迁移数据库中的相对路径为绝对路径"""
    try:
        from .core.database import SessionLocal
        from .models import Audio

        db = SessionLocal()
        try:
            audios = db.query(Audio).all()
            updated_count = 0
            for audio in audios:
                if audio.file_path and not os.path.isabs(audio.file_path):
                    old_path = audio.file_path
                    new_path = os.path.abspath(old_path)
                    audio.file_path = new_path
                    updated_count += 1
                    logger.info(f"路径迁移: {old_path} -> {new_path}")
            if updated_count > 0:
                db.commit()
                logger.info(f"已迁移 {updated_count} 条记录的路径")
            else:
                logger.debug("无需迁移路径")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"路径迁移失败: {e}", exc_info=True)


# 创建应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="智语端侧智能语音笔记助手API",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)


async def _periodic_cleanup():
    """定时清理过期会话的后台任务"""
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_minutes * 60)
            from .services.memory_service import get_memory_service
            from .core.database import SessionLocal
            db = SessionLocal()
            try:
                result = get_memory_service().cleanup_expired_sessions(db)
                if result.get("cleaned_sessions", 0) > 0:
                    logger.info(f"清理过期会话: {result}")
            finally:
                db.close()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"清理任务异常: {e}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    """应用启动时执行的初始化操作"""
    logger.info("应用启动中...")
    # 确保数据库目录存在并自动创建表
    db_dir = os.path.dirname(settings.database_url.replace("sqlite:///", ""))
    os.makedirs(db_dir, exist_ok=True)
    from .core.database import engine, Base
    Base.metadata.create_all(bind=engine)
    # 清理已废弃的 notes 表（笔记已迁移到文件系统 data/notes/*.md）
    with engine.connect() as conn:
        try:
            from sqlalchemy import text
            conn.execute(text("DROP TABLE IF EXISTS notes"))
            conn.commit()
            logger.info("已移除废弃的 notes 表")
        except Exception:
            pass
    logger.info(f"数据库表已就绪: {db_dir}")
    # 确保上传目录是绝对路径
    upload_dir = settings.get_upload_dir()
    logger.info(f"上传目录: {upload_dir}")
    # 迁移旧的相对路径记录
    migrate_relative_paths()
    # 增量同步文档索引（BM25 从持久化数据重建，ChromaDB 仅更新变化的文件）
    try:
        from .services.doc_index_service import get_doc_index_service
        doc_index = get_doc_index_service()
        result = doc_index.sync_docs()
        logger.info(f"文档增量同步完成: {result}")
    except Exception as e:
        logger.warning(f"文档同步失败（不影响启动）: {e}", exc_info=True)
    # 启动过期会话定时清理任务
    asyncio.create_task(_periodic_cleanup())
    logger.info(f"过期会话清理任务已启动（间隔 {settings.cleanup_interval_minutes} 分钟，TTL {settings.session_ttl_hours} 小时）")
    logger.info("应用启动完成")

# 添加请求日志中间件（最先添加，最后执行）
app.add_middleware(LoggingMiddleware)

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
app.include_router(notes_router, prefix="/notes", tags=["笔记管理"])
app.include_router(agent_router, prefix="/agent", tags=["智能助手"])
app.include_router(health_router, prefix="/health", tags=["健康检查"])
app.include_router(docs_router, prefix="/api/documents", tags=["文档管理"])
app.include_router(summary_router, prefix="/summary", tags=["纪要总结"])

# 挂载前端静态文件（必须放在最后）
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
"""FastAPI 生命周期和后台任务管理。"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings

logger = logging.getLogger(__name__)


def _initialize_database() -> None:
    from .database import Base, engine
    from .. import models as _models  # noqa: F401
    from .schema import ensure_schema

    database_path = settings.database_url.replace("sqlite:///", "")
    database_dir = os.path.dirname(database_path)
    if database_dir:
        os.makedirs(database_dir, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    schema_version = ensure_schema(engine)
    logger.info("数据库 schema 已就绪，版本: %s", schema_version)


def _migrate_relative_paths() -> None:
    """将旧音频记录中的相对路径迁移为绝对路径。"""
    from .database import SessionLocal
    from ..models import Audio

    db = SessionLocal()
    try:
        updated_count = 0
        for audio in db.query(Audio).all():
            if audio.file_path and not os.path.isabs(audio.file_path):
                audio.file_path = os.path.abspath(audio.file_path)
                updated_count += 1
        if updated_count:
            db.commit()
            logger.info("已迁移 %s 条音频路径记录", updated_count)
    finally:
        db.close()


def _sync_document_index() -> None:
    from ..services.doc_index_service import get_doc_index_service

    result = get_doc_index_service().sync_docs()
    logger.info("文档增量同步完成: %s", result)


def _recover_wiki_index_tasks() -> None:
    from .database import SessionLocal
    from ..services.page_service import get_page_service

    db = SessionLocal()
    try:
        result = get_page_service(db).recover_index_tasks()
        logger.info("Wiki 索引任务恢复完成: %s", result)
    finally:
        db.close()


async def _periodic_cleanup() -> None:
    """定时清理过期对话会话。"""
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_minutes * 60)
            from .database import SessionLocal
            from ..services.memory_service import get_memory_service

            db = SessionLocal()
            try:
                result = get_memory_service().cleanup_expired_sessions(db)
                if result.get("cleaned_sessions", 0):
                    logger.info("清理过期会话: %s", result)
            finally:
                db.close()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("清理任务异常: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化持久化资源，并统一管理后台任务的启停。"""
    logger.info("应用启动中...")
    _initialize_database()
    settings.get_upload_dir()
    _migrate_relative_paths()

    if settings.sync_legacy_docs_on_startup:
        try:
            await asyncio.to_thread(_sync_document_index)
        except Exception as exc:
            logger.warning("旧文档同步失败（不影响启动）: %s", exc, exc_info=True)
    else:
        logger.info("旧 data/docs 自动索引已关闭；需要时请执行显式迁移")

    try:
        _recover_wiki_index_tasks()
    except Exception as exc:
        logger.warning("Wiki 索引任务恢复失败（不影响启动）: %s", exc, exc_info=True)

    from ..services.wiki_index_worker import run_wiki_index_worker

    tasks = [
        asyncio.create_task(run_wiki_index_worker(), name="wiki-index-worker"),
        asyncio.create_task(_periodic_cleanup(), name="session-cleanup-worker"),
    ]
    app.state.background_tasks = tasks
    logger.info("应用启动完成")
    try:
        yield
    finally:
        logger.info("应用关闭中...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        app.state.background_tasks = []
        logger.info("应用已关闭")

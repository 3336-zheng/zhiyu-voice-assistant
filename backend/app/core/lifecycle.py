"""FastAPI 生命周期和后台任务管理。"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import settings

logger = logging.getLogger(__name__)


def _initialize_database() -> None:
    """按数据库方言决定建表方式。

    SQLite 面向本地开发与测试，直接由 ORM 元数据建表，克隆仓库即可运行。

    MySQL 面向多实例部署，schema 一律交给 Alembic，应用启动不执行任何 DDL：
    多个实例同时启动会并发执行建表语句而互相撞车，且 MySQL 的 DDL 不在事务内，
    中途失败无法回滚，会留下一个半完成的库，比直接启动失败更难排查。
    """
    from .database import Base, engine, is_sqlite
    from .. import models as _models  # noqa: F401

    if not is_sqlite():
        logger.info("数据库 schema 由 Alembic 管理，启动阶段不建表（部署前执行 alembic upgrade head）")
        return

    database_path = settings.database_url.replace("sqlite:///", "")
    database_dir = os.path.dirname(database_path)
    if database_dir:
        os.makedirs(database_dir, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("SQLite schema 已就绪（由 ORM 元数据建表）")


def _migrate_relative_paths() -> None:
    """将旧音频记录中的相对路径迁移为绝对路径。

    这是一次性的历史数据订正，正常情况下查不到任何行。过滤下推到 SQL，
    启动时才不必为了「找那几条可能不存在的记录」把整张表拉进内存。

    多实例同时启动会更新同一批行，靠 MySQL 的行锁串行化；目标值相同
    （各容器的工作目录都是 /app），后写的覆盖成同样的结果，没有一致性风险。
    """
    from .database import SessionLocal
    from ..models import Audio

    db = SessionLocal()
    try:
        # 绝对路径以 "/" 开头（Linux 容器内不存在盘符路径）。SQL 只做预过滤，
        # 权威判据仍是 os.path.isabs——两者不一致时以它为准。
        candidates = (
            db.query(Audio)
            .filter(Audio.file_path.isnot(None))
            .filter(~Audio.file_path.like("/%"))
            .all()
        )
        updated_count = 0
        for audio in candidates:
            if os.path.isabs(audio.file_path):
                continue
            audio.file_path = os.path.abspath(audio.file_path)
            updated_count += 1
        if updated_count:
            db.commit()
            logger.info("已迁移 %s 条音频路径记录", updated_count)
    finally:
        db.close()


def _sync_document_index() -> None:
    from ..services.ingestion.doc_index_service import get_doc_index_service

    result = get_doc_index_service().sync_docs()
    logger.info("文档增量同步完成: %s", result)


def _rebuild_bm25_index() -> None:
    """从 ChromaDB 恢复 BM25 内存索引，避免重启后只剩向量检索。"""
    from ..services.ingestion.doc_index_service import get_doc_index_service

    get_doc_index_service().rebuild_bm25_from_persistent()


async def _periodic_cleanup() -> None:
    """定时清理过期对话会话。

    多实例下每个实例都会跑这个循环，所以用选主锁保证同一时刻只有一个在清理：
    清理本身是幂等的，但 N 份并发的全表查询只会互相加锁、白占连接。
    """
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval_minutes * 60)
            await _cleanup_once()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("清理任务异常: %s", exc, exc_info=True)


async def _cleanup_once() -> None:
    """执行一轮清理。抢到执行权才真的做，没抢到就跳过这一轮。"""
    from .leader_lock import try_acquire

    # TTL 取间隔的 0.9 倍：正好在下一轮开始前过期，每轮都能重新竞争；
    # 持有者崩溃时最多一个间隔之后有人接手。取满一个间隔的话，持有者
    # 自己下一轮再来抢也会失败，清理会停摆一整轮。
    ttl_seconds = int(settings.cleanup_interval_minutes * 60 * 0.9)
    if not await try_acquire("session-cleanup", ttl_seconds):
        return

    def _run() -> dict:
        from .database import SessionLocal
        from ..services.memory.memory_service import get_memory_service

        db = SessionLocal()
        try:
            return get_memory_service().cleanup_expired_sessions(db)
        finally:
            db.close()

    # 清理是同步的数据库操作，放进线程池，别让它阻塞事件循环——
    # 多实例下每次清理都要扫全表，卡住的是正在处理用户请求的同一个循环。
    result = await asyncio.to_thread(_run)
    if result.get("cleaned_sessions", 0):
        logger.info("清理过期会话: %s", result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化持久化资源，并统一管理后台任务的启停。"""
    logger.info("应用启动中...")
    from .telemetry import configure_telemetry

    configure_telemetry()
    _initialize_database()
    from ..services.runtime.agent_runtime_service import get_agent_runtime_service

    agent_runtime = get_agent_runtime_service()
    # 这里**不再**回收「运行中」的任务：Agent 运行现在跑在 worker 进程里，
    # API 重启不该把它们标记成失败（滚动重启会把正在进行的对话全杀掉）。
    # 回收改由 agent worker 在自己的启动流程里做，见 tasks/worker.py。
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
            await asyncio.to_thread(_rebuild_bm25_index)
        except Exception as exc:
            logger.warning("BM25 持久化索引恢复失败（不影响启动）: %s", exc, exc_info=True)

    # 投递器必须先于任务恢复就绪：恢复会把卡在 processing 的任务改回 pending
    # 并立即投递，投递器没起来的话它们就只能等兜底扫描了。
    from ..tasks.queue import close_dispatcher, init_dispatcher

    try:
        await init_dispatcher()
    except Exception as exc:
        # Redis 不可用不该拦住应用启动——任务留在表里，等队列恢复后由兜底扫描补投。
        logger.warning("任务投递器初始化失败（索引任务将由兜底扫描补投）: %s", exc)

    # 这里**不再**做「把 processing 的索引任务改回 pending」的启动恢复。
    #
    # 那个逻辑（原 recover_index_tasks）无差别地捞所有 processing 任务，前提是
    # 「进程重启了，它手上的任务必然都死了」——这只在单进程时代成立。索引执行
    # 搬到 arq worker 之后（Stage 5），执行者是另一个进程：API 重启时 worker 可能
    # 正在正常索引某个页面，把它改回 pending 并重投，同一个页面就会被索引两次；
    # 更糟的是 worker 随后写 completed、这里刚写了 pending，谁后写谁赢，任务状态
    # 可能永久停在 pending 被反复重投。
    #
    # 回收改由 worker 的兜底扫描做（tasks/index_tasks.py 的 sweep_pending_tasks），
    # 按 locked_at 判断执行者是否真的失联，每 60 秒一次，多实例下也安全。

    tasks = [
        asyncio.create_task(_periodic_cleanup(), name="session-cleanup-worker"),
    ]
    app.state.background_tasks = tasks
    logger.info("应用启动完成")
    try:
        yield
    finally:
        logger.info("应用关闭中...")
        await agent_runtime.shutdown()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        app.state.background_tasks = []
        await close_dispatcher()
        from ..services.runtime.agent_state import get_agent_state_store

        get_agent_state_store().close()
        from .observability import close_recent_trace_store

        close_recent_trace_store()
        from .leader_lock import close as close_leader_lock

        # 选主锁的连接池也要收掉，否则每次刷新（uvicorn 的热重载）都会漏一个池。
        # 不用主动释放锁：留着 TTL 让它自然过期，比在这里多一次网络往返简单。
        await close_leader_lock()
        logger.info("应用已关闭")

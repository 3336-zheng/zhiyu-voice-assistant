"""Wiki 索引后台 worker。"""

import asyncio
import logging

from ..core.config import settings
from ..core.database import SessionLocal
from .page_service import get_page_service

logger = logging.getLogger(__name__)


def process_index_batch() -> dict:
    """在独立数据库会话中处理一批索引任务。"""
    db = SessionLocal()
    try:
        return get_page_service(db).process_pending_index_tasks(
            limit=settings.wiki_index_batch_size
        )
    finally:
        db.close()


async def run_wiki_index_worker() -> None:
    """持续轮询索引队列；模型不可用时依靠任务退避，不阻塞 API。"""
    logger.info(
        "Wiki 索引 worker 已启动（间隔 %.1f 秒，每批 %s）",
        settings.wiki_index_poll_seconds,
        settings.wiki_index_batch_size,
    )
    while True:
        try:
            result = await asyncio.to_thread(process_index_batch)
            if result.get("total", 0):
                logger.info("Wiki 索引 worker 处理结果: %s", result)
            await asyncio.sleep(settings.wiki_index_poll_seconds)
        except asyncio.CancelledError:
            logger.info("Wiki 索引 worker 已停止")
            return
        except Exception as exc:
            logger.error("Wiki 索引 worker 异常: %s", exc, exc_info=True)
            await asyncio.sleep(settings.wiki_index_poll_seconds)

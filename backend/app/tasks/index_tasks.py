"""索引任务的执行体。

任务逻辑本身一行没改，还是调 WikiIndexTaskService.process。这里只负责开数据库
会话、读一次退避时间，把「什么时候该再试」交回给表。

退避完全由 WikiIndexTask.next_attempt_at 决定，**不用 arq 自己的重试机制**：
两套退避并存的话，实际间隔取决于谁先到，线上就说不清一条任务到底要等多久。
所以 arq 那边 max_tries=1，重投由这里显式发起。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _process(task_id: str) -> Tuple[Dict[str, Any], float]:
    """执行一条任务，并顺带算出它距离下次重试还有多少秒。

    在同一个会话里把退避时间读出来，省得为了拿一个时间戳再开一次连接。
    """
    from backend.app.models.wiki import WikiIndexTask
    from backend.app.services.wiki.page_service import get_page_service

    db = SessionLocal()
    try:
        result = get_page_service(db).process_index_task(task_id)
        retry_in = 0.0
        if result.get("status") == "failed":
            task = db.get(WikiIndexTask, task_id)
            if task is not None and task.next_attempt_at is not None:
                retry_in = max(
                    0.0, (task.next_attempt_at - _utc_now()).total_seconds()
                )
        return result, retry_in
    finally:
        db.close()


async def index_page(ctx: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """执行一条 Wiki 页面索引任务，失败时按表里的退避时间重新投递。"""
    result, retry_in = await asyncio.to_thread(_process, task_id)

    if result.get("status") == "failed":
        logger.warning(
            "索引任务失败，%.0f 秒后重试: task=%s attempts=%s error=%s",
            retry_in, task_id, result.get("attempts"), result.get("error"),
        )
        # 不抛异常：抛了 arq 会用自己的重试策略，和表里的 next_attempt_at 打架。
        await ctx["redis"].enqueue_job(
            "index_page", task_id, _defer_by=retry_in, _queue_name=settings.arq_index_queue
        )
    return result


def _load_due_task_ids(limit: int) -> List[str]:
    from backend.app.services.wiki.page_service import get_page_service

    db = SessionLocal()
    try:
        return get_page_service(db).due_index_task_ids(limit=limit)
    finally:
        db.close()


def _recover_stale() -> int:
    from backend.app.services.wiki.page_service import get_page_service

    db = SessionLocal()
    try:
        return get_page_service(db).recover_stale_index_tasks(
            settings.wiki_index_stale_seconds
        )["recovered"]
    finally:
        db.close()


async def sweep_pending_tasks(ctx: Dict[str, Any]) -> int:
    """兜底扫描：回收卡死的任务，并把没能投进队列的到期任务补投一遍。

    正常路径下任务在事务提交后就被直接投递了，这里捡的是异常路径——投递时
    Redis 抖动了一下、进程在提交后立刻退出、或者上一次重投本身失败了。
    所以它不需要跑得很勤，每分钟一次足够；漏掉的任务多等一分钟，
    比在每条写入路径上都做同步重试要划算得多。

    回收卡死任务必须放在这里而不能只放在启动恢复里：worker 是独立进程，
    它崩溃时 API 进程还活着，不会有重启来触发恢复，那些任务会永远停在 processing。
    """
    recovered = await asyncio.to_thread(_recover_stale)
    task_ids = await asyncio.to_thread(
        _load_due_task_ids, settings.wiki_index_sweep_limit
    )
    for task_id in task_ids:
        await ctx["redis"].enqueue_job(
            "index_page", task_id, _queue_name=settings.arq_index_queue
        )
    if recovered or task_ids:
        logger.info(
            "兜底扫描：回收 %s 条卡死任务，补投 %s 条索引任务", recovered, len(task_ids)
        )
    return len(task_ids)

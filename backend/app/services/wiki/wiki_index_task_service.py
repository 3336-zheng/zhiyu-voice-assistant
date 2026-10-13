"""Wiki 持久化索引任务服务。"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.errors import ErrorCode
from backend.app.core.logging_config import log_event
from backend.app.models.wiki import WikiIndexTask, WikiPage
from backend.app.tasks.queue import dispatch_index_tasks
from .page_errors import PageNotFoundError

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WikiIndexTaskService:
    """负责索引任务入队、执行、退避重试和重启恢复。"""

    def __init__(
        self,
        db: Session,
        *,
        page_reader: Callable[[str], Dict[str, Any]],
        index_service_getter: Callable[[], Any],
        dispatcher: Callable[[List[str]], None] = dispatch_index_tasks,
    ):
        self.db = db
        self.page_reader = page_reader
        self.index_service_getter = index_service_getter
        # 投递器做成依赖，测试里可以换成记录调用的替身；
        # 默认那个在投递器未初始化时会安静跳过，把任务留给兜底扫描。
        self.dispatcher = dispatcher

    @staticmethod
    def new_task(page_id: str, revision: int, action: str) -> WikiIndexTask:
        return WikiIndexTask(
            id=str(uuid.uuid4()),
            page_id=page_id,
            revision=revision,
            action=action,
            status="pending",
        )

    def process(self, task_id: str) -> Dict[str, Any]:
        # 原子抢占：把「看状态」和「改成 processing」压成一条 UPDATE。
        # 分成读—改—写三步的话，两个 worker 可能都读到 pending 然后都执行，
        # 同一个页面被索引两次。单实例下不会暴露，多开之后必踩。
        claimed = (
            self.db.query(WikiIndexTask)
            .filter(
                WikiIndexTask.id == task_id,
                WikiIndexTask.status.in_(("pending", "failed")),
            )
            .update(
                {"status": "processing", "locked_at": utc_now()},
                synchronize_session=False,
            )
        )
        self.db.commit()

        task = self.db.get(WikiIndexTask, task_id)
        if task is None:
            raise PageNotFoundError(f"索引任务不存在: {task_id}")
        if not claimed:
            # 已经被别的 worker 领走，或状态已不允许执行。这次什么都不做。
            logger.info(
                "索引任务已被领取或状态不允许执行: %s (当前 %s)", task_id, task.status
            )
            return self._task_result(task)

        page = self.db.get(WikiPage, task.page_id)
        task.attempts += 1
        task.updated_at = utc_now()
        try:
            index_service = self.index_service_getter()
            if task.action == "delete" or page is None or page.status == "deleted":
                index_service.remove_page(task.page_id)
            else:
                index_service.index_page(self.page_reader(page.id))
            task.status = "completed"
            task.error = None
            task.next_attempt_at = None
            task.locked_at = None
            if page:
                page.index_status = "ready" if page.status != "deleted" else "deleted"
                page.index_error = None
            self.db.commit()
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "页面索引任务失败",
                component="index",
                operation="index_page",
                error_code=ErrorCode.INDEX_PROVIDER_ERROR,
                retryable=True,
                exception=exc,
                task_id=task.id,
            )
            task.status = "failed"
            task.error = str(exc)
            backoff = min(
                settings.wiki_index_max_backoff_seconds,
                5 * (2 ** min(task.attempts - 1, 6)),
            )
            task.next_attempt_at = utc_now() + timedelta(seconds=backoff)
            task.locked_at = None
            if page:
                page.index_status = "failed"
                page.index_error = str(exc)
            self.db.commit()
        return self._task_result(task)

    def retry(self, limit: int = 100) -> Dict[str, Any]:
        tasks = (
            self.db.query(WikiIndexTask)
            .filter(WikiIndexTask.status.in_(["pending", "failed"]))
            .order_by(WikiIndexTask.created_at)
            .limit(limit)
            .all()
        )
        for task in tasks:
            task.next_attempt_at = None
            task.status = "pending"
        self.db.commit()
        results = [self.process(task.id) for task in tasks]
        return self._batch_result(results)

    def queue_reindex(self) -> Dict[str, Any]:
        pages = self.db.query(WikiPage).all()
        queued = 0
        affected: List[str] = []
        for page in pages:
            page.index_status = "pending"
            page.index_error = None
            action = "delete" if page.status == "deleted" else "upsert"
            existing = (
                self.db.query(WikiIndexTask)
                .filter(
                    WikiIndexTask.page_id == page.id,
                    WikiIndexTask.revision == page.revision,
                    WikiIndexTask.action == action,
                    WikiIndexTask.status.in_(["pending", "processing", "failed"]),
                )
                .first()
            )
            if existing:
                existing.status = "pending"
                existing.next_attempt_at = None
                existing.locked_at = None
                # 被重置回 pending 的旧任务同样要投递，否则它们只能等兜底扫描。
                affected.append(existing.id)
                continue
            task = self.new_task(page.id, page.revision, action)
            self.db.add(task)
            affected.append(task.id)
            queued += 1
        self.db.commit()
        self.dispatch(affected)
        return {"queued": queued, "total": len(pages), "status": "queued"}

    def recover(self) -> Dict[str, Any]:
        tasks = self.db.query(WikiIndexTask).filter(WikiIndexTask.status == "processing").all()
        for task in tasks:
            task.status = "pending"
            task.locked_at = None
            task.next_attempt_at = None
        self.db.commit()
        # 改回 pending 只是把任务放回待执行，没人投递的话它们会一直躺着——
        # 轮询时代靠下一轮扫描捡起来，现在必须显式投一次。
        self.dispatch([task.id for task in tasks])
        return {"recovered": len(tasks)}

    def recover_stale(self, threshold_seconds: int) -> Dict[str, Any]:
        """回收卡死在 processing 的任务。

        worker 独立成进程之后，它崩溃时没有机会执行 recover()——此时 API 进程
        还活着，不会有启动恢复来兜底，这些任务会永远停在 processing。所以按
        locked_at 判断：超过阈值还没有回音，就认为执行它的 worker 已经不在了。

        阈值必须大于任务超时时间，否则正在正常执行的长任务会被误判成卡死并重跑。
        """
        cutoff = utc_now() - timedelta(seconds=threshold_seconds)
        tasks = (
            self.db.query(WikiIndexTask)
            .filter(WikiIndexTask.status == "processing")
            .filter(
                (WikiIndexTask.locked_at.is_(None))
                | (WikiIndexTask.locked_at <= cutoff)
            )
            .all()
        )
        for task in tasks:
            task.status = "pending"
            task.locked_at = None
            task.next_attempt_at = None
        self.db.commit()
        if tasks:
            logger.warning("回收 %s 条卡在 processing 的索引任务", len(tasks))
        return {"recovered": len(tasks)}

    def dispatch(self, task_ids: List[str]) -> None:
        """把已提交的任务投进队列。投递是 best-effort，失败由兜底扫描补。"""
        self.dispatcher(task_ids)

    def due_task_ids(self, limit: int = 100) -> List[str]:
        """列出到期可执行的任务 id，供兜底扫描补投。

        只取 id 不取整行：扫描的目的是把它们送进队列，任务内容由 worker
        执行时再读，避免扫描本身把一批正文拖进内存。
        """
        now = utc_now()
        rows = (
            self.db.query(WikiIndexTask.id)
            .filter(WikiIndexTask.status.in_(("pending", "failed")))
            .filter(
                (WikiIndexTask.next_attempt_at.is_(None))
                | (WikiIndexTask.next_attempt_at <= now)
            )
            .order_by(WikiIndexTask.created_at)
            .limit(limit)
            .all()
        )
        return [row[0] for row in rows]

    def process_pending(self, limit: int = 5) -> Dict[str, Any]:
        """同步处理一批到期任务。

        状态抢占交给 process() 自己做，这里**不能**先把状态改成 processing——
        那样 process() 里的原子 UPDATE 会因为状态不再是 pending/failed 而放弃，
        任务就永远执行不了。
        """
        now = utc_now()
        task_ids = [
            task.id
            for task in (
                self.db.query(WikiIndexTask)
                .filter(WikiIndexTask.status.in_(["pending", "failed"]))
                .filter(
                    (WikiIndexTask.next_attempt_at.is_(None))
                    | (WikiIndexTask.next_attempt_at <= now)
                )
                .order_by(WikiIndexTask.created_at)
                .limit(limit)
                .all()
            )
        ]
        return self._batch_result([self.process(task_id) for task_id in task_ids])

    @staticmethod
    def _task_result(task: WikiIndexTask) -> Dict[str, Any]:
        return {
            "task_id": task.id,
            "page_id": task.page_id,
            "revision": task.revision,
            "action": task.action,
            "status": task.status,
            "attempts": task.attempts,
            "error": task.error,
        }

    @staticmethod
    def _batch_result(results: list[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "total": len(results),
            "completed": sum(item["status"] == "completed" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "results": results,
        }

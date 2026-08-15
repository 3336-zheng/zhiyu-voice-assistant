"""Wiki 持久化索引任务服务。"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.errors import ErrorCode
from backend.app.core.logging_config import log_event
from backend.app.models.wiki import WikiIndexTask, WikiPage
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
    ):
        self.db = db
        self.page_reader = page_reader
        self.index_service_getter = index_service_getter

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
        task = self.db.get(WikiIndexTask, task_id)
        if task is None:
            raise PageNotFoundError(f"索引任务不存在: {task_id}")
        page = self.db.get(WikiPage, task.page_id)
        task.attempts += 1
        task.status = "processing"
        task.locked_at = utc_now()
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
                continue
            self.db.add(self.new_task(page.id, page.revision, action))
            queued += 1
        self.db.commit()
        return {"queued": queued, "total": len(pages), "status": "queued"}

    def recover(self) -> Dict[str, Any]:
        tasks = self.db.query(WikiIndexTask).filter(WikiIndexTask.status == "processing").all()
        for task in tasks:
            task.status = "pending"
            task.locked_at = None
            task.next_attempt_at = None
        self.db.commit()
        return {"recovered": len(tasks)}

    def process_pending(self, limit: int = 5) -> Dict[str, Any]:
        now = utc_now()
        tasks = (
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
        results = []
        for task in tasks:
            task.status = "processing"
            task.locked_at = now
            self.db.commit()
            results.append(self.process(task.id))
        return self._batch_result(results)

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

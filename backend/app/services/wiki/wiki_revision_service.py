"""Wiki 页面历史版本服务。"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.models.wiki import WikiPage, WikiPageRevision
from .page_errors import PageNotFoundError


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class WikiRevisionService:
    """负责版本快照的创建和读取。"""

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def new_revision(
        page: WikiPage,
        content: str,
        metadata: Dict[str, Any],
        change_summary: Optional[str],
    ) -> WikiPageRevision:
        return WikiPageRevision(
            page_id=page.id,
            revision=page.revision,
            title=page.title,
            content=content,
            page_metadata=metadata,
            change_summary=change_summary,
        )

    def list_revisions(self, page_id: str) -> List[Dict[str, Any]]:
        if self.db.get(WikiPage, page_id) is None:
            raise PageNotFoundError(f"页面不存在: {page_id}")
        rows = (
            self.db.query(WikiPageRevision)
            .filter(WikiPageRevision.page_id == page_id)
            .order_by(WikiPageRevision.revision.desc())
            .all()
        )
        return [
            {
                "revision": row.revision,
                "title": row.title,
                "change_summary": row.change_summary,
                "created_at": _isoformat(row.created_at),
            }
            for row in rows
        ]

    def get_revision(self, page_id: str, revision: int) -> Dict[str, Any]:
        row = (
            self.db.query(WikiPageRevision)
            .filter(
                WikiPageRevision.page_id == page_id,
                WikiPageRevision.revision == revision,
            )
            .one_or_none()
        )
        if row is None:
            raise PageNotFoundError(f"页面版本不存在: {page_id}@{revision}")
        return {
            "page_id": page_id,
            "revision": row.revision,
            "title": row.title,
            "content": row.content,
            "metadata": row.page_metadata,
            "change_summary": row.change_summary,
            "created_at": _isoformat(row.created_at),
        }

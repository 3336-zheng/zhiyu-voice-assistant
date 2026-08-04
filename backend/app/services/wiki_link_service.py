"""Wiki 双向链接服务。"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from ..models.wiki import WikiPage, WikiPageLink
from .page_errors import PageNotFoundError, PageValidationError
from .wiki_file_store import WikiFileStore

logger = logging.getLogger(__name__)
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


class WikiLinkService:
    """负责解析 Wiki Link、出链和反向链接。"""

    def __init__(self, db: Session, file_store: WikiFileStore):
        self.db = db
        self.file_store = file_store

    def rebuild_all(self) -> None:
        """重建页面关系，使新页面可解析之前的悬空链接。"""
        pages = self.db.query(WikiPage).filter(WikiPage.status != "deleted").all()
        title_map: Dict[str, List[str]] = {}
        for page in pages:
            for name in [page.title, *(page.aliases or [])]:
                title_map.setdefault(str(name).casefold(), []).append(page.id)
        self.db.query(WikiPageLink).delete(synchronize_session=False)
        for page in pages:
            path = Path(page.file_path)
            if not path.exists():
                continue
            try:
                _, content = self.file_store.parse_page(path.read_text(encoding="utf-8"))
            except PageValidationError:
                logger.warning("跳过 Front Matter 异常页面的链接解析: %s", path)
                continue
            targets = {
                match.strip()
                for match in WIKI_LINK_PATTERN.findall(content)
                if match.strip()
            }
            for target_title in sorted(targets):
                matches = title_map.get(target_title.casefold(), [])
                target_page_id = matches[0] if len(matches) == 1 else None
                self.db.add(
                    WikiPageLink(
                        source_page_id=page.id,
                        target_page_id=target_page_id,
                        target_title=target_title,
                    )
                )
        self.db.commit()

    def get_links(self, page_id: str) -> Dict[str, List[Dict[str, Any]]]:
        if self.db.get(WikiPage, page_id) is None:
            raise PageNotFoundError(f"页面不存在: {page_id}")
        outgoing_rows = (
            self.db.query(WikiPageLink)
            .filter(WikiPageLink.source_page_id == page_id)
            .order_by(WikiPageLink.target_title)
            .all()
        )
        backlink_rows = (
            self.db.query(WikiPageLink)
            .filter(WikiPageLink.target_page_id == page_id)
            .order_by(WikiPageLink.source_page_id)
            .all()
        )
        outgoing = [
            {
                "target_page_id": row.target_page_id,
                "target_title": row.target_title,
                "resolved": row.target_page_id is not None,
            }
            for row in outgoing_rows
        ]
        backlinks = []
        for row in backlink_rows:
            source = self.db.get(WikiPage, row.source_page_id)
            if source and source.status != "deleted":
                backlinks.append({"page_id": source.id, "title": source.title})
        return {"outgoing": outgoing, "backlinks": backlinks}

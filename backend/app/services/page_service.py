"""统一的 Wiki 页面编排服务。"""

import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.wiki import WikiPage
from .page_errors import (
    AmbiguousPageError,
    PageConflictError,
    PageNotFoundError,
    PageServiceError,
    PageValidationError,
)
from .wiki_file_store import WikiFileStore
from .wiki_index_task_service import WikiIndexTaskService
from .wiki_link_service import WikiLinkService
from .wiki_revision_service import WikiRevisionService

ACTIVE_STATUS = "active"


def utc_now() -> datetime:
    """返回带时区的 UTC 时间。"""
    return datetime.now(timezone.utc)


def _search_snippet(content: str, query: str, radius: int = 56) -> str:
    """返回 Markdown 正文命中词附近的单行片段。"""
    normalized = re.sub(r"\s+", " ", content or "").strip()
    index = normalized.casefold().find(query.casefold())
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(normalized), index + len(query) + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(normalized) else ""
    return f"{prefix}{normalized[start:end]}{suffix}"


class PageService:
    """协调 Wiki 页面写入，并向 API 提供稳定门面。"""

    def __init__(
        self,
        db: Session,
        pages_dir: Optional[str] = None,
        index_service: Any = None,
    ):
        self.db = db
        self.pages_dir = Path(pages_dir or settings.wiki_pages_dir).resolve()
        self._index_service = index_service
        self.file_store = WikiFileStore(self.pages_dir)
        self.revisions = WikiRevisionService(db)
        self.links = WikiLinkService(db, self.file_store)
        self.index_tasks = WikiIndexTaskService(
            db,
            page_reader=self.get_page,
            index_service_getter=lambda: self.index_service,
        )

    @property
    def index_service(self):
        """延迟加载索引服务，避免模型初始化影响纯页面操作。"""
        if self._index_service is None:
            from .page_index_service import get_page_index_service

            self._index_service = get_page_index_service()
        return self._index_service

    def create_page(
        self,
        *,
        title: str,
        content: str,
        notebook: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        aliases: Optional[Iterable[str]] = None,
        status: str = ACTIVE_STATUS,
        source_type: str = "manual",
        source_uri: Optional[str] = None,
        change_summary: Optional[str] = None,
        sync_index: bool = False,
        page_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建页面；索引失败不会回滚已经保存的主数据。"""
        title = self._validate_title(title)
        content = self._validate_content(content)
        page_id = page_id or str(uuid.uuid4())
        self._validate_page_id(page_id)
        if self.db.get(WikiPage, page_id):
            raise PageConflictError(f"页面 ID 已存在: {page_id}")

        now = utc_now()
        page_path = self._page_path(page_id)
        normalized_tags = self._normalize_strings(tags)
        normalized_aliases = self._normalize_strings(aliases, exclude={title})
        metadata = self._build_metadata(
            page_id=page_id,
            title=title,
            notebook=notebook,
            tags=normalized_tags,
            aliases=normalized_aliases,
            status=status,
            source_type=source_type,
            source_uri=source_uri,
            revision=1,
            created_at=now,
            updated_at=now,
        )
        serialized = self.serialize_page(metadata, content)
        self._atomic_write(page_path, serialized)

        page = WikiPage(
            id=page_id,
            title=title,
            notebook=notebook,
            tags=normalized_tags,
            aliases=normalized_aliases,
            status=status,
            source_type=source_type,
            source_uri=source_uri,
            file_path=str(page_path),
            revision=1,
            content_hash=self._content_hash(content),
            index_status="pending",
            created_at=now,
            updated_at=now,
        )
        revision = self._new_revision(page, content, metadata, change_summary or "创建页面")
        task = self._new_index_task(page_id, 1, "upsert")

        try:
            self.db.add_all([page, revision, task])
            self.db.commit()
        except Exception:
            self.db.rollback()
            page_path.unlink(missing_ok=True)
            raise

        self._rebuild_all_links()
        if sync_index:
            self.process_index_task(task.id)
        return self.get_page(page_id)

    def upsert_page_by_source(
        self,
        *,
        title: str,
        content: str,
        source_type: str,
        source_uri: str,
        notebook: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        aliases: Optional[Iterable[str]] = None,
        change_summary: Optional[str] = None,
        sync_index: bool = False,
    ) -> Dict[str, Any]:
        """按来源幂等写入；内容变化时更新原页面并保留版本。"""
        existing = (
            self.db.query(WikiPage)
            .filter(
                WikiPage.source_type == source_type,
                WikiPage.source_uri == source_uri,
                WikiPage.status != "deleted",
            )
            .order_by(WikiPage.updated_at.desc())
            .first()
        )
        normalized_content = self._validate_content(content)
        if existing is None:
            result = self.create_page(
                title=title,
                content=normalized_content,
                notebook=notebook,
                tags=tags,
                aliases=aliases,
                source_type=source_type,
                source_uri=source_uri,
                change_summary=change_summary,
                sync_index=sync_index,
            )
            result["deduplicated"] = False
            return result
        if existing.content_hash == self._content_hash(normalized_content):
            result = self.get_page(existing.id)
            result["deduplicated"] = True
            return result
        result = self.update_page(
            existing.id,
            expected_revision=existing.revision,
            title=title,
            content=normalized_content,
            notebook=notebook,
            tags=tags,
            aliases=aliases,
            source_type=source_type,
            source_uri=source_uri,
            change_summary=change_summary or "来源内容更新",
            sync_index=sync_index,
        )
        result["deduplicated"] = False
        return result

    def get_page(self, page_id: str, *, include_deleted: bool = False) -> Dict[str, Any]:
        """按稳定 ID 获取页面。"""
        page = self._get_page_row(page_id, include_deleted=include_deleted)
        content = ""
        metadata = self._row_metadata(page)
        path = Path(page.file_path)
        if path.exists():
            metadata, content = self.parse_page(path.read_text(encoding="utf-8"))
            self._validate_file_identity(page, metadata)
        elif page.status != "deleted":
            raise PageValidationError(f"页面文件不存在: {page.file_path}")
        return self._page_to_dict(page, content, metadata)

    def find_page(self, reference: str, *, include_deleted: bool = False) -> Dict[str, Any]:
        """兼容 UUID、UUID.md、标题、别名和旧文件名查找。"""
        reference = (reference or "").strip()
        if not reference:
            raise PageValidationError("页面引用不能为空")
        candidate_id = Path(reference).stem
        page = self.db.get(WikiPage, candidate_id)
        if page and (include_deleted or page.status != "deleted"):
            return self.get_page(page.id, include_deleted=include_deleted)

        matches = []
        lowered = candidate_id.casefold()
        query = self.db.query(WikiPage)
        if not include_deleted:
            query = query.filter(WikiPage.status != "deleted")
        for row in query.all():
            names = [row.title, *(row.aliases or [])]
            if any(str(name).casefold() == lowered for name in names):
                matches.append(row)
        if not matches:
            raise PageNotFoundError(f"页面不存在: {reference}")
        if len(matches) > 1:
            raise AmbiguousPageError(f"页面引用存在歧义: {reference}")
        return self.get_page(matches[0].id, include_deleted=include_deleted)

    def list_pages(
        self,
        *,
        notebook: Optional[str] = None,
        tag: Optional[str] = None,
        query: Optional[str] = None,
        source_type: Optional[str] = None,
        include_deleted: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """分页列出页面，并支持元数据及 Markdown 正文过滤。"""
        rows_query = self.db.query(WikiPage)
        if not include_deleted:
            rows_query = rows_query.filter(WikiPage.status != "deleted")
        if notebook:
            rows_query = rows_query.filter(WikiPage.notebook == notebook)
        if source_type:
            rows_query = rows_query.filter(WikiPage.source_type == source_type)
        rows = rows_query.order_by(WikiPage.updated_at.desc()).all()

        normalized_query = (query or "").strip()
        lowered_query = normalized_query.casefold() if normalized_query else None
        filtered = []
        for row in rows:
            if tag and tag not in (row.tags or []):
                continue
            match_snippet = None
            if lowered_query:
                names = [row.title, *(row.aliases or []), *(row.tags or [])]
                metadata_matches = any(
                    lowered_query in str(value).casefold() for value in names
                )
                content_matches = False
                try:
                    _, content = self.file_store.read_page(row)
                    content_matches = lowered_query in content.casefold()
                    if content_matches:
                        match_snippet = _search_snippet(content, normalized_query)
                except (OSError, PageValidationError):
                    content_matches = False
                if not metadata_matches and not content_matches:
                    continue
            filtered.append((row, match_snippet))

        items = []
        for row, match_snippet in filtered[offset:offset + limit]:
            item = self._page_summary(row)
            item["match_snippet"] = match_snippet
            items.append(item)
        return {"total": len(filtered), "offset": offset, "limit": limit, "items": items}

    def update_page(
        self,
        page_id: str,
        *,
        expected_revision: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        notebook: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        aliases: Optional[Iterable[str]] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        source_uri: Optional[str] = None,
        change_summary: Optional[str] = None,
        sync_index: bool = False,
    ) -> Dict[str, Any]:
        """更新页面并生成新版本；必须提供当前版本号。"""
        page = self._get_page_row(page_id)
        current = self.get_page(page_id)
        if page.revision != expected_revision:
            raise PageConflictError(
                f"页面版本冲突，当前为 {page.revision}，请求基于 {expected_revision}"
            )

        new_title = self._validate_title(title) if title is not None else page.title
        new_content = self._validate_content(content) if content is not None else current["content"]
        new_aliases = self._normalize_strings(aliases) if aliases is not None else list(page.aliases or [])
        if new_title != page.title and page.title not in new_aliases:
            new_aliases.append(page.title)
        new_aliases = self._normalize_strings(new_aliases, exclude={new_title})
        values = {
            "title": new_title,
            "content": new_content,
            "notebook": notebook if notebook is not None else page.notebook,
            "tags": self._normalize_strings(tags) if tags is not None else list(page.tags or []),
            "aliases": new_aliases,
            "status": status if status is not None else page.status,
            "source_type": source_type if source_type is not None else page.source_type,
            "source_uri": source_uri if source_uri is not None else page.source_uri,
        }
        return self._apply_update(
            page,
            current,
            values,
            change_summary or "更新页面",
            sync_index=sync_index,
        )

    def rename_page(
        self,
        page_id: str,
        *,
        title: str,
        expected_revision: int,
        sync_index: bool = False,
    ) -> Dict[str, Any]:
        """重命名页面，旧标题会自动保留为别名。"""
        return self.update_page(
            page_id,
            title=title,
            expected_revision=expected_revision,
            change_summary=f"重命名页面为 {title}",
            sync_index=sync_index,
        )

    def delete_page(
        self,
        page_id: str,
        *,
        expected_revision: int,
        sync_index: bool = False,
    ) -> Dict[str, Any]:
        """软删除页面主记录并删除当前 Markdown，历史版本继续保留。"""
        page = self._get_page_row(page_id)
        if page.revision != expected_revision:
            raise PageConflictError(
                f"页面版本冲突，当前为 {page.revision}，请求基于 {expected_revision}"
            )
        current = self.get_page(page_id)
        previous_serialized = Path(page.file_path).read_text(encoding="utf-8")
        now = utc_now()
        page.revision += 1
        page.status = "deleted"
        page.deleted_at = now
        page.updated_at = now
        page.index_status = "pending"
        metadata = self._row_metadata(page)
        revision = self._new_revision(page, current["content"], metadata, "删除页面")
        task = self._new_index_task(page.id, page.revision, "delete")
        Path(page.file_path).unlink(missing_ok=True)
        try:
            self.db.add_all([revision, task])
            self.db.commit()
        except Exception:
            self.db.rollback()
            self._atomic_write(Path(page.file_path), previous_serialized)
            raise
        self._rebuild_all_links()
        if sync_index:
            self.process_index_task(task.id)
        return self._page_to_dict(page, current["content"], metadata)

    def list_revisions(self, page_id: str) -> List[Dict[str, Any]]:
        """列出页面历史版本，不返回大段正文。"""
        return self.revisions.list_revisions(page_id)

    def get_revision(self, page_id: str, revision: int) -> Dict[str, Any]:
        """读取指定历史版本的完整快照。"""
        return self.revisions.get_revision(page_id, revision)

    def rollback_page(
        self,
        page_id: str,
        *,
        target_revision: int,
        expected_revision: int,
        sync_index: bool = False,
    ) -> Dict[str, Any]:
        """将历史快照恢复为一个新的版本。"""
        page = self._get_page_row(page_id, include_deleted=True)
        if page.revision != expected_revision:
            raise PageConflictError(
                f"页面版本冲突，当前为 {page.revision}，请求基于 {expected_revision}"
            )
        snapshot = self.get_revision(page_id, target_revision)
        current = self._current_snapshot(page)
        meta = snapshot["metadata"] or {}
        values = {
            "title": self._validate_title(snapshot["title"]),
            "content": snapshot["content"],
            "notebook": meta.get("notebook"),
            "tags": self._normalize_strings(meta.get("tags")),
            "aliases": self._normalize_strings(meta.get("aliases")),
            "status": ACTIVE_STATUS,
            "source_type": meta.get("source_type", page.source_type),
            "source_uri": meta.get("source_uri"),
        }
        page.deleted_at = None
        return self._apply_update(
            page,
            current,
            values,
            f"回滚到版本 {target_revision}",
            sync_index=sync_index,
        )

    def get_links(self, page_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """返回出链和反向链接，未解析链接也会保留。"""
        return self.links.get_links(page_id)

    def process_index_task(self, task_id: str) -> Dict[str, Any]:
        """执行一条持久化索引任务并记录成功或失败状态。"""
        return self.index_tasks.process(task_id)

    def retry_index_tasks(self, limit: int = 100) -> Dict[str, Any]:
        """人工立即重试待处理和失败任务。"""
        return self.index_tasks.retry(limit)

    def queue_reindex(self) -> Dict[str, Any]:
        """将全部页面加入索引队列，实际处理由后台 worker 完成。"""
        return self.index_tasks.queue_reindex()

    def recover_index_tasks(self) -> Dict[str, Any]:
        """将进程中断时遗留的 processing 任务恢复为 pending。"""
        return self.index_tasks.recover()

    def process_pending_index_tasks(self, limit: int = 5) -> Dict[str, Any]:
        """处理一批到期任务，供后台 worker 调用。"""
        return self.index_tasks.process_pending(limit)

    def import_legacy_directory(
        self,
        directory: str,
        *,
        source_type: str,
        notebook: Optional[str] = None,
        sync_index: bool = False,
    ) -> Dict[str, Any]:
        """幂等导入旧 Markdown/TXT 目录，相同来源不会重复导入。"""
        root = Path(directory).resolve()
        if not root.exists():
            return {"total": 0, "imported": 0, "skipped": 0, "errors": []}
        results = {"total": 0, "imported": 0, "skipped": 0, "errors": []}
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            results["total"] += 1
            source_uri = str(path)
            exists = (
                self.db.query(WikiPage)
                .filter(
                    WikiPage.source_uri == source_uri,
                    WikiPage.status != "deleted",
                )
                .first()
            )
            if exists:
                results["skipped"] += 1
                continue
            try:
                raw = path.read_text(encoding="utf-8")
                title, content = self._legacy_title_content(path, raw)
                self.create_page(
                    title=title,
                    content=content,
                    notebook=notebook,
                    source_type=source_type,
                    source_uri=source_uri,
                    change_summary=f"从旧目录导入 {path.name}",
                    sync_index=sync_index,
                )
                results["imported"] += 1
            except Exception as exc:
                results["errors"].append({"filename": path.name, "error": str(exc)})
        return results

    @staticmethod
    def serialize_page(metadata: Dict[str, Any], content: str) -> str:
        """将页面序列化为 UTF-8 Markdown + YAML Front Matter。"""
        return WikiFileStore.serialize_page(metadata, content)

    @staticmethod
    def parse_page(raw: str) -> tuple[Dict[str, Any], str]:
        """解析并校验 YAML Front Matter。"""
        return WikiFileStore.parse_page(raw)

    def _apply_update(
        self,
        page: WikiPage,
        current: Dict[str, Any],
        values: Dict[str, Any],
        change_summary: str,
        *,
        sync_index: bool,
    ) -> Dict[str, Any]:
        old_path = Path(page.file_path)
        previous_serialized = old_path.read_text(encoding="utf-8") if old_path.exists() else None
        now = utc_now()
        next_revision = page.revision + 1
        metadata = self._build_metadata(
            page_id=page.id,
            title=values["title"],
            notebook=values["notebook"],
            tags=values["tags"],
            aliases=values["aliases"],
            status=values["status"],
            source_type=values["source_type"],
            source_uri=values["source_uri"],
            revision=next_revision,
            created_at=page.created_at,
            updated_at=now,
        )
        self._atomic_write(old_path, self.serialize_page(metadata, values["content"]))

        page.title = values["title"]
        page.notebook = values["notebook"]
        page.tags = values["tags"]
        page.aliases = values["aliases"]
        page.status = values["status"]
        page.source_type = values["source_type"]
        page.source_uri = values["source_uri"]
        page.revision = next_revision
        page.content_hash = self._content_hash(values["content"])
        page.updated_at = now
        page.deleted_at = None if values["status"] != "deleted" else page.deleted_at
        page.index_status = "pending"
        page.index_error = None
        revision = self._new_revision(page, values["content"], metadata, change_summary)
        task = self._new_index_task(page.id, next_revision, "upsert")
        try:
            self.db.add_all([revision, task])
            self.db.commit()
        except Exception:
            self.db.rollback()
            if previous_serialized is None:
                old_path.unlink(missing_ok=True)
            else:
                self._atomic_write(old_path, previous_serialized)
            raise
        self._rebuild_all_links()
        if sync_index:
            self.process_index_task(task.id)
        return self.get_page(page.id)

    def _rebuild_all_links(self) -> None:
        """重建页面关系，使新页面可解析之前的悬空链接。"""
        self.links.rebuild_all()

    def _get_page_row(self, page_id: str, *, include_deleted: bool = False) -> WikiPage:
        page = self.db.get(WikiPage, page_id)
        if page is None or (page.status == "deleted" and not include_deleted):
            raise PageNotFoundError(f"页面不存在: {page_id}")
        return page

    def _current_snapshot(self, page: WikiPage) -> Dict[str, Any]:
        path = Path(page.file_path)
        if path.exists():
            metadata, content = self.parse_page(path.read_text(encoding="utf-8"))
            return self._page_to_dict(page, content, metadata)
        revision = self.get_revision(page.id, page.revision)
        return self._page_to_dict(page, revision["content"], revision["metadata"])

    @staticmethod
    def _new_revision(
        page: WikiPage,
        content: str,
        metadata: Dict[str, Any],
        change_summary: Optional[str],
    ):
        return WikiRevisionService.new_revision(page, content, metadata, change_summary)

    @staticmethod
    def _new_index_task(page_id: str, revision: int, action: str):
        return WikiIndexTaskService.new_task(page_id, revision, action)

    def _page_path(self, page_id: str) -> Path:
        return self.file_store.page_path(page_id)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        WikiFileStore.atomic_write(path, content)

    @staticmethod
    def _validate_title(title: str) -> str:
        return WikiFileStore.validate_title(title)

    @staticmethod
    def _validate_content(content: str) -> str:
        return WikiFileStore.validate_content(content)

    @staticmethod
    def _validate_page_id(page_id: str) -> None:
        WikiFileStore.validate_page_id(page_id)

    @staticmethod
    def _normalize_strings(
        values: Optional[Iterable[str]],
        *,
        exclude: Optional[set[str]] = None,
    ) -> List[str]:
        return WikiFileStore.normalize_strings(values, exclude=exclude)

    @staticmethod
    def _content_hash(content: str) -> str:
        return WikiFileStore.content_hash(content)

    @staticmethod
    def _build_metadata(
        *,
        page_id: str,
        title: str,
        notebook: Optional[str],
        tags: List[str],
        aliases: List[str],
        status: str,
        source_type: str,
        source_uri: Optional[str],
        revision: int,
        created_at: datetime,
        updated_at: datetime,
    ) -> Dict[str, Any]:
        return WikiFileStore.build_metadata(
            page_id=page_id,
            title=title,
            notebook=notebook,
            tags=tags,
            aliases=aliases,
            status=status,
            source_type=source_type,
            source_uri=source_uri,
            revision=revision,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _row_metadata(self, page: WikiPage) -> Dict[str, Any]:
        return self.file_store.row_metadata(page)

    def _page_to_dict(
        self,
        page: WikiPage,
        content: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.file_store.page_to_dict(page, content, metadata)

    @staticmethod
    def _page_summary(page: WikiPage) -> Dict[str, Any]:
        return WikiFileStore.page_summary(page)

    @staticmethod
    def _validate_file_identity(page: WikiPage, metadata: Dict[str, Any]) -> None:
        WikiFileStore.validate_file_identity(page, metadata)

    @staticmethod
    def _legacy_title_content(path: Path, raw: str) -> tuple[str, str]:
        return WikiFileStore.legacy_title_content(path, raw)


def get_page_service(db: Session, **kwargs) -> PageService:
    """创建绑定当前数据库会话的页面服务。"""
    return PageService(db=db, **kwargs)

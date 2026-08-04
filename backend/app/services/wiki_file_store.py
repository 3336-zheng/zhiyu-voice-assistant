"""Wiki Markdown 主数据存储。"""

import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ..models.wiki import WikiPage
from .page_errors import PageValidationError


def _isoformat(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class WikiFileStore:
    """负责 Markdown 文件、Front Matter 和页面序列化。"""

    def __init__(self, pages_dir: str | Path):
        self.pages_dir = Path(pages_dir).resolve()
        self.pages_dir.mkdir(parents=True, exist_ok=True)

    def page_path(self, page_id: str) -> Path:
        return self.pages_dir / f"{page_id}.md"

    def read_page(self, page: WikiPage) -> tuple[Dict[str, Any], str]:
        path = Path(page.file_path)
        if not path.exists():
            raise PageValidationError(f"页面文件不存在: {page.file_path}")
        metadata, content = self.parse_page(path.read_text(encoding="utf-8"))
        self.validate_file_identity(page, metadata)
        return metadata, content

    @staticmethod
    def serialize_page(metadata: Dict[str, Any], content: str) -> str:
        """将页面序列化为 UTF-8 Markdown + YAML Front Matter。"""
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        return f"---\n{frontmatter}\n---\n\n{content.rstrip()}\n"

    @staticmethod
    def parse_page(raw: str) -> tuple[Dict[str, Any], str]:
        """解析并校验 YAML Front Matter。"""
        if not raw.startswith("---\n"):
            raise PageValidationError("页面缺少 YAML Front Matter")
        marker = raw.find("\n---\n", 4)
        if marker == -1:
            raise PageValidationError("页面 Front Matter 未闭合")
        yaml_text = raw[4:marker]
        try:
            metadata = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as exc:
            raise PageValidationError(f"页面 Front Matter 格式错误: {exc}") from exc
        if not isinstance(metadata, dict):
            raise PageValidationError("页面 Front Matter 必须是对象")
        required = {"id", "title", "revision"}
        missing = required - set(metadata)
        if missing:
            fields = ", ".join(sorted(missing))
            raise PageValidationError(f"页面 Front Matter 缺少字段: {fields}")
        content = raw[marker + 5 :].lstrip("\n")
        if content.endswith("\n"):
            content = content[:-1]
        return metadata, content

    @staticmethod
    def atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def validate_title(title: str) -> str:
        title = (title or "").strip()
        if not title:
            raise PageValidationError("页面标题不能为空")
        if len(title) > 255:
            raise PageValidationError("页面标题不能超过 255 个字符")
        if "\x00" in title:
            raise PageValidationError("页面标题包含非法字符")
        return title

    @staticmethod
    def validate_content(content: str) -> str:
        if content is None:
            raise PageValidationError("页面内容不能为空")
        if not isinstance(content, str):
            raise PageValidationError("页面内容必须是字符串")
        return content.rstrip()

    @staticmethod
    def validate_page_id(page_id: str) -> None:
        try:
            parsed = uuid.UUID(page_id)
        except (ValueError, AttributeError) as exc:
            raise PageValidationError("页面 ID 必须是 UUID") from exc
        if str(parsed) != page_id:
            raise PageValidationError("页面 ID 必须使用标准 UUID 格式")

    @staticmethod
    def normalize_strings(
        values: Optional[Iterable[str]],
        *,
        exclude: Optional[set[str]] = None,
    ) -> List[str]:
        result = []
        seen = {item.casefold() for item in (exclude or set())}
        for value in values or []:
            cleaned = str(value).strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
        return result

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def build_metadata(
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
        return {
            "id": page_id,
            "title": title,
            "notebook": notebook,
            "tags": tags,
            "aliases": aliases,
            "status": status,
            "source_type": source_type,
            "source_uri": source_uri,
            "revision": revision,
            "created_at": _isoformat(created_at),
            "updated_at": _isoformat(updated_at),
        }

    def row_metadata(self, page: WikiPage) -> Dict[str, Any]:
        return self.build_metadata(
            page_id=page.id,
            title=page.title,
            notebook=page.notebook,
            tags=list(page.tags or []),
            aliases=list(page.aliases or []),
            status=page.status,
            source_type=page.source_type,
            source_uri=page.source_uri,
            revision=page.revision,
            created_at=page.created_at,
            updated_at=page.updated_at,
        )

    @staticmethod
    def page_to_dict(
        page: WikiPage,
        content: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "id": page.id,
            "page_id": page.id,
            "title": page.title,
            "notebook": page.notebook,
            "tags": list(page.tags or []),
            "aliases": list(page.aliases or []),
            "status": page.status,
            "source_type": page.source_type,
            "source_uri": page.source_uri,
            "revision": page.revision,
            "content": content,
            "content_hash": page.content_hash,
            "file_path": page.file_path,
            "filename": f"{page.id}.md",
            "index_status": page.index_status,
            "index_error": page.index_error,
            "created_at": _isoformat(page.created_at),
            "updated_at": _isoformat(page.updated_at),
            "deleted_at": _isoformat(page.deleted_at),
            "metadata": metadata,
        }

    @staticmethod
    def page_summary(page: WikiPage) -> Dict[str, Any]:
        return {
            "id": page.id,
            "page_id": page.id,
            "title": page.title,
            "notebook": page.notebook,
            "tags": list(page.tags or []),
            "aliases": list(page.aliases or []),
            "status": page.status,
            "source_type": page.source_type,
            "source_uri": page.source_uri,
            "revision": page.revision,
            "filename": f"{page.id}.md",
            "index_status": page.index_status,
            "index_error": page.index_error,
            "created_at": _isoformat(page.created_at),
            "updated_at": _isoformat(page.updated_at),
        }

    @staticmethod
    def validate_file_identity(page: WikiPage, metadata: Dict[str, Any]) -> None:
        if metadata.get("id") != page.id:
            raise PageValidationError(f"页面文件 ID 与数据库不一致: {page.id}")
        if int(metadata.get("revision", 0)) != page.revision:
            raise PageValidationError(f"页面文件版本与数据库不一致: {page.id}")

    @staticmethod
    def legacy_title_content(path: Path, raw: str) -> tuple[str, str]:
        lines = raw.splitlines()
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip() or path.stem
            content = "\n".join(lines[1:]).lstrip()
            return title, content
        return path.stem, raw

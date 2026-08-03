"""个人 Wiki 页面、版本、链接和索引任务模型。"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from ..core.database import Base


def utc_now() -> datetime:
    """返回带时区的 UTC 时间。"""
    return datetime.now(timezone.utc)


class WikiPage(Base):
    """Wiki 当前页面元数据，正文保存在 Markdown 文件中。"""

    __tablename__ = "wiki_pages"

    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False, index=True)
    notebook = Column(String(255), nullable=True, index=True)
    tags = Column(JSON, nullable=False, default=list)
    aliases = Column(JSON, nullable=False, default=list)
    status = Column(String(32), nullable=False, default="active", index=True)
    source_type = Column(String(64), nullable=False, default="manual", index=True)
    source_uri = Column(String(1024), nullable=True)
    file_path = Column(String(1024), nullable=False, unique=True)
    revision = Column(Integer, nullable=False, default=1)
    content_hash = Column(String(64), nullable=False, index=True)
    index_status = Column(String(32), nullable=False, default="pending", index=True)
    index_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    revisions = relationship(
        "WikiPageRevision",
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="WikiPageRevision.revision",
    )


class WikiPageRevision(Base):
    """页面完整快照，用于差异比较和回滚。"""

    __tablename__ = "wiki_page_revisions"
    __table_args__ = (
        UniqueConstraint("page_id", "revision", name="uq_wiki_page_revision"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    page_id = Column(
        String(36),
        ForeignKey("wiki_pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    page_metadata = Column(JSON, nullable=False, default=dict)
    change_summary = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    page = relationship("WikiPage", back_populates="revisions")


class WikiPageLink(Base):
    """页面中的 Wiki Link，允许目标暂时不存在或存在歧义。"""

    __tablename__ = "wiki_page_links"
    __table_args__ = (
        UniqueConstraint(
            "source_page_id",
            "target_title",
            name="uq_wiki_page_link_target_title",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_page_id = Column(
        String(36),
        ForeignKey("wiki_pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_page_id = Column(
        String(36),
        ForeignKey("wiki_pages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    target_title = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class WikiIndexTask(Base):
    """可靠记录页面索引更新，失败后可重试。"""

    __tablename__ = "wiki_index_tasks"

    id = Column(String(36), primary_key=True)
    page_id = Column(String(36), nullable=False, index=True)
    revision = Column(Integer, nullable=False)
    action = Column(String(16), nullable=False)  # upsert / delete
    status = Column(String(16), nullable=False, default="pending", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    next_attempt_at = Column(DateTime(timezone=True), nullable=True, index=True)
    locked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class AgentPendingAction(Base):
    """等待用户确认的 Agent 写入计划。"""

    __tablename__ = "agent_pending_actions"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)
    query = Column(Text, nullable=False)
    plan_data = Column(JSON, nullable=False)
    preview = Column(JSON, nullable=False, default=list)
    status = Column(String(16), nullable=False, default="pending", index=True)
    result_data = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

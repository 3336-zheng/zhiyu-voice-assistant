"""轻量、非破坏性的数据库版本迁移。"""

from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

SCHEMA_VERSION = 4


def _has_column(engine: Engine, table: str, column: str) -> bool:
    """检查列是否存在，兼容已有 SQLite 数据库。"""
    return any(item["name"] == column for item in inspect(engine).get_columns(table))


def ensure_schema(engine: Engine) -> int:
    """应用可重复执行的增量迁移，不删除已有表或数据。"""
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description VARCHAR(255) NOT NULL,
                    applied_at DATETIME NOT NULL
                )
                """
            )
        )
        current = connection.execute(
            text("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        ).scalar_one()

        if current < 1:
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 1,
                    "description": "统一 Wiki 页面基线；保留已有业务表",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 1

        # 版本 2 为索引 worker 增加可恢复的退避和租约字段。
        if current < 2:
            if not _has_column(engine, "wiki_index_tasks", "next_attempt_at"):
                connection.execute(
                    text("ALTER TABLE wiki_index_tasks ADD COLUMN next_attempt_at DATETIME")
                )
            if not _has_column(engine, "wiki_index_tasks", "locked_at"):
                connection.execute(
                    text("ALTER TABLE wiki_index_tasks ADD COLUMN locked_at DATETIME")
                )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 2,
                    "description": "增加 Wiki 索引任务退避和恢复字段",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 2

        # 版本 3 保存 ASR 分段时间戳，用于答案回溯到原音频。
        if current < 3:
            if not _has_column(engine, "audios", "transcription_segments"):
                connection.execute(
                    text("ALTER TABLE audios ADD COLUMN transcription_segments JSON")
                )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 3,
                    "description": "增加音频转录分段时间戳",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 3

        # 版本 4 保存 MCP 外部研究、来源快照及页面来源关系。
        if current < 4:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS external_research_runs (
                        id VARCHAR(36) PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        query TEXT NOT NULL,
                        status VARCHAR(16) NOT NULL,
                        search_queries JSON NOT NULL,
                        answer TEXT,
                        draft_title VARCHAR(255),
                        draft_content TEXT,
                        error TEXT,
                        page_id VARCHAR(36),
                        created_at DATETIME NOT NULL,
                        completed_at DATETIME,
                        FOREIGN KEY(page_id) REFERENCES wiki_pages(id) ON DELETE SET NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS external_research_sources (
                        id VARCHAR(36) PRIMARY KEY,
                        run_id VARCHAR(36) NOT NULL,
                        title VARCHAR(500) NOT NULL,
                        url VARCHAR(2048) NOT NULL,
                        snippet TEXT,
                        content TEXT,
                        content_hash VARCHAR(64) NOT NULL,
                        provider VARCHAR(255) NOT NULL,
                        tool_name VARCHAR(255) NOT NULL,
                        retrieved_at DATETIME NOT NULL,
                        created_at DATETIME NOT NULL,
                        CONSTRAINT uq_external_research_source_url UNIQUE(run_id, url),
                        FOREIGN KEY(run_id) REFERENCES external_research_runs(id) ON DELETE CASCADE
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS wiki_page_sources (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        page_id VARCHAR(36) NOT NULL,
                        research_source_id VARCHAR(36) NOT NULL,
                        created_at DATETIME NOT NULL,
                        CONSTRAINT uq_wiki_page_source UNIQUE(page_id, research_source_id),
                        FOREIGN KEY(page_id) REFERENCES wiki_pages(id) ON DELETE CASCADE,
                        FOREIGN KEY(research_source_id) REFERENCES external_research_sources(id) ON DELETE CASCADE
                    )
                    """
                )
            )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 4,
                    "description": "增加 MCP 外部研究和来源追溯",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 4

    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"数据库版本 {current} 高于当前应用支持的版本 {SCHEMA_VERSION}"
        )
    return current

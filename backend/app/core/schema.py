"""轻量、非破坏性的数据库版本迁移。"""

from datetime import datetime, timezone
import re

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

SCHEMA_VERSION = 8


def _conversation_title(content: str | None, limit: int = 80) -> str | None:
    """从首条用户消息生成稳定、紧凑的会话标题。"""
    normalized = re.sub(r"\s+", " ", content or "").strip()
    if not normalized:
        return None
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit - 3].rstrip()}..."


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

        # 版本 5 增加增量会话摘要游标和 Agent 请求运行记录。
        if current < 5:
            if not _has_column(engine, "conversations", "summary_message_id"):
                connection.execute(
                    text("ALTER TABLE conversations ADD COLUMN summary_message_id INTEGER")
                )
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        request_id VARCHAR(128) PRIMARY KEY,
                        session_id VARCHAR(64),
                        query TEXT NOT NULL,
                        intent VARCHAR(50),
                        status VARCHAR(20) NOT NULL,
                        execution_time_ms INTEGER,
                        timeline JSON,
                        retrieval_stats JSON,
                        model_usage JSON,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
                    "version": 5,
                    "description": "增加增量会话摘要和 Agent 运行统计",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 5

        # 版本 6 增加可恢复的 Agent 运行状态和终态事件快照。
        if current < 6:
            agent_run_columns = {
                "response": "TEXT",
                "error": "TEXT",
                "events": "JSON",
                "runtime_snapshot": "JSON",
                "updated_at": "DATETIME",
                "completed_at": "DATETIME",
            }
            for column, column_type in agent_run_columns.items():
                if not _has_column(engine, "agent_runs", column):
                    connection.execute(
                        text(f"ALTER TABLE agent_runs ADD COLUMN {column} {column_type}")
                    )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 6,
                    "description": "增加 Agent 运行状态、事件和终态快照",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 6

        # 版本 7 增加可搜索的会话标题，并为已有会话回填首条用户消息。
        if current < 7:
            inspector = inspect(engine)
            if inspector.has_table("conversations"):
                if not _has_column(engine, "conversations", "title"):
                    connection.execute(
                        text("ALTER TABLE conversations ADD COLUMN title VARCHAR(255)")
                    )
                if inspector.has_table("conversation_messages"):
                    rows = connection.execute(
                        text(
                            """
                            SELECT c.session_id,
                                   (
                                       SELECT m.content
                                       FROM conversation_messages AS m
                                       WHERE m.session_id = c.session_id
                                         AND m.role = 'user'
                                       ORDER BY m.id ASC
                                       LIMIT 1
                                   ) AS first_user_message
                            FROM conversations AS c
                            WHERE c.title IS NULL OR TRIM(c.title) = ''
                            """
                        )
                    ).mappings()
                    for row in rows:
                        title = _conversation_title(row["first_user_message"])
                        if title:
                            connection.execute(
                                text(
                                    "UPDATE conversations SET title = :title "
                                    "WHERE session_id = :session_id"
                                ),
                                {"title": title, "session_id": row["session_id"]},
                            )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 7,
                    "description": "增加可搜索的会话标题并回填历史数据",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 7

        # 版本 8 保存回答反馈、确认写入、索引和自动复测闭环。
        if current < 8:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS answer_feedbacks (
                        id VARCHAR(36) PRIMARY KEY,
                        request_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(64) NOT NULL,
                        category VARCHAR(32) NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        question TEXT NOT NULL,
                        answer_snapshot TEXT NOT NULL,
                        retrieval_snapshot JSON NOT NULL,
                        user_note TEXT,
                        target_page_id VARCHAR(36),
                        external_research_run_id VARCHAR(36),
                        pending_action_id VARCHAR(36),
                        draft_title VARCHAR(255),
                        draft_content TEXT,
                        write_result JSON,
                        index_result JSON,
                        retest_request_id VARCHAR(128),
                        retest_answer TEXT,
                        retest_snapshot JSON,
                        error TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        completed_at DATETIME,
                        CONSTRAINT uq_answer_feedback_request UNIQUE(request_id)
                    )
                    """
                )
            )
            for index_name, column in {
                "ix_answer_feedbacks_request_id": "request_id",
                "ix_answer_feedbacks_session_id": "session_id",
                "ix_answer_feedbacks_category": "category",
                "ix_answer_feedbacks_status": "status",
                "ix_answer_feedbacks_target_page_id": "target_page_id",
                "ix_answer_feedbacks_external_research_run_id": "external_research_run_id",
                "ix_answer_feedbacks_pending_action_id": "pending_action_id",
                "ix_answer_feedbacks_retest_request_id": "retest_request_id",
            }.items():
                connection.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON answer_feedbacks ({column})"
                    )
                )
            connection.execute(
                text(
                    "INSERT INTO schema_migrations(version, description, applied_at) "
                    "VALUES (:version, :description, :applied_at)"
                ),
                {
                    "version": 8,
                    "description": "增加回答反馈、纠错写入和自动复测记录",
                    "applied_at": datetime.now(timezone.utc),
                },
            )
            current = 8

    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"数据库版本 {current} 高于当前应用支持的版本 {SCHEMA_VERSION}"
        )
    return current

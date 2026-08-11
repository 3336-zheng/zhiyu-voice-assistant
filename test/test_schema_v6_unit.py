"""数据库 v5 到 v8 增量迁移与 Token 预算边界测试。"""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from backend.app.core.schema import ensure_schema
from backend.app.services.token_budget_service import estimate_tokens, limit_context


class SchemaV8TestCase(unittest.TestCase):
    def test_v5_database_migrates_to_v8_without_losing_run(self):
        with tempfile.TemporaryDirectory() as root:
            engine = create_engine(f"sqlite:///{Path(root) / 'schema.db'}")
            with engine.begin() as connection:
                connection.execute(text(
                    "CREATE TABLE schema_migrations ("
                    "version INTEGER PRIMARY KEY, description VARCHAR(255) NOT NULL, "
                    "applied_at DATETIME NOT NULL)"
                ))
                connection.execute(text(
                    "INSERT INTO schema_migrations VALUES "
                    "(5, 'Agent 运行统计', CURRENT_TIMESTAMP)"
                ))
                connection.execute(text(
                    "CREATE TABLE agent_runs ("
                    "request_id VARCHAR(128) PRIMARY KEY, session_id VARCHAR(64), "
                    "query TEXT NOT NULL, intent VARCHAR(50), status VARCHAR(20) NOT NULL, "
                    "execution_time_ms INTEGER, timeline JSON, retrieval_stats JSON, "
                    "model_usage JSON, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
                ))
                connection.execute(text(
                    "INSERT INTO agent_runs(request_id, session_id, query, status) "
                    "VALUES ('run-v5', 'session-v5', '保留数据', 'success')"
                ))

            self.assertEqual(ensure_schema(engine), 8)
            self.assertEqual(ensure_schema(engine), 8)
            columns = {item["name"] for item in inspect(engine).get_columns("agent_runs")}
            self.assertTrue(
                {"response", "error", "events", "runtime_snapshot", "updated_at", "completed_at"}
                <= columns
            )
            with engine.connect() as connection:
                row = connection.execute(text(
                    "SELECT query, status FROM agent_runs WHERE request_id = 'run-v5'"
                )).one()
            self.assertEqual(tuple(row), ("保留数据", "success"))
            engine.dispose()

    def test_v6_database_adds_and_backfills_conversation_title(self):
        with tempfile.TemporaryDirectory() as root:
            engine = create_engine(f"sqlite:///{Path(root) / 'schema.db'}")
            with engine.begin() as connection:
                connection.execute(text(
                    "CREATE TABLE schema_migrations ("
                    "version INTEGER PRIMARY KEY, description VARCHAR(255) NOT NULL, "
                    "applied_at DATETIME NOT NULL)"
                ))
                connection.execute(text(
                    "INSERT INTO schema_migrations VALUES "
                    "(6, 'Agent Runtime', CURRENT_TIMESTAMP)"
                ))
                connection.execute(text(
                    "CREATE TABLE conversations ("
                    "id INTEGER PRIMARY KEY, session_id VARCHAR(64) NOT NULL, "
                    "summary TEXT, summary_message_id INTEGER, message_count INTEGER, "
                    "created_at DATETIME, updated_at DATETIME)"
                ))
                connection.execute(text(
                    "CREATE TABLE conversation_messages ("
                    "id INTEGER PRIMARY KEY, session_id VARCHAR(64) NOT NULL, "
                    "role VARCHAR(20) NOT NULL, content TEXT NOT NULL)"
                ))
                connection.execute(text(
                    "INSERT INTO conversations(id, session_id, message_count) "
                    "VALUES (1, 'history-session', 2)"
                ))
                connection.execute(text(
                    "INSERT INTO conversation_messages(id, session_id, role, content) "
                    "VALUES (1, 'history-session', 'user', "
                    "'  帮我   总结这份 RAG 文档  '), "
                    "(2, 'history-session', 'assistant', '回答')"
                ))

            self.assertEqual(ensure_schema(engine), 8)
            columns = {item["name"] for item in inspect(engine).get_columns("conversations")}
            self.assertIn("title", columns)
            with engine.connect() as connection:
                title = connection.execute(text(
                    "SELECT title FROM conversations WHERE session_id = 'history-session'"
                )).scalar_one()
            self.assertEqual(title, "帮我 总结这份 RAG 文档")
            engine.dispose()

    def test_v7_database_adds_answer_feedback_table(self):
        with tempfile.TemporaryDirectory() as root:
            engine = create_engine(f"sqlite:///{Path(root) / 'schema.db'}")
            with engine.begin() as connection:
                connection.execute(text(
                    "CREATE TABLE schema_migrations ("
                    "version INTEGER PRIMARY KEY, description VARCHAR(255) NOT NULL, "
                    "applied_at DATETIME NOT NULL)"
                ))
                connection.execute(text(
                    "INSERT INTO schema_migrations VALUES "
                    "(7, '会话搜索', CURRENT_TIMESTAMP)"
                ))

            self.assertEqual(ensure_schema(engine), 8)
            inspector = inspect(engine)
            self.assertIn("answer_feedbacks", inspector.get_table_names())
            columns = {item["name"] for item in inspector.get_columns("answer_feedbacks")}
            self.assertTrue(
                {
                    "request_id",
                    "retrieval_snapshot",
                    "pending_action_id",
                    "index_result",
                    "retest_request_id",
                    "retest_snapshot",
                } <= columns
            )
            engine.dispose()

    def test_context_limit_never_exceeds_small_or_mixed_text_budget(self):
        content = "中文内容 mixed ASCII 12345 " * 40
        for budget in range(0, 32):
            limited = limit_context(content, budget)
            self.assertLessEqual(estimate_tokens(limited.text), budget)
            self.assertEqual(limited.used_tokens, estimate_tokens(limited.text))
            self.assertEqual(limited.token_budget, budget)
            self.assertTrue(limited.truncated)


if __name__ == "__main__":
    unittest.main()

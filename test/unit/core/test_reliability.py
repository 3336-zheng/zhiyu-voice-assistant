"""可靠性能力测试。"""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from sqlalchemy import create_engine

from backend.app.core.database import Base
from backend.app.services.system.backup_service import BackupValidationError, create_backup, restore_backup
from backend.app.services.retrieval.evidence_service import assess_evidence
from backend.app.services.wiki.page_service import PageService
from backend.app.services.memory.memory_service import MemoryService
from backend.app.agent.executor import Executor


class ReliabilityTestCase(unittest.TestCase):
    def test_executor_does_not_load_retrieval_models_until_needed(self):
        from unittest.mock import patch

        with patch(
            "backend.app.agent.tool_registry.get_hybrid_retrieval_service"
        ) as retrieval_factory:
            executor = Executor()
            try:
                retrieval_factory.assert_not_called()
                _ = executor.hybrid_retrieval
                retrieval_factory.assert_called_once_with()
            finally:
                executor._thread_pool.shutdown(wait=True)

    def test_incremental_summary_preserves_raw_messages(self):
        with tempfile.TemporaryDirectory() as root:
            engine = create_engine(f"sqlite:///{Path(root) / 'memory.db'}")
            Base.metadata.create_all(engine)
            from sqlalchemy.orm import sessionmaker
            from unittest.mock import patch

            session = sessionmaker(bind=engine)()
            service = MemoryService()
            service.summary_threshold = 6

            class FakeLLM:
                @staticmethod
                def chat(messages, max_tokens=300, **kwargs):
                    return "增量摘要"

            try:
                for index in range(8):
                    service.add_message(
                        "summary-session",
                        "user" if index % 2 == 0 else "assistant",
                        f"消息 {index}",
                        db=session,
                    )
                with patch(
                    "backend.app.services.ai.llm_service.get_llm_service",
                    return_value=FakeLLM(),
                ):
                    service.summarize_if_needed("summary-session", db=session)

                from backend.app.models.conversation import Conversation, ConversationMessage

                conversation = session.query(Conversation).filter_by(
                    session_id="summary-session"
                ).one()
                self.assertEqual(conversation.summary, "增量摘要")
                self.assertIsNotNone(conversation.summary_message_id)
                self.assertEqual(
                    session.query(ConversationMessage).filter_by(
                        session_id="summary-session"
                    ).count(),
                    8,
                )
                history = service.get_history("summary-session", db=session)
                self.assertEqual(history[0]["role"], "system")
                self.assertEqual(len(history), 6)
            finally:
                session.close()

    def test_page_index_is_queued_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            engine = create_engine(f"sqlite:///{Path(root) / 'wiki.db'}")
            Base.metadata.create_all(engine)
            from sqlalchemy.orm import sessionmaker

            session = sessionmaker(bind=engine)()

            class FakeIndexer:
                def remove_page(self, page_id):
                    return True

                def index_page(self, page):
                    return {"status": "indexed"}

            try:
                service = PageService(session, pages_dir=str(Path(root) / "pages"), index_service=FakeIndexer())
                page = service.create_page(title="异步页面", content="正文")
                self.assertEqual(page["index_status"], "pending")
                result = service.process_pending_index_tasks()
                self.assertEqual(result["completed"], 1)
                self.assertEqual(service.get_page(page["id"])["index_status"], "ready")
            finally:
                session.close()

    def test_evidence_gate_rejects_empty_and_low_score(self):
        empty = assess_evidence([])
        self.assertEqual(empty.status, "insufficient")
        low = assess_evidence([{"chunk_id": "c1", "content": "片段", "rerank_score": 0.1}])
        self.assertEqual(low.status, "insufficient")
        enough = assess_evidence([{"page_id": "p1", "content": "片段", "rerank_score": 0.8}])
        self.assertEqual(enough.status, "sufficient")

    def test_backup_rejects_path_traversal_and_round_trips(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            database = root_path / "data/database/notes.db"
            database.parent.mkdir(parents=True)
            import sqlite3

            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE sample (value TEXT)")
            connection.execute("INSERT INTO sample VALUES ('ok')")
            connection.commit()
            connection.close()

            from backend.app.services.system import backup_service

            previous = {
                "database_url": backup_service.settings.database_url,
                "wiki_pages_dir": backup_service.settings.wiki_pages_dir,
                "upload_dir": backup_service.settings.upload_dir,
                "backup_dir": backup_service.settings.backup_dir,
            }
            backup_service.settings.database_url = f"sqlite:///{database}"
            backup_service.settings.wiki_pages_dir = str(root_path / "data/wiki/pages")
            backup_service.settings.upload_dir = str(root_path / "data/uploads")
            backup_service.settings.backup_dir = str(root_path / "backups")
            try:
                result = create_backup()
                self.assertTrue(Path(result["path"]).exists())
                restored_root = root_path / "restored"
                restored = restore_backup(result["path"], str(restored_root))
                self.assertGreaterEqual(restored["restored"], 2)

                malicious = root_path / "malicious.zip"
                with zipfile.ZipFile(malicious, "w") as archive:
                    archive.writestr("manifest.json", json.dumps({"format": "zhiyu-backup"}))
                    archive.writestr("../../outside.txt", "blocked")
                with self.assertRaises(BackupValidationError):
                    restore_backup(str(malicious), str(restored_root), overwrite=True)
            finally:
                for name, value in previous.items():
                    setattr(backup_service.settings, name, value)


if __name__ == "__main__":
    unittest.main()

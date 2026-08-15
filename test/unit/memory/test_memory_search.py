"""对话标题、正文搜索与历史恢复测试。"""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.services.memory.memory_service import MemoryService


class MemorySearchTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        engine = create_engine(f"sqlite:///{Path(self.temp_dir.name) / 'memory.db'}")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        self.service = MemoryService()

    def tearDown(self):
        self.session.close()
        self.temp_dir.cleanup()

    def test_first_user_message_becomes_title_and_content_is_searchable(self):
        self.service.add_message(
            "session-search",
            "user",
            "  帮我   总结 RAG 课程  ",
            db=self.session,
        )
        assistant = self.service.add_message(
            "session-search",
            "assistant",
            "答案包含父子分块和统一融合。",
            metadata={"sources": [{"title": "课程记录"}], "evidence_status": "sufficient"},
            db=self.session,
        )

        by_title = self.service.list_sessions(self.session, query="总结 RAG")
        self.assertEqual(by_title[0]["title"], "帮我 总结 RAG 课程")
        self.assertIn("总结 RAG", by_title[0]["match_snippet"])

        by_content = self.service.list_sessions(self.session, query="父子分块")
        self.assertEqual(by_content[0]["matched_message_id"], assistant.id)
        self.assertIn("父子分块", by_content[0]["match_snippet"])

        history = self.service.get_session_messages("session-search", self.session)
        self.assertEqual(len(history["messages"]), 2)
        self.assertEqual(history["messages"][1]["metadata"]["evidence_status"], "sufficient")

    def test_like_wildcards_are_treated_as_literal_text(self):
        self.service.add_message("literal", "user", "如何处理 100% 命中", db=self.session)
        self.service.add_message("other", "user", "普通问题", db=self.session)

        results = self.service.list_sessions(self.session, query="100%")
        self.assertEqual([item["session_id"] for item in results], ["literal"])


if __name__ == "__main__":
    unittest.main()

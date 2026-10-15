"""对话标题、正文搜索与历史恢复测试。"""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from test.support.db import make_test_engine
from backend.app.services.memory.memory_service import MemoryService


class MemorySearchTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        engine = make_test_engine()
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

    def test_没有消息的会话不出现在列表里(self):
        """空会话不该出现在可信问答的历史里。

        `get_or_create_session` 先 commit 会话行、再插消息，中间失败就会留下
        一行没有任何消息的空壳——点进去什么也没有。这里直接构造这种情况。
        """
        self.service.add_message("has-message", "user", "有内容的问题", db=self.session)
        # 只要会话行，不走 add_message（那才会写消息）。
        self.service.get_or_create_session("empty-shell", self.session)
        self.session.commit()

        ids = [item["session_id"] for item in self.service.list_sessions(self.session)]
        self.assertIn("has-message", ids)
        self.assertNotIn("empty-shell", ids)

    def test_空会话不影响带关键词的搜索(self):
        """搜索也不能把空壳捞出来——它是另一条过滤分支。"""
        self.service.add_message("real", "user", "父子分块怎么切", db=self.session)
        shell = self.service.get_or_create_session("empty-shell", self.session)
        shell.title = "父子分块（空壳）"
        self.session.commit()

        ids = [item["session_id"] for item in self.service.list_sessions(self.session, query="父子分块")]
        self.assertEqual(ids, ["real"])


if __name__ == "__main__":
    unittest.main()

"""运行追踪「对话列表」的过滤测试。

连真实 MySQL——过滤条件是 SQL 层的 LIKE 转义和子查询并集，用假 session
测不出 SQL 的实际行为。

背景：这个「测试脚本会话过滤」最初写成把 `session_` 前缀条件同时套进外层和
子查询，结果「有真实消息」那个分支被自己过滤空了，并集实际只剩前缀那一半，
静默少了一半会话。下面的用例就是钉住这个并集语义。
"""

import unittest

from sqlalchemy.orm import Session

from backend.app.models.conversation import Conversation, ConversationMessage
from backend.app.models.observability import AgentRun
from backend.app.services.observability.conversation_filter import (
    apply_conversation_visibility_filter,
)
from test.support.db import TestDatabaseUnavailable, make_test_engine


def _run(request_id: str, session_id: str, query: str = "问题") -> AgentRun:
    return AgentRun(request_id=request_id, session_id=session_id, query=query, status="completed")


def _message(session_id: str, content: str = "内容") -> ConversationMessage:
    return ConversationMessage(session_id=session_id, role="user", content=content)


class ConversationVisibilityFilterTest(unittest.TestCase):
    """默认只看真实对话：session_ 前缀 或 有消息，取并集。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.engine = make_test_engine()
        except TestDatabaseUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()

    def setUp(self) -> None:
        self.db = Session(self.engine)
        # 每个用例从干净状态开始：整库清空，不用关心上一条用例留了什么。
        for model in (ConversationMessage, Conversation, AgentRun):
            self.db.query(model).delete()
        self.db.commit()

        rows = [
            # 两种情况都满足的正常真实会话
            _run("r1", "session_111_a"),
            # 只满足「session_ 前缀」，没有任何消息（8 月那批真人历史就是这样）
            _run("r2", "session_222_b"),
            # 只满足「有消息」，ID 不是前端格式（将来别的入口可能出现这种）
            _run("r3", "external-agent-001"),
            # 都不满足：脚本会话，无消息、ID 也不符合
            _run("r4", "bench-simple-20260818"),
            _run("r5", "perf-fast-session-01"),
        ]
        self.db.add_all(rows)
        self.db.add_all([_message("external-agent-001")])
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()

    def _visible_ids(self, include_empty: bool = False) -> set:
        query = self.db.query(AgentRun.session_id)
        # 接住返回值——filter() 不改原对象，丢了就等于没过滤。
        query = apply_conversation_visibility_filter(query, include_empty=include_empty)
        return {row[0] for row in query.distinct().all()}

    def test_默认留下真实会话与有消息的会话(self):
        self.assertEqual(
            self._visible_ids(),
            {"session_111_a", "session_222_b", "external-agent-001"},
        )

    def test_有消息但非前端格式的会话不会漏掉(self):
        """并集里「有消息」那一半。最初写成子查询也套前缀条件时，这里会失败。"""
        self.assertIn("external-agent-001", self._visible_ids())

    def test_没有消息的真实会话不会漏掉(self):
        """并集里「前缀」那一半。8 月那批真人历史正是这种。"""
        self.assertIn("session_222_b", self._visible_ids())

    def test_脚本会话被滤掉(self):
        visible = self._visible_ids()
        self.assertNotIn("bench-simple-20260818", visible)
        self.assertNotIn("perf-fast-session-01", visible)

    def test_include_empty_返回全部(self):
        self.assertEqual(
            self._visible_ids(include_empty=True),
            {"session_111_a", "session_222_b", "external-agent-001",
             "bench-simple-20260818", "perf-fast-session-01"},
        )

    def test_下划线不被当成通配符(self):
        """LIKE 里 `_` 匹配任意单字符。不转义的话 `sessionXabc` 会被误放行。"""
        self.db.add(_run("r6", "sessionXabc"))
        self.db.commit()
        self.assertNotIn("sessionXabc", self._visible_ids())


if __name__ == "__main__":
    unittest.main()

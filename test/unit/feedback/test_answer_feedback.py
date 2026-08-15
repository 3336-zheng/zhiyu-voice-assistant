"""回答反馈、确认写入、索引和自动复测闭环测试。"""

import tempfile
import time
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.agent.agent import PlanExecuteAgent
from backend.app.agent.executor import Executor
from backend.app.agent.models import AgentResponse
from backend.app.agent.responder import Responder
from backend.app.agent.tool_registry import AgentToolRegistry
from backend.app.core.database import Base
from backend.app.models.feedback import AnswerFeedback
from backend.app.models.observability import AgentRun
from backend.app.models.wiki import ExternalResearchRun, ExternalResearchSource, WikiPage
from backend.app.services.feedback.answer_feedback_service import AnswerFeedbackService
from backend.app.services.wiki.page_service import PageService


class FakeIndexer:
    def __init__(self):
        self.indexed = []
        self.failures_remaining = 0

    def index_page(self, page):
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("模拟索引服务暂时不可用")
        self.indexed.append((page["id"], page["revision"]))
        return {"status": "indexed"}

    @staticmethod
    def remove_page(page_id):
        return True


class FakeResearchService:
    async def research(self, query, session_id, db):
        run_id = str(uuid.uuid4())
        run = ExternalResearchRun(
            id=run_id,
            session_id=session_id,
            query=query,
            status="completed",
            search_queries=[query],
            answer="RAG 应使用可追溯来源。[1]",
            draft_title="RAG 纠错补充",
            draft_content="# RAG 纠错补充\n\n使用可追溯来源。[1]\n\n## 参考来源\n1. [官方资料](https://example.com/rag)",
            completed_at=datetime.now(timezone.utc),
        )
        run.sources.append(
            ExternalResearchSource(
                id=str(uuid.uuid4()),
                title="官方资料",
                url="https://example.com/rag",
                snippet="可追溯来源",
                content="RAG 应使用可追溯来源。",
                content_hash="a" * 64,
                provider="fake",
                tool_name="fetch",
            )
        )
        db.add(run)
        db.commit()
        return {
            "run_id": run_id,
            "draft_title": run.draft_title,
            "draft_content": run.draft_content,
        }


class FailingResearchService:
    async def research(self, query, session_id, db):
        raise RuntimeError("模拟外部研究失败")


class FakeRuntime:
    def __init__(self):
        self.started = []

    async def start(self, query, session_id):
        run_id = str(uuid.uuid4())
        self.started.append((query, session_id, run_id))
        return SimpleNamespace(run_id=run_id)


class StaticResponder(Responder):
    def __init__(self):
        pass

    @staticmethod
    def generate_response(query, plan, execution_result, context):
        return AgentResponse(
            query=query,
            response="知识修订已写入",
            plan=plan,
            execution_result=execution_result,
            confidence=1.0,
            timestamp=datetime.now(),
        )


class AnswerFeedbackTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = create_engine(f"sqlite:///{self.root / 'feedback.db'}")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.indexer = FakeIndexer()
        self.page_factory = lambda db: PageService(
            db,
            pages_dir=str(self.root / "pages"),
            index_service=self.indexer,
        )
        registry = AgentToolRegistry(page_factory=self.page_factory)
        executor = Executor(tool_registry=registry)
        self.addCleanup(executor._thread_pool.shutdown, wait=True)
        self.agent = PlanExecuteAgent.__new__(PlanExecuteAgent)
        self.agent.executor = executor
        self.agent.responder = StaticResponder()
        self.agent._memory_service = False
        self.runtime = FakeRuntime()
        self.service = AnswerFeedbackService(
            research_service=FakeResearchService(),
            runtime_service=self.runtime,
            page_factory=self.page_factory,
        )

    async def asyncTearDown(self):
        self.db.close()
        self.engine.dispose()
        self.temporary.cleanup()

    async def test_feedback_persists_snapshot_and_retests_after_confirmed_index(self):
        request_id = "request-original"
        self.db.add(
            AgentRun(
                request_id=request_id,
                session_id="session-feedback",
                query="RAG 如何保证来源可信？",
                status="completed",
                response="原回答",
                events=[
                    {
                        "type": "run_completed",
                        "data": {
                            "response": {
                                "response": "原回答",
                                "sources": [
                                    {
                                        "page_id": "cited-page",
                                        "title": "旧页面",
                                        "snippet": "旧证据",
                                    }
                                ],
                                "retrieval_stats": {"fused_candidates": 3},
                                "evidence_status": "sufficient",
                            }
                        },
                    }
                ],
            )
        )
        self.db.commit()

        created = self.service.create(
            request_id=request_id,
            session_id="session-feedback",
            category="knowledge_missing",
            user_note="缺少来源校验说明",
            target_page_id=None,
            db=self.db,
        )
        self.assertEqual(created["before"]["sources"][0]["title"], "旧页面")
        self.assertEqual(created["before"]["retrieval_stats"]["fused_candidates"], 3)

        prepared = await self.service.prepare(
            created["id"],
            "session-feedback",
            self.db,
            self.agent,
        )
        self.assertEqual(prepared["status"], "pending_confirmation")
        self.assertEqual(self.db.query(WikiPage).count(), 0)

        confirmed = await self.service.confirm(
            created["id"],
            "session-feedback",
            self.db,
            self.agent,
        )
        self.assertEqual(confirmed["status"], "retesting")
        self.assertEqual(self.db.query(WikiPage).count(), 1)
        self.assertEqual(len(self.indexer.indexed), 1)
        self.assertEqual(self.runtime.started[0][0], "RAG 如何保证来源可信？")

        retest_id = confirmed["retest"]["request_id"]
        self.db.add(
            AgentRun(
                request_id=retest_id,
                session_id="session-feedback",
                query="RAG 如何保证来源可信？",
                status="completed",
                response="新回答",
                events=[
                    {
                        "type": "run_completed",
                        "data": {
                            "response": {
                                "response": "新回答",
                                "sources": [{"page_id": confirmed["target_page_id"], "title": "RAG 纠错补充"}],
                                "retrieval_stats": {"fused_candidates": 4},
                            }
                        },
                    }
                ],
            )
        )
        self.db.commit()

        resolved = self.service.get(created["id"], "session-feedback", self.db)
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["retest"]["answer"], "新回答")
        self.assertEqual(resolved["retest"]["retrieval_stats"]["fused_candidates"], 4)
        self.assertIsNotNone(self.db.get(AnswerFeedback, created["id"]).completed_at)

    async def test_duplicate_feedback_for_same_request_is_idempotent(self):
        request_id = "request-idempotent"
        self.db.add(
            AgentRun(
                request_id=request_id,
                session_id="session-feedback",
                query="什么是可信 RAG？",
                status="completed",
                response="原回答",
                events=[],
            )
        )
        self.db.commit()

        first = self.service.create(
            request_id=request_id,
            session_id="session-feedback",
            category="knowledge_missing",
            user_note="第一次反馈",
            target_page_id=None,
            db=self.db,
        )
        repeated = self.service.create(
            request_id=request_id,
            session_id="session-feedback",
            category="answer_irrelevant",
            user_note="重复提交不应覆盖第一次反馈",
            target_page_id=None,
            db=self.db,
        )

        self.assertEqual(repeated["id"], first["id"])
        self.assertEqual(repeated["category"], "knowledge_missing")
        self.assertEqual(repeated["user_note"], "第一次反馈")
        self.assertEqual(self.db.query(AnswerFeedback).count(), 1)

    async def test_draft_failure_is_persisted_and_prepare_can_retry(self):
        request_id = "request-draft-retry"
        self.db.add(
            AgentRun(
                request_id=request_id,
                session_id="session-feedback",
                query="如何修复缺失知识？",
                status="completed",
                response="原回答",
                events=[],
            )
        )
        self.db.commit()
        created = self.service.create(
            request_id=request_id,
            session_id="session-feedback",
            category="knowledge_missing",
            user_note=None,
            target_page_id=None,
            db=self.db,
        )
        self.service.research_service = FailingResearchService()

        with self.assertRaisesRegex(RuntimeError, "模拟外部研究失败"):
            await self.service.prepare(
                created["id"],
                "session-feedback",
                self.db,
                self.agent,
            )

        failed = self.service.get(created["id"], "session-feedback", self.db)
        self.assertEqual(failed["status"], "draft_failed")
        self.assertIn("模拟外部研究失败", failed["error"])

        self.service.research_service = FakeResearchService()
        retried = await self.service.prepare(
            created["id"],
            "session-feedback",
            self.db,
            self.agent,
        )
        self.assertEqual(retried["status"], "pending_confirmation")
        self.assertIsNone(retried["error"])

    async def test_index_failure_retries_without_writing_page_twice(self):
        request_id = "request-index-retry"
        self.db.add(
            AgentRun(
                request_id=request_id,
                session_id="session-feedback",
                query="索引失败后如何恢复？",
                status="completed",
                response="原回答",
                events=[],
            )
        )
        self.db.commit()
        created = self.service.create(
            request_id=request_id,
            session_id="session-feedback",
            category="knowledge_missing",
            user_note=None,
            target_page_id=None,
            db=self.db,
        )
        await self.service.prepare(
            created["id"],
            "session-feedback",
            self.db,
            self.agent,
        )
        self.indexer.failures_remaining = 1

        failed = await self.service.confirm(
            created["id"],
            "session-feedback",
            self.db,
            self.agent,
        )
        self.assertEqual(failed["status"], "index_failed")
        self.assertEqual(self.db.query(WikiPage).count(), 1)
        self.assertEqual(self.runtime.started, [])

        retried = await self.service.retry(
            created["id"],
            "session-feedback",
            self.db,
        )
        self.assertEqual(retried["status"], "retesting")
        self.assertEqual(self.db.query(WikiPage).count(), 1)
        self.assertEqual(len(self.indexer.indexed), 1)
        self.assertEqual(len(self.runtime.started), 1)
        self.assertEqual(retried["index_result"]["attempts"], 2)


if __name__ == "__main__":
    unittest.main()

"""Wiki、索引恢复、Agent 确认和音频溯源的 API 集成测试。"""

import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app import RequestContextMiddleware
from backend.app.agent.agent import PlanExecuteAgent
from backend.app.agent.models import (
    AgentResponse,
    ExecutionResult,
    IntentType,
    Plan,
    PlanStep,
    ToolName,
    ToolResult,
)
from backend.app.api import agent as agent_api
from backend.app.api import audio as audio_api
from backend.app.api import pages as pages_api
from backend.app.core.config import settings
from backend.app.core.database import Base, get_db
from backend.app.models import Audio
from backend.app.models.wiki import WikiIndexTask
from backend.app.services.demo_service import initialize_demo_data
from backend.app.services.page_service import PageService


class FlakyIndexer:
    """第一次写入失败，后续成功。"""

    def __init__(self):
        self.failures_remaining = 1

    def index_page(self, page):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("模拟索引服务暂时不可用")
        return {"status": "indexed", "page_id": page["id"]}

    def remove_page(self, page_id):
        return True


class CountingExecutor:
    def __init__(self):
        self.call_count = 0

    async def execute(self, plan, db):
        self.call_count += 1
        result = {"id": "confirmed-page", "title": "确认写入"}
        return ExecutionResult(
            plan=plan,
            results=[
                ToolResult(
                    step_id=1,
                    tool_name=ToolName.CREATE_NOTE,
                    success=True,
                    result=result,
                )
            ],
            completed_steps=1,
            total_steps=1,
            success=True,
            execution_log=[],
            final_data={"created_note": result},
        )


class StaticResponder:
    @staticmethod
    def generate_response(query, plan, execution_result, context):
        return AgentResponse(
            query=query,
            response="页面创建成功",
            plan=plan,
            execution_result=execution_result,
            confidence=1.0,
            timestamp=datetime.now(),
        )


class APIIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_upload_dir = settings.upload_dir
        settings.upload_dir = str(self.root / "uploads")
        self.engine = create_engine(
            f"sqlite:///{self.root / 'integration.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.indexer = FlakyIndexer()

        app = FastAPI()
        app.add_middleware(RequestContextMiddleware)
        app.include_router(pages_api.router, prefix="/api/pages")
        app.include_router(agent_api.router, prefix="/agent")
        app.include_router(audio_api.router, prefix="/audio")

        def override_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        def override_page_service():
            db = self.session_factory()
            try:
                yield PageService(
                    db,
                    pages_dir=str(self.root / "pages"),
                    index_service=self.indexer,
                )
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[pages_api.page_service] = override_page_service
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        settings.upload_dir = self.previous_upload_dir
        self.temporary.cleanup()

    def test_page_conflict_and_index_retry_through_api(self):
        created = self.client.post(
            "/api/pages",
            json={"title": "并发更新", "content": "版本一"},
            headers={"X-Request-ID": "integration-request-001"},
        )
        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.headers["X-Request-ID"], "integration-request-001")
        self.assertIn("total;dur=", created.headers["Server-Timing"])
        page = created.json()

        conflict = self.client.put(
            f"/api/pages/{page['id']}",
            json={"expected_revision": 2, "content": "过期写入"},
        )
        self.assertEqual(conflict.status_code, 409)

        failed = self.client.post("/api/pages/index-tasks/retry")
        self.assertEqual(failed.status_code, 200)
        self.assertEqual(failed.json()["failed"], 1)

        completed = self.client.post("/api/pages/index-tasks/retry")
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["completed"], 1)

    def test_processing_task_is_recovered_after_service_restart(self):
        db = self.session_factory()
        try:
            service = PageService(
                db,
                pages_dir=str(self.root / "pages"),
                index_service=self.indexer,
            )
            service.create_page(title="恢复测试", content="正文")
            task = db.query(WikiIndexTask).one()
            task.status = "processing"
            task.locked_at = datetime.now()
            db.commit()
        finally:
            db.close()

        restarted_db = self.session_factory()
        try:
            restarted = PageService(
                restarted_db,
                pages_dir=str(self.root / "pages"),
                index_service=self.indexer,
            )
            result = restarted.recover_index_tasks()
            self.assertEqual(result["recovered"], 1)
            task = restarted_db.query(WikiIndexTask).one()
            self.assertEqual(task.status, "pending")
            self.assertIsNone(task.locked_at)
        finally:
            restarted_db.close()

    def test_agent_duplicate_confirmation_executes_once_through_api(self):
        agent = PlanExecuteAgent.__new__(PlanExecuteAgent)
        agent.executor = CountingExecutor()
        agent.responder = StaticResponder()
        agent._memory_service = False
        plan = Plan(
            intent=IntentType.CREATE_NOTE,
            original_query="创建确认写入页面",
            steps=[
                PlanStep(
                    step_id=1,
                    tool_name=ToolName.CREATE_NOTE,
                    parameters={"title": "确认写入", "content": "正文"},
                    description="创建页面：确认写入",
                )
            ],
            estimated_steps=1,
            reasoning="集成测试",
        )
        db = self.session_factory()
        try:
            pending = agent._create_pending_response(
                plan.original_query,
                "integration-session",
                plan,
                db,
                time.time(),
            )
        finally:
            db.close()

        payload = {"session_id": "integration-session"}
        endpoint = f"/agent/actions/{pending.pending_action_id}/confirm"
        with patch("backend.app.api.agent.get_agent", return_value=agent):
            first = self.client.post(endpoint, json=payload)
            second = self.client.post(endpoint, json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["response"], second.json()["response"])
        self.assertEqual(agent.executor.call_count, 1)

    def test_demo_data_and_audio_provenance_are_idempotent(self):
        db = self.session_factory()
        try:
            first = initialize_demo_data(
                db,
                pages_dir=str(self.root / "pages"),
                upload_dir=str(self.root / "uploads"),
            )
            second = initialize_demo_data(
                db,
                pages_dir=str(self.root / "pages"),
                upload_dir=str(self.root / "uploads"),
            )
        finally:
            db.close()

        self.assertEqual(first["created_pages"], 3)
        self.assertEqual(second["created_pages"], 0)
        audio_id = first["audio"]["id"]
        transcript = self.client.get(f"/audio/{audio_id}/transcript?start=2&end=4")
        self.assertEqual(transcript.status_code, 200)
        self.assertGreaterEqual(len(transcript.json()["segments"]), 1)
        audio = self.client.get(f"/audio/{audio_id}/file")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.headers["content-type"], "audio/wav")

        demo_audio = Path(settings.upload_dir) / "zhiyu-demo-lesson.wav"
        traversal = self.client.post(
            "/audio/upload/",
            files={"file": ("../../escape.wav", demo_audio.read_bytes(), "audio/wav")},
        )
        self.assertEqual(traversal.status_code, 200)
        db = self.session_factory()
        try:
            uploaded = db.query(Audio).filter(Audio.filename == "escape.wav").one()
            self.assertEqual(Path(uploaded.file_path).resolve().parent, Path(settings.upload_dir).resolve())
            outside_path = self.root / "outside.wav"
            outside_path.write_bytes(demo_audio.read_bytes())
            outside = Audio(
                filename="outside.wav",
                original_filename="outside.wav",
                file_path=str(outside_path),
                file_size=outside_path.stat().st_size,
            )
            db.add(outside)
            db.commit()
            db.refresh(outside)
            outside_id = outside.id
        finally:
            db.close()
        forbidden = self.client.get(f"/audio/{outside_id}/file")
        self.assertEqual(forbidden.status_code, 403)


if __name__ == "__main__":
    unittest.main()

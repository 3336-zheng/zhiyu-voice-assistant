"""Wiki、索引恢复、Agent 确认和音频溯源的 API 集成测试。"""

import tempfile
import time
import unittest
import wave
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
from backend.app.services.hybrid_retrieval_service import _audio_provenance
from backend.app.services.page_service import PageService


def write_test_wav(path: Path, duration_seconds: int = 4) -> None:
    """写入一个轻量静音 WAV，供音频接口集成测试使用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 8_000
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(b"\x00\x00" * sample_rate * duration_seconds)


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
        with patch("backend.app.api.agent_actions.get_agent", return_value=agent):
            first = self.client.post(endpoint, json=payload)
            second = self.client.post(endpoint, json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["response"], second.json()["response"])
        self.assertEqual(agent.executor.call_count, 1)

    def test_audio_provenance_and_path_safety(self):
        audio_path = Path(settings.upload_dir) / "lecture.wav"
        write_test_wav(audio_path)
        db = self.session_factory()
        try:
            audio_record = Audio(
                filename=audio_path.name,
                original_filename=audio_path.name,
                file_path=str(audio_path),
                file_size=audio_path.stat().st_size,
                duration=4.0,
                language="zh",
                transcription="混合检索结合关键词与语义召回。",
                transcription_segments=[
                    {"start": 2.0, "end": 4.0, "text": "混合检索结合关键词与语义召回。"},
                ],
            )
            db.add(audio_record)
            db.commit()
            db.refresh(audio_record)
            audio_id = audio_record.id
            page = PageService(
                db,
                pages_dir=str(self.root / "pages"),
                index_service=self.indexer,
            ).create_page(
                title="课堂检索笔记",
                content="# 课堂检索笔记\n\n[00:02-00:04] 混合检索结合关键词与语义召回。",
                source_type="class_audio",
                source_uri=f"audio:{audio_id}",
            )
        finally:
            db.close()

        provenance = _audio_provenance(page["source_uri"], page["content"])
        self.assertEqual(provenance["audio_id"], audio_id)
        self.assertEqual(provenance["audio_start"], 2.0)
        self.assertEqual(provenance["audio_end"], 4.0)

        transcript = self.client.get(f"/audio/{audio_id}/transcript?start=2&end=4")
        self.assertEqual(transcript.status_code, 200)
        self.assertGreaterEqual(len(transcript.json()["segments"]), 1)
        audio = self.client.get(f"/audio/{audio_id}/file")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.headers["content-type"], "audio/wav")

        traversal = self.client.post(
            "/audio/upload/",
            files={"file": ("../../escape.wav", audio_path.read_bytes(), "audio/wav")},
        )
        self.assertEqual(traversal.status_code, 200)
        db = self.session_factory()
        try:
            uploaded = db.query(Audio).filter(Audio.filename == "escape.wav").one()
            self.assertEqual(Path(uploaded.file_path).resolve().parent, Path(settings.upload_dir).resolve())
            outside_path = self.root / "outside.wav"
            outside_path.write_bytes(audio_path.read_bytes())
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

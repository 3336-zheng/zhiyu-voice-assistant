"""Agent Runtime 状态、事件、取消和恢复测试。"""

import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.agent.events import AgentEventType, AgentRunCancelled
from backend.app.agent.models import AgentResponse
from backend.app.core.config import settings
from backend.app.core.database import Base
from backend.app.models.observability import AgentRun
from backend.app.services.agent_runtime_service import (
    AgentRunConflict,
    AgentRunStatus,
    AgentRuntimeService,
)


class StreamingAgent:
    def __init__(self):
        self.call_count = 0

    async def run(self, query, session_id, db, event_callback, cancel_check, raise_errors):
        self.call_count += 1
        event_callback(AgentEventType.STAGE_STARTED, {"stage": "agent.generation"})
        event_callback(AgentEventType.TOKEN, {"content": "智"})
        event_callback(AgentEventType.TOKEN, {"content": "语"})
        event_callback(
            AgentEventType.STAGE_COMPLETED,
            {"stage": "agent.generation", "status": "completed"},
        )
        return AgentResponse(
            query=query,
            response="智语",
            session_id=session_id,
            timestamp=datetime.now(),
            execution_time_ms=3,
        )


class BlockingAgent:
    def __init__(self):
        self.started = asyncio.Event()

    async def run(self, query, session_id, db, event_callback, cancel_check, raise_errors):
        self.started.set()
        while True:
            if cancel_check():
                raise AgentRunCancelled("Agent 运行已取消")
            await asyncio.sleep(0.005)


class SlowAgent:
    async def run(self, query, session_id, db, event_callback, cancel_check, raise_errors):
        await asyncio.sleep(10)


class AgentRuntimeTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "runtime.db"
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

    async def asyncTearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    async def test_event_order_matches_single_streamed_response_and_replays_terminal(self):
        agent = StreamingAgent()
        runtime = AgentRuntimeService(
            agent_factory=lambda: agent,
            session_factory=self.session_factory,
        )

        run = await runtime.start("测试问题", "session-stream")
        await run.task

        self.assertEqual(agent.call_count, 1)
        self.assertEqual(run.status, AgentRunStatus.COMPLETED)
        self.assertEqual(
            [event.sequence for event in run.events],
            list(range(1, len(run.events) + 1)),
        )
        streamed_text = "".join(
            event.data["content"]
            for event in run.events
            if event.type == AgentEventType.TOKEN
        )
        terminal = run.events[-1]
        self.assertEqual(terminal.type, AgentEventType.RUN_COMPLETED)
        self.assertEqual(streamed_text, terminal.data["response"]["response"])

        replay_runtime = AgentRuntimeService(session_factory=self.session_factory)
        replayed = [
            event
            async for event in replay_runtime.iter_events(
                run.run_id,
                run.session_id,
            )
            if event is not None
        ]
        self.assertNotIn(AgentEventType.TOKEN, [event.type for event in replayed])
        self.assertEqual(replayed[-1].type, AgentEventType.RUN_COMPLETED)
        self.assertEqual(replayed[-1].data["response"]["response"], "智语")

    async def test_same_session_rejects_concurrent_run_and_supports_cancellation(self):
        agent = BlockingAgent()
        runtime = AgentRuntimeService(
            agent_factory=lambda: agent,
            session_factory=self.session_factory,
        )
        run = await runtime.start("第一个问题", "session-conflict")
        await asyncio.wait_for(agent.started.wait(), timeout=1)

        with self.assertRaises(AgentRunConflict):
            await runtime.start("第二个问题", "session-conflict")

        cancelling = await runtime.cancel(run.run_id, run.session_id)
        self.assertEqual(cancelling["status"], AgentRunStatus.CANCELLING.value)
        await asyncio.wait_for(run.task, timeout=1)
        self.assertEqual(run.status, AgentRunStatus.CANCELLED)
        self.assertEqual(run.events[-1].type, AgentEventType.RUN_CANCELLED)

    async def test_total_timeout_produces_terminal_error(self):
        runtime = AgentRuntimeService(
            agent_factory=SlowAgent,
            session_factory=self.session_factory,
        )
        with patch.object(settings, "agent_run_timeout_seconds", 0.02):
            run = await runtime.start("超时问题", "session-timeout")
            await asyncio.wait_for(run.task, timeout=1)

        self.assertEqual(run.status, AgentRunStatus.TIMED_OUT)
        self.assertEqual(run.events[-1].type, AgentEventType.RUN_ERROR)
        self.assertEqual(run.events[-1].data["code"], "run_timeout")

    async def test_restart_recovery_persists_replayable_error_event(self):
        session = self.session_factory()
        try:
            session.add(
                AgentRun(
                    request_id="interrupted-run",
                    session_id="session-restart",
                    query="未完成问题",
                    status=AgentRunStatus.RUNNING.value,
                    events=[],
                )
            )
            session.commit()
        finally:
            session.close()

        runtime = AgentRuntimeService(session_factory=self.session_factory)
        self.assertEqual(runtime.recover_interrupted_runs(), 1)
        snapshot = await runtime.get("interrupted-run", "session-restart")
        self.assertEqual(snapshot["status"], AgentRunStatus.FAILED.value)
        replayed = [
            event
            async for event in runtime.iter_events(
                "interrupted-run",
                "session-restart",
            )
            if event is not None
        ]
        self.assertEqual(replayed[-1].type, AgentEventType.RUN_ERROR)
        self.assertEqual(replayed[-1].data["code"], "service_restarted")


if __name__ == "__main__":
    unittest.main()

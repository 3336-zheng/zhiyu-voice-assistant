"""Agent Runtime 状态、事件、取消和恢复测试。"""

import asyncio
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.agent.events import AgentEventType, AgentRunCancelled, AgentRuntimeEvent
from backend.app.agent.models import AgentResponse
from backend.app.core.config import settings
from backend.app.core.database import Base
from backend.app.models.observability import AgentRun
from backend.app.services.runtime.agent_runtime_service import (
    AgentRunConflict,
    AgentRunStatus,
    AgentRuntimeService,
)


class FakeAgentStateStore:
    """内存版状态存储，接口与 AgentStateStore 对齐。

    单元测试不该依赖 Redis 进程，真实 Stream 的语义（XREAD 游标、阻塞、
    MAXLEN 截断）由 test_agent_state.py 对着真 Redis 验证；这里只需要
    接口行为一致，好让运行时服务的逻辑能被单独检查。
    """

    # 模仿真实存储的阻塞读：没有新事件时等一小会儿再返回。
    # 不这么做的话 iter_events 会变成忙等，测试表现为 CPU 打满。
    IDLE_SLEEP_SECONDS = 0.005

    def __init__(self) -> None:
        self.runs = {}
        self.streams = {}
        self.sessions = {}
        self._entry = 0

    # ---------------------------------------------------------- 运行注册表

    def create_run(
        self, run_id, session_id, query, status, allow_external_research, started_at
    ):
        self.runs[run_id] = {
            "run_id": run_id,
            "session_id": session_id,
            "query": query,
            "status": status,
            "sequence": 0,
            "allow_external_research": allow_external_research,
            "started_at": started_at,
            "finished_at": None,
            "error": None,
            "response": None,
        }
        self.streams.setdefault(run_id, [])

    def set_status(self, run_id, status, error=None):
        run = self.runs.get(run_id)
        if run is not None:
            run["status"] = status
            if error is not None:
                run["error"] = error

    def save_terminal(self, run_id, status, error, response, finished_at):
        run = self.runs.get(run_id)
        if run is not None:
            run.update(
                status=status,
                error=error or None,
                response=response,
                finished_at=finished_at,
            )

    def get_run(self, run_id):
        run = self.runs.get(run_id)
        return dict(run) if run is not None else None

    def delete_run(self, run_id):
        self.runs.pop(run_id, None)
        self.streams.pop(run_id, None)

    # ---------------------------------------------------------- 会话锁

    def acquire_session(self, session_id, run_id):
        if session_id in self.sessions:
            return False
        self.sessions[session_id] = run_id
        return True

    def renew_session(self, session_id, run_id):
        return self.sessions.get(session_id) == run_id

    def release_session(self, session_id, run_id):
        if self.sessions.get(session_id) == run_id:
            self.sessions.pop(session_id, None)

    def get_session_run(self, session_id):
        return self.sessions.get(session_id)

    # ---------------------------------------------------------- 事件流

    def append_event(self, run_id, event):
        self._entry += 1
        self.streams.setdefault(run_id, []).append((str(self._entry), event))
        run = self.runs.get(run_id)
        if run is not None:
            run["sequence"] = event["sequence"]

    def read_events(self, run_id, last_id="0", after_sequence=0, block_ms=0, count=500):
        entries = self.streams.get(run_id, [])
        if last_id == "0":
            start = 0
        else:
            start = len(entries)
            for index, (entry_id, _) in enumerate(entries):
                if entry_id == last_id:
                    start = index + 1
                    break
        pending = entries[start : start + count]
        if not pending:
            time.sleep(self.IDLE_SLEEP_SECONDS)
            return [], last_id
        return (
            [payload for _, payload in pending if payload["sequence"] > after_sequence],
            pending[-1][0],
        )

    def delete_events(self, run_id):
        self.streams.pop(run_id, None)

    def has_stream(self, run_id):
        # 与 Redis 一致：键只在第一次 XADD 之后才存在，空列表等同于不存在
        return bool(self.streams.get(run_id))

    def close(self):
        pass

    # 便于断言：完整的事件序列
    def events_of(self, run_id):
        return [payload for _, payload in self.streams.get(run_id, [])]


class StreamingAgent:
    def __init__(self):
        self.call_count = 0

    async def run(
        self,
        query,
        session_id,
        db,
        event_callback,
        cancel_check,
        raise_errors,
        allow_external_research=False,
    ):
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

    async def run(
        self,
        query,
        session_id,
        db,
        event_callback,
        cancel_check,
        raise_errors,
        allow_external_research=False,
    ):
        self.started.set()
        while True:
            if cancel_check():
                raise AgentRunCancelled("Agent 运行已取消")
            await asyncio.sleep(0.005)


class SlowAgent:
    async def run(
        self,
        query,
        session_id,
        db,
        event_callback,
        cancel_check,
        raise_errors,
        allow_external_research=False,
    ):
        await asyncio.sleep(10)


class AgentRuntimeTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "runtime.db"
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.state = FakeAgentStateStore()

    def _runtime(self, **kwargs) -> AgentRuntimeService:
        kwargs.setdefault("session_factory", self.session_factory)
        kwargs.setdefault("state_store", self.state)
        return AgentRuntimeService(**kwargs)

    def _stream_events(self, run_id):
        return [
            AgentRuntimeEvent.model_validate(payload)
            for payload in self.state.events_of(run_id)
        ]

    async def asyncTearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    async def test_event_order_matches_single_streamed_response_and_replays_terminal(self):
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)

        run = await runtime.start("测试问题", "session-stream")
        await run.task

        self.assertEqual(agent.call_count, 1)
        self.assertEqual(run.status, AgentRunStatus.COMPLETED)

        events = self._stream_events(run.run_id)
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        streamed_text = "".join(
            event.data["content"]
            for event in events
            if event.type == AgentEventType.TOKEN
        )
        terminal = events[-1]
        self.assertEqual(terminal.type, AgentEventType.RUN_COMPLETED)
        self.assertEqual(streamed_text, terminal.data["response"]["response"])

        # 换一个实例（本地没有运行句柄）读同一条流：事件流在 Redis 上，
        # 任何实例都该拿到完整序列——这正是多实例下 SSE 能落到任意实例的前提。
        other_instance = self._runtime()
        replayed = [
            event
            async for event in other_instance.iter_events(run.run_id, run.session_id)
            if event is not None
        ]
        self.assertEqual(
            [event.sequence for event in replayed],
            [event.sequence for event in events],
        )
        self.assertEqual(replayed[-1].type, AgentEventType.RUN_COMPLETED)
        self.assertEqual(replayed[-1].data["response"]["response"], "智语")

    async def test_stream_expired_falls_back_to_persisted_events(self):
        # 事件流有 TTL，过期后读不到终止事件。这时必须回落到数据库里的
        # 持久化事件，否则客户端会一直等一个永不到来的终止事件。
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("测试问题", "session-expired")
        await run.task

        self.state.delete_events(run.run_id)

        other_instance = self._runtime()
        replayed = [
            event
            async for event in other_instance.iter_events(run.run_id, run.session_id)
            if event is not None
        ]
        self.assertTrue(replayed)
        # 数据库里存的是非 TOKEN 事件（TOKEN 量大且终态里已有完整回答）
        self.assertNotIn(AgentEventType.TOKEN, [event.type for event in replayed])
        self.assertEqual(replayed[-1].type, AgentEventType.RUN_COMPLETED)

    async def test_same_session_rejects_concurrent_run_and_supports_cancellation(self):
        agent = BlockingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("第一个问题", "session-conflict")
        await asyncio.wait_for(agent.started.wait(), timeout=1)

        with self.assertRaises(AgentRunConflict):
            await runtime.start("第二个问题", "session-conflict")

        cancelling = await runtime.cancel(run.run_id, run.session_id)
        self.assertEqual(cancelling["status"], AgentRunStatus.CANCELLING.value)
        await asyncio.wait_for(run.task, timeout=1)
        self.assertEqual(run.status, AgentRunStatus.CANCELLED)
        self.assertEqual(
            self._stream_events(run.run_id)[-1].type, AgentEventType.RUN_CANCELLED
        )

    async def test_run_completion_releases_session_lock(self):
        # 锁没释放的话，用户跑完一轮再问第二个问题会被判「会话忙」。
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("第一个问题", "session-release")
        await run.task

        self.assertIsNone(self.state.get_session_run("session-release"))
        second = await runtime.start("第二个问题", "session-release")
        self.assertEqual(second.status, AgentRunStatus.PENDING)

    async def test_stale_session_lock_from_finished_run_is_taken_over(self):
        # 进程在释放锁之前退出时，锁会留到 TTL 到期。若只认锁不看状态，
        # 这个会话会被一把已经结束的运行锁死。
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        finished = await runtime.start("旧问题", "session-stale")
        await finished.task

        # 手动把锁塞回去，模拟「没释放成功」
        self.state.sessions["session-stale"] = finished.run_id

        run = await runtime.start("新问题", "session-stale")
        self.assertIsNotNone(run)
        await run.task

    async def test_total_timeout_produces_terminal_error(self):
        runtime = self._runtime(agent_factory=SlowAgent)
        with patch.object(settings, "agent_run_timeout_seconds", 0.02):
            run = await runtime.start("超时问题", "session-timeout")
            await asyncio.wait_for(run.task, timeout=1)

        self.assertEqual(run.status, AgentRunStatus.TIMED_OUT)
        last = self._stream_events(run.run_id)[-1]
        self.assertEqual(last.type, AgentEventType.RUN_ERROR)
        self.assertEqual(last.data["code"], "run_timeout")
        # 终态也要落到注册表里，否则别的实例查到的还是 running
        self.assertEqual(
            self.state.get_run(run.run_id)["status"], AgentRunStatus.TIMED_OUT.value
        )

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

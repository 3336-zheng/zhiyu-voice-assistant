"""Agent Runtime 状态、事件、取消和恢复测试。"""

import asyncio
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import sessionmaker

from backend.app.agent.events import (
    TERMINAL_EVENT_TYPES,
    AgentEventType,
    AgentRunCancelled,
    AgentRuntimeEvent,
)
from backend.app.agent.models import AgentResponse
from backend.app.core.config import settings
from test.support.db import make_test_engine
from backend.app.models.observability import AgentRun
from backend.app.services.runtime.agent_runtime_service import (
    AgentRunConflict,
    AgentRunNotFound,
    AgentRunStatus,
    AgentRuntimeError,
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
        self.cancels = {}
        self._entry = 0
        # 操作顺序日志：状态与事件是两个存储，谁先谁后决定了客户端
        # 收到终止事件时能不能查到终态，需要能被断言
        self.ops = []

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
        self.ops.append(("save_terminal", status))
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
        self.ops.append(("append_event", event["type"]))
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

    # ---------------------------------------------------------- 取消信号

    def request_cancel(self, run_id, ttl_seconds):
        self.cancels[run_id] = ttl_seconds

    def is_cancel_requested(self, run_id):
        return run_id in self.cancels

    def clear_cancel(self, run_id):
        self.cancels.pop(run_id, None)

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


class TracingAgent:
    """返回带完整追踪信息的响应，用于验证这些字段有没有落库。"""

    async def run(self, query, session_id, db, event_callback, cancel_check,
                  raise_errors, allow_external_research=False):
        event_callback(AgentEventType.TOKEN, {"content": "答"})
        return AgentResponse(
            query=query,
            response="答",
            session_id=session_id,
            timestamp=datetime.now(),
            execution_time_ms=7,
            timeline=[{"stage": "retrieval.recall", "duration_ms": 3.0}],
            retrieval_stats={"fused": 5, "reranked": 3},
            model_usage={"call_count": 2, "total_tokens": 123},
            token_budget={"used": 100, "limit": 4000},
        )


class AgentRuntimeTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "runtime.db"
        self.engine = make_test_engine()
        self.session_factory = sessionmaker(bind=self.engine)
        self.state = FakeAgentStateStore()
        self.dispatched = []

    def _runtime(self, **kwargs) -> AgentRuntimeService:
        """构造一个运行时服务。

        执行现在发生在 worker 进程里，所以 start() 只创建运行并投递。
        测试里用一个记录调用的替身投递器代替 arq，再显式调 execute_run
        来扮演 worker——这样「API 只创建」「worker 才执行」这条边界是被
        真的走过的，而不是靠 start() 顺手把活干了。
        """
        kwargs.setdefault("session_factory", self.session_factory)
        kwargs.setdefault("state_store", self.state)
        kwargs.setdefault("run_dispatcher", self._dispatch)
        return AgentRuntimeService(**kwargs)

    async def _dispatch(self, run_id: str) -> bool:
        self.dispatched.append(run_id)
        return True

    async def _run_to_completion(self, runtime: AgentRuntimeService, run_id: str, timeout: float = 5.0):
        """扮演 agent worker：把投递出去的运行执行掉。"""
        return await asyncio.wait_for(runtime.execute_run(run_id), timeout=timeout)

    def _execute_in_background(self, runtime: AgentRuntimeService, run_id: str) -> asyncio.Task:
        """扮演 agent worker，但不等待它结束——用于测试运行中途的交互。"""
        return asyncio.create_task(runtime.execute_run(run_id))

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
        self.assertEqual(self.dispatched, [run.run_id], "start 应当把运行投递出去")
        await self._run_to_completion(runtime, run.run_id)

        self.assertEqual(agent.call_count, 1)
        # 断言共享状态而不是 start() 返回的那个对象：执行在 worker 进程里，
        # 它新建了自己的运行句柄，API 手上这份只是一张回执。
        self.assertEqual(
            self.state.get_run(run.run_id)["status"], AgentRunStatus.COMPLETED.value
        )

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
        await self._run_to_completion(runtime, run.run_id)

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
        worker = self._execute_in_background(runtime, run.run_id)
        await asyncio.wait_for(agent.started.wait(), timeout=1)

        with self.assertRaises(AgentRunConflict):
            await runtime.start("第二个问题", "session-conflict")

        cancelling = await runtime.cancel(run.run_id, run.session_id)
        self.assertEqual(cancelling["status"], AgentRunStatus.CANCELLING.value)
        await asyncio.wait_for(worker, timeout=5)
        self.assertEqual(
            self.state.get_run(run.run_id)["status"], AgentRunStatus.CANCELLED.value
        )
        self.assertEqual(
            self._stream_events(run.run_id)[-1].type, AgentEventType.RUN_CANCELLED
        )

    async def _await_cancelled(self, run, worker_future, timeout: float = 5.0) -> None:
        """等运行结束，并确认它是被「协作式取消」停下的。

        只断言 status == CANCELLED 是不够的：asyncio.wait_for 超时会取消任务，
        而 _execute 捕获 CancelledError 时同样把状态设成 CANCELLED（错误信息是
        「服务退出」）。两者混在一起时，取消信号根本没送达也会让断言通过。
        所以这里断言只有协作式取消路径才会写下的错误信息。
        """
        await asyncio.wait_for(worker_future, timeout=timeout)
        state = self.state.get_run(run.run_id)
        self.assertEqual(state["status"], AgentRunStatus.CANCELLED.value)
        self.assertEqual(
            state["error"],
            "运行已由用户停止",
            "运行不是被取消信号停下的（可能是测试超时把任务取消了）",
        )

    async def test_另一个实例可以取消运行(self):
        # 本阶段的核心能力之一：取消请求落在哪个实例上都该生效。
        # 执行它的实例靠轮询把 Redis 上的信号同步到本地。
        agent = BlockingAgent()
        executor = self._runtime(agent_factory=lambda: agent)
        run = await executor.start("被取消的问题", "session-cross-cancel")
        worker = self._execute_in_background(executor, run.run_id)
        await asyncio.wait_for(agent.started.wait(), timeout=1)

        # 另一个实例：本地没有运行句柄，只有共享的状态存储
        other_instance = self._runtime()
        snapshot = await other_instance.cancel(run.run_id, run.session_id)
        self.assertEqual(snapshot["status"], AgentRunStatus.CANCELLING.value)

        await self._await_cancelled(run, worker)
        self.assertEqual(
            self.state.get_run(run.run_id)["status"], AgentRunStatus.CANCELLED.value
        )

    async def test_取消信号在运行结束后被清掉(self):
        # 不清的话它会留到 TTL 到期，期间的行为难以解释
        agent = BlockingAgent()
        executor = self._runtime(agent_factory=lambda: agent)
        run = await executor.start("被取消的问题", "session-clear-cancel")
        worker = self._execute_in_background(executor, run.run_id)
        await asyncio.wait_for(agent.started.wait(), timeout=1)

        other_instance = self._runtime()
        await other_instance.cancel(run.run_id, run.session_id)
        await self._await_cancelled(run, worker)

        self.assertFalse(self.state.is_cancel_requested(run.run_id))

    async def test_取消已结束的运行是幂等的(self):
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("已完成的问题", "session-idempotent")
        await self._run_to_completion(runtime, run.run_id)

        snapshot = await runtime.cancel(run.run_id, run.session_id)
        self.assertEqual(snapshot["status"], AgentRunStatus.COMPLETED.value)

    async def test_取消不存在或会话不符的运行会报错(self):
        agent = BlockingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("问题", "session-owner")
        worker = self._execute_in_background(runtime, run.run_id)
        await asyncio.wait_for(agent.started.wait(), timeout=1)
        try:
            with self.assertRaises(AgentRunNotFound):
                await runtime.cancel(run.run_id, "session-other")
            with self.assertRaises(AgentRunNotFound):
                await runtime.cancel("no-such-run", "session-owner")
        finally:
            await runtime.cancel(run.run_id, run.session_id)
            await asyncio.wait_for(worker, timeout=5)

    async def test_投递失败时释放会话锁并把运行标记为失败(self):
        # 不收拾的话运行会永远停在 pending，而会话锁还握着——
        # 用户之后每次提问都被判「会话忙」，要等锁的 TTL 到期才能恢复。
        async def broken_dispatch(run_id: str) -> bool:
            raise OSError("队列连不上")

        runtime = AgentRuntimeService(
            session_factory=self.session_factory,
            state_store=self.state,
            run_dispatcher=broken_dispatch,
        )
        with self.assertRaises(AgentRuntimeError):
            await runtime.start("问题", "session-dispatch-fail")

        self.assertIsNone(
            self.state.get_session_run("session-dispatch-fail"),
            "投递失败后会话锁没有释放",
        )
        states = list(self.state.runs.values())
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["status"], AgentRunStatus.FAILED.value)

    async def test_投递被拒时同样收尾(self):
        # arq 对已存在的 job_id 返回 None。正常路径走不到（run_id 是新 uuid），
        # 但真发生时也不能让会话被锁住。
        async def rejected_dispatch(run_id: str) -> bool:
            return False

        runtime = AgentRuntimeService(
            session_factory=self.session_factory,
            state_store=self.state,
            run_dispatcher=rejected_dispatch,
        )
        with self.assertRaises(AgentRunConflict):
            await runtime.start("问题", "session-dispatch-rejected")
        self.assertIsNone(self.state.get_session_run("session-dispatch-rejected"))

    async def test_worker_跳过已经取消的运行(self):
        # 投递到执行之间用户取消了，worker 拿到任务时不该再跑一遍
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("问题", "session-skip")
        await runtime.cancel(run.run_id, run.session_id)

        await runtime.execute_run(run.run_id)
        self.assertEqual(agent.call_count, 0, "已经取消的运行不该被执行")

    async def test_终止事件一定在终态写完之后才发出(self):
        """状态与事件是两个存储，谁先谁后决定了客户端会不会读到过期值。

        客户端收到 run_completed 之后往往会立刻查一次状态来渲染结果；
        事件先发的话那一查会看到 running。

        这里断言操作顺序而不是「读到终止事件后去查状态」——后者依赖时序，
        两次存储之间的窗口不一定每次都被踩到，测试会时灵时不灵。
        """
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("问题", "session-terminal-order")
        await self._run_to_completion(runtime, run.run_id)

        ops = self.state.ops
        terminal_index = max(
            index
            for index, (kind, value) in enumerate(ops)
            if kind == "append_event" and value == AgentEventType.RUN_COMPLETED.value
        )
        saves_before = [
            value
            for kind, value in ops[:terminal_index]
            if kind == "save_terminal" and value == AgentRunStatus.COMPLETED.value
        ]
        self.assertTrue(
            saves_before,
            "终止事件发出时终态还没写——客户端会读到过期的 running",
        )

    async def test_执行搬到_worker_后追踪字段仍然落库(self):
        """计划里标记的「已知坑」：这三个字段会不会因为执行换了进程而变空。

        它们来自 run.response，而落库（_persist_terminal）与执行（_execute）
        始终在同一个进程里——所以执行搬到 worker 之后依然成立。
        这条测试把结论钉住，避免以后有人把落库挪到 API 侧时静默丢数据。
        """
        runtime = self._runtime(agent_factory=TracingAgent)
        run = await runtime.start("问题", "session-tracing")
        await self._run_to_completion(runtime, run.run_id)

        session = self.session_factory()
        try:
            row = session.get(AgentRun, run.run_id)
        finally:
            session.close()

        self.assertIsNotNone(row)
        self.assertEqual(row.timeline, [{"stage": "retrieval.recall", "duration_ms": 3.0}])
        self.assertEqual(row.retrieval_stats, {"fused": 5, "reranked": 3})
        self.assertEqual(row.model_usage, {"call_count": 2, "total_tokens": 123})
        self.assertEqual((row.runtime_snapshot or {}).get("token_budget"), {"used": 100, "limit": 4000})

    async def test_运行已过期时安静跳过而不是报错(self):
        """注册表有 TTL，任务可能在队列里排到过期才被消费。

        这是**预期路径**不是故障：worker 停机重启后重放旧任务、
        或任务排队超过保留期，都会走到这里。当成异常抛的话会以 ERROR 刷屏
        并触发队列的重试告警，把真正的故障淹掉。
        """
        runtime = self._runtime()
        # 不存在的 run_id：不该抛异常
        await runtime.execute_run("expired-run-id")

    async def test_run_completion_releases_session_lock(self):
        # 锁没释放的话，用户跑完一轮再问第二个问题会被判「会话忙」。
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        run = await runtime.start("第一个问题", "session-release")
        await self._run_to_completion(runtime, run.run_id)

        self.assertIsNone(self.state.get_session_run("session-release"))
        second = await runtime.start("第二个问题", "session-release")
        self.assertEqual(second.status, AgentRunStatus.PENDING)

    async def test_stale_session_lock_from_finished_run_is_taken_over(self):
        # 进程在释放锁之前退出时，锁会留到 TTL 到期。若只认锁不看状态，
        # 这个会话会被一把已经结束的运行锁死。
        agent = StreamingAgent()
        runtime = self._runtime(agent_factory=lambda: agent)
        finished = await runtime.start("旧问题", "session-stale")
        await self._run_to_completion(runtime, finished.run_id)

        # 手动把锁塞回去，模拟「没释放成功」
        self.state.sessions["session-stale"] = finished.run_id

        run = await runtime.start("新问题", "session-stale")
        self.assertIsNotNone(run)
        await self._run_to_completion(runtime, run.run_id)

    async def test_total_timeout_produces_terminal_error(self):
        runtime = self._runtime(agent_factory=SlowAgent)
        with patch.object(settings, "agent_run_timeout_seconds", 0.02):
            run = await runtime.start("超时问题", "session-timeout")
            await self._run_to_completion(runtime, run.run_id)

        self.assertEqual(
            self.state.get_run(run.run_id)["status"], AgentRunStatus.TIMED_OUT.value
        )
        last = self._stream_events(run.run_id)[-1]
        self.assertEqual(last.type, AgentEventType.RUN_ERROR)
        self.assertEqual(last.data["code"], "run_timeout")
        # 终态也要落到注册表里，否则别的实例查到的还是 running
        self.assertEqual(
            self.state.get_run(run.run_id)["status"], AgentRunStatus.TIMED_OUT.value
        )

    # ------------------------------------------------------ 启动回收

    def _insert_run(self, run_id, session_id, status, events=None):
        """往数据库插一条运行记录。返回后连接已关闭，模拟「另一个进程写的」。"""
        session = self.session_factory()
        try:
            session.add(
                AgentRun(
                    request_id=run_id,
                    session_id=session_id,
                    query="问题",
                    status=status,
                    events=events if events is not None else [],
                )
            )
            session.commit()
        finally:
            session.close()

    async def test_运行中的运行会被回收(self):
        """执行进程重启后，它手上没跑完的运行必须标记失败。

        否则这些运行没有人再推进，客户端会一直挂着等一个永远不会到来的
        终止事件。
        """
        self._insert_run("dead-run", "s-dead", AgentRunStatus.RUNNING.value)
        self.assertEqual(self._runtime().recover_interrupted_runs(), 1)

    async def test_取消中的运行同样会被回收(self):
        # CANCELLING 也是「已被 worker 领走正在执行」，重启后同样没人推进
        self._insert_run("cancelling-run", "s-c", AgentRunStatus.CANCELLING.value)
        self.assertEqual(self._runtime().recover_interrupted_runs(), 1)

    async def test_pending_运行不被回收(self):
        """还没被领走的运行不该回收。

        它可能只是排在长队列里，进程重启并不影响它——队列里的任务不会丢，
        worker 恢复后会正常消费。顺手杀掉会误伤这种运行。
        """
        self._insert_run("queued-run", "s-queued", AgentRunStatus.PENDING.value)
        self.assertEqual(self._runtime().recover_interrupted_runs(), 0)

    async def test_终态运行不受影响(self):
        for index, status in enumerate(["completed", "failed", "cancelled"]):
            self._insert_run(f"done-{index}", f"s-d{index}", status)
        self.assertEqual(self._runtime().recover_interrupted_runs(), 0)

    async def test_回收时同步_redis_并释放会话锁(self):
        """不同步的话前端会在注册表 TTL 内一直看到 running。

        查状态走 Redis，而回收只改了数据库——两处不一致会持续到 TTL 到期
        （默认 1 小时）。释放会话锁则是让用户不必等锁到期才能重新提问。
        """
        self.state.create_run("dead-run", "s-lock", "问题", "running", False, 0.0)
        self.state.sessions["s-lock"] = "dead-run"
        self._insert_run("dead-run", "s-lock", AgentRunStatus.RUNNING.value)

        self.assertEqual(self._runtime().recover_interrupted_runs(), 1)
        self.assertEqual(self.state.get_run("dead-run")["status"], "failed")
        self.assertIsNone(self.state.get_session_run("s-lock"))

    async def test_redis_里没有这条运行时不写残缺记录(self):
        """Redis 的 hset 会自动建键，盲目写会造出一条只有半数字段的记录。

        注册表已过期、或创建时 Redis 不可用时就是这种情况；写进去的话
        查询时连 session_id 都对不上，比不写更糟。
        """
        self._insert_run("orphan-run", "s-orphan", AgentRunStatus.RUNNING.value)
        self.assertEqual(self._runtime().recover_interrupted_runs(), 1)
        self.assertIsNone(self.state.get_run("orphan-run"))

    async def test_restart_recovery_persists_replayable_error_event(self):
        """被回收的运行会留下可回放的终止事件。

        客户端不该只看到状态变 failed 却收不到任何事件——那样 SSE 会一直
        挂等。这里钉住事件也补齐了，并且带明确的错误码。
        """
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
        self.assertEqual(replayed[-1].data["code"], "worker_restarted")


if __name__ == "__main__":
    unittest.main()

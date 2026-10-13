"""Agent 运行态、事件回放、取消和终态持久化。

运行状态存在 Redis 上（见 agent_state.py），所以同一会话的运行可以被任意实例
查询、订阅和取消，不再要求请求落在发起它的那个实例上。

**执行仍然发生在这个进程里。** `_runs` 保留为「本实例正在执行的运行」的句柄，
它持有 asyncio 任务和本地序号计数器；但**它不是读取方的事实来源**——
`get` / `iter_events` 一律走 Redis，这样跨实例才看得到。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from backend.app.agent.events import (
    AgentEventType,
    AgentRunCancelled,
    AgentRuntimeEvent,
    TERMINAL_EVENT_TYPES,
)
from backend.app.agent.models import AgentResponse
from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.core.errors import ErrorCode, safe_error_message
from backend.app.core.logging_config import log_event
from backend.app.core.observability import (
    record_error,
    reset_request,
    start_request,
    get_recent_trace,
    store_current_trace,
)
from backend.app.models.observability import AgentRun
from .agent_state import (
    BLOCK_MS,
    SESSION_LOCK_RENEW_SECONDS,
    AgentStateStore,
    AgentStateUnavailable,
    get_agent_state_store,
)

logger = logging.getLogger(__name__)

# 事件流无新事件时发心跳的间隔。每次最多阻塞 1 秒读一次（见 agent_state），
# 这里控制的是「读了多少轮都没事件才发心跳」。
HEARTBEAT_SECONDS = 15


class AgentRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES = {
    AgentRunStatus.COMPLETED,
    AgentRunStatus.CANCELLED,
    AgentRunStatus.FAILED,
    AgentRunStatus.TIMED_OUT,
}

# 同一组状态的字符串形式。会话锁存的是 Redis 里的字符串状态，
# 判断「锁的主人还在不在跑」时用它比较。
TERMINAL_STATUS_VALUES = {status.value for status in TERMINAL_STATUSES}


class AgentRuntimeError(RuntimeError):
    """Agent 运行时基础异常。"""


class AgentRunNotFound(AgentRuntimeError):
    """运行不存在或不属于指定会话。"""


class AgentRunConflict(AgentRuntimeError):
    """同一会话已经存在活动运行。"""


@dataclass
class RuntimeRun:
    """本实例正在执行的运行。

    事件缓冲和 asyncio 唤醒已搬到 Redis（见 agent_state.py）；这里留下的
    `sequence` 是本地计数器——一次运行只有一个执行者，本地递增就够，
    不需要为每个 token 换来一次 Redis 自增。
    """

    run_id: str
    session_id: str
    query: str
    allow_external_research: bool = False
    status: AgentRunStatus = AgentRunStatus.PENDING
    sequence: int = 0
    persisted_events: List[Dict[str, Any]] = field(default_factory=list)
    cancel_signal: threading.Event = field(default_factory=threading.Event)
    task: Optional[asyncio.Task] = None
    response: Optional[AgentResponse] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    has_token_output: bool = False
    # 状态存储不可用时记下原因。事件流一断，SSE 客户端什么也拿不到，
    # 这次运行已经没有意义——cancel_check 会据此尽快中止它，
    # 而不是让它拖到 180 秒超时。
    state_broken: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class AgentRuntimeService:
    """以 Redis 承载运行态，以数据库保存终态快照。"""

    def __init__(
        self,
        agent_factory: Optional[Callable[[], Any]] = None,
        session_factory: Callable[[], Any] = SessionLocal,
        state_store: Optional[AgentStateStore] = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._session_factory = session_factory
        self._state = state_store or get_agent_state_store()
        # 只保存「本实例正在执行」的运行。读取方不查它，一律走 Redis。
        self._runs: Dict[str, RuntimeRun] = {}
        self._lock = asyncio.Lock()

    def _get_agent(self):
        if self._agent_factory is not None:
            return self._agent_factory()
        from backend.app.agent.agent import get_agent

        return get_agent()

    @staticmethod
    def _runtime_snapshot() -> Dict[str, Any]:
        """记录影响结果的运行配置，不包含提示词正文或密钥。"""
        return {
            "app_version": settings.app_version,
            "model": settings.llm_model,
            "fallback_enabled": settings.llm_fallback_enabled,
            "fallback_model": settings.llm_fallback_model or None,
            "agent_max_iterations": settings.agent_max_iterations,
            "run_timeout_seconds": settings.agent_run_timeout_seconds,
            "tool_context_token_budget": settings.agent_tool_context_token_budget,
            "llm_context_window_tokens": settings.llm_context_window_tokens,
            "llm_max_tokens": settings.llm_max_tokens,
            "llm_response_max_tokens": settings.llm_response_max_tokens,
            "rag_v2_enabled": settings.rag_v2_enabled,
            "rag_parent_child_enabled": settings.rag_parent_child_enabled,
            "rag_context_token_budget": settings.rag_context_token_budget,
        }

    async def start(
        self,
        query: str,
        session_id: str,
        *,
        allow_external_research: bool = False,
    ) -> RuntimeRun:
        """创建并启动一次运行；同一会话只允许一个活动运行。"""
        async with self._lock:
            self._cleanup_finished_runs()
            run_id = str(uuid.uuid4())
            self._claim_session(session_id, run_id)

            run = RuntimeRun(
                run_id=run_id,
                session_id=session_id,
                query=query,
                allow_external_research=bool(allow_external_research),
            )
            self._runs[run_id] = run
            # 先写注册表再追加事件：SSE 可能在任何时刻连上来，
            # 注册表不在的话它会以为这条运行不存在。
            self._state.create_run(
                run_id=run_id,
                session_id=session_id,
                query=query,
                status=run.status.value,
                allow_external_research=run.allow_external_research,
                started_at=run.started_at,
            )
            self._append_event(
                run,
                AgentEventType.RUN_STARTED,
                {"query": query, "status": AgentRunStatus.PENDING.value},
            )
            await asyncio.to_thread(self._persist_start, run)
            run.task = asyncio.create_task(
                self._execute(run),
                name=f"agent-run-{run.run_id}",
            )
            return run

    def _claim_session(self, session_id: str, run_id: str) -> None:
        """抢占会话锁，抢不到就报冲突。

        锁有可能被一条已经结束、但没来得及释放的运行占着（进程退出、
        Redis 短暂不可用）。所以抢不到时要回查那把锁的主人还在不在跑，
        还在跑才拒绝，否则把锁接过来——不然一个已经结束的运行会把这个
        会话永久锁死到 TTL 到期。
        """
        if self._state.acquire_session(session_id, run_id):
            return

        holder = self._state.get_session_run(session_id)
        holder_state = self._state.get_run(holder) if holder else None
        if holder and holder_state and holder_state["status"] not in TERMINAL_STATUS_VALUES:
            raise AgentRunConflict("当前会话已有正在运行的请求，请等待完成或先停止")

        logger.warning(
            "会话锁被已结束的运行占用，接管: session_id=%s 原持有者=%s 状态=%s",
            session_id, holder, (holder_state or {}).get("status"),
        )
        self._state.release_session(session_id, holder)
        if not self._state.acquire_session(session_id, run_id):
            # 接管过程中被别人抢先了。
            raise AgentRunConflict("当前会话已有正在运行的请求，请等待完成或先停止")

    def _cleanup_finished_runs(self) -> None:
        """清掉本实例里已经结束的运行句柄。

        注册表和事件流的保留期由 Redis 的 TTL 负责，这里只管本地字典——
        它是执行句柄，不是读取方的事实来源。
        """
        cutoff = time.time() - settings.agent_run_retention_seconds
        expired = [
            run_id
            for run_id, run in self._runs.items()
            if run.terminal and (run.finished_at or run.started_at) < cutoff
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)

    def _append_event(
        self,
        run: RuntimeRun,
        event_type: AgentEventType | str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentRuntimeEvent:
        normalized_type = AgentEventType(event_type)
        run.sequence += 1
        event = AgentRuntimeEvent(
            type=normalized_type,
            run_id=run.run_id,
            session_id=run.session_id,
            sequence=run.sequence,
            data=data or {},
        )
        if normalized_type == AgentEventType.TOKEN:
            run.has_token_output = True
        else:
            run.persisted_events.append(event.model_dump(mode="json"))

        # 序号先在本地递增再写 Redis：一次运行只有一个执行者，本地计数器
        # 足够保证单调；换成 Redis 自增会让每个 token 多一次网络往返。
        try:
            self._state.append_event(run.run_id, event.model_dump(mode="json"))
        except AgentStateUnavailable as exc:
            # 事件写不进去，读端也就什么都看不到，这次运行已经不可用了。
            # 记下来让 cancel_check 尽快中止它，别拖着跑到超时。
            logger.error(
                "Agent 事件写入失败，运行将中止: run_id=%s error=%s", run.run_id, exc
            )
            run.state_broken = str(exc)
            run.cancel_signal.set()
        return event

    async def _execute(self, run: RuntimeRun) -> None:
        run.status = AgentRunStatus.RUNNING
        await asyncio.to_thread(self._persist_status, run.run_id, run.status, None)
        self._set_status(run.run_id, run.status)
        loop = asyncio.get_running_loop()

        def event_callback(event_type: AgentEventType | str, data: Dict[str, Any]) -> None:
            if run.cancel_signal.is_set():
                raise AgentRunCancelled("Agent 运行已取消")
            loop.call_soon_threadsafe(self._append_event, run, event_type, data)

        def cancel_check() -> bool:
            # state_broken 时 _append_event 已经把 cancel_signal 置上了，
            # 所以这里不用再单独判断。
            return run.cancel_signal.is_set()

        # 运行最长 180 秒，而会话锁 TTL 只有 60 秒，必须续期。
        heartbeat = asyncio.create_task(
            self._renew_session_loop(run),
            name=f"agent-session-heartbeat-{run.run_id}",
        )

        (
            _,
            request_token,
            timings_token,
            timeline_token,
            usage_token,
            context_token,
        ) = start_request(run.run_id)
        db = self._session_factory()
        try:
            response = await asyncio.wait_for(
                self._get_agent().run(
                    run.query,
                    session_id=run.session_id,
                    db=db,
                    event_callback=event_callback,
                    cancel_check=cancel_check,
                    raise_errors=True,
                    allow_external_research=run.allow_external_research,
                ),
                timeout=settings.agent_run_timeout_seconds,
            )
            await asyncio.sleep(0)
            if run.cancel_signal.is_set():
                raise AgentRunCancelled("Agent 运行已取消")
            run.response = response
            if not run.has_token_output and response.response:
                self._append_event(
                    run,
                    AgentEventType.TOKEN,
                    {"content": response.response},
                )
            run.status = AgentRunStatus.COMPLETED
            self._append_event(
                run,
                AgentEventType.RUN_COMPLETED,
                {"response": response.model_dump(mode="json")},
            )
        except AgentRunCancelled:
            if run.state_broken:
                # 运行是被状态存储故障中止的，不是用户取消——报成失败，
                # 否则用户会以为是自己点了停止。
                run.status = AgentRunStatus.FAILED
                run.error = f"运行时状态存储不可用: {run.state_broken}"
                self._append_event(
                    run,
                    AgentEventType.RUN_ERROR,
                    {"message": run.error, "code": "state_unavailable"},
                )
            else:
                run.status = AgentRunStatus.CANCELLED
                run.error = "运行已由用户停止"
                self._append_event(
                    run,
                    AgentEventType.RUN_CANCELLED,
                    {"message": run.error},
                )
        except asyncio.TimeoutError:
            run.cancel_signal.set()
            run.status = AgentRunStatus.TIMED_OUT
            run.error = f"Agent 运行超过 {settings.agent_run_timeout_seconds:g} 秒"
            record_error(
                stage="agent.run",
                component="agent",
                operation="run",
                error_code=ErrorCode.AGENT_TIMEOUT,
                retryable=True,
                message=run.error,
            )
            self._append_event(
                run,
                AgentEventType.RUN_ERROR,
                {"message": run.error, "code": "run_timeout"},
            )
        except asyncio.CancelledError:
            run.cancel_signal.set()
            run.status = AgentRunStatus.CANCELLED
            run.error = "服务退出，运行已停止"
            self._append_event(
                run,
                AgentEventType.RUN_CANCELLED,
                {"message": run.error},
            )
        except Exception as exc:
            run.status = AgentRunStatus.FAILED
            run.error = safe_error_message(exc)
            record_error(
                stage="agent.run",
                component="agent",
                operation="run",
                error_code=ErrorCode.AGENT_EXECUTION_ERROR,
                retryable=False,
                exception=exc,
            )
            log_event(
                logger,
                logging.ERROR,
                "Agent 运行失败",
                component="agent",
                operation="run",
                error_code=ErrorCode.AGENT_EXECUTION_ERROR,
                retryable=False,
                exception=exc,
                run_id=run.run_id,
            )
            self._append_event(
                run,
                AgentEventType.RUN_ERROR,
                {"message": run.error, "code": type(exc).__name__},
            )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

            run.finished_at = time.time()
            duration_ms = (run.finished_at - run.started_at) * 1000
            store_current_trace(
                request_id=run.run_id,
                method="AGENT",
                path="agent.run",
                status_code={
                    AgentRunStatus.COMPLETED: 200,
                    AgentRunStatus.CANCELLED: 499,
                    AgentRunStatus.TIMED_OUT: 504,
                }.get(run.status, 500),
                total_ms=duration_ms,
                status=run.status.value,
                trace_type="agent",
                query=run.query,
                answer_quality=(
                    {
                        "confidence": run.response.confidence,
                        "evidence_status": run.response.evidence_status,
                        "evidence_score": run.response.evidence_score,
                        "evidence_source_count": run.response.evidence_source_count,
                        "evidence_reason": run.response.evidence_reason,
                    }
                    if run.response
                    else None
                ),
                retrieval_stats=run.response.retrieval_stats if run.response else None,
                timeline=run.response.timeline if run.response else None,
                model_usage=run.response.model_usage if run.response else None,
                token_budget=run.response.token_budget if run.response else None,
            )
            db.close()
            await asyncio.to_thread(self._persist_terminal, run)
            self._finalize_state(run)
            reset_request(
                request_token,
                timings_token,
                timeline_token,
                usage_token,
                context_token,
            )

    async def _renew_session_loop(self, run: RuntimeRun) -> None:
        """运行期间定期续期会话锁。

        锁 TTL 只有 60 秒而运行可以跑 180 秒。不续期的话锁会在运行中途过期，
        同一会话的第二个请求就能并发跑起来——锁等于没加。
        """
        while True:
            await asyncio.sleep(SESSION_LOCK_RENEW_SECONDS)
            try:
                if not self._state.renew_session(run.session_id, run.run_id):
                    logger.warning(
                        "会话锁已易主，停止续期: session_id=%s run_id=%s",
                        run.session_id, run.run_id,
                    )
                    return
            except AgentStateUnavailable as exc:
                logger.warning("会话锁续期失败: %s", exc)
                return

    def _set_status(self, run_id: str, status: AgentRunStatus) -> None:
        """把状态同步到 Redis；状态存储不可用时只记日志。

        状态写不进去不该让执行本身崩掉——终态还有数据库那一路兜底，
        而且 get() 在 Redis 查不到时会回落到数据库。
        """
        try:
            self._state.set_status(run_id, status.value)
        except AgentStateUnavailable as exc:
            logger.warning("写入运行状态失败: run_id=%s error=%s", run_id, exc)

    def _finalize_state(self, run: RuntimeRun) -> None:
        """运行收尾：写终态、释放会话锁。"""
        try:
            self._state.save_terminal(
                run_id=run.run_id,
                status=run.status.value,
                error=run.error,
                response=(
                    run.response.model_dump(mode="json") if run.response else None
                ),
                finished_at=run.finished_at or time.time(),
            )
            self._state.release_session(run.session_id, run.run_id)
        except AgentStateUnavailable as exc:
            # 锁没能释放会一直占到 TTL 到期；这是可接受的降级，
            # 但要留下痕迹，否则用户会遇到「刚跑完却说会话忙」而查不出原因。
            logger.error(
                "运行收尾写入状态失败，会话锁将等到 TTL 到期: run_id=%s error=%s",
                run.run_id, exc,
            )

    async def cancel(self, run_id: str, session_id: str) -> Dict[str, Any]:
        """幂等触发协作式取消。

        取消信号目前仍在执行进程的内存里，所以**只能取消本实例发起的运行**；
        别的实例发起的运行会走到「不存在于当前进程」的报错。跨实例取消需要把
        信号也放到 Redis（下一步做）。
        """
        run = await self._get_authorized_run(run_id, session_id)
        if run.terminal:
            return self._snapshot(run)
        run.cancel_signal.set()
        run.status = AgentRunStatus.CANCELLING
        await asyncio.to_thread(self._persist_status, run.run_id, run.status, None)
        self._set_status(run.run_id, run.status)
        return self._snapshot(run)

    async def get(self, run_id: str, session_id: str) -> Dict[str, Any]:
        """读取当前运行或已经持久化的终态快照。

        先查 Redis——这样别的实例发起的运行也查得到。Redis 不可用或运行
        已经从 Redis 过期时回落到数据库，那里至少还有终态。
        """
        state = await self._load_state(run_id)
        if state is not None:
            if state["session_id"] != session_id:
                raise AgentRunNotFound("Agent 运行不存在")
            return {
                "run_id": state["run_id"],
                "session_id": state["session_id"],
                "status": state["status"],
                "last_sequence": state["sequence"],
                "response": state["response"],
                "error": state["error"],
            }
        persisted = await asyncio.to_thread(self._load_persisted, run_id, session_id)
        if persisted is None:
            raise AgentRunNotFound("Agent 运行不存在")
        return persisted

    async def _load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """读取运行状态；Redis 不可用时返回 None 交给数据库兜底。"""
        try:
            return await asyncio.to_thread(self._state.get_run, run_id)
        except AgentStateUnavailable as exc:
            logger.warning("读取运行状态失败，回落到数据库: run_id=%s error=%s", run_id, exc)
            return None

    async def _get_authorized_run(self, run_id: str, session_id: str) -> RuntimeRun:
        run = self._runs.get(run_id)
        if run is None or run.session_id != session_id:
            raise AgentRunNotFound("Agent 运行不存在或已不在当前进程中")
        return run

    async def iter_events(
        self,
        run_id: str,
        session_id: str,
        after_sequence: int = 0,
    ) -> AsyncIterator[Optional[AgentRuntimeEvent]]:
        """按 sequence 增量返回事件；空值表示需要发送 SSE 心跳。

        读的是 Redis Stream，所以本实例执行的运行和别的实例执行的运行走的是
        同一条路径，断线重连到哪台机器都能从 after_sequence 续上。
        """
        cursor_id = "0"
        cursor_sequence = max(0, after_sequence)
        idle_seconds = 0.0

        # 开流前先确认事件流还在。流被 TTL 清掉后 XREAD 会一直返回空，
        # 而「读到空」和「运行还没输出」在读取端无法区分，客户端就会一直
        # 等一个永远不会到来的终止事件。这时改用数据库里的持久化事件收尾。
        if not await self._stream_present(run_id):
            async for item in self._replay_persisted(run_id, session_id, after_sequence):
                yield item
            return

        while True:
            events, cursor_id = await self._read_stream(run_id, cursor_id, cursor_sequence)
            for event in events:
                cursor_sequence = max(cursor_sequence, event.sequence)
                yield event
                if event.type in TERMINAL_EVENT_TYPES:
                    return

            if events:
                idle_seconds = 0.0
                continue

            # 读不到新事件：可能是运行还活着但暂时没有输出，也可能运行已经结束
            # 而终止事件已经被 MAXLEN 截掉。用运行状态区分。
            state = await self._load_state(run_id)
            if state is None or (
                state["status"] in TERMINAL_STATUS_VALUES
                and cursor_sequence >= state["sequence"]
            ):
                async for item in self._replay_persisted(run_id, session_id, cursor_sequence):
                    yield item
                return

            idle_seconds += BLOCK_MS / 1000
            if idle_seconds >= HEARTBEAT_SECONDS:
                idle_seconds = 0.0
                yield None

    async def _stream_present(self, run_id: str) -> bool:
        """事件流是否存在；状态存储不可用时当作存在，别误判成已过期。"""
        try:
            return await asyncio.to_thread(self._state.has_stream, run_id)
        except AgentStateUnavailable as exc:
            logger.warning("探测事件流失败: run_id=%s error=%s", run_id, exc)
            return True

    async def _read_stream(
        self,
        run_id: str,
        cursor_id: str,
        after_sequence: int,
    ) -> tuple[List[AgentRuntimeEvent], str]:
        """读一批事件；Redis 不可用时当作没有新事件（由状态探测决定何时收尾）。"""
        try:
            raw_events, new_cursor = await asyncio.to_thread(
                self._state.read_events, run_id, cursor_id, after_sequence
            )
        except AgentStateUnavailable as exc:
            logger.warning("读取事件流失败: run_id=%s error=%s", run_id, exc)
            return [], cursor_id

        parsed: List[AgentRuntimeEvent] = []
        for raw in raw_events:
            try:
                parsed.append(AgentRuntimeEvent.model_validate(raw))
            except Exception:  # noqa: BLE001 — 单条脏数据不该掐断整条流
                logger.warning("事件负载不符合协议，已跳过: run_id=%s", run_id)
        return parsed, new_cursor

    async def _replay_persisted(
        self,
        run_id: str,
        session_id: str,
        after_sequence: int,
    ) -> AsyncIterator[AgentRuntimeEvent]:
        """回放数据库里保存的终态事件，用于 Redis 流已经过期的情况。"""
        persisted = await asyncio.to_thread(self._load_persisted, run_id, session_id)
        if persisted is None:
            raise AgentRunNotFound("Agent 运行不存在")
        for payload in persisted.get("events", []):
            event = AgentRuntimeEvent.model_validate(payload)
            if event.sequence > after_sequence:
                yield event

    async def shutdown(self) -> None:
        """应用退出时取消仍在运行的任务并等待状态收敛。"""
        tasks = []
        for run in self._runs.values():
            if run.task and not run.task.done():
                run.cancel_signal.set()
                run.task.cancel()
                tasks.append(run.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def recover_interrupted_runs(self) -> int:
        """服务重启后将无法继续的进程内运行标记为失败。"""
        db = self._session_factory()
        try:
            rows = db.query(AgentRun).filter(
                AgentRun.status.in_([
                    AgentRunStatus.PENDING.value,
                    AgentRunStatus.RUNNING.value,
                    AgentRunStatus.CANCELLING.value,
                ])
            ).all()
            now = datetime.now(timezone.utc)
            for row in rows:
                row.status = AgentRunStatus.FAILED.value
                row.error = "服务重启，运行未能继续"
                events = list(row.events or [])
                sequence = max(
                    (item.get("sequence", 0) for item in events if isinstance(item, dict)),
                    default=0,
                )
                events.append(
                    AgentRuntimeEvent(
                        type=AgentEventType.RUN_ERROR,
                        run_id=row.request_id,
                        session_id=row.session_id or "unknown",
                        sequence=sequence + 1,
                        data={
                            "message": row.error,
                            "code": "service_restarted",
                        },
                    ).model_dump(mode="json")
                )
                row.events = events
                row.completed_at = now
                row.updated_at = now
            if rows:
                db.commit()
            return len(rows)
        finally:
            db.close()

    @staticmethod
    def _snapshot(run: RuntimeRun) -> Dict[str, Any]:
        return {
            "run_id": run.run_id,
            "session_id": run.session_id,
            "status": run.status.value,
            "last_sequence": run.sequence,
            "response": run.response.model_dump(mode="json") if run.response else None,
            "error": run.error,
        }

    def _persist_start(self, run: RuntimeRun) -> None:
        db = self._session_factory()
        try:
            db.merge(
                AgentRun(
                    request_id=run.run_id,
                    session_id=run.session_id,
                    query=run.query,
                    status=run.status.value,
                    runtime_snapshot=self._runtime_snapshot(),
                )
            )
            db.commit()
        finally:
            db.close()

    def _persist_status(
        self,
        run_id: str,
        status: AgentRunStatus,
        error: Optional[str],
    ) -> None:
        db = self._session_factory()
        try:
            row = db.get(AgentRun, run_id)
            if row is not None:
                row.status = status.value
                row.error = error
                row.updated_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()

    def _persist_terminal(self, run: RuntimeRun) -> None:
        db = self._session_factory()
        try:
            row = db.get(AgentRun, run.run_id)
            if row is None:
                row = AgentRun(
                    request_id=run.run_id,
                    session_id=run.session_id,
                    query=run.query,
                    status=run.status.value,
                )
                db.add(row)
            row.status = run.status.value
            row.response = run.response.response if run.response else None
            row.error = run.error
            row.events = run.persisted_events
            row.execution_time_ms = int(
                ((run.finished_at or time.time()) - run.started_at) * 1000
            )
            row.completed_at = datetime.now(timezone.utc)
            row.updated_at = row.completed_at
            if run.response:
                row.timeline = run.response.timeline
                row.retrieval_stats = run.response.retrieval_stats
                row.model_usage = run.response.model_usage
                row.intent = run.response.plan.intent.value if run.response.plan else None
                snapshot = dict(row.runtime_snapshot or {})
                snapshot["token_budget"] = run.response.token_budget or {}
                row.runtime_snapshot = snapshot
            else:
                trace = get_recent_trace(run.run_id) or {}
                row.timeline = trace.get("timeline") or row.timeline
                row.retrieval_stats = trace.get("retrieval_stats") or row.retrieval_stats
                row.model_usage = trace.get("model_usage") or row.model_usage
                if trace.get("answer_quality"):
                    snapshot = dict(row.runtime_snapshot or {})
                    snapshot["answer_quality"] = trace["answer_quality"]
                    row.runtime_snapshot = snapshot
            db.commit()
        except Exception:
            db.rollback()
            logger.warning("保存 Agent 终态失败 run_id=%s", run.run_id, exc_info=True)
        finally:
            db.close()

    def _load_persisted(
        self,
        run_id: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        db = self._session_factory()
        try:
            row = db.query(AgentRun).filter_by(
                request_id=run_id,
                session_id=session_id,
            ).first()
            if row is None:
                return None
            return {
                "run_id": row.request_id,
                "session_id": row.session_id,
                "status": row.status,
                "last_sequence": max(
                    (item.get("sequence", 0) for item in (row.events or [])),
                    default=0,
                ),
                "response": row.response,
                "error": row.error,
                "events": row.events or [],
            }
        finally:
            db.close()


runtime_instance: Optional[AgentRuntimeService] = None


def get_agent_runtime_service() -> AgentRuntimeService:
    """获取进程内 Agent Runtime 单例。"""
    global runtime_instance
    if runtime_instance is None:
        runtime_instance = AgentRuntimeService()
    return runtime_instance

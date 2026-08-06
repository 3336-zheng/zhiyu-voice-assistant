"""单进程 Agent 运行态、事件回放、取消和终态持久化。"""

from __future__ import annotations

import asyncio
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
from backend.app.core.observability import reset_request, start_request
from backend.app.models.observability import AgentRun

logger = logging.getLogger(__name__)


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


class AgentRuntimeError(RuntimeError):
    """Agent 运行时基础异常。"""


class AgentRunNotFound(AgentRuntimeError):
    """运行不存在或不属于指定会话。"""


class AgentRunConflict(AgentRuntimeError):
    """同一会话已经存在活动运行。"""


@dataclass
class RuntimeRun:
    run_id: str
    session_id: str
    query: str
    status: AgentRunStatus = AgentRunStatus.PENDING
    sequence: int = 0
    events: List[AgentRuntimeEvent] = field(default_factory=list)
    persisted_events: List[Dict[str, Any]] = field(default_factory=list)
    cancel_signal: threading.Event = field(default_factory=threading.Event)
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    response: Optional[AgentResponse] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    has_token_output: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class AgentRuntimeService:
    """以进程内事件缓冲服务当前请求，以 SQLite 保存终态快照。"""

    def __init__(
        self,
        agent_factory: Optional[Callable[[], Any]] = None,
        session_factory: Callable[[], Any] = SessionLocal,
    ) -> None:
        self._agent_factory = agent_factory
        self._session_factory = session_factory
        self._runs: Dict[str, RuntimeRun] = {}
        self._active_sessions: Dict[str, str] = {}
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
            "rag_v2_enabled": settings.rag_v2_enabled,
            "rag_parent_child_enabled": settings.rag_parent_child_enabled,
            "rag_context_token_budget": settings.rag_context_token_budget,
        }

    async def start(self, query: str, session_id: str) -> RuntimeRun:
        """创建并启动一次运行；同一会话只允许一个活动运行。"""
        async with self._lock:
            self._cleanup_finished_runs()
            active_run_id = self._active_sessions.get(session_id)
            if active_run_id:
                active = self._runs.get(active_run_id)
                if active and not active.terminal:
                    raise AgentRunConflict("当前会话已有正在运行的请求，请等待完成或先停止")

            run = RuntimeRun(
                run_id=str(uuid.uuid4()),
                session_id=session_id,
                query=query,
            )
            self._runs[run.run_id] = run
            self._active_sessions[session_id] = run.run_id
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

    def _cleanup_finished_runs(self) -> None:
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
        run.events.append(event)
        if len(run.events) > settings.agent_event_buffer_size:
            del run.events[: len(run.events) - settings.agent_event_buffer_size]
        if normalized_type == AgentEventType.TOKEN:
            run.has_token_output = True
        else:
            run.persisted_events.append(event.model_dump(mode="json"))
        run.updated.set()
        return event

    async def _execute(self, run: RuntimeRun) -> None:
        run.status = AgentRunStatus.RUNNING
        await asyncio.to_thread(self._persist_status, run.run_id, run.status, None)
        loop = asyncio.get_running_loop()

        def event_callback(event_type: AgentEventType | str, data: Dict[str, Any]) -> None:
            if run.cancel_signal.is_set():
                raise AgentRunCancelled("Agent 运行已取消")
            loop.call_soon_threadsafe(self._append_event, run, event_type, data)

        def cancel_check() -> bool:
            return run.cancel_signal.is_set()

        (
            _,
            request_token,
            timings_token,
            timeline_token,
            usage_token,
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
            run.error = str(exc)
            logger.error("Agent 运行失败 run_id=%s", run.run_id, exc_info=True)
            self._append_event(
                run,
                AgentEventType.RUN_ERROR,
                {"message": run.error, "code": type(exc).__name__},
            )
        finally:
            run.finished_at = time.time()
            db.close()
            await asyncio.to_thread(self._persist_terminal, run)
            reset_request(request_token, timings_token, timeline_token, usage_token)
            async with self._lock:
                if self._active_sessions.get(run.session_id) == run.run_id:
                    self._active_sessions.pop(run.session_id, None)
            run.updated.set()

    async def cancel(self, run_id: str, session_id: str) -> Dict[str, Any]:
        """幂等触发协作式取消。"""
        run = await self._get_authorized_run(run_id, session_id)
        if run.terminal:
            return self._snapshot(run)
        run.cancel_signal.set()
        run.status = AgentRunStatus.CANCELLING
        await asyncio.to_thread(self._persist_status, run.run_id, run.status, None)
        run.updated.set()
        return self._snapshot(run)

    async def get(self, run_id: str, session_id: str) -> Dict[str, Any]:
        """读取当前运行或已经持久化的终态快照。"""
        run = self._runs.get(run_id)
        if run is not None:
            if run.session_id != session_id:
                raise AgentRunNotFound("Agent 运行不存在")
            return self._snapshot(run)
        persisted = await asyncio.to_thread(self._load_persisted, run_id, session_id)
        if persisted is None:
            raise AgentRunNotFound("Agent 运行不存在")
        return persisted

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
        """按 sequence 增量返回事件；空值表示需要发送 SSE 心跳。"""
        run = self._runs.get(run_id)
        if run is None:
            persisted = await asyncio.to_thread(self._load_persisted, run_id, session_id)
            if persisted is None:
                raise AgentRunNotFound("Agent 运行不存在")
            for payload in persisted.get("events", []):
                event = AgentRuntimeEvent.model_validate(payload)
                if event.sequence > after_sequence:
                    yield event
            return
        if run.session_id != session_id:
            raise AgentRunNotFound("Agent 运行不存在")

        cursor = max(0, after_sequence)
        while True:
            available = [event for event in run.events if event.sequence > cursor]
            for event in available:
                cursor = event.sequence
                yield event
            if run.terminal and cursor >= run.sequence:
                return

            run.updated.clear()
            if any(event.sequence > cursor for event in run.events):
                continue
            try:
                await asyncio.wait_for(run.updated.wait(), timeout=15)
            except asyncio.TimeoutError:
                yield None

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
            row.completed_at = datetime.now(timezone.utc)
            row.updated_at = row.completed_at
            if run.response:
                row.execution_time_ms = run.response.execution_time_ms
                row.timeline = run.response.timeline
                row.retrieval_stats = run.response.retrieval_stats
                row.model_usage = run.response.model_usage
                row.intent = run.response.plan.intent.value if run.response.plan else None
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

"""Agent 运行时状态的 Redis 存储。

Stage 6 把原先存在进程内存里的三样东西搬到这里，让多个 API 实例能协作：

1. **运行注册表**（原 `AgentRuntimeService._runs`）：状态、查询、序号、终态响应
2. **会话锁**（原 `_active_sessions`）：同一会话同时只允许一个运行
3. **事件流**（原 `RuntimeRun.events`）：SSE 的事件来源，带断点续传

**为什么写事件用同步客户端**：`_append_event` 由事件循环调用，而 Redis Stream 的
顺序就是 SSE 的顺序。同步客户端在同一个线程里串行发命令，顺序天然等于调用顺序；
换成 redis.asyncio 的话，并发命令会落到连接池的不同连接上，到达服务端的次序不保证，
事件就会乱序——而乱序的事件流对前端是不可用的。

**为什么阻塞读只有 1 秒**：`XREAD BLOCK` 会占住一个线程。按 15 秒（心跳间隔）
一次读到底的话，线程池会被 SSE 连接占满，把检索、DB 那些 `to_thread` 一起饿死。
改成每次最多阻塞 1 秒、在事件循环里循环，线程占用有界。心跳仍按满 15 秒发。
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# 建连快，读写留出阻塞读的余量：socket_timeout 必须大于单次 XREAD BLOCK 的时长，
# 否则每次阻塞读都会被客户端自己判超时。
_CONNECT_TIMEOUT_SECONDS = 0.3
_READ_TIMEOUT_SECONDS = 2.0

# 单次阻塞读的时长。见模块开头「为什么阻塞读只有 1 秒」。
BLOCK_MS = 1000
# 单次最多取回的事件条数。追赶历史时会被调用方循环消费。
READ_COUNT = 500

RUN_KEY = "zhiyu:agent:run:{run_id}"
SESSION_KEY = "zhiyu:agent:session:{session_id}"
EVENTS_KEY = "zhiyu:agent:events:{run_id}"
CANCEL_KEY = "zhiyu:agent:cancel:{run_id}"

# 会话锁的存活时间与续期间隔。运行最长 180 秒，所以必须续期；
# TTL 取 3 倍续期间隔，容忍两次续期丢失。
SESSION_LOCK_TTL_SECONDS = 60
SESSION_LOCK_RENEW_SECONDS = 20


class AgentStateUnavailable(RuntimeError):
    """Redis 不可用，Agent 运行时状态无法读写。

    这里不做「退回进程内」的降级：那会让多实例协作静默失效——请求落到哪个实例
    就只能看见哪个实例的运行，正是本阶段要消灭的那类问题。宁可明确报错。
    """


class AgentStateStore:
    """Agent 运行时状态的 Redis 访问层。"""

    def __init__(self, client_factory=None) -> None:
        self._client = None
        self._client_factory = client_factory
        self._lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                if self._client_factory is not None:
                    client = self._client_factory()
                else:
                    import redis

                    client = redis.Redis.from_url(
                        settings.redis_url,
                        socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
                        socket_timeout=_READ_TIMEOUT_SECONDS,
                        decode_responses=True,
                    )
                    client.ping()
            except Exception as exc:  # noqa: BLE001 — 统一转成对外错误类型
                raise AgentStateUnavailable(
                    f"无法连接 Redis，Agent 运行时状态不可用: {exc}"
                ) from exc
            self._client = client
            return client

    def reset_for_tests(self) -> None:
        with self._lock:
            self._client = None

    # ------------------------------------------------------------------ 运行注册表

    def create_run(
        self,
        run_id: str,
        session_id: str,
        query: str,
        status: str,
        allow_external_research: bool,
        started_at: float,
    ) -> None:
        client = self._get_client()
        key = RUN_KEY.format(run_id=run_id)
        client.hset(
            key,
            mapping={
                "run_id": run_id,
                "session_id": session_id,
                "query": query,
                "status": status,
                "sequence": 0,
                "allow_external_research": int(allow_external_research),
                "started_at": started_at,
                "finished_at": "",
                "error": "",
                "response": "",
            },
        )
        client.expire(key, settings.agent_run_retention_seconds)

    def set_status(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        client = self._get_client()
        key = RUN_KEY.format(run_id=run_id)
        mapping: Dict[str, Any] = {"status": status}
        if error is not None:
            mapping["error"] = error
        client.hset(key, mapping=mapping)
        client.expire(key, settings.agent_run_retention_seconds)

    def save_terminal(
        self,
        run_id: str,
        status: str,
        error: Optional[str],
        response: Optional[Dict[str, Any]],
        finished_at: float,
    ) -> None:
        """写入终态。到这里运行不再变化，TTL 从这一刻重新算。"""
        client = self._get_client()
        key = RUN_KEY.format(run_id=run_id)
        client.hset(
            key,
            mapping={
                "status": status,
                "error": error or "",
                "response": json.dumps(response, ensure_ascii=False) if response else "",
                "finished_at": finished_at,
            },
        )
        client.expire(key, settings.agent_run_retention_seconds)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        raw = self._get_client().hgetall(RUN_KEY.format(run_id=run_id))
        if not raw:
            return None
        response = raw.get("response") or ""
        return {
            "run_id": raw.get("run_id") or run_id,
            "session_id": raw.get("session_id") or "",
            "query": raw.get("query") or "",
            "status": raw.get("status") or "",
            "sequence": int(raw.get("sequence") or 0),
            "allow_external_research": bool(int(raw.get("allow_external_research") or 0)),
            "started_at": float(raw.get("started_at") or 0),
            "finished_at": float(raw["finished_at"]) if raw.get("finished_at") else None,
            "error": raw.get("error") or None,
            "response": json.loads(response) if response else None,
        }

    def delete_run(self, run_id: str) -> None:
        client = self._get_client()
        client.delete(
            RUN_KEY.format(run_id=run_id),
            EVENTS_KEY.format(run_id=run_id),
            CANCEL_KEY.format(run_id=run_id),
        )

    # ------------------------------------------------------------------ 取消信号

    def request_cancel(self, run_id: str, ttl_seconds: int) -> None:
        """请求取消一次运行。

        写在 Redis 上而不是直接改执行进程的内存：发起取消的请求可能落在
        任何一个实例上，而运行的执行者只有一个。执行者靠轮询把它同步到
        本地的 threading.Event（见 agent_runtime_service._watch_cancel）。
        """
        self._get_client().set(CANCEL_KEY.format(run_id=run_id), "1", ex=ttl_seconds)

    def is_cancel_requested(self, run_id: str) -> bool:
        return bool(self._get_client().exists(CANCEL_KEY.format(run_id=run_id)))

    def clear_cancel(self, run_id: str) -> None:
        self._get_client().delete(CANCEL_KEY.format(run_id=run_id))

    # ------------------------------------------------------------------ 会话锁

    def acquire_session(self, session_id: str, run_id: str) -> bool:
        """抢占会话锁。返回 False 表示该会话已有活动运行。"""
        client = self._get_client()
        acquired = client.set(
            SESSION_KEY.format(session_id=session_id),
            run_id,
            nx=True,
            ex=SESSION_LOCK_TTL_SECONDS,
        )
        return bool(acquired)

    def renew_session(self, session_id: str, run_id: str) -> bool:
        """续期会话锁。

        只在锁仍属于本次运行时才续——否则锁过期后已被别的运行抢走，
        无条件 EXPIRE 会把别人的锁也一起延长。
        """
        client = self._get_client()
        key = SESSION_KEY.format(session_id=session_id)
        if client.get(key) != run_id:
            return False
        client.expire(key, SESSION_LOCK_TTL_SECONDS)
        return True

    def release_session(self, session_id: str, run_id: str) -> None:
        client = self._get_client()
        key = SESSION_KEY.format(session_id=session_id)
        if client.get(key) == run_id:
            client.delete(key)

    def get_session_run(self, session_id: str) -> Optional[str]:
        value = self._get_client().get(SESSION_KEY.format(session_id=session_id))
        return value or None

    # ------------------------------------------------------------------ 事件流

    def append_event(self, run_id: str, event: Dict[str, Any]) -> None:
        """追加一个事件，并同步更新运行注册表里的序号。

        用管道把三件事压成一次往返：写流、记序号、续 TTL。
        每个 token 都会走到这里，往返次数直接影响流式输出的手感。
        """
        client = self._get_client()
        events_key = EVENTS_KEY.format(run_id=run_id)
        run_key = RUN_KEY.format(run_id=run_id)
        pipe = client.pipeline()
        pipe.xadd(
            events_key,
            {"payload": json.dumps(event, ensure_ascii=False)},
            maxlen=settings.agent_event_buffer_size,
            approximate=True,
        )
        pipe.hset(run_key, "sequence", event["sequence"])
        pipe.expire(events_key, settings.agent_run_retention_seconds)
        pipe.expire(run_key, settings.agent_run_retention_seconds)
        pipe.execute()

    def read_events(
        self,
        run_id: str,
        last_id: str = "0",
        after_sequence: int = 0,
        block_ms: int = BLOCK_MS,
        count: int = READ_COUNT,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """从 `last_id` 之后读取事件，返回 (事件列表, 新游标)。

        游标一开始是 "0"，此时 XREAD 会把流里已有的条目一次性带回来，
        之后再阻塞等待新条目——新连接追赶历史与实时推送用的是同一条路径。

        按 sequence 过滤而不是只依赖流位置：断点续传时客户端给的是序号，
        而流里可能因为 MAXLEN 截断缺了前面几条，两者要对齐。
        """
        client = self._get_client()
        rows = client.xread(
            {EVENTS_KEY.format(run_id=run_id): last_id},
            count=count,
            block=block_ms,
        )
        if not rows:
            return [], last_id

        events: List[Dict[str, Any]] = []
        cursor = last_id
        for _stream_name, entries in rows:
            for entry_id, fields in entries:
                cursor = entry_id
                payload = fields.get("payload")
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except (TypeError, ValueError):
                    logger.warning("事件负载无法解析，已跳过: run_id=%s id=%s", run_id, entry_id)
                    continue
                if int(event.get("sequence") or 0) > after_sequence:
                    events.append(event)
        return events, cursor

    def delete_events(self, run_id: str) -> None:
        self._get_client().delete(EVENTS_KEY.format(run_id=run_id))

    def has_stream(self, run_id: str) -> bool:
        """事件流是否还在。

        被 TTL 清掉之后 XREAD 会一直返回空，读取方无法从「读到空」区分出
        「运行还没输出」和「流已经没了」。所以需要单独问一次，否则客户端
        会一直等一个永远不会到来的终止事件。
        """
        return bool(self._get_client().exists(EVENTS_KEY.format(run_id=run_id)))

    # ------------------------------------------------------------------ 生命周期

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001 — 关闭失败不该影响退出
                    logger.debug("关闭 Agent 状态存储连接失败", exc_info=True)
                self._client = None


_store: Optional[AgentStateStore] = None


def get_agent_state_store() -> AgentStateStore:
    global _store
    if _store is None:
        _store = AgentStateStore()
    return _store

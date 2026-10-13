"""近期追踪快照的 Redis 存储。

进程内那张 `_recent_traces` 表只在单个实例里有效，多实例下查追踪等于抽奖——
用户在实例 A 发起的查询，去实例 B 的观测页面上看不到。

**这里的降级方向与 Agent 运行状态相反**：追踪是诊断数据，不是正确性数据。
Redis 不可用时退回进程内的表，应用照常工作（只是看不到别的实例的记录）。
Agent 运行状态则相反——读不到就没法跨实例协作，所以那边宁可报错也不降级。

容量按「条数 + 时间」双重限制：条数上限跟进程内那张表一致（最近 500 条），
时间上限兜底，避免一条长期没人读的追踪永久占着内存。
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_SECONDS = 0.3
_TRACE_TTL_SECONDS = 24 * 3600
# 连接失败后多久再试。追踪写入在请求路径上，不能每次请求都白付一次连接超时。
_RETRY_INTERVAL_SECONDS = 30.0

INDEX_KEY = "zhiyu:trace:index"
TRACE_KEY = "zhiyu:trace:{request_id}"


class RecentTraceStore:
    """追踪快照的 Redis 存储；不可用时安静退回进程内。"""

    def __init__(self, limit: int, client_factory=None) -> None:
        self._limit = limit
        self._client_factory = client_factory
        self._client = None
        self._lock = threading.Lock()
        self._retry_after = 0.0

    def _get_client(self):
        with self._lock:
            if self._client is not None:
                return self._client
            if time.monotonic() < self._retry_after:
                return None
            try:
                if self._client_factory is not None:
                    client = self._client_factory()
                else:
                    import redis

                    client = redis.Redis.from_url(
                        settings.redis_url,
                        socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
                        socket_timeout=_CONNECT_TIMEOUT_SECONDS,
                        decode_responses=True,
                    )
                    client.ping()
            except Exception as exc:  # noqa: BLE001 — 追踪不该让请求失败
                self._retry_after = time.monotonic() + _RETRY_INTERVAL_SECONDS
                logger.warning(
                    "追踪存储不可用，暂时只保留本进程的记录: %s", exc
                )
                return None
            self._client = client
            return client

    def reset_for_tests(self) -> None:
        with self._lock:
            self._client = None
            self._retry_after = 0.0

    def put(self, request_id: str, snapshot: Dict[str, Any], completed_at: float) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            payload = json.dumps(snapshot, ensure_ascii=False, default=str)
            pipe = client.pipeline()
            pipe.set(TRACE_KEY.format(request_id=request_id), payload, ex=_TRACE_TTL_SECONDS)
            pipe.zadd(INDEX_KEY, {request_id: completed_at})
            # 只保留最新的 limit 条：0 到 -(limit+1) 是要删掉的旧记录
            pipe.zremrangebyrank(INDEX_KEY, 0, -(self._limit + 1))
            pipe.expire(INDEX_KEY, _TRACE_TTL_SECONDS)
            pipe.execute()
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return None
        try:
            payload = client.get(TRACE_KEY.format(request_id=request_id))
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            return None
        if not payload:
            return None
        try:
            return json.loads(payload)
        except (TypeError, ValueError):
            logger.warning("追踪快照无法解析: request_id=%s", request_id)
            return None

    def list_recent(self, limit: int) -> List[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return []
        # 多取一些再过滤：调用方按 trace_type 筛，而索引里混着所有类型，
        # 只取 limit 条的话过滤完可能不够。
        fetch = min(max(limit * 3, limit), self._limit)
        try:
            request_ids = client.zrevrange(INDEX_KEY, 0, fetch - 1)
            if not request_ids:
                return []
            payloads = client.mget(
                [TRACE_KEY.format(request_id=item) for item in request_ids]
            )
        except Exception as exc:  # noqa: BLE001
            self._degrade(exc)
            return []

        snapshots: List[Dict[str, Any]] = []
        for payload in payloads:
            if not payload:
                continue
            try:
                snapshots.append(json.loads(payload))
            except (TypeError, ValueError):
                continue
        return snapshots

    def _degrade(self, exc: Exception) -> None:
        """读写失败后退回进程内，并进入退避窗口。"""
        with self._lock:
            self._client = None
            self._retry_after = time.monotonic() + _RETRY_INTERVAL_SECONDS
        logger.warning("追踪存储读写失败，暂时只保留本进程的记录: %s", exc)

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001
                    logger.debug("关闭追踪存储连接失败", exc_info=True)
                self._client = None

"""Redis 分布式模型并发闸门。

API 与三个 worker 是独立进程，各自调用同一个模型网关。进程内的信号量拦不住
跨进程的并发，高峰期四个进程合计可能打出远超供应商限额的请求，触发 429。

这里用 Redis ZSET 做一个带租约的分布式信号量：

- 键按 ``kind + provider + model`` 划分，计数的是**所有进程合计**的并发；
- 每个持有者写入一个唯一成员，分数是获取时间；
- 获取前先清理分数早于租约的成员——持有者崩溃后不会永久占位；
- 释放就是删除自己的成员。

**失败策略是降级放行**：Redis 不可用时记录限频告警并直接放行。闸门是防 429 的
优化，不该成为新的单点故障——它挂了，业务照常，只是失去了跨进程保护。
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

from backend.app.core.config import settings
from backend.app.core.observability import timed_stage

logger = logging.getLogger(__name__)

MODEL_GATE_KEY = "zhiyu:model_gate:{kind}:{provider}:{model}"

# 轮询间隔。50ms 在「尽快拿到许可」和「别把 Redis 打满」之间取折中。
_POLL_INTERVAL_SECONDS = 0.05
# 降级告警的限频窗口：同一进程同一类闸门 60 秒最多一条。
_DEGRADE_LOG_INTERVAL_SECONDS = 60.0
_CONNECT_TIMEOUT_SECONDS = 0.3
_READ_TIMEOUT_SECONDS = 2.0

_client: Optional[Any] = None
_client_lock = threading.Lock()

# 获取许可必须一次往返完成：清理过期成员、判断容量、写入、设租约。
# 拆成多次调用时，多个进程可能同时读到「还有空位」然后各自写入，实际并发会
# 超过配置上限——那正是闸门要防的场景。
_ACQUIRE_SCRIPT = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[2])
local count = redis.call('ZCARD', KEYS[1])
if count < tonumber(ARGV[3]) then
  redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
  redis.call('EXPIRE', KEYS[1], ARGV[5])
  return 1
end
return 0
"""

# 每类闸门上一次降级告警的时间。key 为 kind，避免 LLM/Embedding 互相刷屏。
_last_degrade_log: dict[str, float] = {}
_degrade_log_lock = threading.Lock()


def _get_client():
    """进程级 Redis 客户端。连接失败抛异常，由 acquire 统一降级。"""
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        import redis

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=_READ_TIMEOUT_SECONDS,
            decode_responses=True,
        )
        client.ping()
        _client = client
        return client


def reset_for_tests() -> None:
    """测试钩子：关闭并清空缓存的客户端。"""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001 — 关闭失败不影响测试
                pass
            _client = None
    with _degrade_log_lock:
        _last_degrade_log.clear()


def _limit_for(kind: str, override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    if kind == "llm":
        return settings.model_gate_llm_max_concurrency
    if kind == "embedding":
        return settings.model_gate_embedding_max_concurrency
    if kind == "rerank":
        return settings.model_gate_rerank_max_concurrency
    return 1


class ModelGate:
    """一次模型调用的许可。可作为上下文管理器使用。

    语义说明：``acquire()`` 返回 False 表示等待超时，但调用方**仍然继续执行**。
    这是刻意的降级——宁可让请求自己去撞供应商的 429，也不要让用户无限卡住。
    """

    def __init__(
        self,
        kind: str,
        provider: str,
        model: str,
        *,
        client_factory=None,
        max_concurrency: Optional[int] = None,
        wait_timeout_seconds: Optional[float] = None,
        lease_seconds: Optional[int] = None,
    ) -> None:
        self.kind = (kind or "").strip() or "unknown"
        self.provider = (provider or "").strip() or "unknown"
        self.model_name = (model or "").strip() or "unknown"
        self.key = MODEL_GATE_KEY.format(
            kind=self.kind,
            provider=self.provider,
            model=self.model_name,
        )
        self._client_factory = client_factory
        self._max_concurrency = max_concurrency
        self._wait_timeout_seconds = wait_timeout_seconds
        self._lease_seconds = lease_seconds
        self._member = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        self._held = False

    # ------------------------------------------------------------------ 公开接口

    def acquire(self, timeout_seconds: Optional[float] = None) -> bool:
        """尝试获取许可；超时返回 False，Redis 异常时降级返回 True。"""
        if not settings.model_gate_enabled:
            return False

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._wait_timeout_seconds
            if self._wait_timeout_seconds is not None
            else settings.model_gate_wait_timeout_seconds
        )
        lease = (
            self._lease_seconds
            if self._lease_seconds is not None
            else settings.model_gate_lease_seconds
        )
        limit = _limit_for(self.kind, self._max_concurrency)
        deadline = time.monotonic() + max(0.0, float(timeout))

        with timed_stage("model_gate.wait"):
            while True:
                try:
                    client = self._client()
                    now = time.time()
                    # 单次脚本：先回收持有者崩溃留下的过期成员，再判容量并写入。
                    acquired = client.eval(
                        _ACQUIRE_SCRIPT,
                        1,
                        self.key,
                        now,
                        now - lease,
                        limit,
                        self._member,
                        lease,
                    )
                    if int(acquired) == 1:
                        self._held = True
                        return True
                except Exception as exc:  # noqa: BLE001 — 统一降级
                    self._warn_degraded(exc)
                    return True

                if time.monotonic() >= deadline:
                    logger.warning(
                        "模型闸门等待超时，放行请求: kind=%s provider=%s model=%s limit=%s",
                        self.kind,
                        self.provider,
                        self.model_name,
                        limit,
                    )
                    return False
                time.sleep(_POLL_INTERVAL_SECONDS)

    def release(self) -> None:
        """释放许可。未持有许可或 Redis 异常时静默返回。"""
        if not self._held:
            return
        self._held = False
        try:
            self._client().zrem(self.key, self._member)
        except Exception as exc:  # noqa: BLE001 — 释放失败靠租约回收
            logger.debug("模型闸门释放失败（将由租约回收）: %s", type(exc).__name__)

    # ------------------------------------------------------------------ 上下文管理

    def __enter__(self) -> "ModelGate":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.release()
        return False

    # ------------------------------------------------------------------ 内部

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        return _get_client()

    def _warn_degraded(self, exc: Exception) -> None:
        """Redis 故障时限频告警。同一 kind 60 秒内只记一条。"""
        now = time.monotonic()
        with _degrade_log_lock:
            last = _last_degrade_log.get(self.kind, 0.0)
            if now - last < _DEGRADE_LOG_INTERVAL_SECONDS:
                return
            _last_degrade_log[self.kind] = now
        logger.warning(
            "模型闸门不可用，降级放行: kind=%s error=%s",
            self.kind,
            type(exc).__name__,
        )


def acquire_model_slot(kind: str, provider: str, model: str) -> ModelGate:
    """为一次模型调用创建闸门。作为上下文管理器使用。"""
    return ModelGate(kind, provider, model)

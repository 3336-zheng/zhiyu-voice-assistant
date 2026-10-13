"""选主锁：让「同一时刻只该有一个实例执行」的定时任务在多实例下不重复跑。

Phase 5 把应用改造成可多开之后，lifespan 里启动的定时任务会跟着变成 N 份。
清理过期会话本身是幂等的——重复执行最坏是白跑一次，不会删错东西。但白跑
N-1 份没有意义，而且并发跑全表查询会互相加锁、白占连接。这里用 Redis 的
SET NX EX 做一个最小的选主：每个实例每轮都来抢一次，抢到的执行，没抢到的跳过。

**锁的 TTL 要短于任务间隔**，否则持有者拿到锁之后锁不过期，它自己下一轮再来
抢也会失败，任务会一直停摆到 TTL 结束。取间隔的 0.9 倍：正常时锁在下一轮
开始前刚好过期，每轮都重新竞争；持有者崩溃时最多一个间隔之后有人接手。

Redis 不可用时返回 False（跳过这一轮），**不做「退回本地执行」的降级**——
那正是要避免的情况：每个实例都以为自己该跑。清理晚一轮没有影响。

注意这只解决「多实例重复执行」，不解决「API 重启会打断任务」：定时任务
放在 API 进程里，滚动重启期间那一轮就是没人跑。彻底的做法是搬进 arq 的
cron（arq 自己保证 cron 只被一个 worker 执行），当前规模下不值得。
"""

from __future__ import annotations

import logging
import os
import socket

from .config import settings

logger = logging.getLogger(__name__)

# 键名带 zhiyu 前缀：Redis 实例可能被其他项目共用
KEY_PREFIX = "zhiyu:leader:"

# 连接与命令超时。这个调用在定时任务的路径上，不是用户请求的关键路径，
# 但也不能让 Redis 卡住整个循环，所以同样设短。
_CONNECT_TIMEOUT_SECONDS = 0.5
_COMMAND_TIMEOUT_SECONDS = 1.0

_client = None


def _instance_id() -> str:
    """当前进程的身份，写进锁的值里，便于排查「这一轮是谁在跑」。"""
    return f"{socket.gethostname()}:{os.getpid()}"


async def _get_client():
    """拿可用的异步 Redis 客户端；拿不到返回 None（调用方跳过本轮）。"""
    global _client
    if _client is not None:
        return _client
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=_COMMAND_TIMEOUT_SECONDS,
            decode_responses=True,
        )
        await client.ping()
    except Exception as exc:
        logger.warning("选主锁连不上 Redis，本轮定时任务跳过: %s", exc)
        return None
    _client = client
    return client


async def _drop_client() -> None:
    """丢弃当前客户端，让下一轮重建。"""
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            await client.aclose()
        except Exception:
            pass


async def try_acquire(name: str, ttl_seconds: int) -> bool:
    """抢一轮执行权，抢到返回 True，没抢到或 Redis 不可用返回 False。

    多实例同时抢时只有一个会成功——这是 Redis SET NX 本身的保证。
    """
    client = await _get_client()
    if client is None:
        return False
    try:
        acquired = await client.set(
            KEY_PREFIX + name, _instance_id(), nx=True, ex=max(int(ttl_seconds), 1)
        )
    except Exception as exc:
        # 命令失败多半是连接断了，把客户端丢掉让下一轮重建，否则会一直用它重试。
        await _drop_client()
        logger.warning("选主锁读写失败，本轮定时任务跳过: %s", exc)
        return False
    return bool(acquired)


async def close() -> None:
    """关闭连接池。由 lifespan 的收尾流程调用。"""
    await _drop_client()


def reset_for_tests() -> None:
    """重置内部状态，仅供测试使用。"""
    global _client
    _client = None

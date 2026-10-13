"""全局索引版本号。

检索索引（Chroma 向量库 + BM25 词表）有一部分状态存在进程内存里。单实例时这没问题，
多实例时就出事：实例 A 写入后只更新了自己的内存，B 和 C 毫不知情，继续用陈旧索引
返回旧结果——不报错，只是搜不到新内容。

这里不搬运索引本身，只在 Redis 上放一个计数器：谁改了索引就 +1，各实例检索前比对
版本号，发现落后就自己重建。之所以敢这么做，是因为实测 973 个 chunk 全量重建只要
约 400 毫秒（其中 jieba 分词 216 毫秒、从 Chroma 拉取 175 毫秒、BM25Okapi 构造 8 毫秒）。
把词表序列化进 Redis 只能省掉分词那一段，却要额外背上序列化格式兼容、快照体积、
pickle 反序列化的安全面和写入侧的分布式锁——在这个数据量下是负收益。

等 chunk 数涨到一万左右（分词约 2.2 秒）时再考虑存快照。

版本号带来源前缀（"r:" 来自 Redis，"l:" 来自进程内），这样从 Redis 模式掉到降级模式
时版本串必然不同，会触发一次重建。宁可多重建一次，也不要漏掉一次。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)

# Redis 里的键名。加 zhiyu 前缀是因为 Redis 实例可能被其他项目共用。
REDIS_KEY = "zhiyu:index:version"

# 连接和读写的超时。必须设得很短：这个调用在检索的关键路径上，
# Redis 出问题时宁可立刻降级，也不能让用户的每次搜索都卡在默认超时上。
_TIMEOUT_SECONDS = 0.3

# 降级后多久再试一次 Redis。没有这个退避，Redis 挂掉期间每次检索都要
# 白白付一次连接超时的代价。
_RETRY_INTERVAL_SECONDS = 5.0


class _IndexVersion:
    """索引版本号，优先用 Redis，不可用时退回进程内计数器。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client = None
        self._local_version = 0
        # 下一次允许尝试连接 Redis 的时刻（time.monotonic 基准）
        self._retry_after = 0.0
        self._degraded_logged = False

    def _get_client(self):
        """拿到可用的 Redis 客户端，拿不到就返回 None（调用方负责降级）。"""
        with self._lock:
            if self._client is not None:
                return self._client
            if time.monotonic() < self._retry_after:
                return None
            try:
                import redis

                client = redis.Redis.from_url(
                    settings.redis_url,
                    socket_connect_timeout=_TIMEOUT_SECONDS,
                    socket_timeout=_TIMEOUT_SECONDS,
                    decode_responses=True,
                )
                client.ping()
            except Exception as exc:
                self._enter_degraded(exc)
                return None

            self._client = client
            if self._degraded_logged:
                logger.info("索引版本号已恢复使用 Redis")
                self._degraded_logged = False
            return client

    def _enter_degraded(self, exc: Exception) -> None:
        """转入降级模式，并在退避窗口内不再重试。"""
        with self._lock:
            self._client = None
            self._retry_after = time.monotonic() + _RETRY_INTERVAL_SECONDS
            if not self._degraded_logged:
                # 只在进入降级时警告一次，避免每次检索都刷日志
                logger.warning(
                    "索引版本号无法使用 Redis，降级为进程内计数器"
                    "（多实例下各实例的索引可能不一致）: %s",
                    exc,
                )
                self._degraded_logged = True

    def bump(self) -> None:
        """索引变更后调用，让所有实例知道该重建了。"""
        with self._lock:
            # 本地计数器始终递增：降级时它是唯一的版本来源，
            # 正常时它也保证了「Redis 恢复后本地版本不会倒退」。
            self._local_version += 1

        client = self._get_client()
        if client is None:
            return
        try:
            client.incr(REDIS_KEY)
        except Exception as exc:
            self._enter_degraded(exc)

    def current(self) -> str:
        """返回当前版本号，形如 "r:123"（Redis）或 "l:5"（进程内）。"""
        client = self._get_client()
        if client is not None:
            try:
                value = client.get(REDIS_KEY)
                # 键不存在说明还没有人写过索引，当作 0
                return f"r:{int(value) if value is not None else 0}"
            except Exception as exc:
                self._enter_degraded(exc)

        with self._lock:
            return f"l:{self._local_version}"

    def reset_for_tests(self) -> None:
        """清掉客户端与退避状态，供测试隔离使用。"""
        with self._lock:
            self._client = None
            self._local_version = 0
            self._retry_after = 0.0
            self._degraded_logged = False


_index_version = _IndexVersion()


def bump() -> None:
    """索引发生变更时调用。"""
    _index_version.bump()


def current() -> str:
    """读取当前索引版本号。"""
    return _index_version.current()


def reset_for_tests() -> None:
    """重置内部状态，仅供测试使用。"""
    _index_version.reset_for_tests()

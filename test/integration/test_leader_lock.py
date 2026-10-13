"""选主锁测试。

多实例下每个 API 实例都会跑 lifespan 里的定时清理，这个锁保证同一时刻只有
一个在跑。要验的核心只有一条：**同一时刻只有一个抢得到**——抢不到的必须
老老实实跳过，而不是「抢不到就自己跑一份」。
"""

import asyncio
import unittest
from unittest.mock import patch

from backend.app.core import leader_lock

TEST_REDIS_URL = "redis://127.0.0.1:6382/10"

# 一个必定连不上的地址，用来验 Redis 不可用时的降级行为
UNREACHABLE_REDIS_URL = "redis://127.0.0.1:1/0"


def _redis_available() -> bool:
    try:
        import redis

        client = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=0.5)
        client.ping()
        return True
    except Exception:  # noqa: BLE001 — 探测而已
        return False


REDIS_AVAILABLE = _redis_available()


@unittest.skipUnless(REDIS_AVAILABLE, "需要本机 Redis（docker compose up -d redis）")
class LeaderLockTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import redis.asyncio as aioredis

        leader_lock.reset_for_tests()
        self._url_patch = patch.object(leader_lock.settings, "redis_url", TEST_REDIS_URL)
        self._url_patch.start()
        self.redis = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
        await self.redis.flushdb()

    async def asyncTearDown(self):
        await leader_lock.close()
        await self.redis.flushdb()
        await self.redis.aclose()
        self._url_patch.stop()
        leader_lock.reset_for_tests()

    async def test_同一时刻只有一个实例抢得到(self):
        self.assertTrue(await leader_lock.try_acquire("session-cleanup", 60))
        # 第二个实例（这里是同进程的第二次调用，对 Redis 而言完全一样）
        self.assertFalse(await leader_lock.try_acquire("session-cleanup", 60))

    async def test_不同任务名互不影响(self):
        self.assertTrue(await leader_lock.try_acquire("session-cleanup", 60))
        self.assertTrue(await leader_lock.try_acquire("another-job", 60))

    async def test_锁过期后可以重新抢到(self):
        self.assertTrue(await leader_lock.try_acquire("session-cleanup", 1))
        self.assertFalse(await leader_lock.try_acquire("session-cleanup", 1))
        await asyncio.sleep(1.2)
        # 持有者崩溃时就是靠这条 TTL 让别的实例接手的
        self.assertTrue(await leader_lock.try_acquire("session-cleanup", 1))

    async def test_锁的值记录了持有者(self):
        await leader_lock.try_acquire("session-cleanup", 60)
        holder = await self.redis.get(leader_lock.KEY_PREFIX + "session-cleanup")
        self.assertEqual(holder, leader_lock._instance_id())

    async def test_Redis不可用时返回False而不抛异常(self):
        await leader_lock.close()
        with patch.object(
            leader_lock.settings, "redis_url", UNREACHABLE_REDIS_URL
        ):
            # 关键是不能抛：清理任务在一轮里抛出去会被外层 catch 掉刷日志，
            # 但定时任务的语义是「跳过这一轮」，不是「出错」。
            self.assertFalse(await leader_lock.try_acquire("session-cleanup", 60))

    async def test_没有客户端时关闭是空操作(self):
        leader_lock.reset_for_tests()
        await leader_lock.close()  # 不该抛


if __name__ == "__main__":
    unittest.main()

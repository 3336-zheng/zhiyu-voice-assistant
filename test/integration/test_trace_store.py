"""追踪快照的 Redis 存储测试。

这里验的是「跨实例可见」和「Redis 挂了不影响应用」两件事：
- 跨实例：进程内的表只有本实例看得到，多实例下查追踪等于抽奖
- 降级：追踪是诊断数据不是正确性数据，Redis 不可用时退回进程内即可，
  不能让请求失败——这一点与 Agent 运行状态的取向相反
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from backend.app.core.trace_store import INDEX_KEY, TRACE_KEY, RecentTraceStore

TEST_REDIS_URL = "redis://127.0.0.1:6382/8"


def _redis_available() -> bool:
    try:
        import redis

        client = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=0.5)
        client.ping()
        return True
    except Exception:  # noqa: BLE001 — 探测而已
        return False


REDIS_AVAILABLE = _redis_available()


def _snapshot(request_id: str, query: str = "问题") -> dict:
    return {
        "request_id": request_id,
        "trace_type": "agent",
        "method": "AGENT",
        "path": "agent.run",
        "status": "completed",
        "status_code": 200,
        "total_ms": 12.5,
        "query": query,
        "completed_at": time.time(),
    }


@unittest.skipUnless(REDIS_AVAILABLE, "需要本机 Redis（docker compose up -d redis）")
class RecentTraceStoreTestCase(unittest.TestCase):
    def setUp(self):
        import redis

        self.client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
        self.client.flushdb()

    def tearDown(self):
        self.client.flushdb()

    def _store(self, limit: int = 500) -> RecentTraceStore:
        return RecentTraceStore(limit=limit, client_factory=lambda: self.client)

    def test_写入后可以读回(self):
        store = self._store()
        store.put("r1", _snapshot("r1"), 100.0)
        self.assertEqual(store.get("r1")["query"], "问题")
        self.assertGreater(self.client.ttl(TRACE_KEY.format(request_id="r1")), 0)

    def test_另一个实例写得进去(self):
        # 这条是本次改造的目的：A 实例产生的追踪，B 实例查得到
        self._store().put("r1", _snapshot("r1"), 100.0)
        other_instance = self._store()
        self.assertEqual(other_instance.get("r1")["request_id"], "r1")

    def test_按完成时间倒序返回(self):
        store = self._store()
        for index, ts in enumerate([100.0, 300.0, 200.0]):
            store.put(f"r{index}", _snapshot(f"r{index}"), ts)
        listed = store.list_recent(10)
        self.assertEqual([item["request_id"] for item in listed], ["r1", "r2", "r0"])

    def test_超出条数上限时淘汰最旧的(self):
        store = self._store(limit=5)
        for index in range(12):
            store.put(f"r{index}", _snapshot(f"r{index}"), float(index))

        listed = store.list_recent(20)
        self.assertEqual(len(listed), 5)
        # 留下的应当是最新的 5 条
        self.assertEqual(
            sorted(item["request_id"] for item in listed),
            ["r10", "r11", "r7", "r8", "r9"],
        )
        self.assertLessEqual(self.client.zcard(INDEX_KEY), 5)

    def test_读不存在的追踪返回_None(self):
        self.assertIsNone(self._store().get("missing"))

    def test_空索引返回空列表(self):
        self.assertEqual(self._store().list_recent(10), [])


class RecentTraceStoreDegradeTestCase(unittest.TestCase):
    """Redis 不可用时的降级。这里不需要真的 Redis。"""

    def _broken_store(self, limit: int = 500) -> RecentTraceStore:
        def boom():
            raise OSError("连接被拒绝")

        return RecentTraceStore(limit=limit, client_factory=boom)

    def test_写失败不抛异常(self):
        # 追踪写入在请求路径上，它失败不该让用户的请求失败
        store = self._broken_store()
        store.put("r1", _snapshot("r1"), 100.0)

    def test_读失败返回空(self):
        store = self._broken_store()
        self.assertIsNone(store.get("r1"))
        self.assertEqual(store.list_recent(10), [])

    def test_失败后进入退避窗口不再重试连接(self):
        # 不设退避的话，Redis 挂掉期间每个请求都要白付一次连接超时
        calls = []

        def counting_boom():
            calls.append(1)
            raise OSError("连接被拒绝")

        store = RecentTraceStore(limit=500, client_factory=counting_boom)
        for index in range(5):
            store.put(f"r{index}", _snapshot(f"r{index}"), float(index))
        self.assertEqual(len(calls), 1, "退避窗口内不该反复尝试连接")

    def test_退避到期后会重新尝试(self):
        store = self._broken_store()
        store.put("r1", _snapshot("r1"), 100.0)

        client = MagicMock()
        with patch.object(store, "_retry_after", 0.0):
            store._client_factory = lambda: client
            store.put("r2", _snapshot("r2"), 200.0)
        client.pipeline.assert_called_once()


if __name__ == "__main__":
    unittest.main()

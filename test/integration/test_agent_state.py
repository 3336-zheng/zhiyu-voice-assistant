"""Agent 状态存储对着真实 Redis 的行为测试。

内存替身只保证接口一致，验不了真正依赖的那些语义：XREAD 游标怎么推进、
从 "0" 能不能追赶历史、MAXLEN 有没有把流封顶、TTL 是不是真的设上了。
这些一旦理解错，表现是「事件流卡住」或「内存慢慢涨上去」，很难在别处发现。

Redis 不可用时整体跳过，不让单元测试因为缺一个外部依赖而变红。
"""

import json
import time
import unittest

from backend.app.services.runtime.agent_state import (
    CANCEL_KEY,
    EVENTS_KEY,
    RUN_KEY,
    SESSION_KEY,
    AgentStateStore,
)

TEST_REDIS_URL = "redis://127.0.0.1:6382/11"


def _redis_available() -> bool:
    try:
        import redis

        client = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=0.5)
        client.ping()
        return True
    except Exception:  # noqa: BLE001 — 探测而已，失败就跳过
        return False


REDIS_AVAILABLE = _redis_available()


@unittest.skipUnless(REDIS_AVAILABLE, "需要本机 Redis（docker compose up -d redis）")
class AgentStateStoreTestCase(unittest.TestCase):
    def setUp(self):
        import redis

        self.client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
        self.client.flushdb()
        self.store = AgentStateStore(client_factory=lambda: self.client)

    def tearDown(self):
        self.client.flushdb()
        self.store.close()

    def _event(self, run_id: str, sequence: int, event_type: str = "token") -> dict:
        return {
            "type": event_type,
            "run_id": run_id,
            "session_id": "s1",
            "sequence": sequence,
            "created_at": "2026-10-13T00:00:00+00:00",
            "data": {"content": f"第{sequence}段"},
        }

    # ------------------------------------------------------------ 运行注册表

    def test_运行注册表读写往返(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 123.0)
        state = self.store.get_run("r1")
        self.assertEqual(state["session_id"], "s1")
        self.assertEqual(state["query"], "问题")
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["sequence"], 0)

        self.store.save_terminal(
            "r1", "completed", None, {"response": "回答"}, 456.0
        )
        state = self.store.get_run("r1")
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["response"], {"response": "回答"})
        self.assertEqual(state["finished_at"], 456.0)
        self.assertGreater(self.client.ttl(RUN_KEY.format(run_id="r1")), 0)

    def test_不存在的运行返回_None(self):
        self.assertIsNone(self.store.get_run("missing"))

    # ------------------------------------------------------------ 事件流

    def test_事件按序写入并按序读回(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        for seq in range(1, 6):
            self.store.append_event("r1", self._event("r1", seq))

        events, cursor = self.store.read_events("r1", "0", 0, block_ms=100)
        self.assertEqual([e["sequence"] for e in events], [1, 2, 3, 4, 5])

        # 用返回的游标继续读：没有新事件时不应重复拿到旧的
        again, _ = self.store.read_events("r1", cursor, 5, block_ms=100)
        self.assertEqual(again, [])

    def test_游标之后的增量事件(self):
        # 断点续传的核心：客户端给一个序号，只需要它之后的事件
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        for seq in range(1, 4):
            self.store.append_event("r1", self._event("r1", seq))

        events, _ = self.store.read_events("r1", "0", after_sequence=2, block_ms=100)
        self.assertEqual([e["sequence"] for e in events], [3])

    def test_新连接能追赶历史再阻塞等待(self):
        # 从 "0" 开始读：已有的事件应当立刻返回，而不是等到下一次写入
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        self.store.append_event("r1", self._event("r1", 1))

        started = time.time()
        events, _ = self.store.read_events("r1", "0", 0, block_ms=1000)
        elapsed = time.time() - started
        self.assertEqual(len(events), 1)
        self.assertLess(elapsed, 0.5, "已有事件时不该阻塞满整个窗口")

    def test_无新事件时阻塞到超时(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        self.store.append_event("r1", self._event("r1", 1))
        _, cursor = self.store.read_events("r1", "0", 0, block_ms=100)

        started = time.time()
        events, _ = self.store.read_events("r1", cursor, 1, block_ms=300)
        elapsed = time.time() - started
        self.assertEqual(events, [])
        self.assertGreaterEqual(elapsed, 0.25, "没有新事件时应当阻塞等待")

    def test_事件流长度有上界(self):
        # 一次运行可能吐出几千个 token 事件。不设上限的话 Redis 内存会一直涨。
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        for seq in range(1, 3001):
            self.store.append_event("r1", self._event("r1", seq))

        length = self.client.xlen(EVENTS_KEY.format(run_id="r1"))
        # MAXLEN 用近似截断，允许略多于设定值，但必须有界
        self.assertLess(length, 2300, f"事件流没有被封顶，长度 {length}")

    def test_流被清掉后能被探测到(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        self.store.append_event("r1", self._event("r1", 1))
        self.assertTrue(self.store.has_stream("r1"))

        self.store.delete_events("r1")
        self.assertFalse(self.store.has_stream("r1"))

    def test_空负载不会被当成事件(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        self.client.xadd(EVENTS_KEY.format(run_id="r1"), {"payload": ""})
        events, _ = self.store.read_events("r1", "0", 0, block_ms=100)
        self.assertEqual(events, [])

    # ------------------------------------------------------------ 会话锁

    def test_会话锁互斥与释放(self):
        self.assertTrue(self.store.acquire_session("s1", "r1"))
        self.assertFalse(self.store.acquire_session("s1", "r2"))
        self.assertEqual(self.store.get_session_run("s1"), "r1")

        self.store.release_session("s1", "r1")
        self.assertIsNone(self.store.get_session_run("s1"))
        self.assertTrue(self.store.acquire_session("s1", "r2"))

    def test_释放只对自己持有的锁生效(self):
        # 别人的锁不能被误删，否则会出现两个运行同时跑
        self.assertTrue(self.store.acquire_session("s1", "r1"))
        self.store.release_session("s1", "r2")
        self.assertEqual(self.store.get_session_run("s1"), "r1")

    def test_续期只对自己持有的锁生效(self):
        self.assertTrue(self.store.acquire_session("s1", "r1"))
        self.assertTrue(self.store.renew_session("s1", "r1"))
        self.assertFalse(self.store.renew_session("s1", "r2"))

    def test_会话锁带过期时间(self):
        # 运行中途进程崩溃时，锁不能永久留在那里把会话锁死
        self.assertTrue(self.store.acquire_session("s1", "r1"))
        ttl = self.client.ttl(SESSION_KEY.format(session_id="s1"))
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, 60)

    def test_运行键带过期时间(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        self.assertGreater(self.client.ttl(RUN_KEY.format(run_id="r1")), 0)

    # ------------------------------------------------------------ 取消信号

    def test_取消请求可被跨实例看到(self):
        # 发起取消的请求可能落在任何一个实例上，而执行者只有一个。
        # 信号存在 Redis 上，两边看到的是同一个。
        self.assertFalse(self.store.is_cancel_requested("r1"))
        self.store.request_cancel("r1", 120)
        self.assertTrue(self.store.is_cancel_requested("r1"))

        other_instance = AgentStateStore(client_factory=lambda: self.client)
        self.assertTrue(other_instance.is_cancel_requested("r1"))

    def test_取消信号带过期时间(self):
        # 执行者崩溃后没人清它，不能留一个永不过期的键
        self.store.request_cancel("r1", 120)
        ttl = self.client.ttl(CANCEL_KEY.format(run_id="r1"))
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, 120)

    def test_取消信号可以被清掉(self):
        self.store.request_cancel("r1", 120)
        self.store.clear_cancel("r1")
        self.assertFalse(self.store.is_cancel_requested("r1"))

    def test_删除运行会一并清掉取消信号(self):
        self.store.create_run("r1", "s1", "问题", "pending", False, 0.0)
        self.store.request_cancel("r1", 120)
        self.store.delete_run("r1")
        self.assertFalse(self.store.is_cancel_requested("r1"))
        self.assertFalse(self.store.has_stream("r1"))

    # ------------------------------------------------------------ 隔离

    def test_不同运行的事件互不串台(self):
        for run_id in ("r1", "r2"):
            self.store.create_run(run_id, "s1", "问题", "pending", False, 0.0)
            self.store.append_event(run_id, self._event(run_id, 1))

        events, _ = self.store.read_events("r1", "0", 0, block_ms=100)
        self.assertEqual([e["run_id"] for e in events], ["r1"])


if __name__ == "__main__":
    unittest.main()

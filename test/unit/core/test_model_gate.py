"""分布式模型并发闸门的单元测试。"""

import unittest

from backend.app.core.config import settings


class ModelGateConfigTest(unittest.TestCase):
    def test_默认值与设计一致(self):
        self.assertTrue(settings.model_gate_enabled)
        self.assertEqual(settings.model_gate_llm_max_concurrency, 4)
        self.assertEqual(settings.model_gate_embedding_max_concurrency, 8)
        self.assertEqual(settings.model_gate_rerank_max_concurrency, 4)
        self.assertEqual(settings.model_gate_wait_timeout_seconds, 30.0)
        self.assertEqual(settings.model_gate_lease_seconds, 300)

import threading
import time
from unittest.mock import patch

from backend.app.core import model_gate
from backend.app.core.model_gate import ModelGate


class FakeRedis:
    """只实现闸门用到的 zset / expire / ping 子集。"""

    def __init__(self):
        self.zsets = {}
        self.expires = {}
        self.fail = False
        self._lock = threading.Lock()

    def ping(self):
        if self.fail:
            raise ConnectionError("redis down")
        return True

    def zremrangebyscore(self, key, low, high):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            members = self.zsets.get(key, {})
            removed = [m for m, s in members.items() if float(low) <= s <= float(high)]
            for m in removed:
                del members[m]
            return len(removed)

    def zcard(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return len(self.zsets.get(key, {}))

    def zadd(self, key, mapping):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self.zsets.setdefault(key, {}).update(mapping)
        return 1

    def zrem(self, key, member):
        if self.fail:
            raise ConnectionError("redis down")
        with self._lock:
            self.zsets.get(key, {}).pop(member, None)
        return 1

    def expire(self, key, seconds):
        if self.fail:
            raise ConnectionError("redis down")
        self.expires[key] = seconds
        return True


class ModelGateTest(unittest.TestCase):
    def setUp(self):
        model_gate.reset_for_tests()

    def tearDown(self):
        model_gate.reset_for_tests()

    def _gate(self, fake, limit=2, kind="llm"):
        return ModelGate(
            kind,
            "primary",
            "test-model",
            client_factory=lambda: fake,
            max_concurrency=limit,
            wait_timeout_seconds=1.0,
            lease_seconds=60,
        )

    def test_未达上限时可获取并释放(self):
        fake = FakeRedis()
        gate = self._gate(fake)
        self.assertTrue(gate.acquire())
        self.assertEqual(fake.zcard(gate.key), 1)
        gate.release()
        self.assertEqual(fake.zcard(gate.key), 0)

    def test_达到上限后在释放前拿不到许可(self):
        fake = FakeRedis()
        first = self._gate(fake, limit=1)
        second = self._gate(fake, limit=1)
        self.assertTrue(first.acquire())
        self.assertFalse(second.acquire(timeout_seconds=0.2))
        first.release()
        self.assertTrue(second.acquire(timeout_seconds=1.0))
        second.release()

    def test_租约过期后僵尸许可被清理(self):
        fake = FakeRedis()
        gate = self._gate(fake, limit=1)
        key = gate.key
        fake.zsets[key] = {"dead-holder": time.time() - 120}
        self.assertTrue(gate.acquire(timeout_seconds=1.0))
        self.assertNotIn("dead-holder", fake.zsets[key])
        gate.release()

    def test_redis异常时降级放行(self):
        fake = FakeRedis()
        fake.fail = True
        gate = self._gate(fake, limit=1)
        self.assertTrue(gate.acquire())
        gate.release()  # 释放失败也不能抛

    def test_关闭开关时不访问redis(self):
        fake = FakeRedis()
        gate = self._gate(fake)
        with patch.object(model_gate.settings, "model_gate_enabled", False):
            gate.acquire()
            self.assertEqual(fake.zsets, {})

    def test_上下文管理器释放许可(self):
        fake = FakeRedis()
        gate = self._gate(fake, limit=1)
        with gate:
            self.assertEqual(fake.zcard(gate.key), 1)
        self.assertEqual(fake.zcard(gate.key), 0)

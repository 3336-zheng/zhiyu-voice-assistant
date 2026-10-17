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

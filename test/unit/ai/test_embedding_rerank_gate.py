"""Embedding / Rerank 接入并发闸门的测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.app.services.ai.embedding_service import OpenAICompatibleEmbeddingBackend
from backend.app.services.ai.reranker_service import RerankCompatibleBackend


class EmbeddingGateTest(unittest.TestCase):
    def test_每个批次获取一次许可(self):
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])]
        )
        backend = OpenAICompatibleEmbeddingBackend(
            api_key="k",
            api_url="https://example.com/v1",
            model="m",
            batch_size=1,
            client=client,
        )
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.embedding_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            backend.encode_batch(["a", "b"])
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.args[0], "embedding")


class RerankGateTest(unittest.TestCase):
    def test_请求前获取许可(self):
        session = MagicMock()
        session.post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {"results": [{"index": 0, "relevance_score": 0.9}]},
        )
        backend = RerankCompatibleBackend(
            api_key="k",
            api_url="https://example.com/rerank",
            model="m",
            session=session,
        )
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.reranker_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            backend.rerank("q", ["a"], 1)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[0], "rerank")

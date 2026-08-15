"""在线 Embedding/Rerank Provider 与向量集合隔离测试。"""

import unittest
from types import SimpleNamespace

from backend.app.services.retrieval.chroma_service import resolve_embedding_collection_name
from backend.app.services.ai.embedding_service import (
    EmbeddingProviderError,
    EmbeddingService,
    OpenAICompatibleEmbeddingBackend,
)
from backend.app.services.ai.reranker_service import (
    RerankCompatibleBackend,
    RerankerProviderError,
    RerankerService,
)


class FakeEmbeddingsEndpoint:
    def __init__(self, dimension=2, error=None, vector_value=None):
        self.dimension = dimension
        self.error = error
        self.vector_value = vector_value
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        items = [
            SimpleNamespace(
                index=index,
                embedding=[
                    float(len(text)) if self.vector_value is None else self.vector_value
                ] * self.dimension,
            )
            for index, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(items)))


class FakeOpenAIClient:
    def __init__(self, endpoint):
        self.embeddings = endpoint


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class RemoteEmbeddingProviderTestCase(unittest.TestCase):
    def test_batching_preserves_input_order_and_empty_positions(self):
        endpoint = FakeEmbeddingsEndpoint(dimension=2)
        backend = OpenAICompatibleEmbeddingBackend(
            api_key="test-key",
            api_url="https://gateway.example.com/v1",
            model="text-embedding-test",
            dimensions=2,
            batch_size=2,
            client=FakeOpenAIClient(endpoint),
        )

        vectors = EmbeddingService(backend).encode_documents(
            ["alpha", "", "b", "cc"]
        )

        self.assertEqual(vectors, [[5.0, 5.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        self.assertEqual(len(endpoint.calls), 2)
        self.assertTrue(all(call["dimensions"] == 2 for call in endpoint.calls))
        self.assertEqual(endpoint.calls[0]["input"], ["alpha", "b"])
        self.assertEqual(endpoint.calls[1]["input"], ["cc"])

    def test_dimension_mismatch_is_rejected(self):
        backend = OpenAICompatibleEmbeddingBackend(
            api_key="test-key",
            api_url="https://gateway.example.com/v1",
            model="text-embedding-test",
            dimensions=3,
            client=FakeOpenAIClient(FakeEmbeddingsEndpoint(dimension=2)),
        )

        with self.assertRaisesRegex(EmbeddingProviderError, "维度 2"):
            backend.encode_batch(["test"])

    def test_credentials_in_url_are_rejected(self):
        with self.assertRaisesRegex(EmbeddingProviderError, "不能包含用户名或密码"):
            OpenAICompatibleEmbeddingBackend(
                api_key="",
                api_url="https://user:secret@gateway.example.com/v1",
                model="text-embedding-test",
                client=FakeOpenAIClient(FakeEmbeddingsEndpoint()),
            )

    def test_query_credentials_in_url_are_rejected(self):
        with self.assertRaisesRegex(EmbeddingProviderError, "查询参数或片段"):
            OpenAICompatibleEmbeddingBackend(
                api_key="",
                api_url="https://gateway.example.com/v1?api_key=secret",
                model="text-embedding-test",
                client=FakeOpenAIClient(FakeEmbeddingsEndpoint()),
            )

    def test_remote_exception_does_not_expose_provider_message(self):
        error = RuntimeError("provider leaked secret=abc")
        error.status_code = 503
        backend = OpenAICompatibleEmbeddingBackend(
            api_key="test-key",
            api_url="https://gateway.example.com/v1",
            model="text-embedding-test",
            client=FakeOpenAIClient(FakeEmbeddingsEndpoint(error=error)),
        )

        with self.assertRaises(EmbeddingProviderError) as context:
            backend.encode_batch(["test"])
        self.assertEqual(str(context.exception), "在线 Embedding 调用失败（HTTP 503）")
        self.assertNotIn("secret", str(context.exception))
        self.assertTrue(context.exception.__suppress_context__)

    def test_non_finite_embedding_value_is_rejected(self):
        backend = OpenAICompatibleEmbeddingBackend(
            api_key="",
            api_url="https://gateway.example.com/v1",
            model="text-embedding-test",
            client=FakeOpenAIClient(
                FakeEmbeddingsEndpoint(dimension=2, vector_value=float("nan"))
            ),
        )

        with self.assertRaisesRegex(EmbeddingProviderError, "非有限数值"):
            backend.encode_batch(["test"])


class RemoteRerankerProviderTestCase(unittest.TestCase):
    def test_request_and_response_are_normalized(self):
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "results": [
                        {"index": 0, "relevance_score": 0.4},
                        {"index": 2, "score": 0.91},
                        {"index": 1, "relevance_score": 0.72},
                    ]
                },
            )
        )
        backend = RerankCompatibleBackend(
            api_key="rerank-key",
            api_url="https://ai-gateway.vercel.sh/v1/rerank",
            model="cohere/rerank-v3.5",
            timeout_seconds=12,
            session=session,
        )

        results = RerankerService(backend).rerank(
            "可信 RAG",
            ["第一段", "第二段", "第三段"],
            top_k=2,
        )

        self.assertEqual(results, [{"index": 2, "score": 0.91}, {"index": 1, "score": 0.72}])
        url, kwargs = session.calls[0]
        self.assertEqual(url, "https://ai-gateway.vercel.sh/v1/rerank")
        self.assertEqual(kwargs["json"]["model"], "cohere/rerank-v3.5")
        self.assertEqual(kwargs["json"]["top_n"], 2)
        self.assertFalse(kwargs["json"]["return_documents"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer rerank-key")
        self.assertEqual(kwargs["timeout"], 12)
        self.assertEqual(backend.provider_name, "rerank_compatible")

    def test_http_error_does_not_expose_response_body(self):
        backend = RerankCompatibleBackend(
            api_key="rerank-key",
            api_url="https://gateway.example.com/v1/rerank",
            model="rerank-test",
            session=FakeSession(FakeResponse(401, {"secret": "provider detail"})),
        )

        with self.assertRaises(RerankerProviderError) as context:
            backend.rerank("query", ["document"], 1)
        self.assertEqual(str(context.exception), "在线 Rerank 调用失败（HTTP 401）")
        self.assertNotIn("provider detail", str(context.exception))

    def test_invalid_result_index_is_rejected(self):
        backend = RerankCompatibleBackend(
            api_key="",
            api_url="https://gateway.example.com/v1/rerank",
            model="rerank-test",
            session=FakeSession(
                FakeResponse(200, {"results": [{"index": 4, "relevance_score": 0.9}]})
            ),
        )

        with self.assertRaisesRegex(RerankerProviderError, "越界"):
            backend.rerank("query", ["document"], 1)

    def test_non_finite_score_is_rejected(self):
        backend = RerankCompatibleBackend(
            api_key="",
            api_url="https://gateway.example.com/v1/rerank",
            model="rerank-test",
            session=FakeSession(
                FakeResponse(
                    200,
                    {"results": [{"index": 0, "relevance_score": float("inf")}]},
                )
            ),
        )

        with self.assertRaisesRegex(RerankerProviderError, "无效分数"):
            backend.rerank("query", ["document"], 1)


class EmbeddingCollectionIsolationTestCase(unittest.TestCase):
    def test_local_provider_keeps_existing_collection_name(self):
        self.assertEqual(
            resolve_embedding_collection_name("notes", provider="local"),
            "notes",
        )

    def test_remote_profile_is_stable_and_isolated(self):
        first = resolve_embedding_collection_name(
            "notes",
            provider="openai_compatible",
            api_url="https://gateway.example.com/v1",
            model="embedding-a",
            dimensions=1024,
        )
        repeated = resolve_embedding_collection_name(
            "notes",
            provider="openai_compatible",
            api_url="https://gateway.example.com/v1/",
            model="embedding-a",
            dimensions=1024,
        )
        changed = resolve_embedding_collection_name(
            "notes",
            provider="openai_compatible",
            api_url="https://gateway.example.com/v1",
            model="embedding-b",
            dimensions=1024,
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertRegex(first, r"^notes-[0-9a-f]{12}$")


if __name__ == "__main__":
    unittest.main()

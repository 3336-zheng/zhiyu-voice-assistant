"""RAG v2 父块折叠、统一融合与 Token 预算测试。"""

import unittest

from backend.app.services.hybrid_retrieval_service import HybridRetrievalService


class FakeEmbedding:
    @staticmethod
    def encode(query):
        return [float(len(query))]


class FakeBM25:
    def __init__(self):
        self.corpus = {}

    @staticmethod
    def search(query, top_k=20):
        suffix = 0 if "原始" in query else 1
        return [
            (f"page:p1:revision:1:chunk:0:child:{suffix}", 1.0),
            ("page:p2:revision:1:chunk:0", 0.8),
        ][:top_k]


class FakeChromaCollection:
    def __init__(self):
        self.documents = {
            "page:p1:revision:1:chunk:0": "父块一的完整正文" * 12,
            "page:p2:revision:1:chunk:0": "父块二正文",
        }

    def get(self, ids, include):
        existing = [doc_id for doc_id in ids if doc_id in self.documents]
        return {
            "ids": existing,
            "documents": [self.documents[doc_id] for doc_id in existing],
            "metadatas": [
                {
                    "source_type": "wiki_page",
                    "page_id": "p1" if ":p1:" in doc_id else "p2",
                    "page_title": "页面一" if ":p1:" in doc_id else "页面二",
                    "parent_chunk_id": doc_id,
                    "chunk_level": "parent",
                }
                for doc_id in existing
            ],
        }


class FakeChroma:
    def __init__(self):
        self.collection = FakeChromaCollection()

    @staticmethod
    def search(embedding, top_k=20):
        return [
            ("page:p1:revision:1:chunk:0:child:2", 0.95),
            ("page:p2:revision:1:chunk:0", 0.7),
        ][:top_k]


class FakeRRF:
    def __init__(self):
        self.calls = 0
        self.last_lists = None

    def fuse_multi(self, results_list, weights=None, top_k=None):
        self.calls += 1
        self.last_lists = results_list
        ordered = []
        seen = set()
        for results in results_list:
            for doc_id, _ in results:
                if doc_id not in seen:
                    seen.add(doc_id)
                    ordered.append((doc_id, 1.0 / len(ordered + [doc_id])))
        return ordered[:top_k]


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents, top_k=8):
        self.calls += 1
        return [
            {"index": index, "score": 0.9 - index * 0.1}
            for index in range(min(top_k, len(documents)))
        ]


class RagV2TestCase(unittest.TestCase):
    def setUp(self):
        self.service = HybridRetrievalService(
            embedding_service=FakeEmbedding(),
            bm25_service=FakeBM25(),
            chroma_service=FakeChroma(),
            rrf_service=FakeRRF(),
            reranker_service=FakeReranker(),
        )

    def test_multi_query_fuses_and_reranks_once_after_parent_collapse(self):
        outcome = self.service.search_multi(
            ["原始问题", "改写问题"],
            original_query="原始问题",
            top_k=5,
            token_budget=100,
        )
        self.assertEqual(self.service.rrf_service.calls, 1)
        self.assertEqual(self.service.reranker_service.calls, 1)
        flattened_ids = [doc_id for items in self.service.rrf_service.last_lists for doc_id, _ in items]
        self.assertNotIn("page:p1:revision:1:chunk:0:child:0", flattened_ids)
        self.assertIn("page:p1:revision:1:chunk:0", flattened_ids)
        self.assertEqual(outcome["stats"]["query_count"], 2)
        self.assertLessEqual(outcome["stats"]["context_tokens"], 100)
        self.assertTrue(all(":child:" not in item["chunk_id"] for item in outcome["results"]))

    def test_token_budget_truncates_last_result(self):
        selected, used = self.service.apply_token_budget(
            [{"content": "知识" * 100, "snippet": ""}],
            token_budget=30,
        )
        self.assertEqual(used, 30)
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]["context_truncated"])


if __name__ == "__main__":
    unittest.main()

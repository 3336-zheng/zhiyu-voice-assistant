"""RAG v2 父块折叠、统一融合与 Token 预算测试。"""

import unittest

from backend.app.services.retrieval.hybrid_retrieval_service import HybridRetrievalService
from backend.app.agent.fast_path import is_fast_path_query


class FakeEmbedding:
    @staticmethod
    def encode(query):
        return [float(len(query))]


class FakeBatchEmbedding:
    def __init__(self):
        self.calls = 0

    def encode_queries(self, queries):
        self.calls += 1
        return [[float(len(query))] for query in queries]


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

    def test_filter_reranked_results_keeps_close_scores_and_drops_outliers(self):
        filtered = self.service.filter_reranked_results(
            [
                {"index": 0, "score": 0.90},
                {"index": 1, "score": 0.81},
                {"index": 2, "score": 0.72},
                {"index": 3, "score": 0.41},
                {"index": 4, "score": 0.12},
            ],
            final_top_k=5,
            min_score=0.35,
            score_margin=0.20,
        )
        self.assertEqual([item["index"] for item in filtered], [0, 1, 2])

    def test_filter_reranked_results_keeps_best_low_score_for_evidence_gate(self):
        filtered = self.service.filter_reranked_results(
            [{"index": 0, "score": 0.24}, {"index": 1, "score": 0.10}],
            final_top_k=5,
            min_score=0.35,
            score_margin=0.20,
        )
        self.assertEqual([item["index"] for item in filtered], [0])

    def test_fast_path_only_accepts_simple_read_queries(self):
        self.assertTrue(is_fast_path_query("Markdown 作为知识主数据有哪些优势？"))
        self.assertTrue(is_fast_path_query("SQLite 在智语中负责保存什么？"))
        self.assertFalse(is_fast_path_query("请把这段内容保存到知识库"))
        self.assertFalse(is_fast_path_query("比较父子分块和 RRF，并说明什么时候触发重检？"))
        self.assertFalse(is_fast_path_query("请把这段内容总结成复习卡片"))
        self.assertFalse(
            is_fast_path_query(
                "这个结论对吗？",
                [{"role": "user", "content": "上文提到父子分块"}],
            )
        )

    def test_multi_query_uses_one_batch_embedding_and_short_candidate_list(self):
        embedding = FakeBatchEmbedding()
        service = HybridRetrievalService(
            embedding_service=embedding,
            bm25_service=FakeBM25(),
            chroma_service=FakeChroma(),
            rrf_service=FakeRRF(),
            reranker_service=FakeReranker(),
        )

        outcome = service.search_multi(
            ["原始问题", "改写问题"],
            original_query="原始问题",
            top_k=5,
            token_budget=100,
        )
        cached = service.search_multi(
            ["原始问题", "改写问题"],
            original_query="原始问题",
            top_k=5,
            token_budget=100,
        )

        self.assertEqual(embedding.calls, 1)
        self.assertFalse(outcome["stats"]["cache_hit"])
        self.assertTrue(cached["stats"]["cache_hit"])
        self.assertLessEqual(outcome["stats"]["reranked_candidates"], 12)


if __name__ == "__main__":
    unittest.main()

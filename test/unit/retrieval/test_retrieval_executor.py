"""检索共享线程池测试。

改成共享池之后新增的风险是「池内任务又向同一个池提交并等待结果」——那会在
worker 被占满时死锁，而且只在并发够高时才出现，平时跑不出来。这里用一个
被刻意调小的池（max_workers=2）配合远超它的并发量，把这种结构性问题逼出来。
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend.app.services.retrieval.hybrid_retrieval_service import (
    HybridRetrievalService,
)


class _FakeBM25:
    def search(self, query, top_k=20):
        time.sleep(0.01)  # 让召回真的重叠，而不是瞬间返回
        return [(f"doc_{query}", 1.0)]


class _FakeChroma:
    def search(self, embedding, top_k=20):
        time.sleep(0.01)
        return [(f"vec_{embedding[0]}", 0.9)]


class _FakeEmbedding:
    def encode_queries(self, queries):
        return [[float(index)] for index, _ in enumerate(queries)]


def _build_service(max_workers: int) -> HybridRetrievalService:
    with patch(
        "backend.app.services.retrieval.hybrid_retrieval_service.settings.retrieval_max_workers",
        max_workers,
    ):
        return HybridRetrievalService(
            chroma_service=_FakeChroma(),
            bm25_service=_FakeBM25(),
            rrf_service=object(),
            embedding_service=_FakeEmbedding(),
            reranker_service=object(),
        )


class RetrievalExecutorTest(unittest.TestCase):
    def test_线程池按配置的额度创建(self):
        service = _build_service(max_workers=5)
        self.assertEqual(service._executor._max_workers, 5)

    def test_多次检索复用同一个线程池(self):
        # 每次检索新建再销毁线程池正是这次要消除的开销。
        service = _build_service(max_workers=4)
        pool_before = service._executor
        service._recall_many(["查询一"], bm25_top_k=5, embedding_top_k=5)
        service._recall_many(["查询二"], bm25_top_k=5, embedding_top_k=5)
        self.assertIs(service._executor, pool_before)

    def test_并发查询数远超池容量时不会死锁(self):
        # 池只有 2 个 worker，却要同时跑 6 个查询 × 2 路召回 = 12 个任务。
        # 如果召回任务内部再往同一个池提交并等待，这里会挂死而不是变慢。
        service = _build_service(max_workers=2)
        queries = [f"查询{i}" for i in range(6)]

        done = threading.Event()
        outcomes = []

        def run():
            outcomes.append(
                service._recall_many(queries, bm25_top_k=5, embedding_top_k=5)
            )
            done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.assertTrue(done.wait(timeout=15), "共享线程池发生死锁或严重排队")
        self.assertEqual(len(outcomes[0]), len(queries))
        for outcome in outcomes[0]:
            self.assertTrue(outcome["bm25"])
            self.assertTrue(outcome["embedding"])

    def test_多个线程同时进入检索时结果不串位(self):
        # Agent executor 最多 4 个步骤并发进入检索，它们现在共用一个池。
        # 每个查询的 bm25/embedding 结果必须回到它自己那一格。
        service = _build_service(max_workers=8)
        errors = []
        barrier = threading.Barrier(4)

        def worker(index: int):
            try:
                barrier.wait()
                queries = [f"线程{index}查询{i}" for i in range(3)]
                outcomes = service._recall_many(
                    queries, bm25_top_k=5, embedding_top_k=5
                )
                for position, query in enumerate(queries):
                    expected = f"doc_{query}"
                    actual = outcomes[position]["bm25"][0][0]
                    if actual != expected:
                        errors.append(f"下标 {position} 期望 {expected}，实得 {actual}")
            except Exception as exc:  # noqa: BLE001 — 线程内异常要带回主线程
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        self.assertEqual(errors, [], f"并发检索结果错位: {errors}")


if __name__ == "__main__":
    unittest.main()

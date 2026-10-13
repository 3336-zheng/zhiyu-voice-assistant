"""BM25 服务的并发安全与索引刷新测试。

这里盯的是两类不会报错、只会安静出错的问题：
1. doc_id_list 和 tokenized_corpus 靠下标一一对应，并发增删中途错位后，
   BM25 算出「第 5 篇最相关」，取回来的却是别的文档。
2. 多实例下别人改了索引，本实例毫不知情，继续用陈旧索引返回旧结果。
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend.app.services.retrieval.bm25_service import BM25Service


class _SlowDeleteList(list):
    """删完一个元素后主动让出 GIL。

    remove_document 里 doc_id_list 和 tokenized_corpus 是分两句删的，出错窗口
    就夹在这两句之间。裸跑并发测试撞不到这个窗口——它只有几条字节码宽，CPython
    的线程切换基本不会正好落进去，去掉锁也照样能通过，那样的测试是没有意义的。
    这里把窗口人为撑开到毫秒级，让「少了锁就必然错位」变成确定性结果。
    """

    def __delitem__(self, index):
        super().__delitem__(index)
        time.sleep(0.005)


class BM25ConcurrencyTest(unittest.TestCase):
    """并发增删改之后，索引内部的下标对应关系必须仍然成立。"""

    def _assert_invariants(self, service: BM25Service):
        self.assertEqual(
            len(service.doc_id_list),
            len(service.tokenized_corpus),
            "doc_id_list 与 tokenized_corpus 长度不一致，下标已经错位",
        )
        self.assertEqual(
            len(service.doc_id_list),
            len(service.corpus),
            "doc_id_list 与 corpus 数量不一致",
        )
        self.assertEqual(
            len(set(service.doc_id_list)),
            len(service.doc_id_list),
            "doc_id_list 出现重复条目",
        )
        # 每个位置上的分词结果必须是该位置 doc_id 对应正文分出来的
        for index, doc_id in enumerate(service.doc_id_list):
            self.assertIn(doc_id, service.corpus)
            expected = service._tokenize(f"标题{doc_id} {service.corpus[doc_id]}")
            self.assertEqual(
                service.tokenized_corpus[index],
                expected,
                f"下标 {index} 的分词与 doc_id={doc_id} 的正文对不上",
            )

    def test_并发增删改之后下标不错位(self):
        service = BM25Service()
        doc_ids = [f"doc_{i}" for i in range(40)]
        for doc_id in doc_ids:
            service.add_document(doc_id, f"初始内容 关于机器学习 {doc_id}", f"标题{doc_id}")

        errors = []
        barrier = threading.Barrier(12)

        def worker(worker_index: int):
            try:
                barrier.wait()
                for round_index in range(20):
                    doc_id = doc_ids[(worker_index * 7 + round_index * 3) % len(doc_ids)]
                    action = (worker_index + round_index) % 3
                    if action == 0:
                        service.add_document(
                            doc_id, f"新增内容 深度学习 {round_index}", f"标题{doc_id}"
                        )
                    elif action == 1:
                        service.update_document(
                            doc_id, f"更新内容 向量检索 {round_index}", f"标题{doc_id}"
                        )
                    else:
                        service.remove_document(doc_id)
            except Exception as exc:  # noqa: BLE001 — 线程里的异常要带回主线程断言
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [], f"并发操作抛出异常: {errors}")
        self._assert_invariants(service)

    def test_删除的两次下标改动必须是原子的(self):
        # 这是本文件里唯一一个「去掉锁就必定失败」的用例，其余用例撞不到那个窗口。
        service = BM25Service()
        doc_ids = [f"doc_{i}" for i in range(12)]
        for doc_id in doc_ids:
            service.add_document(doc_id, f"机器学习 内容 {doc_id}", f"标题{doc_id}")
        service.doc_id_list = _SlowDeleteList(service.doc_id_list)

        errors = []
        barrier = threading.Barrier(4)

        def remover(doc_id: str):
            try:
                barrier.wait()
                service.remove_document(doc_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=remover, args=(doc_ids[i],)) for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [], f"并发删除抛出异常: {errors}")
        self._assert_invariants(service)

    def test_边检索边写入不会取到越界下标(self):
        # search 如果在锁外读 self.doc_id_list，写入线程的删除会让它比
        # 算分用的矩阵短，argsort 出来的下标就会越界或指向错误文档。
        service = BM25Service()
        for i in range(60):
            service.add_document(f"doc_{i}", f"机器学习 向量检索 内容 {i}", f"标题doc_{i}")

        errors = []
        stop = threading.Event()

        def searcher():
            try:
                while not stop.is_set():
                    for doc_id, score in service.search("机器学习 向量检索", top_k=10):
                        self.assertIsInstance(doc_id, str)
                        self.assertGreater(score, 0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def writer():
            try:
                for i in range(60):
                    service.remove_document(f"doc_{i}")
                    service.add_document(f"doc_{i}", f"机器学习 新内容 {i}", f"标题doc_{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        search_threads = [threading.Thread(target=searcher) for _ in range(3)]
        writer_thread = threading.Thread(target=writer)
        for thread in search_threads:
            thread.start()
        writer_thread.start()
        writer_thread.join()
        stop.set()
        for thread in search_threads:
            thread.join()

        self.assertEqual(errors, [], f"并发检索/写入抛出异常: {errors}")
        self._assert_invariants(service)


class BM25FreshnessTest(unittest.TestCase):
    """ensure_fresh 只在版本落后时重建，且并发下只重建一次。"""

    def _service_with_version(self, version: str):
        service = BM25Service()
        service.add_document("doc_1", "机器学习 内容", "标题")
        return service

    def test_版本落后时触发重建(self):
        service = BM25Service()
        calls = []

        def rebuild():
            calls.append(True)
            service.mark_synced("r:5")

        with patch("backend.app.core.index_version.current", return_value="r:5"):
            self.assertTrue(service.ensure_fresh(rebuild))
        self.assertEqual(len(calls), 1)

    def test_版本一致时不重建(self):
        service = BM25Service()
        service.mark_synced("r:5")
        calls = []

        with patch("backend.app.core.index_version.current", return_value="r:5"):
            self.assertFalse(service.ensure_fresh(lambda: calls.append(True)))
        self.assertEqual(calls, [])

    def test_版本再次变化时会再重建(self):
        service = BM25Service()
        calls = []

        def make_rebuild(version):
            def rebuild():
                calls.append(version)
                service.mark_synced(version)

            return rebuild

        with patch("backend.app.core.index_version.current", return_value="r:5"):
            service.ensure_fresh(make_rebuild("r:5"))
            service.ensure_fresh(make_rebuild("r:5"))
        with patch("backend.app.core.index_version.current", return_value="r:6"):
            service.ensure_fresh(make_rebuild("r:6"))

        self.assertEqual(calls, ["r:5", "r:6"])

    def test_版本倒退也会触发重建(self):
        # 用 != 而不是 > 比较：Redis 被清空后计数器会从头开始，
        # 用大小比较的话本实例会认为自己「更新」，从此再也不重建。
        service = BM25Service()
        service.mark_synced("r:100")
        calls = []

        with patch("backend.app.core.index_version.current", return_value="r:1"):
            service.ensure_fresh(lambda: calls.append(True))
        self.assertEqual(len(calls), 1)

    def test_并发刷新只重建一次(self):
        service = BM25Service()
        calls = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def rebuild():
            with lock:
                calls.append(True)
            time.sleep(0.05)
            service.mark_synced("r:9")

        def worker():
            barrier.wait()
            service.ensure_fresh(rebuild)

        with patch("backend.app.core.index_version.current", return_value="r:9"):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(len(calls), 1, "并发检索各跑了一遍全量重建")

    def test_重建失败时不会把版本标记成已同步(self):
        # 标记成已同步就再也不会重试，失败必须留下「还欠一次重建」的状态。
        service = BM25Service()

        def rebuild():
            raise RuntimeError("Chroma 不可用")

        with patch("backend.app.core.index_version.current", return_value="r:5"):
            with self.assertRaises(RuntimeError):
                service.ensure_fresh(rebuild)
            self.assertIsNone(service._synced_version)

            calls = []
            service.ensure_fresh(lambda: calls.append(True))
            self.assertEqual(len(calls), 1, "失败之后下一次检索没有重试重建")

    def test_清空索引会把已同步版本一起清掉(self):
        service = BM25Service()
        service.mark_synced("r:5")
        service.clear_index()
        self.assertIsNone(service._synced_version)


if __name__ == "__main__":
    unittest.main()

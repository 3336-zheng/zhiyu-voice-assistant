"""知识图谱服务测试。

重点锁三件事，都是「改错了不会报错、只会让图悄悄失真」的地方：
1. 页面向量只算父块——子块重复计入会让长页面权重虚高
2. 阈值真的在过滤边，而不是摆设
3. 没有向量的页面既不能连边，也不能从 pages_without_vectors 里消失
"""

import unittest
from types import SimpleNamespace

from backend.app.services.wiki.graph_service import GraphService, _cosine, _mean_vector


def make_page(page_id, title="页面", notebook="测试本", index_status="ready"):
    """构造一个只带 _node() 会读到的字段的假页面。"""
    return SimpleNamespace(
        id=page_id,
        title=title,
        notebook=notebook,
        tags=["t"],
        index_status=index_status,
    )


class FakeQuery:
    """只实现链路里用到的方法，其余一概不关心。"""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


class FakeSession:
    def __init__(self, pages, links=None):
        self._pages = pages
        self._links = links or []

    def query(self, model):
        # 按模型分发，避免把 link 行当成 page 行返回。
        return FakeQuery(self._pages if model.__name__ == "WikiPage" else self._links)


class FakeCollection:
    def __init__(self, metadatas, embeddings, fail=False):
        self.metadatas = metadatas
        self.embeddings = embeddings
        self.fail = fail
        self.last_where = None

    def get(self, where=None, include=None):
        if self.fail:
            raise RuntimeError("Error finding id")
        self.last_where = where
        return {
            "ids": [str(index) for index in range(len(self.metadatas))],
            "metadatas": self.metadatas,
            "embeddings": self.embeddings,
        }


class FakeChroma:
    def __init__(self, metadatas=(), embeddings=(), fail=False):
        self._collection = FakeCollection(list(metadatas), list(embeddings), fail)


def build_service(pages, metadatas=(), embeddings=(), links=None, chroma_fail=False):
    service = GraphService.__new__(GraphService)
    service.db = FakeSession(pages, links)
    service.chroma = FakeChroma(metadatas, embeddings, chroma_fail)
    return service


class MeanVectorTestCase(unittest.TestCase):
    """页面向量与余弦的纯函数行为。"""

    def test_空输入返回None(self):
        self.assertIsNone(_mean_vector([]))

    def test_零维向量返回None(self):
        # 维度为 0 时后续除法没有意义，必须挡住而不是返回空列表。
        self.assertIsNone(_mean_vector([[]]))

    def test_取算术平均(self):
        self.assertEqual(_mean_vector([[2.0, 4.0], [4.0, 8.0]]), [3.0, 6.0])

    def test_维度不一致的向量被跳过而不是让整体失真(self):
        # 混入不同模型写入的向量时，只跳过它，剩下的照常平均。
        self.assertEqual(_mean_vector([[1.0, 1.0], [1.0, 1.0, 1.0]]), [1.0, 1.0])

    def test_基准维度取第一个向量(self):
        """维度以首个向量为准。首个若本身畸形（短），后面的正常向量会被全部丢弃——
        这是已知的取舍：真实数据里同一页面同一次索引写入，维度不会不一致。"""
        self.assertEqual(_mean_vector([[1.0], [2.0, 2.0]]), [1.0])

    def test_余弦对零向量返回0(self):
        self.assertEqual(_cosine([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_余弦对长度不等返回0(self):
        self.assertEqual(_cosine([1.0, 0.0], [1.0, 0.0, 0.0]), 0.0)


class GraphBuildTestCase(unittest.TestCase):
    """build() 的端到端行为。"""

    PAGE_A = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
    PAGE_B = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"
    PAGE_C = "cccccccc-3333-4333-8333-cccccccccccc"

    def test_阈值过滤掉相似度不足的边(self):
        """A=[1,0] B=[1,0] C=[0,1]。A-B 相似度 1.0，A-C 与 B-C 都是 0。"""
        pages = [make_page(self.PAGE_A), make_page(self.PAGE_B), make_page(self.PAGE_C)]
        metas = [
            {"page_id": self.PAGE_A, "chunk_level": "parent"},
            {"page_id": self.PAGE_B, "chunk_level": "parent"},
            {"page_id": self.PAGE_C, "chunk_level": "parent"},
        ]
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        service = build_service(pages, metas, vectors)

        graph = service.build(threshold=0.9)

        semantic = [edge for edge in graph["edges"] if edge["type"] == "semantic"]
        self.assertEqual(len(semantic), 1)
        self.assertEqual({semantic[0]["source"], semantic[0]["target"]}, {self.PAGE_A, self.PAGE_B})
        self.assertEqual(semantic[0]["weight"], 1.0)

    def test_调低阈值会多出边(self):
        """A=[1,0] D=[1,1] 相似度 0.7071，落在 0.7 与 0.75 之间。"""
        pages = [make_page(self.PAGE_A), make_page(self.PAGE_B)]
        metas = [
            {"page_id": self.PAGE_A, "chunk_level": "parent"},
            {"page_id": self.PAGE_B, "chunk_level": "parent"},
        ]
        vectors = [[1.0, 0.0], [1.0, 1.0]]
        service = build_service(pages, metas, vectors)

        self.assertEqual(len(service.build(threshold=0.7)["edges"]), 1)
        self.assertEqual(len(service.build(threshold=0.75)["edges"]), 0)

    def test_子块不参与页面向量(self):
        """A 若把子块 [0,1] 一起平均，向量会变成 [0.5,0.5]，与 B 的相似度掉到 0.707。"""
        pages = [make_page(self.PAGE_A), make_page(self.PAGE_B)]
        metas = [
            {"page_id": self.PAGE_A, "chunk_level": "parent"},
            {"page_id": self.PAGE_A, "chunk_level": "child"},
            {"page_id": self.PAGE_B, "chunk_level": "parent"},
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        service = build_service(pages, metas, vectors)

        graph = service.build(threshold=0.9)

        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["edges"][0]["weight"], 1.0)

    def test_没有向量的页面进缺向量名单且不连边(self):
        pages = [make_page(self.PAGE_A, index_status="pending"), make_page(self.PAGE_B)]
        metas = [{"page_id": self.PAGE_A, "chunk_level": "parent"}]
        vectors = [[1.0, 0.0]]
        service = build_service(pages, metas, vectors)

        graph = service.build(threshold=0.5)

        self.assertEqual(graph["pages_without_vectors"], [self.PAGE_B])
        self.assertEqual(graph["edges"], [])
        # 节点仍然在图上，只是孤立。
        self.assertEqual({node["id"] for node in graph["nodes"]}, {self.PAGE_A, self.PAGE_B})
        self.assertEqual([node["degree"] for node in graph["nodes"]], [0, 0])

    def test_未索引页面在节点上标记为未索引(self):
        pages = [make_page(self.PAGE_A, index_status="pending")]
        service = build_service(pages)

        graph = service.build(threshold=0.5)

        self.assertFalse(graph["nodes"][0]["indexed"])

    def test_度等于连到这个节点的边数(self):
        """三页两两相似时，每个节点的度都该是 2。"""
        pages = [make_page(self.PAGE_A), make_page(self.PAGE_B), make_page(self.PAGE_C)]
        metas = [
            {"page_id": page_id, "chunk_level": "parent"}
            for page_id in (self.PAGE_A, self.PAGE_B, self.PAGE_C)
        ]
        vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        service = build_service(pages, metas, vectors)

        graph = service.build(threshold=0.5)

        self.assertEqual(len(graph["edges"]), 3)
        self.assertEqual([node["degree"] for node in graph["nodes"]], [2, 2, 2])

    def test_wiki链接作为独立类型的边并入(self):
        pages = [make_page(self.PAGE_A), make_page(self.PAGE_B)]
        links = [
            SimpleNamespace(source_page_id=self.PAGE_A, target_page_id=self.PAGE_B),
        ]
        service = build_service(pages, links=links)

        graph = service.build(threshold=0.5)

        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["edges"][0]["type"], "link")
        self.assertEqual(graph["edges"][0]["weight"], 1.0)

    def test_悬空链接不产生边(self):
        """链接指向已删除页面时，target 不在本页集合里，不能画成边。"""
        pages = [make_page(self.PAGE_A)]
        links = [
            SimpleNamespace(source_page_id=self.PAGE_A, target_page_id="不存在-的-页面"),
        ]
        service = build_service(pages, links=links)

        graph = service.build(threshold=0.5)

        self.assertEqual(graph["edges"], [])

    def test_向量查询条件必须包一层and(self):
        """Chroma 不接受平铺的多条件字典，会直接抛「只能有一个操作符」。"""
        pages = [make_page(self.PAGE_A)]
        service = build_service(pages)
        service.build(threshold=0.5)

        where = service.chroma._collection.last_where
        self.assertIn("$and", where)
        self.assertEqual(len(where["$and"]), 2)

    def test_向量读取失败时降级而不是整张图报错(self):
        """Chroma 读不出来时，页面和手写链接仍然要能看到。

        实际触发过一次：嵌入式模式下 Chroma 是进程内的本地索引文件，
        另一个进程写入后本进程句柄失效，报 "Error finding id"。
        只读视图不该因为可选数据源挂掉而整个 503。
        """
        pages = [make_page(self.PAGE_A), make_page(self.PAGE_B)]
        links = [
            SimpleNamespace(source_page_id=self.PAGE_A, target_page_id=self.PAGE_B),
        ]
        service = build_service(pages, links=links, chroma_fail=True)

        graph = service.build(threshold=0.5)

        self.assertEqual(len(graph["nodes"]), 2)
        # 语义边没了，但用户手写的 link 边还在。
        self.assertEqual([edge["type"] for edge in graph["edges"]], ["link"])
        # 全部报成缺向量，前端会画成空心：能看出「还没索引」而不是「跟谁都不像」。
        self.assertEqual(graph["pages_without_vectors"], [self.PAGE_A, self.PAGE_B])

    def test_空知识库不报错(self):
        service = build_service([])

        graph = service.build(threshold=0.5)

        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["edges"], [])
        self.assertEqual(graph["pages_without_vectors"], [])

    def test_阈值回显实际使用的值(self):
        """前端用回显值初始化滑块，回显错了滑块位置就是错的。"""
        pages = [make_page(self.PAGE_A)]
        service = build_service(pages)

        self.assertEqual(service.build(threshold=0.81)["threshold"], 0.81)


if __name__ == "__main__":
    unittest.main()

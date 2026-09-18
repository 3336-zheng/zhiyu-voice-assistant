"""Chroma 客户端工厂与封装边界测试。

这里盯的是 Stage 4 引入的两类风险：
1. 模式切换本身出错——server 连不上却当成启动成功，表现为检索召回变 0 而不报错。
2. 封装被重新穿透——有人拿到原生 Collection 直接写，绕过 mark_index_changed()，
   索引变了版本号没涨，其他实例不会重建。
"""

import unittest
from unittest.mock import MagicMock, patch

import pydantic

from backend.app.core.config import Settings
from backend.app.services.retrieval import chroma_service as chroma_module


class ChromaClientFactoryTest(unittest.TestCase):
    """_create_client 按 chroma_mode 选择客户端类型。"""

    def test_embedded_模式创建本地持久化客户端(self):
        with patch.object(chroma_module.settings, "chroma_mode", "embedded"):
            with patch.object(chroma_module.chromadb, "PersistentClient") as persistent:
                with patch.object(chroma_module.chromadb, "HttpClient") as http:
                    chroma_module._create_client("data/database/chromadb")

        persistent.assert_called_once()
        self.assertEqual(
            persistent.call_args.kwargs["path"], "data/database/chromadb"
        )
        http.assert_not_called()

    def test_server_模式创建_http_客户端并带上配置的地址(self):
        with patch.object(chroma_module.settings, "chroma_mode", "server"):
            with patch.object(chroma_module.settings, "chroma_host", "chroma"):
                with patch.object(chroma_module.settings, "chroma_port", 8000):
                    with patch.object(chroma_module.chromadb, "PersistentClient") as persistent:
                        with patch.object(chroma_module.chromadb, "HttpClient") as http:
                            chroma_module._create_client("data/database/chromadb")

        http.assert_called_once()
        self.assertEqual(http.call_args.kwargs["host"], "chroma")
        self.assertEqual(http.call_args.kwargs["port"], 8000)
        persistent.assert_not_called()

    def test_server_模式会主动探活(self):
        # 不探活的话，连不上也能「启动成功」，直到第一次检索才发现召回是 0。
        client = MagicMock()
        with patch.object(chroma_module.settings, "chroma_mode", "server"):
            with patch.object(chroma_module.chromadb, "HttpClient", return_value=client):
                chroma_module._create_client("data/database/chromadb")
        client.heartbeat.assert_called_once()

    def test_server_连不上时抛出并提示代理可能性(self):
        # 本机开着系统代理时 httpx 会把请求送进代理拿到 502，而 curl 不读系统代理，
        # 于是现象是「curl 能通、应用连不上」。报错里必须点出这条线索。
        client = MagicMock()
        client.heartbeat.side_effect = OSError("连接被拒绝")
        with patch.object(chroma_module.settings, "chroma_mode", "server"):
            with patch.object(chroma_module.settings, "chroma_host", "chroma"):
                with patch.object(chroma_module.chromadb, "HttpClient", return_value=client):
                    with self.assertRaises(RuntimeError) as ctx:
                        chroma_module._create_client("data/database/chromadb")

        message = str(ctx.exception)
        self.assertIn("chroma", message)
        self.assertIn("NO_PROXY", message)


class ChromaEncapsulationTest(unittest.TestCase):
    """写入口必须自己标记索引变更，调用方不该需要记得这件事。"""

    def _service_with_fake_collection(self):
        service = chroma_module.ChromaService.__new__(chroma_module.ChromaService)
        service._collection = MagicMock()
        return service

    def test_add_chunks_会标记索引变更(self):
        service = self._service_with_fake_collection()
        with patch.object(chroma_module.index_version, "bump") as bump:
            ok = service.add_chunks(
                ids=["doc_a_0"],
                embeddings=[[0.1, 0.2]],
                documents=["正文"],
                metadatas=[{"source_type": "doc"}],
            )
        self.assertTrue(ok)
        bump.assert_called_once()

    def test_空批次不写入也不标记变更(self):
        service = self._service_with_fake_collection()
        with patch.object(chroma_module.index_version, "bump") as bump:
            self.assertTrue(service.add_chunks([], [], [], []))
        service._collection.add.assert_not_called()
        bump.assert_not_called()

    def test_get_all_chunks_把过滤下推给_chroma(self):
        # 拉全量回来再在 Python 里筛，在 server 模式下要白白传输一遍再丢掉。
        service = self._service_with_fake_collection()
        service._collection.get.return_value = {
            "ids": ["doc_a_0"],
            "documents": ["正文"],
            "metadatas": [{"source_type": "doc"}],
        }
        chunks = service.get_all_chunks(["doc", "wiki_page"])

        where = service._collection.get.call_args.kwargs["where"]
        self.assertEqual(where, {"source_type": {"$in": ["doc", "wiki_page"]}})
        include = service._collection.get.call_args.kwargs["include"]
        self.assertNotIn("embeddings", include, "重建 BM25 用不到向量，不该拉回来")
        self.assertEqual(chunks[0]["id"], "doc_a_0")

    def test_get_all_chunks_失败时必须抛出而不是返回空列表(self):
        # 返回空列表会让调用方把 BM25 索引清空并标记为已同步，
        # 于是一次临时故障让检索永久退化成零结果，且再也不会重建。
        service = self._service_with_fake_collection()
        service._collection.get.side_effect = OSError("Chroma 不可用")
        with self.assertRaises(OSError):
            service.get_all_chunks(["doc"])

    def test_get_chunks_by_ids_返回以_id_为键的字典(self):
        service = self._service_with_fake_collection()
        service._collection.get.return_value = {
            "ids": ["a", "b"],
            "documents": ["正文 a", "正文 b"],
            "metadatas": [{"source_type": "doc"}, {"source_type": "wiki_page"}],
        }
        chunks = service.get_chunks_by_ids(["a", "b", "缺失"])
        self.assertEqual(set(chunks), {"a", "b"})
        self.assertEqual(chunks["a"]["content"], "正文 a")

    def test_取不到任何_id_时不发请求(self):
        service = self._service_with_fake_collection()
        self.assertEqual(service.get_chunks_by_ids([]), {})
        service._collection.get.assert_not_called()


class ChromaServerConfigTest(unittest.TestCase):
    """配置错误要在启动时暴露，而不是拖到第一次检索。"""

    def test_server_模式缺少地址时拒绝启动(self):
        with self.assertRaises(pydantic.ValidationError):
            Settings(chroma_mode="server", chroma_host="   ")

    def test_embedded_模式不要求地址(self):
        settings = Settings(chroma_mode="embedded", chroma_host="")
        self.assertEqual(settings.chroma_mode, "embedded")

    def test_模式只接受两个取值(self):
        with self.assertRaises(pydantic.ValidationError):
            Settings(chroma_mode="http")


if __name__ == "__main__":
    unittest.main()

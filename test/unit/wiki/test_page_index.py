"""Wiki 页面稳定索引测试。"""

import unittest

from backend.app.services.wiki.page_index_service import PageIndexService, split_parent_into_children


class FakeCollection:
    def __init__(self):
        self.add_calls = []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeChroma:
    def __init__(self):
        self.collection = FakeCollection()
        self.deleted_filters = []

    def delete_by_filter(self, where):
        self.deleted_filters.append(where)
        return True


class FakeEmbedding:
    @staticmethod
    def encode_documents(texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeBM25:
    def __init__(self):
        self.corpus = {}

    def add_document(self, doc_id, content, title=""):
        self.corpus[doc_id] = content
        return True

    def remove_document(self, doc_id):
        self.corpus.pop(doc_id, None)
        return True


class PageIndexServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.service = PageIndexService.__new__(PageIndexService)
        self.service.embedding_service = FakeEmbedding()
        self.service.chroma_service = FakeChroma()
        self.service.bm25_service = FakeBM25()

    def test_stable_chunk_ids_and_revision_replacement(self):
        page = {
            "id": "11111111-1111-4111-8111-111111111111",
            "revision": 3,
            "title": "RAG 基础",
            "content": "# 第一章\n\n向量检索。\n\n## 证据\n\n引用必须可追溯。",
            "filename": "11111111-1111-4111-8111-111111111111.md",
            "tags": ["RAG", "LLM"],
            "notebook": "人工智能",
            "source_uri": "audio:lesson.wav",
            "updated_at": "2026-08-03T10:00:00+00:00",
        }
        result = self.service.index_page(page)
        self.assertEqual(result["status"], "indexed")
        add_call = self.service.chroma_service.collection.add_calls[-1]
        parent_ids = [
            chunk_id for chunk_id in add_call["ids"]
            if ":child:" not in chunk_id
        ]
        self.assertTrue(parent_ids)
        for index, chunk_id in enumerate(parent_ids):
            self.assertEqual(chunk_id, f"page:{page['id']}:revision:3:chunk:{index}")
        metadata = add_call["metadatas"][0]
        self.assertEqual(metadata["page_id"], page["id"])
        self.assertEqual(metadata["page_revision"], 3)
        self.assertEqual(metadata["page_title"], "RAG 基础")
        self.assertIn("section_path", metadata)
        self.assertEqual(metadata["chunk_level"], "parent")

        old_ids = set(self.service.bm25_service.corpus)
        page["revision"] = 4
        page["content"] = "# 新版本\n\n更新后的正文。"
        self.service.index_page(page)
        new_ids = set(self.service.bm25_service.corpus)
        self.assertFalse(old_ids & new_ids)
        self.assertTrue(all(":revision:4:" in item for item in new_ids))
        self.assertEqual(
            self.service.chroma_service.deleted_filters[-1],
            {"page_id": page["id"]},
        )

    def test_parent_child_split_respects_window_and_overlap(self):
        text = "第一段。" * 80
        children = split_parent_into_children(text, max_chars=120, overlap_chars=20)
        self.assertGreater(len(children), 1)
        self.assertTrue(all(0 < len(child) <= 120 for child in children))
        self.assertEqual(split_parent_into_children("短内容", 120, 20), [])


if __name__ == "__main__":
    unittest.main()

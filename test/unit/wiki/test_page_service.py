"""PageService 核心一致性测试。"""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.services.wiki.page_service import (
    AmbiguousPageError,
    PageConflictError,
    PageValidationError,
    PageService,
)


class FakePageIndexService:
    """不加载真实模型的可控索引器。"""

    def __init__(self):
        self.indexed = []
        self.removed = []
        self.fail_index = False
        self.fail_delete = False

    def index_page(self, page):
        if self.fail_index:
            raise RuntimeError("模拟索引写入失败")
        self.indexed.append((page["id"], page["revision"]))
        return {"status": "indexed"}

    def remove_page(self, page_id):
        if self.fail_delete:
            raise RuntimeError("模拟索引删除失败")
        self.removed.append(page_id)
        return True

    def clear_page_index(self):
        self.indexed.clear()
        self.removed.clear()
        return True


class PageServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "wiki-test.db"
        engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        self.indexer = FakePageIndexService()
        self.service = PageService(
            self.session,
            pages_dir=str(Path(self.temp_dir.name) / "pages"),
            index_service=self.indexer,
        )

    def tearDown(self):
        self.session.close()
        self.temp_dir.cleanup()

    def test_create_update_conflict_and_rollback(self):
        created = self.service.create_page(
            title="RAG 基础",
            content="第一版内容",
            tags=["RAG", "RAG", " LLM "],
            sync_index=True,
        )
        self.assertEqual(created["revision"], 1)
        self.assertEqual(created["tags"], ["RAG", "LLM"])
        self.assertEqual(created["index_status"], "ready")
        self.assertEqual(self.indexer.indexed, [(created["id"], 1)])

        metadata, content = self.service.parse_page(
            Path(created["file_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["id"], created["id"])
        self.assertEqual(content, "第一版内容")

        updated = self.service.update_page(
            created["id"],
            expected_revision=1,
            content="第二版内容",
        )
        self.assertEqual(updated["revision"], 2)
        with self.assertRaises(PageConflictError):
            self.service.update_page(
                created["id"],
                expected_revision=1,
                content="覆盖并发修改",
            )

        rolled_back = self.service.rollback_page(
            created["id"],
            target_revision=1,
            expected_revision=2,
        )
        self.assertEqual(rolled_back["revision"], 3)
        self.assertEqual(rolled_back["content"], "第一版内容")
        self.assertEqual(len(self.service.list_revisions(created["id"])), 3)

    def test_links_backlinks_rename_and_duplicate_titles(self):
        source = self.service.create_page(
            title="课程首页",
            content="继续阅读 [[RAG 基础]]，另见 [[尚未创建]]。",
        )
        target = self.service.create_page(title="RAG 基础", content="知识正文")
        links = self.service.get_links(source["id"])
        resolved = {item["target_title"]: item for item in links["outgoing"]}
        self.assertEqual(resolved["RAG 基础"]["target_page_id"], target["id"])
        self.assertFalse(resolved["尚未创建"]["resolved"])
        self.assertEqual(
            self.service.get_links(target["id"])["backlinks"],
            [{"page_id": source["id"], "title": "课程首页"}],
        )

        renamed = self.service.rename_page(
            target["id"],
            title="检索增强生成",
            expected_revision=1,
        )
        self.assertIn("RAG 基础", renamed["aliases"])
        links_after_rename = self.service.get_links(source["id"])
        old_link = next(
            item for item in links_after_rename["outgoing"]
            if item["target_title"] == "RAG 基础"
        )
        self.assertEqual(old_link["target_page_id"], target["id"])

        self.service.create_page(title="检索增强生成", content="同名页面")
        with self.assertRaises(AmbiguousPageError):
            self.service.find_page("检索增强生成")

    def test_source_deduplication_and_bad_frontmatter(self):
        first = self.service.upsert_page_by_source(
            title="上传资料",
            content="相同内容",
            source_type="document",
            source_uri="upload:lesson.md",
        )
        duplicate = self.service.upsert_page_by_source(
            title="上传资料",
            content="相同内容",
            source_type="document",
            source_uri="upload:lesson.md",
        )
        self.assertTrue(duplicate["deduplicated"])
        self.assertEqual(first["id"], duplicate["id"])
        self.assertEqual(duplicate["revision"], 1)

        changed = self.service.upsert_page_by_source(
            title="上传资料",
            content="变化内容",
            source_type="document",
            source_uri="upload:lesson.md",
        )
        self.assertFalse(changed["deduplicated"])
        self.assertEqual(changed["revision"], 2)

        Path(changed["file_path"]).write_text("---\ntitle: broken\n---\n正文", encoding="utf-8")
        with self.assertRaises(PageValidationError):
            self.service.get_page(changed["id"])

    def test_list_pages_searches_metadata_and_markdown_content(self):
        body_match = self.service.create_page(
            title="课堂记录",
            content="这里记录父子分块与统一融合的实现细节。",
            notebook="学习",
        )
        title_match = self.service.create_page(
            title="检索预算",
            content="另一份正文",
            tags=["RAG"],
        )
        self.service.create_page(title="无关页面", content="普通内容")

        body_results = self.service.list_pages(query="父子分块")
        self.assertEqual(body_results["total"], 1)
        self.assertEqual(body_results["items"][0]["id"], body_match["id"])
        self.assertIn("父子分块", body_results["items"][0]["match_snippet"])

        title_results = self.service.list_pages(query="检索预算")
        self.assertEqual(title_results["total"], 1)
        self.assertEqual(title_results["items"][0]["id"], title_match["id"])
        self.assertIsNone(title_results["items"][0]["match_snippet"])

    def test_index_failure_retry_and_delete_cleanup(self):
        self.indexer.fail_index = True
        created = self.service.create_page(title="索引恢复", content="正文", sync_index=True)
        self.assertEqual(created["index_status"], "failed")
        self.assertTrue(created["index_error"])

        self.indexer.fail_index = False
        retried = self.service.retry_index_tasks()
        self.assertEqual(retried["failed"], 0)
        self.assertEqual(self.service.get_page(created["id"])["index_status"], "ready")

        self.indexer.fail_delete = True
        deleted = self.service.delete_page(
            created["id"],
            expected_revision=created["revision"],
            sync_index=True,
        )
        self.assertEqual(deleted["status"], "deleted")
        self.assertEqual(deleted["index_status"], "failed")
        self.assertFalse(Path(deleted["file_path"]).exists())

        self.indexer.fail_delete = False
        retried_delete = self.service.retry_index_tasks()
        self.assertEqual(retried_delete["failed"], 0)
        restored = self.service.rollback_page(
            created["id"],
            target_revision=1,
            expected_revision=deleted["revision"],
        )
        self.assertEqual(restored["status"], "active")
        self.assertTrue(Path(restored["file_path"]).exists())


if __name__ == "__main__":
    unittest.main()

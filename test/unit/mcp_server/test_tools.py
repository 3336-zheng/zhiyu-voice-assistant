"""MCP 只读工具的单元测试：不连接 MySQL，全部使用假服务。"""

import asyncio
import unittest
from unittest.mock import MagicMock

from backend.app.mcp_server.tools import WikiMCPTools
from backend.app.services.wiki.page_errors import PageNotFoundError


def _run(coro):
    return asyncio.run(coro)


class WikiMCPToolsTest(unittest.TestCase):
    def _tools(self, retrieval=None, page=None, session=None):
        return WikiMCPTools(
            page_service_factory=lambda _db: page,
            retrieval_factory=lambda: retrieval,
            session_factory=lambda: session or MagicMock(),
        )

    def test_search_wiki_精简结果并遵守top_k(self):
        retrieval = MagicMock()
        retrieval.search_hybrid.return_value = [
            {"page_id": "p1", "title": "T1", "section_title": "S1",
             "content": "x" * 100, "score": 0.9, "tags": ["a"],
             "source_url": "/api/pages/p1"},
        ]
        tools = self._tools(retrieval=retrieval)
        result = _run(tools.search_wiki("查询", top_k=3))
        self.assertTrue(result["ok"])
        self.assertEqual(retrieval.search_hybrid.call_args.kwargs["top_k"], 3)
        self.assertEqual(result["results"][0]["page_id"], "p1")

    def test_search_wiki_空查询返回空结果(self):
        retrieval = MagicMock()
        tools = self._tools(retrieval=retrieval)
        result = _run(tools.search_wiki("   "))
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"], [])
        retrieval.search_hybrid.assert_not_called()

    def test_search_wiki_top_k超限被截断(self):
        retrieval = MagicMock()
        retrieval.search_hybrid.return_value = []
        tools = self._tools(retrieval=retrieval)
        _run(tools.search_wiki("q", top_k=999))
        self.assertEqual(retrieval.search_hybrid.call_args.kwargs["top_k"], 50)

    def test_read_page_正文超限截断(self):
        page = MagicMock()
        page.find_page.return_value = {
            "id": "p1", "title": "标题", "status": "active",
            "revision": 2, "tags": [], "content": "字" * 20_000,
        }
        tools = self._tools(page=page)
        result = _run(tools.read_page("标题"))
        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["content"]), 8000)

    def test_read_page_不存在时返回可读错误(self):
        page = MagicMock()
        page.find_page.side_effect = PageNotFoundError("页面不存在")
        tools = self._tools(page=page)
        result = _run(tools.read_page("没有这页"))
        self.assertFalse(result["ok"])
        self.assertIn("不存在", result["error"])
        self.assertNotIn("Traceback", result["error"])

    def test_list_pages_只返回摘要字段(self):
        page = MagicMock()
        page.list_pages.return_value = {
            "total": 1, "items": [
                {"id": "p1", "title": "T", "notebook": "n", "tags": [],
                 "status": "active", "revision": 1, "updated_at": "x",
                 "content": "正文不应出现"},
            ],
        }
        tools = self._tools(page=page)
        result = _run(tools.list_pages(notebook="n", limit=5))
        self.assertTrue(result["ok"])
        self.assertEqual(result["total"], 1)
        self.assertNotIn("content", result["pages"][0])

"""MCP Server 组装测试：工具注册与开关。"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from backend.app.mcp_server.server import create_server


def _list_tools(server):
    return asyncio.run(server.list_tools())


class CreateServerTest(unittest.TestCase):
    def test_注册三个只读工具(self):
        tools = MagicMock()
        tools.search_wiki = AsyncMock(return_value={"ok": True, "results": []})
        tools.read_page = AsyncMock(return_value={"ok": True})
        tools.list_pages = AsyncMock(return_value={"ok": True, "pages": []})
        server = create_server(tools)
        names = {tool.name for tool in _list_tools(server)}
        self.assertEqual(names, {"search_wiki", "read_page", "list_pages"})

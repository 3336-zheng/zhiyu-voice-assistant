"""构建只读 MCP Server：三个工具全部委托给 WikiMCPTools。"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from backend.app.core.config import settings
from backend.app.mcp_server.tools import WikiMCPTools


def create_server(tools: WikiMCPTools | None = None) -> MCPServer:
    """创建 MCP Server；tools 可注入，便于测试。"""
    tools = tools or WikiMCPTools()
    server = MCPServer(
        name="zhiyu",
        version=settings.app_version,
        instructions="智语本地知识库的只读检索工具，不提供写入能力。",
    )

    @server.tool()
    async def search_wiki(query: str, top_k: int = 5) -> dict:
        """在智语知识库中做混合检索（BM25 + 向量）。返回带页面来源的片段。"""
        return await tools.search_wiki(query, top_k)

    @server.tool()
    async def read_page(page_id_or_title: str) -> dict:
        """按页面 ID、标题或别名读取完整 Markdown 正文。"""
        return await tools.read_page(page_id_or_title)

    @server.tool()
    async def list_pages(
        notebook: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> dict:
        """分页列出知识库页面摘要，可按笔记本、标签或关键词筛选。"""
        return await tools.list_pages(notebook, tag, query, limit)

    return server

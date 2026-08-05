"""受限的 stdio MCP 客户端，仅暴露外部研究所需的只读工具。"""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.config import settings

logger = logging.getLogger(__name__)

URLGuard = Callable[[str], Awaitable[Optional[str]]]


class MCPClientError(RuntimeError):
    """MCP Server 不可用、协议异常或返回内容不可用。"""


class MCPClientService:
    """连接一个显式配置的 MCP Server，并只调用搜索和抓取工具。"""

    async def collect(self, queries: List[str], url_guard: URLGuard) -> List[Dict[str, str]]:
        """在同一 MCP 会话中搜索并抓取经过安全校验的公开 URL。"""
        if not settings.mcp_research_available():
            raise MCPClientError("外部研究服务未启用或配置不完整")

        server = StdioServerParameters(
            command=settings.mcp_server_command.strip(),
            args=settings.get_mcp_server_args(),
            env=settings.get_mcp_server_env(),
        )
        try:
            async with asyncio.timeout(settings.mcp_total_timeout_seconds):
                with open(os.devnull, "w", encoding="utf-8") as errlog:
                    async with AsyncExitStack() as stack:
                        read_stream, write_stream = await stack.enter_async_context(
                            stdio_client(server, errlog=errlog)
                        )
                        session = await stack.enter_async_context(
                            ClientSession(read_stream, write_stream)
                        )
                        await self._with_timeout(session.initialize())
                        await self._verify_tools(session)
                        return await self._collect_sources(session, queries, url_guard)
        except TimeoutError as exc:
            raise MCPClientError("外部研究服务响应超时") from exc
        except MCPClientError:
            raise
        except Exception as exc:
            logger.warning("MCP 外部研究调用失败: %s", type(exc).__name__, exc_info=True)
            raise MCPClientError("外部研究服务调用失败") from exc

    async def _verify_tools(self, session: ClientSession) -> None:
        listed = await self._with_timeout(session.list_tools())
        available = {tool.name for tool in listed.tools}
        required = {settings.mcp_search_tool, settings.mcp_fetch_tool}
        missing = sorted(required - available)
        if missing:
            raise MCPClientError(f"MCP Server 缺少已配置工具: {', '.join(missing)}")

    async def _collect_sources(
        self,
        session: ClientSession,
        queries: List[str],
        url_guard: URLGuard,
    ) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        seen_urls = set()
        search_limit = max(1, min(settings.mcp_max_sources, 10))

        for query in queries[: max(1, settings.mcp_max_queries)]:
            payload = await self._call_tool(
                session,
                settings.mcp_search_tool,
                {
                    settings.mcp_search_query_arg: query,
                    settings.mcp_search_limit_arg: search_limit,
                },
            )
            for item in self._search_items(payload):
                raw_url = item.get("url", "")
                guarded_url = await url_guard(raw_url)
                if not guarded_url or guarded_url in seen_urls:
                    continue
                seen_urls.add(guarded_url)
                item["url"] = guarded_url
                candidates.append(item)
                if len(candidates) >= settings.mcp_max_sources:
                    break
            if len(candidates) >= settings.mcp_max_sources:
                break

        collected: List[Dict[str, str]] = []
        for item in candidates:
            payload = await self._call_tool(
                session,
                settings.mcp_fetch_tool,
                {settings.mcp_fetch_url_arg: item["url"]},
            )
            collected.append(
                {
                    **item,
                    "content": self._fetch_content(payload) or item.get("snippet", ""),
                    "tool_name": settings.mcp_fetch_tool,
                }
            )
        return collected

    async def _call_tool(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        if tool_name not in {settings.mcp_search_tool, settings.mcp_fetch_tool}:
            raise MCPClientError("拒绝调用未授权的 MCP 工具")
        result = await self._with_timeout(
            session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=settings.mcp_timeout_seconds,
            )
        )
        if getattr(result, "is_error", False):
            raise MCPClientError(f"MCP 工具执行失败: {tool_name}")
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured
        texts = [
            block.text
            for block in getattr(result, "content", [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        combined = "\n".join(texts).strip()
        if not combined:
            return {}
        try:
            return json.loads(combined)
        except json.JSONDecodeError:
            return {"content": combined}

    async def _with_timeout(self, awaitable):
        async with asyncio.timeout(settings.mcp_timeout_seconds):
            return await awaitable

    @classmethod
    def _search_items(cls, payload: Any) -> List[Dict[str, str]]:
        values: Any = payload
        if isinstance(payload, dict):
            for key in ("results", "items", "data", "sources"):
                if isinstance(payload.get(key), list):
                    values = payload[key]
                    break
            else:
                values = [payload]
        if not isinstance(values, list):
            return []

        normalized = []
        for value in values:
            if not isinstance(value, dict):
                continue
            url = cls._first_text(value, "url", "link", "href")
            if not url:
                continue
            normalized.append(
                {
                    "url": url,
                    "title": cls._first_text(value, "title", "name") or url,
                    "snippet": cls._first_text(
                        value, "snippet", "description", "summary", "text", "content"
                    ),
                }
            )
        return normalized

    @classmethod
    def _fetch_content(cls, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            direct = cls._first_text(payload, "content", "text", "markdown", "body")
            if direct:
                return direct
            for key in ("result", "data", "page"):
                nested = payload.get(key)
                if nested is not None:
                    content = cls._fetch_content(nested)
                    if content:
                        return content
        if isinstance(payload, list):
            return "\n".join(filter(None, (cls._fetch_content(item) for item in payload)))
        return ""

    @staticmethod
    def _first_text(value: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""


def get_mcp_client_service() -> MCPClientService:
    """创建无状态 MCP 客户端服务。"""
    return MCPClientService()

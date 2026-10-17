"""MCP 只读工具：检索、读页、列页。

三个工具全部只读：不调用 LLM、不写库、不访问外部网络。数据库与检索都是同步
实现，所以每个工具用 ``asyncio.to_thread`` 执行，避免阻塞 MCP 事件循环。

依赖通过构造参数注入，单测可以完全不碰 MySQL。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.services.ai.embedding_service import EmbeddingProviderError
from backend.app.services.ai.reranker_service import RerankerProviderError
from backend.app.services.wiki.page_errors import PageServiceError

logger = logging.getLogger(__name__)

# 单次检索最多返回的片段数。外部工具一次调用就产生多次 Embedding，
# 上限防止客户端传入离谱的 top_k。
MAX_SEARCH_TOP_K = 50
# 列表最多返回的页面数。
MAX_LIST_LIMIT = 100

_RESULT_FIELDS = (
    "page_id",
    "title",
    "section_title",
    "content",
    "snippet",
    "score",
    "tags",
    "source_url",
)
_PAGE_SUMMARY_FIELDS = (
    "id",
    "title",
    "notebook",
    "tags",
    "status",
    "revision",
    "updated_at",
)


class WikiMCPTools:
    """三个只读工具的实现。服务工厂可注入，便于测试与复用。"""

    def __init__(
        self,
        *,
        page_service_factory: Optional[Callable[[Any], Any]] = None,
        retrieval_factory: Optional[Callable[[], Any]] = None,
        session_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._page_service_factory = page_service_factory
        self._retrieval_factory = retrieval_factory
        self._session_factory = session_factory

    # ------------------------------------------------------------------ 工具

    async def search_wiki(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """在知识库中做混合检索（BM25 + 向量），返回带来源的片段。"""
        normalized = (query or "").strip()
        if not normalized:
            return {"ok": True, "results": []}
        top_k = _clamp_int(top_k, default=5, maximum=MAX_SEARCH_TOP_K)

        def _run() -> Dict[str, Any]:
            db = self._session()
            try:
                results = self._retrieval().search_hybrid(
                    query=normalized,
                    top_k=top_k,
                    db=db,
                )
                return {
                    "ok": True,
                    "results": [
                        {key: item.get(key) for key in _RESULT_FIELDS if key in item}
                        for item in results
                    ],
                }
            finally:
                db.close()

        return await self._run_sync(_run)

    async def read_page(self, page_id_or_title: str) -> Dict[str, Any]:
        """按页面 ID、标题或别名读取完整 Markdown 正文。"""
        reference = (page_id_or_title or "").strip()
        if not reference:
            return {"ok": False, "error": "页面引用不能为空"}

        def _run() -> Dict[str, Any]:
            db = self._session()
            try:
                page = self._page_service(db).find_page(reference)
                content = page.get("content") or ""
                limit = settings.mcp_server_max_content_chars
                return {
                    "ok": True,
                    "id": page.get("id"),
                    "title": page.get("title"),
                    "status": page.get("status"),
                    "revision": page.get("revision"),
                    "tags": page.get("tags") or [],
                    "notebook": page.get("notebook"),
                    "content": content[:limit],
                    "truncated": len(content) > limit,
                }
            finally:
                db.close()

        return await self._run_sync(_run)

    async def list_pages(
        self,
        notebook: Optional[str] = None,
        tag: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """分页列出页面摘要，可按笔记本、标签或关键词筛选。"""
        limit = _clamp_int(limit, default=20, maximum=MAX_LIST_LIMIT)

        def _run() -> Dict[str, Any]:
            db = self._session()
            try:
                result = self._page_service(db).list_pages(
                    notebook=notebook or None,
                    tag=tag or None,
                    query=query or None,
                    offset=0,
                    limit=limit,
                )
                return {
                    "ok": True,
                    "total": result.get("total", 0),
                    "pages": [
                        {
                            key: item.get(key)
                            for key in _PAGE_SUMMARY_FIELDS
                            if key in item
                        }
                        for item in result.get("items", [])
                    ],
                }
            finally:
                db.close()

        return await self._run_sync(_run)

    # ------------------------------------------------------------------ 内部

    async def _run_sync(self, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """在线程中执行同步逻辑，并把领域异常转成可读的工具错误。"""
        try:
            return await asyncio.to_thread(fn)
        except (PageServiceError, EmbeddingProviderError, RerankerProviderError) as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:  # noqa: BLE001 — 不向外部客户端暴露堆栈
            logger.exception("MCP 工具执行失败")
            return {"ok": False, "error": "工具执行失败"}

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        from backend.app.core.database import SessionLocal

        return SessionLocal()

    def _page_service(self, db):
        if self._page_service_factory is not None:
            return self._page_service_factory(db)
        from backend.app.services.wiki.page_service import get_page_service

        return get_page_service(db)

    def _retrieval(self):
        if self._retrieval_factory is not None:
            return self._retrieval_factory()
        from backend.app.services.retrieval.hybrid_retrieval_service import (
            get_hybrid_retrieval_service,
        )

        return get_hybrid_retrieval_service()


def _clamp_int(value: Any, *, default: int, maximum: int) -> int:
    """把外部传入的数值收敛到 [1, maximum]，非法值回退默认值。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))

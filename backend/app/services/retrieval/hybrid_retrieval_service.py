"""
混合检索服务 (Hybrid Retrieval Service)
整合 BM25 + Embedding + RRF + Reranker 的完整检索流程
"""
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import re
from urllib.parse import quote

from backend.app.core.config import settings
from backend.app.services.retrieval.chroma_service import get_chroma_service
from backend.app.services.retrieval.bm25_service import get_bm25_service
from backend.app.services.retrieval.rrf_service import get_rrf_service
from backend.app.services.ai.embedding_service import get_embedding_service
from backend.app.services.ai.reranker_service import get_reranker_service
from backend.app.services.retrieval.token_budget_service import estimate_tokens, truncate_text
from backend.app.services.wiki.markdown_normalizer import clean_markdown_link_label
from backend.app.core.observability import timed_stage
from backend.app.core.ttl_cache import TTLCache

logger = logging.getLogger(__name__)
AUDIO_SOURCE_PATTERN = re.compile(r"^audio:(\d+)")
TIME_RANGE_PATTERN = re.compile(
    r"\[(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\]"
)
CHILD_CHUNK_PATTERN = re.compile(r"^(?P<parent>.+:chunk:\d+):child:\d+$")


def _timecode_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def _audio_provenance(source_uri: str, content: str) -> Dict[str, Any]:
    source_match = AUDIO_SOURCE_PATTERN.match(source_uri or "")
    if not source_match:
        return {}
    audio_id = int(source_match.group(1))
    time_match = TIME_RANGE_PATTERN.search(content or "")
    start = _timecode_seconds(time_match.group("start")) if time_match else 0.0
    end = _timecode_seconds(time_match.group("end")) if time_match else None
    fragment = f"#t={start:g}" + (f",{end:g}" if end is not None else "")
    query = f"start={start:g}" + (f"&end={end:g}" if end is not None else "")
    return {
        "audio_id": audio_id,
        "audio_start": start,
        "audio_end": end,
        "audio_url": f"/audio/{audio_id}/file{fragment}",
        "transcript_url": f"/audio/{audio_id}/transcript?{query}",
    }


class HybridRetrievalService:
    """
    混合检索服务
    流程: BM25 + Embedding 并行检索 → RRF 融合 → BGE-reranker 精排
    """

    def __init__(
        self,
        *,
        chroma_service=None,
        bm25_service=None,
        rrf_service=None,
        embedding_service=None,
        reranker_service=None,
    ):
        """初始化检索依赖；未注入时使用生产环境的单例服务。"""
        self.chroma_service = (
            chroma_service if chroma_service is not None else get_chroma_service()
        )
        self.bm25_service = (
            bm25_service if bm25_service is not None else get_bm25_service()
        )
        self.rrf_service = rrf_service if rrf_service is not None else get_rrf_service()
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else get_embedding_service()
        )
        self.reranker_service = (
            reranker_service if reranker_service is not None else get_reranker_service()
        )
        self._retrieval_cache: TTLCache[str, Dict[str, Any]] = TTLCache(
            settings.retrieval_cache_ttl_seconds,
            settings.retrieval_cache_max_entries,
        )

    def _fetch_doc_chunks(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        从 ChromaDB 批量获取文档块详情

        Args:
            doc_ids: 文档 ID 列表

        Returns:
            Dict: {doc_id: {"content": ..., "metadata": ...}}
        """
        if not doc_ids:
            return {}
        try:
            chunk_results = self.chroma_service.collection.get(
                ids=doc_ids,
                include=["documents", "metadatas"]
            )
            chunks_dict = {}
            if chunk_results["ids"]:
                for i, cid in enumerate(chunk_results["ids"]):
                    chunks_dict[cid] = {
                        "content": chunk_results["documents"][i] if chunk_results["documents"] else "",
                        "metadata": chunk_results["metadatas"][i] if chunk_results["metadatas"] else {}
                    }
            return chunks_dict
        except Exception as e:
            logger.error(f"获取文档块失败: {e}")
            return {}

    @staticmethod
    def _parent_chunk_id(doc_id: str) -> str:
        """将子块 ID 折叠为稳定父块 ID，其他来源保持不变。"""
        match = CHILD_CHUNK_PATTERN.match(doc_id)
        return match.group("parent") if match else doc_id

    @classmethod
    def _collapse_ranked_children(
        cls,
        results: List[tuple[str, float]],
    ) -> List[tuple[str, float]]:
        """按原始排名折叠同一父块，避免多个子块重复占据候选位。"""
        collapsed: List[tuple[str, float]] = []
        seen = set()
        for doc_id, score in results:
            parent_id = cls._parent_chunk_id(doc_id)
            if parent_id in seen:
                continue
            seen.add(parent_id)
            collapsed.append((parent_id, score))
        return collapsed

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """保留旧入口，统一委托 Token 预算服务。"""
        return estimate_tokens(text)

    @staticmethod
    def _truncate_to_tokens(text: str, token_budget: int) -> str:
        """保留旧入口，统一委托 Token 预算服务。"""
        return truncate_text(text, token_budget)

    @classmethod
    def apply_token_budget(
        cls,
        results: List[Dict[str, Any]],
        token_budget: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        """按排序顺序装配上下文，最后一块可截断但不拆散来源信息。"""
        selected: List[Dict[str, Any]] = []
        used_tokens = 0
        for result in results:
            content = result.get("content", "")
            remaining = token_budget - used_tokens
            if remaining <= 0:
                break
            truncated = cls._truncate_to_tokens(content, remaining)
            if not truncated:
                break
            item = dict(result)
            item["content"] = truncated
            item["snippet"] = truncated[:300]
            item["context_truncated"] = len(truncated) < len(content)
            item_tokens = cls._estimate_tokens(truncated)
            item["context_tokens"] = item_tokens
            selected.append(item)
            used_tokens += item_tokens
            if item["context_truncated"]:
                break
        return selected, used_tokens

    @staticmethod
    def filter_reranked_results(
        reranked: List[Dict[str, Any]],
        *,
        final_top_k: int,
        min_score: Optional[float] = None,
        score_margin: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """过滤明显低相关结果，同时保留最高分证据供证据门禁判断。"""
        if final_top_k <= 0:
            return []

        minimum = (
            settings.retrieval_rerank_min_score
            if min_score is None
            else min_score
        )
        margin = (
            settings.retrieval_rerank_score_margin
            if score_margin is None
            else score_margin
        )
        valid = []
        for item in reranked:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            normalized = dict(item)
            normalized["score"] = score
            valid.append(normalized)

        if not valid:
            return []

        valid.sort(key=lambda item: item["score"], reverse=True)
        best_score = valid[0]["score"]
        threshold = max(float(minimum), best_score - float(margin))
        selected = [item for item in valid if item["score"] >= threshold]
        # 无论模型分数多低都保留最高分，避免把低相关结果误判为“无结果”。
        if not selected:
            selected = valid[:1]
        return selected[:final_top_k]

    def _recall(self, query: str, bm25_top_k: int, embedding_top_k: int) -> Dict[str, Any]:
        """并行执行单个查询的稀疏与稠密召回，不做融合和精排。"""
        with timed_stage("retrieval.embedding"):
            query_embedding = self.embedding_service.encode(query)

        def bm25_search():
            try:
                return self.bm25_service.search(query, top_k=bm25_top_k)
            except Exception as exc:
                logger.error("BM25 检索失败: %s", exc)
                return []

        def embedding_search():
            try:
                return self.chroma_service.search(query_embedding, top_k=embedding_top_k)
            except Exception as exc:
                logger.error("Embedding 检索失败: %s", exc)
                return []

        with timed_stage("retrieval.recall"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_bm25 = executor.submit(bm25_search)
                future_embedding = executor.submit(embedding_search)
                bm25_results = future_bm25.result()
                embedding_results = future_embedding.result()
        return {"bm25": bm25_results, "embedding": embedding_results}

    def _encode_queries(self, queries: List[str]) -> List[List[float]]:
        """优先使用批量查询向量，兼容旧的单条测试替身。"""
        encoder = getattr(self.embedding_service, "encode_queries", None)
        if callable(encoder):
            return encoder(queries)
        encoder = getattr(self.embedding_service, "encode_batch", None)
        if callable(encoder):
            return encoder(queries)
        with ThreadPoolExecutor(
            max_workers=min(settings.retrieval_max_workers, max(1, len(queries)))
        ) as executor:
            futures = [executor.submit(self.embedding_service.encode, query) for query in queries]
            return [future.result() for future in futures]

    def _recall_many(
        self,
        queries: List[str],
        bm25_top_k: int,
        embedding_top_k: int,
    ) -> List[Dict[str, Any]]:
        """批量生成向量，并发执行所有查询的 BM25 与向量召回。"""
        with timed_stage("retrieval.embedding"):
            query_embeddings = self._encode_queries(queries)

        outcomes = [{"bm25": [], "embedding": []} for _ in queries]

        def bm25_search(index: int) -> tuple[int, str, list]:
            try:
                return index, "bm25", self.bm25_service.search(
                    queries[index], top_k=bm25_top_k
                )
            except Exception as exc:
                logger.error("BM25 检索失败: %s", exc)
                return index, "bm25", []

        def embedding_search(index: int) -> tuple[int, str, list]:
            try:
                return index, "embedding", self.chroma_service.search(
                    query_embeddings[index], top_k=embedding_top_k
                )
            except Exception as exc:
                logger.error("Embedding 检索失败: %s", exc)
                return index, "embedding", []

        with timed_stage("retrieval.recall"):
            max_workers = min(
                settings.retrieval_max_workers,
                max(2, len(queries) * 2),
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for index in range(len(queries)):
                    futures.append(executor.submit(bm25_search, index))
                    futures.append(executor.submit(embedding_search, index))
                for future in futures:
                    index, source, results = future.result()
                    outcomes[index][source] = results
        return outcomes

    def _index_version(self) -> str:
        """组合集合计数与 generation，识别同数量内容更新。"""
        try:
            count = self.chroma_service.collection.count()
            generation = getattr(self.chroma_service, "get_generation", lambda: 0)()
            return f"{int(count)}:{int(generation)}"
        except Exception:
            getter = getattr(self.bm25_service, "get_document_count", None)
            if callable(getter):
                return f"{int(getter())}:bm25"
            return f"{len(getattr(self.bm25_service, 'corpus', {}) or {})}:bm25"

    def _retrieval_cache_key(
        self,
        queries: List[str],
        original_query: Optional[str],
        final_top_k: int,
        context_budget: int,
    ) -> str:
        payload = {
            "queries": queries,
            "original_query": original_query or "",
            "final_top_k": final_top_k,
            "context_budget": context_budget,
            "bm25_top_k": settings.bm25_top_k,
            "embedding_top_k": settings.embedding_top_k,
            "rerank_candidate_top_k": settings.rag_rerank_candidate_top_k,
            "rerank_min_score": settings.retrieval_rerank_min_score,
            "rerank_score_margin": settings.retrieval_rerank_score_margin,
            "index_version": self._index_version(),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def search_multi(
        self,
        queries: List[str],
        original_query: Optional[str] = None,
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """多查询统一召回、融合、父块折叠和单次精排。"""
        normalized_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if not normalized_queries:
            return {"results": [], "stats": {"query_count": 0}}

        final_top_k = top_k or settings.rag_final_top_k
        context_budget = token_budget or settings.rag_context_token_budget
        cache_key = self._retrieval_cache_key(
            normalized_queries,
            original_query,
            final_top_k,
            context_budget,
        )
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            cached_stats = dict(cached.get("stats") or {})
            cached_stats["cache_hit"] = True
            return {"results": cached.get("results", []), "stats": cached_stats}

        recall_lists: List[List[tuple[str, float]]] = []
        bm25_count = 0
        embedding_count = 0
        recalled_items = self._recall_many(
            normalized_queries,
            settings.bm25_top_k,
            settings.embedding_top_k,
        )
        for recalled in recalled_items:
            bm25_count += len(recalled["bm25"])
            embedding_count += len(recalled["embedding"])
            recall_lists.extend(
                [
                    self._collapse_ranked_children(recalled["bm25"]),
                    self._collapse_ranked_children(recalled["embedding"]),
                ]
            )

        populated_lists = [items for items in recall_lists if items]
        if not populated_lists:
            return {
                "results": [],
                "stats": {
                    "query_count": len(normalized_queries),
                    "bm25_hits": bm25_count,
                    "embedding_hits": embedding_count,
                    "fused_candidates": 0,
                    "reranked_candidates": 0,
                    "selected_results": 0,
                    "context_tokens": 0,
                    "token_budget": context_budget,
                    "context_truncated": False,
                    "cache_hit": False,
                },
            }

        rerank_candidate_top_k = max(
            final_top_k,
            settings.rag_rerank_candidate_top_k,
        )
        with timed_stage("retrieval.fusion"):
            fused = self.rrf_service.fuse_multi(
                populated_lists,
                top_k=rerank_candidate_top_k,
            )
        doc_ids = [doc_id for doc_id, _ in fused]
        with timed_stage("retrieval.fetch_chunks"):
            chunks_dict = self._fetch_doc_chunks(doc_ids)

        candidates = []
        for doc_id, rrf_score in fused:
            chunk = chunks_dict.get(doc_id)
            if chunk:
                metadata = chunk.get("metadata", {})
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "content": chunk.get("content", ""),
                        "title": metadata.get("page_title", metadata.get("section_title", "")),
                        "source_type": metadata.get("source_type", "doc"),
                        "metadata": metadata,
                        "rrf_score": rrf_score,
                    }
                )
                continue
            content = self.bm25_service.corpus.get(doc_id, "")
            if content:
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "content": content,
                        "title": doc_id,
                        "source_type": "doc",
                        "metadata": {},
                        "rrf_score": rrf_score,
                    }
                )

        if not candidates:
            return {
                "results": [],
                "stats": {
                    "query_count": len(normalized_queries),
                    "context_tokens": 0,
                    "token_budget": context_budget,
                    "context_truncated": False,
                    "cache_hit": False,
                },
            }

        rerank_query = original_query or normalized_queries[0]
        with timed_stage("retrieval.rerank"):
            reranked = self.reranker_service.rerank(
                query=rerank_query,
                documents=[item["content"] for item in candidates],
                top_k=min(final_top_k, len(candidates)),
            )
        reranked = self.filter_reranked_results(
            reranked,
            final_top_k=final_top_k,
        )

        formatted = []
        for rank, rerank_item in enumerate(reranked, start=1):
            candidate = candidates[rerank_item["index"]]
            result = self._format_result(
                candidate,
                candidate["metadata"],
                rank,
                "rerank_score",
                rerank_item["score"],
            )
            result["rrf_score"] = candidate["rrf_score"]
            formatted.append(result)

        selected, used_tokens = self.apply_token_budget(formatted, context_budget)
        stats = {
            "query_count": len(normalized_queries),
            "bm25_hits": bm25_count,
            "embedding_hits": embedding_count,
            "fused_candidates": len(fused),
            "reranked_candidates": len(candidates),
            "quality_filtered_candidates": len(reranked),
            "selected_results": len(selected),
            "context_tokens": used_tokens,
            "token_budget": context_budget,
            "context_truncated": (
                len(selected) < len(formatted)
                or any(item.get("context_truncated", False) for item in selected)
            ),
            "cache_hit": False,
        }
        logger.info("RAG v2 检索完成: %s", stats)
        outcome = {"results": selected, "stats": stats}
        self._retrieval_cache.set(cache_key, outcome)
        return outcome

    @staticmethod
    def _normalize_tags(tags: Any) -> List[str]:
        """将 ChromaDB 中的标签字符串转换为结构化列表。"""
        if isinstance(tags, list):
            return tags
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split(",") if tag.strip()]
        return []

    def _format_result(
        self,
        doc: Dict[str, Any],
        metadata: Dict[str, Any],
        rank: int,
        score_name: str,
        score: float,
    ) -> Dict[str, Any]:
        """统一返回可追溯的页面、版本、章节和原文片段。"""
        page_id = metadata.get("page_id")
        if not page_id:
            page_id = re.match(
                r"^page:([0-9a-f-]{36}):",
                str(doc.get("doc_id", "")),
            )
            page_id = page_id.group(1) if page_id else None
        section_title = clean_markdown_link_label(metadata.get("section_title", ""))
        section_path = clean_markdown_link_label(
            metadata.get("section_path", section_title)
        )
        page_title = clean_markdown_link_label(metadata.get("page_title", ""))
        fallback_title = clean_markdown_link_label(
            metadata.get("filename", doc.get("title", ""))
        )
        title = page_title or section_title or fallback_title or page_id or "知识库页面"
        source_url = None
        if page_id:
            source_url = f"/api/pages/{page_id}"
            if section_title:
                source_url += f"#section={quote(section_title, safe='')}"
        source_uri = metadata.get("source_uri", "")
        result = {
            "id": doc["doc_id"],
            "chunk_id": doc["doc_id"],
            "page_id": page_id,
            "page_revision": metadata.get("page_revision"),
            "title": title,
            "content": doc["content"],
            "snippet": doc["content"][:300],
            "summary": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
            "tags": self._normalize_tags(metadata.get("tags", [])),
            "source_type": doc["source_type"],
            "source_uri": source_uri,
            "source_url": source_url,
            "filename": metadata.get("filename", ""),
            "section_title": section_title,
            "section_path": section_path or section_title,
            score_name: score,
            "rank": rank,
        }
        result.update(_audio_provenance(source_uri, doc["content"]))
        return result

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_top_k: int = None,
        embedding_top_k: int = None,
        rrf_top_k: int = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索

        Args:
            query: 查询字符串
            top_k: 最终返回结果数量
            bm25_top_k: BM25 检索数量
            embedding_top_k: Embedding 检索数量
            rrf_top_k: RRF 融合后候选数量

        Returns:
            List[Dict]: 检索结果列表，包含完整信息和分数
        """
        # 使用默认配置
        if bm25_top_k is None:
            bm25_top_k = settings.bm25_top_k
        if embedding_top_k is None:
            embedding_top_k = settings.embedding_top_k
        if rrf_top_k is None:
            rrf_top_k = settings.rrf_top_k

        try:
            logger.info("开始混合检索，查询长度=%s", len(query))

            # Step 1: 生成查询向量
            with timed_stage("retrieval.embedding"):
                query_embedding = self.embedding_service.encode(query)

            # Step 2: 并行执行 BM25 和 Embedding 检索
            def bm25_search():
                try:
                    results = self.bm25_service.search(query, top_k=bm25_top_k)
                    logger.debug(f"BM25 检索完成: {len(results)} 条")
                    return results
                except Exception as e:
                    logger.error(f"BM25 检索失败: {e}")
                    return []

            def embedding_search():
                try:
                    results = self.chroma_service.search(query_embedding, top_k=embedding_top_k)
                    logger.debug(f"Embedding 检索完成: {len(results)} 条")
                    return results
                except Exception as e:
                    logger.error(f"Embedding 检索失败: {e}")
                    return []

            # 并行执行
            with timed_stage("retrieval.recall"):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_bm25 = executor.submit(bm25_search)
                    future_embedding = executor.submit(embedding_search)

                    bm25_results = future_bm25.result()
                    embedding_results = future_embedding.result()

            logger.info(f"BM25: {len(bm25_results)} 条, Embedding: {len(embedding_results)} 条")

            # 如果某个检索为空，直接使用另一个的结果
            if not bm25_results and not embedding_results:
                logger.warning("BM25 和 Embedding 均未返回结果")
                return []

            if not bm25_results:
                logger.info("BM25 无结果，使用纯 Embedding 结果")
                rrf_results = embedding_results[:rrf_top_k]
            elif not embedding_results:
                logger.info("Embedding 无结果，使用纯 BM25 结果")
                rrf_results = bm25_results[:rrf_top_k]
            else:
                # Step 3: RRF 融合
                with timed_stage("retrieval.fusion"):
                    rrf_results = self.rrf_service.fuse(
                        bm25_results,
                        embedding_results,
                        top_k=rrf_top_k,
                    )
                logger.info(f"RRF 融合完成: {len(rrf_results)} 条")

            # Step 4: 从 ChromaDB 获取所有候选文档块详情
            doc_ids = [doc_id for doc_id, _ in rrf_results]
            if not doc_ids:
                return []

            with timed_stage("retrieval.fetch_chunks"):
                chunks_dict = self._fetch_doc_chunks(doc_ids)

            # Step 5: 准备候选文档并执行 BGE-reranker 精排
            candidate_docs = []
            for doc_id, _ in rrf_results:
                if doc_id in chunks_dict:
                    chunk = chunks_dict[doc_id]
                    candidate_docs.append({
                        "doc_id": doc_id,
                        "content": chunk["content"],
                        "title": chunk["metadata"].get("page_title", chunk["metadata"].get("section_title", "")),
                        "source_type": chunk["metadata"].get("source_type", "doc")
                    })
                else:
                    # BM25 独有的文档（从内存 corpus 取内容）
                    content = self.bm25_service.corpus.get(doc_id, "")
                    if content:
                        candidate_docs.append({
                            "doc_id": doc_id,
                            "content": content,
                            "title": doc_id,
                            "source_type": "doc"
                        })

            if not candidate_docs:
                logger.warning("未找到候选文档")
                return []

            # 执行重排序
            with timed_stage("retrieval.rerank"):
                rerank_results = self.reranker_service.rerank(
                    query=query,
                    documents=[doc["content"] for doc in candidate_docs],
                    top_k=min(top_k, len(candidate_docs))
                )
            rerank_results = self.filter_reranked_results(
                rerank_results,
                final_top_k=top_k,
            )

            # Step 6: 组装最终结果
            final_results = []
            for i, rerank_item in enumerate(rerank_results):
                idx = rerank_item["index"]
                rerank_score = rerank_item["score"]
                doc = candidate_docs[idx]
                chunk = chunks_dict.get(doc["doc_id"], {})
                metadata = chunk.get("metadata", {})

                final_results.append(
                    self._format_result(doc, metadata, i + 1, "rerank_score", rerank_score)
                )

            logger.info(f"混合检索完成，返回 {len(final_results)} 条结果")
            return final_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}", exc_info=True)
            return []

    def search_pure_bm25(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        纯 BM25 检索（用于对比测试）

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果
        """
        try:
            bm25_results = self.bm25_service.search(query, top_k=top_k)
            if not bm25_results:
                return []

            doc_ids = [doc_id for doc_id, _ in bm25_results]
            chunks_dict = self._fetch_doc_chunks(doc_ids)
            results = []
            for i, (doc_id, score) in enumerate(bm25_results):
                content = self.bm25_service.corpus.get(doc_id, "")
                chunk = chunks_dict.get(doc_id, {})
                metadata = chunk.get("metadata", {})
                doc = {
                    "doc_id": doc_id,
                    "title": metadata.get("page_title", metadata.get("section_title", doc_id)),
                    "content": chunk.get("content", content),
                    "source_type": metadata.get("source_type", "doc"),
                }
                results.append(self._format_result(doc, metadata, i + 1, "bm25_score", score))

            return results

        except Exception as e:
            logger.error(f"纯 BM25 检索失败: {e}")
            return []

    def search_pure_embedding(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        纯 Embedding 检索（用于对比测试）

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果
        """
        try:
            query_embedding = self.embedding_service.encode(query)
            embedding_results = self.chroma_service.search(query_embedding, top_k=top_k)
            if not embedding_results:
                return []

            doc_ids = [doc_id for doc_id, _ in embedding_results]
            chunks_dict = self._fetch_doc_chunks(doc_ids)

            results = []
            for i, (doc_id, score) in enumerate(embedding_results):
                chunk = chunks_dict.get(doc_id, {})
                metadata = chunk.get("metadata", {})
                content = chunk.get("content", "")

                doc = {
                    "doc_id": doc_id,
                    "title": metadata.get("page_title", metadata.get("section_title", doc_id)),
                    "content": content,
                    "source_type": metadata.get("source_type", "doc"),
                }
                results.append(self._format_result(doc, metadata, i + 1, "embedding_score", score))

            return results

        except Exception as e:
            logger.error(f"纯 Embedding 检索失败: {e}")
            return []

    def search_with_filter(
        self,
        query: str,
        tag_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        带元数据过滤的混合检索

        Args:
            query: 查询字符串
            tag_filter: 标签过滤
            date_from: 开始日期（ISO格式）
            date_to: 结束日期（ISO格式）
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果
        """
        try:
            query_embedding = self.embedding_service.encode(query)

            # 构建 ChromaDB 过滤条件
            where_clause = {}
            if tag_filter:
                where_clause["tags"] = {"$contains": tag_filter}

            # BM25 检索（不支持元数据过滤）
            bm25_results = self.bm25_service.search(query, top_k=settings.bm25_top_k)

            # Embedding 检索（支持元数据过滤）
            embedding_results = self.chroma_service.search(
                query_embedding,
                top_k=settings.embedding_top_k,
                where=where_clause if where_clause else None
            )

            # RRF 融合
            rrf_results = self.rrf_service.fuse(bm25_results, embedding_results, top_k=settings.rrf_top_k)

            doc_ids = [doc_id for doc_id, _ in rrf_results]
            if not doc_ids:
                return []

            chunks_dict = self._fetch_doc_chunks(doc_ids)

            # 准备候选文档
            candidate_docs = []
            for doc_id, _ in rrf_results:
                if doc_id in chunks_dict:
                    chunk = chunks_dict[doc_id]
                    candidate_docs.append({
                        "doc_id": doc_id,
                        "content": chunk["content"],
                        "title": chunk["metadata"].get("page_title", chunk["metadata"].get("section_title", "")),
                        "source_type": chunk["metadata"].get("source_type", "doc")
                    })
                else:
                    content = self.bm25_service.corpus.get(doc_id, "")
                    if content:
                        candidate_docs.append({
                            "doc_id": doc_id,
                            "content": content,
                            "title": doc_id,
                            "source_type": "doc"
                        })

            if not candidate_docs:
                return []

            rerank_results = self.reranker_service.rerank(
                query=query,
                documents=[doc["content"] for doc in candidate_docs],
                top_k=min(top_k, len(candidate_docs))
            )
            rerank_results = self.filter_reranked_results(
                rerank_results,
                final_top_k=top_k,
            )

            # 组装结果
            final_results = []
            for i, rerank_item in enumerate(rerank_results):
                idx = rerank_item["index"]
                rerank_score = rerank_item["score"]
                doc = candidate_docs[idx]
                chunk = chunks_dict.get(doc["doc_id"], {})
                metadata = chunk.get("metadata", {})

                final_results.append(
                    self._format_result(doc, metadata, i + 1, "rerank_score", rerank_score)
                )

            return final_results

        except Exception as e:
            logger.error(f"带过滤的混合检索失败: {e}")
            return []

    def compare_search_methods(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        对比三种检索方法的结果

        Returns:
            Dict: {"bm25": [...], "embedding": [...], "hybrid": [...]}
        """
        return {
            "bm25": self.search_pure_bm25(query, top_k),
            "embedding": self.search_pure_embedding(query, top_k),
            "hybrid": self.search_hybrid(query, top_k)
        }


# 全局服务实例
hybrid_retrieval_service = None


def get_hybrid_retrieval_service() -> HybridRetrievalService:
    """获取混合检索服务实例（单例模式）"""
    global hybrid_retrieval_service
    if hybrid_retrieval_service is None:
        hybrid_retrieval_service = HybridRetrievalService()
    return hybrid_retrieval_service

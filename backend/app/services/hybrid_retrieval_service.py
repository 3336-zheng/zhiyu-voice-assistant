"""
混合检索服务 (Hybrid Retrieval Service)
整合 BM25 + Embedding + RRF + Reranker 的完整检索流程
"""
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
import re
from urllib.parse import quote

from backend.app.core.config import settings
from backend.app.services.chroma_service import get_chroma_service
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.rrf_service import get_rrf_service
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.reranker_service import get_reranker_service
from backend.app.core.observability import timed_stage

logger = logging.getLogger(__name__)
AUDIO_SOURCE_PATTERN = re.compile(r"^audio:(\d+)")
TIME_RANGE_PATTERN = re.compile(
    r"\[(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\]"
)


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

    def __init__(self):
        """初始化混合检索服务"""
        self.chroma_service = get_chroma_service()
        self.bm25_service = get_bm25_service()
        self.rrf_service = get_rrf_service()
        self.embedding_service = get_embedding_service()
        self.reranker_service = get_reranker_service()

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
        section_title = metadata.get("section_title", "")
        page_title = metadata.get("page_title")
        title = page_title or section_title or metadata.get("filename", doc.get("title", ""))
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
            "section_path": metadata.get("section_path", section_title),
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
            logger.info(f"开始混合检索，查询: '{query[:50]}...'" if len(query) > 50 else f"开始混合检索，查询: '{query}'")

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

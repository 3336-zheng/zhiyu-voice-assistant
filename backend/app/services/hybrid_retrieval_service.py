"""
混合检索服务 (Hybrid Retrieval Service)
整合 BM25 + Embedding + RRF + Reranker 的完整检索流程
"""
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.services.chroma_service import get_chroma_service
from backend.app.services.bm25_service import get_bm25_service
from backend.app.services.rrf_service import get_rrf_service
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.reranker_service import get_reranker_service
from backend.app.models.note import Note
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


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

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        bm25_top_k: int = None,
        embedding_top_k: int = None,
        rrf_top_k: int = None,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索

        Args:
            query: 查询字符串
            top_k: 最终返回结果数量
            bm25_top_k: BM25 检索数量
            embedding_top_k: Embedding 检索数量
            rrf_top_k: RRF 融合后候选数量
            db: 数据库会话

        Returns:
            List[Dict]: 检索结果列表，包含笔记完整信息和分数
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
                rrf_results = self.rrf_service.fuse(bm25_results, embedding_results, top_k=rrf_top_k)
                logger.info(f"RRF 融合完成: {len(rrf_results)} 条")

            # Step 4: 按 ID 类型分组，分别获取详情
            doc_ids = [doc_id for doc_id, _ in rrf_results]
            if not doc_ids:
                return []

            note_ids = []
            doc_chunk_ids = []
            for doc_id in doc_ids:
                if doc_id.startswith("note_"):
                    try:
                        note_ids.append(int(doc_id.split("_", 1)[1]))
                    except ValueError:
                        pass
                elif doc_id.startswith("doc_"):
                    doc_chunk_ids.append(doc_id)

            # 获取笔记详情（从 SQLite）
            notes_dict = {}
            if note_ids:
                if db is None:
                    db = SessionLocal()
                    should_close = True
                else:
                    should_close = False
                try:
                    notes = db.query(Note).filter(Note.id.in_(note_ids)).all()
                    notes_dict = {f"note_{note.id}": note for note in notes}
                finally:
                    if should_close:
                        db.close()

            # 获取文档块详情（从 ChromaDB）
            doc_chunks_dict = {}
            if doc_chunk_ids:
                try:
                    chunk_results = self.chroma_service.collection.get(
                        ids=doc_chunk_ids,
                        include=["documents", "metadatas"]
                    )
                    if chunk_results["ids"]:
                        for i, cid in enumerate(chunk_results["ids"]):
                            doc_chunks_dict[cid] = {
                                "content": chunk_results["documents"][i],
                                "metadata": chunk_results["metadatas"][i] if chunk_results["metadatas"] else {}
                            }
                except Exception as e:
                    logger.error(f"获取文档块失败: {e}")

            # Step 5: BGE-reranker 精排
            candidate_docs = []
            for doc_id, _ in rrf_results:
                if doc_id in notes_dict:
                    note = notes_dict[doc_id]
                    candidate_docs.append({
                        "doc_id": doc_id,
                        "content": note.content,
                        "title": note.title or "",
                        "source_type": "note"
                    })
                elif doc_id in doc_chunks_dict:
                    chunk = doc_chunks_dict[doc_id]
                    candidate_docs.append({
                        "doc_id": doc_id,
                        "content": chunk["content"],
                        "title": chunk["metadata"].get("section_title", ""),
                        "source_type": "doc"
                    })

            if not candidate_docs:
                logger.warning("未找到候选文档")
                return []

            # 执行重排序
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

                if doc["source_type"] == "note":
                    note = notes_dict.get(doc["doc_id"])
                    if note:
                        final_results.append({
                            "id": note.id,
                            "title": note.title,
                            "content": note.content,
                            "summary": note.summary,
                            "tags": note.tags,
                            "source_type": "note",
                            "created_at": note.created_at.isoformat() if note.created_at else None,
                            "updated_at": note.updated_at.isoformat() if note.updated_at else None,
                            "rerank_score": rerank_score,
                            "rank": i + 1
                        })
                else:
                    # 文档块
                    chunk = doc_chunks_dict.get(doc["doc_id"], {})
                    metadata = chunk.get("metadata", {})
                    final_results.append({
                        "id": doc["doc_id"],
                        "title": metadata.get("section_title", metadata.get("filename", "")),
                        "content": doc["content"],
                        "summary": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "tags": ["文档"],
                        "source_type": "doc",
                        "filename": metadata.get("filename", ""),
                        "section_title": metadata.get("section_title", ""),
                        "created_at": None,
                        "updated_at": None,
                        "rerank_score": rerank_score,
                        "rank": i + 1
                    })

            logger.info(f"混合检索完成，返回 {len(final_results)} 条结果")
            return final_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}", exc_info=True)
            return []

    def search_pure_bm25(
        self,
        query: str,
        top_k: int = 5,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        纯 BM25 检索（用于对比测试）

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            db: 数据库会话

        Returns:
            List[Dict]: 检索结果
        """
        try:
            # BM25 检索
            bm25_results = self.bm25_service.search(query, top_k=top_k)
            if not bm25_results:
                return []

            # 按 ID 类型分组
            int_note_ids = []
            for doc_id, _ in bm25_results:
                if doc_id.startswith("note_"):
                    try:
                        int_note_ids.append(int(doc_id.split("_", 1)[1]))
                    except ValueError:
                        pass

            notes_dict = {}
            if int_note_ids:
                if db is None:
                    db = SessionLocal()
                    should_close = True
                else:
                    should_close = False
                try:
                    notes = db.query(Note).filter(Note.id.in_(int_note_ids)).all()
                    notes_dict = {f"note_{note.id}": note for note in notes}
                finally:
                    if should_close:
                        db.close()

            # 组装结果
            results = []
            for i, (doc_id, score) in enumerate(bm25_results):
                if doc_id in notes_dict:
                    note = notes_dict[doc_id]
                    results.append({
                        "id": note.id,
                        "title": note.title,
                        "content": note.content,
                        "summary": note.summary,
                        "tags": note.tags,
                        "source_type": "note",
                        "bm25_score": score,
                        "rank": i + 1
                    })
                else:
                    # doc chunk — 从 BM25 corpus 取内容
                    content = self.bm25_service.corpus.get(doc_id, "")
                    results.append({
                        "id": doc_id,
                        "title": doc_id,
                        "content": content,
                        "summary": content[:200] + "..." if len(content) > 200 else content,
                        "tags": ["文档"],
                        "source_type": "doc",
                        "bm25_score": score,
                        "rank": i + 1
                    })

            return results

        except Exception as e:
            logger.error(f"纯 BM25 检索失败: {e}")
            return []

    def search_pure_embedding(
        self,
        query: str,
        top_k: int = 5,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        纯 Embedding 检索（用于对比测试）

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            db: 数据库会话

        Returns:
            List[Dict]: 检索结果
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_service.encode(query)

            # Embedding 检索
            embedding_results = self.chroma_service.search(query_embedding, top_k=top_k)
            if not embedding_results:
                return []

            # 按 ID 类型分组
            int_note_ids = []
            doc_chunk_ids = []
            for doc_id, _ in embedding_results:
                if doc_id.startswith("note_"):
                    try:
                        int_note_ids.append(int(doc_id.split("_", 1)[1]))
                    except ValueError:
                        pass
                elif doc_id.startswith("doc_"):
                    doc_chunk_ids.append(doc_id)

            # 获取笔记详情
            notes_dict = {}
            if int_note_ids:
                if db is None:
                    db = SessionLocal()
                    should_close = True
                else:
                    should_close = False
                try:
                    notes = db.query(Note).filter(Note.id.in_(int_note_ids)).all()
                    notes_dict = {f"note_{note.id}": note for note in notes}
                finally:
                    if should_close:
                        db.close()

            # 获取文档块详情
            doc_chunks_dict = {}
            if doc_chunk_ids:
                try:
                    chunk_results = self.chroma_service.collection.get(
                        ids=doc_chunk_ids,
                        include=["documents", "metadatas"]
                    )
                    if chunk_results["ids"]:
                        for i, cid in enumerate(chunk_results["ids"]):
                            doc_chunks_dict[cid] = {
                                "content": chunk_results["documents"][i],
                                "metadata": chunk_results["metadatas"][i] if chunk_results["metadatas"] else {}
                            }
                except Exception as e:
                    logger.error(f"获取文档块失败: {e}")

            # 组装结果
            results = []
            for i, (doc_id, score) in enumerate(embedding_results):
                if doc_id in notes_dict:
                    note = notes_dict[doc_id]
                    results.append({
                        "id": note.id,
                        "title": note.title,
                        "content": note.content,
                        "summary": note.summary,
                        "tags": note.tags,
                        "source_type": "note",
                        "embedding_score": score,
                        "rank": i + 1
                    })
                elif doc_id in doc_chunks_dict:
                    chunk = doc_chunks_dict[doc_id]
                    metadata = chunk.get("metadata", {})
                    results.append({
                        "id": doc_id,
                        "title": metadata.get("section_title", metadata.get("filename", "")),
                        "content": chunk["content"],
                        "summary": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                        "tags": ["文档"],
                        "source_type": "doc",
                        "filename": metadata.get("filename", ""),
                        "embedding_score": score,
                        "rank": i + 1
                    })

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
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        带元数据过滤的混合检索

        Args:
            query: 查询字符串
            tag_filter: 标签过滤
            date_from: 开始日期（ISO格式）
            date_to: 结束日期（ISO格式）
            top_k: 返回结果数量
            db: 数据库会话

        Returns:
            List[Dict]: 检索结果
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_service.encode(query)

            # 构建 ChromaDB 过滤条件
            where_clause = {}
            if tag_filter:
                where_clause["tags"] = {"$contains": tag_filter}

            # BM25 检索（BM25 不支持元数据过滤，需要后处理）
            bm25_results = self.bm25_service.search(query, top_k=settings.bm25_top_k)

            # Embedding 检索（支持元数据过滤）
            embedding_results = self.chroma_service.search(
                query_embedding,
                top_k=settings.embedding_top_k,
                where=where_clause if where_clause else None
            )

            # RRF 融合
            rrf_results = self.rrf_service.fuse(bm25_results, embedding_results, top_k=settings.rrf_top_k)

            # 按 ID 类型分组
            doc_ids = [doc_id for doc_id, _ in rrf_results]
            if not doc_ids:
                return []

            int_note_ids = []
            doc_chunk_ids = []
            for doc_id in doc_ids:
                if doc_id.startswith("note_"):
                    try:
                        int_note_ids.append(int(doc_id.split("_", 1)[1]))
                    except ValueError:
                        pass
                elif doc_id.startswith("doc_"):
                    doc_chunk_ids.append(doc_id)

            # 获取笔记（带过滤）
            notes_dict = {}
            if int_note_ids:
                if db is None:
                    db = SessionLocal()
                    should_close = True
                else:
                    should_close = False
                try:
                    from datetime import datetime
                    query_stmt = db.query(Note).filter(Note.id.in_(int_note_ids))
                    if date_from:
                        from_date = datetime.fromisoformat(date_from)
                        query_stmt = query_stmt.filter(Note.created_at >= from_date)
                    if date_to:
                        to_date = datetime.fromisoformat(date_to)
                        query_stmt = query_stmt.filter(Note.created_at <= to_date)
                    notes = query_stmt.all()
                    notes_dict = {f"note_{note.id}": note for note in notes}
                finally:
                    if should_close:
                        db.close()

            # 获取文档块
            doc_chunks_dict = {}
            if doc_chunk_ids:
                try:
                    chunk_results = self.chroma_service.collection.get(
                        ids=doc_chunk_ids,
                        include=["documents", "metadatas"]
                    )
                    if chunk_results["ids"]:
                        for i, cid in enumerate(chunk_results["ids"]):
                            doc_chunks_dict[cid] = {
                                "content": chunk_results["documents"][i],
                                "metadata": chunk_results["metadatas"][i] if chunk_results["metadatas"] else {}
                            }
                except Exception as e:
                    logger.error(f"获取文档块失败: {e}")

            # 准备候选文档
            candidate_docs = []
            for doc_id, _ in rrf_results:
                if doc_id in notes_dict:
                    note = notes_dict[doc_id]
                    candidate_docs.append({
                        "doc_id": doc_id,
                        "content": note.content,
                        "title": note.title or "",
                        "source_type": "note"
                    })
                elif doc_id in doc_chunks_dict:
                    chunk = doc_chunks_dict[doc_id]
                    candidate_docs.append({
                        "doc_id": doc_id,
                        "content": chunk["content"],
                        "title": chunk["metadata"].get("section_title", ""),
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

                if doc["source_type"] == "note":
                    note = notes_dict.get(doc["doc_id"])
                    if note:
                        final_results.append({
                            "id": note.id,
                            "title": note.title,
                            "content": note.content,
                            "summary": note.summary,
                            "tags": note.tags,
                            "source_type": "note",
                            "created_at": note.created_at.isoformat() if note.created_at else None,
                            "rerank_score": rerank_score,
                            "rank": i + 1
                        })
                else:
                    chunk = doc_chunks_dict.get(doc["doc_id"], {})
                    metadata = chunk.get("metadata", {})
                    final_results.append({
                        "id": doc["doc_id"],
                        "title": metadata.get("section_title", metadata.get("filename", "")),
                        "content": doc["content"],
                        "summary": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "tags": ["文档"],
                        "source_type": "doc",
                        "filename": metadata.get("filename", ""),
                        "created_at": None,
                        "rerank_score": rerank_score,
                        "rank": i + 1
                    })

            return final_results

        except Exception as e:
            logger.error(f"带过滤的混合检索失败: {e}")
            return []

    def compare_search_methods(
        self,
        query: str,
        top_k: int = 5,
        db: Session = None
    ) -> Dict[str, List[Dict]]:
        """
        对比三种检索方法的结果

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            db: 数据库会话

        Returns:
            Dict: {"bm25": [...], "embedding": [...], "hybrid": [...]}
        """
        return {
            "bm25": self.search_pure_bm25(query, top_k, db),
            "embedding": self.search_pure_embedding(query, top_k, db),
            "hybrid": self.search_hybrid(query, top_k, db=db)
        }


# 全局服务实例
hybrid_retrieval_service = None


def get_hybrid_retrieval_service() -> HybridRetrievalService:
    """获取混合检索服务实例（单例模式）"""
    global hybrid_retrieval_service
    if hybrid_retrieval_service is None:
        hybrid_retrieval_service = HybridRetrievalService()
    return hybrid_retrieval_service

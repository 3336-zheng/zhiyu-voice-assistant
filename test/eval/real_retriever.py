"""复用项目检索组件的通用内存评测器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.core.config import settings
from backend.app.services.rrf_service import RRFService
from test.eval.dataset import CorpusDocument


class EvaluationRetriever:
    """共享索引与向量缓存，公平比较四种检索链路。"""

    SUPPORTED_METHODS = ("bm25", "embedding", "hybrid", "hybrid_reranker")

    def __init__(
        self,
        documents: Sequence[CorpusDocument],
        *,
        embedding_service: Any = None,
        reranker_service: Any = None,
    ) -> None:
        if not documents:
            raise ValueError("评测语料不能为空")
        self._documents_by_id = {document.doc_id: document for document in documents}
        self.document_ids = [document.doc_id for document in documents]
        self.documents = [document.text for document in documents]
        self._parent_ids = {
            document.doc_id: str(
                document.metadata.get("parent_chunk_id") or document.doc_id
            )
            for document in documents
        }
        self.bm25 = BM25Okapi([self._tokenize(text) for text in self.documents])
        self.rrf = RRFService(k=settings.rrf_k)
        self._embedding_service = embedding_service
        self._reranker_service = reranker_service
        self._document_embeddings: np.ndarray | None = None
        self._usage = {
            "embedding_document_calls": 0,
            "embedding_documents": 0,
            "embedding_document_characters": 0,
            "embedding_query_calls": 0,
            "embedding_query_characters": 0,
            "rerank_calls": 0,
            "rerank_candidates": 0,
            "rerank_candidate_characters": 0,
            "retrieval_candidates_before_parent_collapse": 0,
            "duplicate_parent_candidates_collapsed": 0,
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.strip() for token in jieba.lcut(text.lower()) if token.strip()]

    @property
    def embedding_service(self) -> Any:
        if self._embedding_service is None:
            from backend.app.services.embedding_service import get_embedding_service

            self._embedding_service = get_embedding_service()
        return self._embedding_service

    @property
    def reranker_service(self) -> Any:
        if self._reranker_service is None:
            from backend.app.services.reranker_service import get_reranker_service

            self._reranker_service = get_reranker_service()
        return self._reranker_service

    def prepare(self, methods: Sequence[str]) -> None:
        """在计时前构建共享语料向量，避免首个方法承担初始化成本。"""
        if any(method != "bm25" for method in methods):
            self._ensure_document_embeddings()

    def _ensure_document_embeddings(self) -> np.ndarray:
        if self._document_embeddings is None:
            self._usage["embedding_document_calls"] += 1
            self._usage["embedding_documents"] += len(self.documents)
            self._usage["embedding_document_characters"] += sum(
                len(text) for text in self.documents
            )
            vectors = self.embedding_service.encode_documents(self.documents)
            self._document_embeddings = np.asarray(vectors, dtype=np.float32)
            if self._document_embeddings.ndim != 2:
                raise ValueError("Embedding 返回的语料向量维度不正确")
        return self._document_embeddings

    def _query_embedding(self, query: str) -> np.ndarray:
        self._usage["embedding_query_calls"] += 1
        self._usage["embedding_query_characters"] += len(query)
        return np.asarray(self.embedding_service.encode(query), dtype=np.float32)

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            zip(self.document_ids, scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[: min(top_k, len(ranked))]

    def _embedding_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        document_embeddings = self._ensure_document_embeddings()
        query_vector = self._query_embedding(query)
        query_norm = float(np.linalg.norm(query_vector))
        document_norms = np.linalg.norm(document_embeddings, axis=1)
        denominator = np.maximum(document_norms * max(query_norm, 1e-12), 1e-12)
        scores = (document_embeddings @ query_vector) / denominator
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.document_ids[index], float(scores[index]))
            for index in ranked_indices
        ]

    def _collapse_to_parents(
        self,
        results: Sequence[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """按排名折叠同一父块，并记录子块导致的候选重复。"""
        collapsed = []
        seen = set()
        for doc_id, score in results:
            parent_id = self._parent_ids.get(doc_id, doc_id)
            if parent_id in seen:
                self._usage["duplicate_parent_candidates_collapsed"] += 1
                continue
            seen.add(parent_id)
            collapsed.append((parent_id, score))
        self._usage["retrieval_candidates_before_parent_collapse"] += len(results)
        return collapsed

    def search(self, query: str, method: str, top_k: int) -> list[str]:
        """按方法返回文档 ID 排名，候选规模与项目运行配置一致。"""
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"不支持的评测方法: {method}")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        bm25_results = self._collapse_to_parents(self._bm25_search(
            query,
            max(settings.bm25_top_k, top_k),
        ))
        if method == "bm25":
            return [doc_id for doc_id, _ in bm25_results[:top_k]]

        embedding_results = self._collapse_to_parents(self._embedding_search(
            query,
            max(settings.embedding_top_k, top_k),
        ))
        if method == "embedding":
            return [doc_id for doc_id, _ in embedding_results[:top_k]]

        fused = self.rrf.fuse(
            bm25_results,
            embedding_results,
            top_k=max(settings.rrf_top_k, top_k),
        )
        if method == "hybrid":
            return [doc_id for doc_id, _ in fused[:top_k]]

        candidate_ids = [doc_id for doc_id, _ in fused]
        candidate_texts = [
            self._documents_by_id[doc_id].text for doc_id in candidate_ids
        ]
        self._usage["rerank_calls"] += 1
        self._usage["rerank_candidates"] += len(candidate_texts)
        self._usage["rerank_candidate_characters"] += sum(
            len(text) for text in candidate_texts
        )
        reranked = self.reranker_service.rerank(
            query,
            candidate_texts,
            top_k=min(top_k, len(candidate_texts)),
        )
        return [candidate_ids[item["index"]] for item in reranked]

    def model_snapshot(self) -> dict[str, object]:
        """返回可复现实验但不包含 Key 和服务地址的模型配置。"""
        embedding_name = settings.embedding_model
        if not embedding_name and settings.embedding_model_path:
            embedding_name = Path(settings.embedding_model_path).name
        reranker_name = settings.reranker_model
        if not reranker_name and settings.reranker_model_path:
            reranker_name = Path(settings.reranker_model_path).name
        return {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": embedding_name,
            "embedding_dimensions": settings.embedding_dimensions or "provider_default",
            "reranker_provider": settings.reranker_provider,
            "reranker_model": reranker_name,
        }

    def usage_snapshot(self) -> dict[str, int]:
        """返回模型工作量，便于结合供应商单价估算成本。"""
        return dict(self._usage)

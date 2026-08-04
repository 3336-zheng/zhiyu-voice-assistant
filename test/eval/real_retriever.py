"""独立、无业务数据写入的真实检索评估器。"""

from typing import Dict, List, Tuple

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.services.rrf_service import RRFService


class EvaluationRetriever:
    """在固定语料上运行项目采用的真实检索算法和本地模型。"""

    SUPPORTED_METHODS = {"bm25", "embedding", "hybrid", "hybrid_reranker"}

    def __init__(self, corpus: Dict[str, str], method: str):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"不支持的评估方法: {method}")
        self.method = method
        self.document_ids = list(corpus)
        self.documents = [corpus[doc_id] for doc_id in self.document_ids]
        tokenized = [self._tokenize(text) for text in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        self.rrf = RRFService()
        self.embedding_service = None
        self.reranker_service = None
        self.document_embeddings = None

        if method != "bm25":
            from backend.app.services.embedding_service import get_embedding_service

            self.embedding_service = get_embedding_service()
            self.document_embeddings = np.asarray(
                self.embedding_service.encode_documents(self.documents),
                dtype=np.float32,
            )
        if method == "hybrid_reranker":
            from backend.app.services.reranker_service import get_reranker_service

            self.reranker_service = get_reranker_service()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.strip() for token in jieba.lcut(text.lower()) if token.strip()]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            zip(self.document_ids, scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def _embedding_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        query_vector = np.asarray(self.embedding_service.encode(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        document_norms = np.linalg.norm(self.document_embeddings, axis=1)
        denominator = np.maximum(document_norms * max(query_norm, 1e-12), 1e-12)
        scores = (self.document_embeddings @ query_vector) / denominator
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.document_ids[index], float(scores[index]))
            for index in ranked_indices
        ]

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """按指定方法返回排序后的文档 ID。"""
        bm25_results = self._bm25_search(query, max(top_k, 20))
        if self.method == "bm25":
            return [doc_id for doc_id, _ in bm25_results[:top_k]]

        embedding_results = self._embedding_search(query, max(top_k, 20))
        if self.method == "embedding":
            return [doc_id for doc_id, _ in embedding_results[:top_k]]

        fused = self.rrf.fuse(
            bm25_results,
            embedding_results,
            top_k=max(top_k, 20),
        )
        if self.method == "hybrid":
            return [doc_id for doc_id, _ in fused[:top_k]]

        candidates = [doc_id for doc_id, _ in fused]
        candidate_documents = [self.documents[self.document_ids.index(doc_id)] for doc_id in candidates]
        reranked = self.reranker_service.rerank(
            query,
            candidate_documents,
            top_k=min(top_k, len(candidate_documents)),
        )
        return [candidates[item["index"]] for item in reranked]

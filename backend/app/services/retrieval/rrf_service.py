"""
RRF (Reciprocal Rank Fusion) 融合排序服务
融合 BM25 和 Embedding 的检索结果
"""
from typing import List, Tuple, Dict
import logging

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class RRFService:
    """
    Reciprocal Rank Fusion 融合排序服务

    公式: score(d) = Σ 1/(k + rank_i(d))

    其中:
    - k 是常数（通常取 60）
    - rank_i(d) 是文档 d 在第 i 个检索结果中的排名
    """

    def __init__(self, k: float = None):
        """
        初始化 RRF 服务

        Args:
            k: RRF 常数，默认从配置读取（通常 60）
        """
        self.k = k or settings.rrf_k

    def fuse(
        self,
        bm25_results: List[Tuple[str, float]],
        embedding_results: List[Tuple[str, float]],
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        融合 BM25 和 Embedding 的检索结果

        Args:
            bm25_results: BM25 结果 [(doc_id, bm25_score), ...]
            embedding_results: Embedding 结果 [(doc_id, embedding_score), ...]
            top_k: 返回结果数量，默认从配置读取

        Returns:
            List[Tuple[str, float]]: 融合后的结果 [(doc_id, rrf_score), ...]，按分数降序
        """
        if top_k is None:
            top_k = settings.rrf_top_k

        try:
            # 构建排名字典: {doc_id: rank}
            bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, start=1)}
            embedding_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(embedding_results, start=1)}

            # 获取所有唯一的文档 ID
            all_doc_ids = set(bm25_ranks.keys()) | set(embedding_ranks.keys())

            # 计算 RRF 分数
            rrf_scores: Dict[str, float] = {}

            for doc_id in all_doc_ids:
                score = 0.0

                # BM25 的贡献（如果有）
                if doc_id in bm25_ranks:
                    rank = bm25_ranks[doc_id]
                    score += 1.0 / (self.k + rank)

                # Embedding 的贡献（如果有）
                if doc_id in embedding_ranks:
                    rank = embedding_ranks[doc_id]
                    score += 1.0 / (self.k + rank)

                rrf_scores[doc_id] = score

            # 按分数降序排序
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            # 返回 top_k 结果
            final_results = sorted_results[:top_k]

            logger.debug(
                f"RRF 融合完成: BM25={len(bm25_results)}, "
                f"Embedding={len(embedding_results)}, "
                f"融合后={len(final_results)}"
            )

            return final_results

        except Exception as e:
            logger.error(f"RRF 融合失败: {e}")
            return []

    def fuse_multi(
        self,
        results_list: List[List[Tuple[str, float]]],
        weights: List[float] = None,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        融合多个检索源的结果（支持权重）

        Args:
            results_list: 多个检索结果列表
            weights: 各检索源的权重，None 表示均等权重
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float]]: 融合后的结果
        """
        if top_k is None:
            top_k = settings.rrf_top_k

        if not results_list:
            return []

        try:
            # 如果没有指定权重，使用均等权重
            if weights is None:
                weights = [1.0] * len(results_list)

            if len(weights) != len(results_list):
                raise ValueError("权重数量必须与结果列表数量相同")

            # 构建排名字典列表
            ranks_list = []
            for results in results_list:
                ranks = {note_id: rank for rank, (note_id, _) in enumerate(results, start=1)}
                ranks_list.append(ranks)

            # 获取所有唯一的文档 ID
            all_note_ids = set()
            for ranks in ranks_list:
                all_note_ids.update(ranks.keys())

            # 计算加权 RRF 分数
            rrf_scores: Dict[str, float] = {}

            for note_id in all_note_ids:
                score = 0.0

                for ranks, weight in zip(ranks_list, weights):
                    if note_id in ranks:
                        rank = ranks[note_id]
                        score += weight * (1.0 / (self.k + rank))

                rrf_scores[note_id] = score

            # 排序并返回
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"多源 RRF 融合失败: {e}")
            return []

    def fuse_with_min_score_filter(
        self,
        bm25_results: List[Tuple[str, float]],
        embedding_results: List[Tuple[str, float]],
        min_bm25_score: float = 0.0,
        min_embedding_score: float = 0.0,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        带分数过滤的 RRF 融合

        Args:
            bm25_results: BM25 结果
            embedding_results: Embedding 结果
            min_bm25_score: BM25 最小分数阈值
            min_embedding_score: Embedding 最小分数阈值
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float]]: 融合后的结果
        """
        if top_k is None:
            top_k = settings.rrf_top_k

        # 过滤低于阈值的
        filtered_bm25 = [(nid, score) for nid, score in bm25_results if score >= min_bm25_score]
        filtered_embedding = [(nid, score) for nid, score in embedding_results if score >= min_embedding_score]

        return self.fuse(filtered_bm25, filtered_embedding, top_k)

    def get_fusion_details(
        self,
        doc_id: str,
        bm25_results: List[Tuple[str, float]],
        embedding_results: List[Tuple[str, float]]
    ) -> Dict:
        """
        获取某个文档的融合详情（用于调试和分析）

        Args:
            doc_id: 文档 ID
            bm25_results: BM25 结果
            embedding_results: Embedding 结果

        Returns:
            Dict: 包含排名和贡献的详细信息
        """
        # 查找在各自列表中的排名
        bm25_rank = None
        bm25_score = None
        for rank, (nid, score) in enumerate(bm25_results, start=1):
            if nid == doc_id:
                bm25_rank = rank
                bm25_score = score
                break

        embedding_rank = None
        embedding_score = None
        for rank, (nid, score) in enumerate(embedding_results, start=1):
            if nid == doc_id:
                embedding_rank = rank
                embedding_score = score
                break

        # 计算 RRF 分数
        rrf_score = 0.0
        if bm25_rank:
            rrf_score += 1.0 / (self.k + bm25_rank)
        if embedding_rank:
            rrf_score += 1.0 / (self.k + embedding_rank)

        return {
            "doc_id": doc_id,
            "rrf_score": rrf_score,
            "bm25": {
                "rank": bm25_rank,
                "score": bm25_score,
                "contribution": 1.0 / (self.k + bm25_rank) if bm25_rank else 0
            },
            "embedding": {
                "rank": embedding_rank,
                "score": embedding_score,
                "contribution": 1.0 / (self.k + embedding_rank) if embedding_rank else 0
            },
            "k": self.k
        }


# 全局服务实例
rrf_service = None


def get_rrf_service() -> RRFService:
    """获取 RRF 服务实例（单例模式）"""
    global rrf_service
    if rrf_service is None:
        rrf_service = RRFService()
    return rrf_service

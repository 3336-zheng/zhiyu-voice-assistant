"""
Query 改写服务
支持 HyDE 和 RAG-Fusion 两种策略，用于检索前优化召回
"""
import logging
from typing import List, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RewriteStrategy(str, Enum):
    """改写策略"""
    HYDE = "hyde"  # HyDE: 生成假设答案再检索
    RAG_FUSION = "rag_fusion"  # RAG-Fusion: 多 query 并行召回 + RRF 融合
    NONE = "none"  # 不改写


class QueryRewriteService:
    """
    Query 改写服务
    支持 HyDE 和 RAG-Fusion 两种策略
    """

    def __init__(self):
        """初始化"""
        self._llm_service = None

    @property
    def llm_service(self):
        """延迟加载 LLM 服务"""
        if self._llm_service is None:
            try:
                from backend.app.services.ai.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            except Exception as e:
                logger.warning(f"LLM 服务加载失败: {e}")
        return self._llm_service

    def rewrite_query(
        self,
        query: str,
        strategy: RewriteStrategy = RewriteStrategy.RAG_FUSION,
        num_queries: int = 3
    ) -> List[str]:
        """
        改写查询

        Args:
            query: 原始查询
            strategy: 改写策略
            num_queries: 生成的查询数量（RAG-Fusion）

        Returns:
            List[str]: 改写后的查询列表（包含原始查询）
        """
        if strategy == RewriteStrategy.NONE:
            return [query]

        try:
            if strategy == RewriteStrategy.HYDE:
                return self._hyde_rewrite(query)
            elif strategy == RewriteStrategy.RAG_FUSION:
                return self._rag_fusion_rewrite(query, num_queries)
            else:
                return [query]
        except Exception as e:
            logger.error(f"Query 改写失败: {e}")
            return [query]

    def _hyde_rewrite(self, query: str) -> List[str]:
        """
        HyDE 改写：生成假设答案，用假设答案检索

        Args:
            query: 原始查询

        Returns:
            List[str]: [原始查询, 假设答案]
        """
        if not self.llm_service:
            logger.warning("LLM 服务不可用，跳过 HyDE 改写")
            return [query]

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个知识库检索助手。请根据用户的问题，生成一个假设性的答案。\n"
                        "要求：\n"
                        "1. 答案应该是你对问题的推测，不需要完全准确\n"
                        "2. 使用与问题相同的语言风格\n"
                        "3. 包含可能的关键词和概念\n"
                        "4. 长度适中（50-150字）"
                    )
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n\n请生成一个假设性的答案："
                }
            ]

            hypothetical_answer = self.llm_service.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )

            logger.info(
                "HyDE 改写完成: 查询长度=%s，假设答案长度=%s",
                len(query),
                len(hypothetical_answer),
            )
            return [query, hypothetical_answer]

        except Exception as e:
            logger.error(f"HyDE 改写失败: {e}")
            return [query]

    def _rag_fusion_rewrite(self, query: str, num_queries: int = 3) -> List[str]:
        """
        RAG-Fusion 改写：生成多个视角的查询

        Args:
            query: 原始查询
            num_queries: 生成的查询数量

        Returns:
            List[str]: 多个视角的查询列表
        """
        if not self.llm_service:
            logger.warning("LLM 服务不可用，跳过 RAG-Fusion 改写")
            return [query]

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"你是一个查询改写助手。请将用户的查询改写为 {num_queries} 个不同视角的查询。\n"
                        "要求：\n"
                        "1. 保持原始查询的核心意图\n"
                        "2. 从不同角度、不同粒度提问\n"
                        "3. 使用同义词、近义词替换\n"
                        "4. 每行一个查询，不要编号\n"
                        "5. 使用与原始查询相同的语言"
                    )
                },
                {
                    "role": "user",
                    "content": f"原始查询：{query}\n\n请改写为 {num_queries} 个不同视角的查询："
                }
            ]

            response = self.llm_service.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            # 解析改写后的查询
            rewritten_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]

            # 过滤掉空查询和与原始查询完全相同的查询
            rewritten_queries = [q for q in rewritten_queries if q and q != query]

            # 确保不超过指定数量
            rewritten_queries = rewritten_queries[:num_queries]

            # 原始查询放在第一位
            result = [query] + rewritten_queries

            logger.info("RAG-Fusion 改写完成: 改写数=%s", len(rewritten_queries))
            return result

        except Exception as e:
            logger.error(f"RAG-Fusion 改写失败: {e}")
            return [query]


# 全局实例
query_rewrite_service = None


def get_query_rewrite_service() -> QueryRewriteService:
    """获取 Query 改写服务实例（单例模式）"""
    global query_rewrite_service
    if query_rewrite_service is None:
        query_rewrite_service = QueryRewriteService()
    return query_rewrite_service

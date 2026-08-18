"""
Query 改写服务
支持 HyDE 和 RAG-Fusion 两种策略，用于检索前优化召回
"""
import logging
import hashlib
import json
from typing import List, Dict, Optional
from enum import Enum

from backend.app.core.config import settings
from backend.app.services.memory.context_assembler import ContextAssembler
from backend.app.core.ttl_cache import TTLCache

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
        self.context_assembler = ContextAssembler(
            context_window_tokens=settings.llm_context_window_tokens,
            history_token_budget=settings.memory_context_token_budget,
            summary_token_budget=settings.memory_summary_token_budget,
        )
        self._cache: TTLCache[str, List[str]] = TTLCache(
            settings.query_rewrite_cache_ttl_seconds,
            settings.query_rewrite_cache_max_entries,
        )

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
        num_queries: int = 3,
        context: Optional[List[Dict[str, str]]] = None,
        goal: Optional[str] = None,
        intent: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        """
        改写查询

        Args:
            query: 原始查询
            strategy: 改写策略
            num_queries: 生成的查询数量（RAG-Fusion）
            context: 会话上下文，用于消解指代和省略
            goal: Planner 提取的用户目标
            intent: Planner 提取的结构化意图

        Returns:
            List[str]: 以消歧后的独立查询开头的检索查询列表
        """
        if strategy == RewriteStrategy.NONE:
            return [self._fallback_query(query, goal)]

        cache_key = self._cache_key(
            query,
            strategy=strategy,
            num_queries=num_queries,
            context=context,
            goal=goal,
            intent=intent,
        )
        cached = None if force_refresh else self._cache.get(cache_key)
        if cached is not None:
            logger.info("Query 改写缓存命中: strategy=%s", strategy.value)
            return cached

        try:
            if strategy == RewriteStrategy.HYDE:
                result = self._hyde_rewrite(query, context, goal, intent)
            elif strategy == RewriteStrategy.RAG_FUSION:
                result = self._rag_fusion_rewrite(
                    query,
                    num_queries,
                    context,
                    goal,
                    intent,
                )
            else:
                result = [self._fallback_query(query, goal)]
            if not force_refresh:
                self._cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Query 改写失败: {e}")
            return [self._fallback_query(query, goal)]

    def _hyde_rewrite(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None,
        goal: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> List[str]:
        """
        HyDE 改写：生成假设答案，用假设答案检索

        Args:
            query: 原始查询

        Returns:
            List[str]: [原始查询, 假设答案]
        """
        if not self.llm_service:
            logger.warning("LLM 服务不可用，跳过 HyDE 改写")
            return [self._fallback_query(query, goal)]

        try:
            messages = self._assemble_messages(
                system_content=(
                    "你是一个知识库检索助手。根据会话上下文、用户当前问题和 Planner 目标，"
                    "先理解完整检索意图，再生成一个用于检索的假设性答案。\n"
                    "要求：\n"
                    "1. 解析代词、省略和承接关系，不得改变用户原意\n"
                    "2. 答案只是检索假设，不需要完全准确\n"
                    "3. 使用与问题相同的语言风格，并包含可能的关键词和概念\n"
                    "4. 长度适中（50-150字）"
                ),
                query=query,
                context=context,
                goal=goal,
                intent=intent,
                instruction="请生成一个假设性的答案：",
                output_token_reserve=200,
            )

            hypothetical_answer = self.llm_service.chat(
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                model=settings.llm_query_rewrite_model or settings.llm_model,
                trace_name="agent.query_rewrite.hyde",
            )

            logger.info(
                "HyDE 改写完成: 查询长度=%s，假设答案长度=%s",
                len(query),
                len(hypothetical_answer),
            )
            return [self._fallback_query(query, goal), hypothetical_answer]

        except Exception as e:
            logger.error(f"HyDE 改写失败: {e}")
            return [self._fallback_query(query, goal)]

    def _rag_fusion_rewrite(
        self,
        query: str,
        num_queries: int = 3,
        context: Optional[List[Dict[str, str]]] = None,
        goal: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> List[str]:
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
            return [self._fallback_query(query, goal)]

        try:
            messages = self._assemble_messages(
                system_content=(
                    f"你是一个查询改写助手。请基于会话上下文、用户当前问题和 Planner 目标，"
                    f"生成一个可独立理解的查询及 {num_queries} 个不同视角的检索查询。\n"
                    "要求：\n"
                    "1. standalone_query 必须消解代词、省略和承接关系，脱离对话也能理解\n"
                    "2. 保持用户核心意图，不补充上下文中不存在的事实\n"
                    "3. queries 从不同角度和粒度表达，并覆盖关键实体及同义词\n"
                    "4. 使用与用户当前问题相同的语言\n"
                    "5. 只返回 JSON 对象："
                    '{"standalone_query":"独立查询","queries":["查询1","查询2"]}'
                ),
                query=query,
                context=context,
                goal=goal,
                intent=intent,
                instruction="请完成指代消解并生成多视角检索查询：",
                output_token_reserve=300,
            )

            parameters = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "standalone_query": {"type": "string", "minLength": 1},
                    "queries": {
                        "type": "array",
                        "maxItems": max(1, num_queries),
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["standalone_query", "queries"],
            }
            structured_call = getattr(self.llm_service, "structured_call", None)
            if callable(structured_call):
                response = structured_call(
                    messages,
                    name="submit_query_rewrite",
                    description="提交独立查询和有限数量的检索变体",
                    parameters=parameters,
                    temperature=0.3,
                    max_tokens=300,
                    model=settings.llm_query_rewrite_model or settings.llm_model,
                    trace_name="agent.query_rewrite",
                )
            else:
                response = self.llm_service.chat_json(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=300,
                )

            standalone_query = str(
                response.get("standalone_query") or self._fallback_query(query, goal)
            ).strip()
            variants = response.get("queries", [])
            if not isinstance(variants, list):
                variants = []
            candidates = [standalone_query, *(str(item).strip() for item in variants)]
            result = list(dict.fromkeys(item for item in candidates if item))[:num_queries + 1]
            if not result:
                result = [self._fallback_query(query, goal)]

            logger.info("RAG-Fusion 改写完成: 改写数=%s", max(0, len(result) - 1))
            return result

        except Exception as e:
            logger.error(f"RAG-Fusion 改写失败: {e}")
            return [self._fallback_query(query, goal)]

    def _assemble_messages(
        self,
        *,
        system_content: str,
        query: str,
        context: Optional[List[Dict[str, str]]],
        goal: Optional[str],
        intent: Optional[str],
        instruction: str,
        output_token_reserve: int,
    ) -> List[Dict[str, str]]:
        """在统一 Token 预算内装配历史、Planner 目标与当前问题。"""
        task_lines = [f"用户当前问题：{query}"]
        if goal:
            task_lines.append(f"Planner 解析目标：{goal}")
        if intent:
            task_lines.append(f"Planner 结构化意图：{intent}")
        task_lines.append(instruction)
        assembled = self.context_assembler.assemble(
            system_messages=[{"role": "system", "content": system_content}],
            history=context,
            current_messages=[{"role": "user", "content": "\n".join(task_lines)}],
            output_token_reserve=output_token_reserve,
        )
        logger.info("Query 改写上下文装配统计: %s", assembled.stats())
        return assembled.messages

    @staticmethod
    def _fallback_query(query: str, goal: Optional[str]) -> str:
        """模型不可用时优先复用 Planner 已解析目标。"""
        return (goal or query).strip()

    @staticmethod
    def _cache_key(
        query: str,
        *,
        strategy: RewriteStrategy,
        num_queries: int,
        context: Optional[List[Dict[str, str]]],
        goal: Optional[str],
        intent: Optional[str],
    ) -> str:
        """根据语义输入生成稳定键，避免改写结果受调用进程状态影响。"""
        payload = {
            "query": query,
            "strategy": strategy.value,
            "num_queries": num_queries,
            "context": context if QueryRewriteService._query_needs_context(query) else [],
            "goal": goal or "",
            "intent": intent or "",
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _query_needs_context(query: str) -> bool:
        """只有存在明显指代时才把会话历史纳入缓存键。"""
        return any(
            marker in (query or "")
            for marker in ("它", "这个", "那个", "上面", "刚才", "前者", "后者", "该问题")
        )


# 全局实例
query_rewrite_service = None


def get_query_rewrite_service() -> QueryRewriteService:
    """获取 Query 改写服务实例（单例模式）"""
    global query_rewrite_service
    if query_rewrite_service is None:
        query_rewrite_service = QueryRewriteService()
    return query_rewrite_service

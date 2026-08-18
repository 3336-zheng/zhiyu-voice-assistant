"""
LangGraph 状态机 Agent（Agentic RAG 版本）
集成 Query 改写（P1-5）和 CRAG 检索后纠错（P1-6）
"""
import logging
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from backend.app.agent.models import Plan, ToolName
from backend.app.agent.executor import get_executor
from backend.app.agent.responder import get_responder
from backend.app.services.retrieval.query_rewrite_service import get_query_rewrite_service, RewriteStrategy
from backend.app.services.retrieval.crag_grader_service import get_crag_grader_service, RelevanceGrade
from backend.app.services.retrieval.evidence_service import assess_evidence
from backend.app.core.config import settings
from backend.app.core.observability import record_timing, timed_stage
from backend.app.agent.events import AgentEventType, AgentRunCancelled

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Agent 状态定义"""
    # 输入
    query: str
    session_id: Optional[str]
    context: List[Dict[str, str]]

    # 中间状态
    plan: Optional[Plan]
    retrieval_query: Optional[str]  # 消歧后供检索、Rerank 和 CRAG 共用的独立查询
    rewritten_queries: List[str]  # P1-5: 改写后的查询
    search_results: Optional[List[Dict]]
    execution_results: Optional[Dict[str, Any]]
    relevance_grade: Optional[str]  # P1-6: CRAG 评分结果
    refined_content: Optional[str]  # P1-6: 精炼后的内容
    retrieval_stats: Optional[Dict[str, Any]]

    # 输出
    answer: Optional[str]
    sources: Optional[List[Dict]]
    confidence: float
    evidence_status: str
    evidence_score: Optional[float]
    evidence_source_count: int
    evidence_reason: Optional[str]

    # 控制
    iter_count: int
    max_iterations: int
    error: Optional[str]
    event_callback: Optional[Any]
    cancel_check: Optional[Any]
    token_callback: Optional[Any]


def _raise_if_cancelled(state: AgentState) -> None:
    cancel_check = state.get("cancel_check")
    if cancel_check and cancel_check():
        raise AgentRunCancelled("Agent 运行已取消")


def _emit_event(
    state: AgentState,
    event_type: AgentEventType,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    callback = state.get("event_callback")
    if callback:
        callback(event_type.value, data or {})


def _emit_stage(
    state: AgentState,
    event_type: AgentEventType,
    stage: str,
    status: Optional[str] = None,
    **extra: Any,
) -> None:
    data = {"stage": stage, **extra}
    if status:
        data["status"] = status
    _emit_event(state, event_type, data)


def query_rewrite_node(state: AgentState) -> AgentState:
    """
    Query 改写节点（P1-5）：生成多个视角的查询

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] Query 改写节点：生成多个视角的查询...")
    _raise_if_cancelled(state)
    _emit_stage(state, AgentEventType.STAGE_STARTED, "agent.query_rewrite")
    rewrite_service = get_query_rewrite_service()

    query = state["query"]
    plan = state.get("plan")
    goal = plan.goal if plan else None
    intent = plan.intent.value if plan else None

    started = time.perf_counter()
    status = "completed"
    try:
        # 使用 RAG-Fusion 策略
        rewritten_queries = rewrite_service.rewrite_query(
            query,
            strategy=RewriteStrategy.RAG_FUSION,
            num_queries=3,
            context=state.get("context", []),
            goal=goal,
            intent=intent,
        )
        retrieval_query = rewritten_queries[0] if rewritten_queries else goal or query

        logger.info("[graph] Query 改写完成: 改写数=%s", len(rewritten_queries) - 1)
        return {
            **state,
            "retrieval_query": retrieval_query,
            "rewritten_queries": rewritten_queries,
            "iter_count": state.get("iter_count", 0) + 1,
        }

    except AgentRunCancelled:
        status = "cancelled"
        raise
    except Exception as e:
        status = "failed"
        logger.error(f"[graph] Query 改写失败: {e}")
        # 失败时优先使用 Planner 已解析目标
        return {
            **state,
            "retrieval_query": goal or query,
            "rewritten_queries": [goal or query],
            "iter_count": state.get("iter_count", 0) + 1,
        }
    finally:
        record_timing("agent.query_rewrite", (time.perf_counter() - started) * 1000)
        _emit_stage(
            state,
            AgentEventType.STAGE_COMPLETED,
            "agent.query_rewrite",
            status,
        )


def retrieve_node(state: AgentState) -> AgentState:
    """
    检索节点：执行混合检索（支持多查询）

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] 检索节点：执行混合检索...")
    _raise_if_cancelled(state)
    _emit_stage(state, AgentEventType.STAGE_STARTED, "agent.retrieve")
    executor = get_executor()

    rewritten_queries = state.get("rewritten_queries", [state["query"]])
    retrieval_query = state.get("retrieval_query") or state["query"]

    started = time.perf_counter()
    status = "completed"
    try:
        if settings.rag_v2_enabled:
            outcome = executor.hybrid_retrieval.search_multi(
                rewritten_queries,
                original_query=retrieval_query,
                top_k=settings.rag_final_top_k,
                token_budget=settings.rag_context_token_budget,
            )
            results = outcome["results"]
            logger.info(
                "[graph] RAG v2 检索完成: 查询数=%s, 结果数=%s",
                len(rewritten_queries),
                len(results),
            )
            return {
                **state,
                "search_results": results,
                "retrieval_stats": outcome["stats"],
            }

        all_results = []

        # 对每个改写后的查询执行检索
        for query in rewritten_queries:
            params = {"query": query, "top_k": 5}
            results = executor.search_knowledge_base(params, None)
            all_results.extend(results)

        # 去重（基于 content）
        seen_contents = set()
        unique_results = []
        for result in all_results:
            content = result.get("content", "")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append(result)

        # 按 rerank_score 排序，取 top 10
        unique_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        unique_results = unique_results[:10]

        logger.info(f"[graph] 检索完成: 查询数={len(rewritten_queries)}, 结果数={len(unique_results)}")
        return {**state, "search_results": unique_results}

    except AgentRunCancelled:
        status = "cancelled"
        raise
    except Exception as e:
        status = "failed"
        logger.error(f"[graph] 检索失败: {e}")
        return {**state, "error": str(e)}
    finally:
        record_timing("agent.retrieve", (time.perf_counter() - started) * 1000)
        _emit_stage(state, AgentEventType.STAGE_COMPLETED, "agent.retrieve", status)


def grade_node(state: AgentState) -> AgentState:
    """
    CRAG 评分节点（P1-6）：评估文档相关性

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] CRAG 评分节点：评估文档相关性...")
    _raise_if_cancelled(state)
    _emit_stage(state, AgentEventType.STAGE_STARTED, "agent.crag_grade")
    grader_service = get_crag_grader_service()

    query = state["query"]
    retrieval_query = state.get("retrieval_query") or query
    plan = state.get("plan")
    goal = plan.goal if plan else None
    intent = plan.intent.value if plan else None
    search_results = state.get("search_results", [])

    started = time.perf_counter()
    status = "completed"
    try:
        # 评估文档相关性
        grade_result = grader_service.grade_documents(
            query,
            search_results,
            retrieval_query=retrieval_query,
            goal=goal,
            intent=intent,
        )

        grade = grade_result["grade"]
        reasoning = grade_result["reasoning"]

        logger.info(f"[graph] CRAG 评分完成: grade={grade.value}, 理由='{reasoning[:50]}...'")

        # 如果是 ambiguous，执行知识精炼
        refined_content = None
        if grade == RelevanceGrade.AMBIGUOUS:
            refined_content = grader_service.refine_documents(
                query,
                search_results,
                retrieval_query=retrieval_query,
                goal=goal,
                intent=intent,
            )
            logger.info(f"[graph] 知识精炼完成: 长度={len(refined_content)}")

        return {
            **state,
            "relevance_grade": grade.value,
            "refined_content": refined_content
        }

    except AgentRunCancelled:
        status = "cancelled"
        raise
    except Exception as e:
        status = "failed"
        logger.error(f"[graph] CRAG 评分失败: {e}")
        # 失败时默认 ambiguous，触发重试
        return {**state, "relevance_grade": "ambiguous"}
    finally:
        record_timing("agent.crag_grade", (time.perf_counter() - started) * 1000)
        _emit_stage(state, AgentEventType.STAGE_COMPLETED, "agent.crag_grade", status)


def generate_node(state: AgentState) -> AgentState:
    """
    生成节点：生成答案

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] 生成节点：生成答案...")
    _raise_if_cancelled(state)
    _emit_stage(state, AgentEventType.STAGE_STARTED, "agent.generation")
    responder = get_responder()

    plan = state.get("plan")
    search_results = state.get("search_results")
    refined_content = state.get("refined_content")

    status = "completed"
    try:
        with timed_stage("agent.evidence"):
            evidence = assess_evidence(search_results)
        if evidence.status != "sufficient":
            return {
                **state,
                "answer": "现有 Wiki 中没有足够证据支持这个问题，暂时不生成推测性答案。",
                "sources": search_results or [],
                "confidence": 0.0,
                **evidence.as_dict(),
            }

        if not plan:
            status = "failed"
            return {**state, "error": "没有执行计划", "answer": "抱歉，无法处理您的请求。"}

        # 如果有精炼后的内容，使用精炼内容
        if refined_content:
            logger.info("[graph] 使用精炼后的内容生成答案")
            # 构建虚拟的检索结果
            search_results = [{"content": refined_content, "rerank_score": 1.0}]

        # 构建执行结果
        from backend.app.agent.models import ExecutionResult, ToolResult

        tool_results = []
        if search_results is not None:
            tool_results.append(ToolResult(
                step_id=1,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                success=True,
                result=search_results
            ))

        execution_result = ExecutionResult(
            plan=plan,
            results=tool_results,
            completed_steps=len(tool_results),
            total_steps=len(plan.steps),
            success=True,
            execution_log=[],
            final_data={"search_results": search_results} if search_results else {}
        )

        # 生成回复
        with timed_stage("agent.generation"):
            response_kwargs = {}
            if state.get("token_callback"):
                response_kwargs["token_callback"] = state["token_callback"]
            response = responder.generate_response(
                state["query"],
                plan,
                execution_result,
                state.get("context", []),
                **response_kwargs,
            )

        logger.info(f"[graph] 生成完成，答案长度: {len(response.response)}")
        return {
            **state,
            "answer": response.response,
            "sources": response.sources,
            "confidence": response.confidence,
            **evidence.as_dict(),
        }

    except AgentRunCancelled:
        status = "cancelled"
        raise
    except Exception as e:
        status = "failed"
        logger.error(f"[graph] 生成失败: {e}")
        return {**state, "error": str(e), "answer": f"抱歉，生成答案时出现错误：{str(e)}"}
    finally:
        _emit_stage(state, AgentEventType.STAGE_COMPLETED, "agent.generation", status)


def should_continue(state: AgentState) -> str:
    """
    条件边：判断是否继续循环

    Args:
        state: 当前状态

    Returns:
        str: 下一个节点名称
    """
    _raise_if_cancelled(state)

    # 检查是否有错误
    if state.get("error"):
        logger.info(f"[graph] 检测到错误，结束: {state['error']}")
        return "end"

    # 检查迭代次数
    if state.get("iter_count", 0) >= state.get("max_iterations", 5):
        logger.info("[graph] 达到最大迭代次数，结束")
        return "end"

    # 检查是否有答案
    if state.get("answer"):
        logger.info("[graph] 已生成答案，结束")
        return "end"

    # 检查 CRAG 评分
    grade = state.get("relevance_grade")
    if grade == "correct":
        logger.info("[graph] CRAG 评分 correct，继续生成")
        return "generate"
    elif grade == "ambiguous":
        logger.info("[graph] CRAG 评分 ambiguous，继续生成（使用精炼内容）")
        return "generate"
    elif grade == "incorrect":
        logger.info("[graph] CRAG 评分 incorrect，需要改写重检")
        return "rewrite"

    # 默认继续
    return "generate"


def create_agent_graph() -> StateGraph:
    """
    创建 Agent 图状态机（Agentic RAG 版本）

    Returns:
        StateGraph: 编译后的图
    """
    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("rewrite", query_rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)

    # 设置入口
    workflow.set_entry_point("rewrite")

    # 添加边
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # 添加条件边：根据 CRAG 评分决定下一步
    workflow.add_conditional_edges(
        "grade",
        should_continue,
        {
            "generate": "generate",
            "rewrite": "rewrite",  # 循环回到改写
            "end": END
        }
    )

    # 添加条件边：生成后判断是否结束
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {
            "end": END,
            "generate": "generate",  # 继续生成
            "rewrite": "rewrite"  # 循环回到改写
        }
    )

    # 编译图
    graph = workflow.compile()
    logger.info("[graph] Agent 图状态机编译完成（Agentic RAG 版本）")

    return graph


# 全局图实例
agent_graph = None


def get_agent_graph() -> StateGraph:
    """获取 Agent 图实例（单例模式）"""
    global agent_graph
    if agent_graph is None:
        agent_graph = create_agent_graph()
    return agent_graph

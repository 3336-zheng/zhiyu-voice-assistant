"""
LangGraph 状态机 Agent
将手写 Plan-and-Execute 重构为图状态机，支持条件分支和循环
"""
import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from backend.app.agent.models import IntentType, Plan, PlanStep, ToolName
from backend.app.agent.planner import get_planner
from backend.app.agent.executor import get_executor
from backend.app.agent.responder import get_responder

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Agent 状态定义"""
    # 输入
    query: str
    session_id: Optional[str]
    context: List[Dict[str, str]]

    # 中间状态
    plan: Optional[Plan]
    search_results: Optional[List[Dict]]
    execution_results: Optional[Dict[str, Any]]

    # 输出
    answer: Optional[str]
    sources: Optional[List[Dict]]
    confidence: float

    # 控制
    iter_count: int
    max_iterations: int
    error: Optional[str]


def route_node(state: AgentState) -> AgentState:
    """
    路由节点：分析意图，生成计划

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] 路由节点：分析意图...")
    planner = get_planner()

    try:
        plan = planner.plan(state["query"], state.get("context"))
        logger.info(f"[graph] 意图识别完成：{plan.intent.value}, 步骤数={len(plan.steps)}")
        return {**state, "plan": plan}
    except Exception as e:
        logger.error(f"[graph] 意图识别失败: {e}")
        return {**state, "error": str(e)}


def retrieve_node(state: AgentState) -> AgentState:
    """
    检索节点：执行混合检索

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] 检索节点：执行混合检索...")
    executor = get_executor()

    plan = state.get("plan")
    if not plan:
        return {**state, "error": "没有执行计划"}

    try:
        # 找到检索步骤
        search_step = None
        for step in plan.steps:
            if step.tool_name == ToolName.SEARCH_KNOWLEDGE_BASE:
                search_step = step
                break

        if search_step:
            # 执行检索
            results = executor.search_knowledge_base(search_step.parameters, None)
            logger.info(f"[graph] 检索完成，结果数: {len(results)}")
            return {**state, "search_results": results}
        else:
            logger.info("[graph] 无需检索，跳过")
            return {**state, "search_results": []}

    except Exception as e:
        logger.error(f"[graph] 检索失败: {e}")
        return {**state, "error": str(e)}


def grade_node(state: AgentState) -> AgentState:
    """
    评分节点：CRAG 相关性评分（占位，P1-6 实现）

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] 评分节点：CRAG 相关性评分（占位）...")

    # TODO: P1-6 实现 CRAG 打分逻辑
    # 当前直接返回，不做评分
    return state


def generate_node(state: AgentState) -> AgentState:
    """
    生成节点：生成答案

    Args:
        state: 当前状态

    Returns:
        AgentState: 更新后的状态
    """
    logger.info("[graph] 生成节点：生成答案...")
    responder = get_responder()

    plan = state.get("plan")
    search_results = state.get("search_results")

    if not plan:
        return {**state, "error": "没有执行计划", "answer": "抱歉，无法处理您的请求。"}

    try:
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
        response = responder.generate_response(
            state["query"],
            plan,
            execution_result,
            state.get("context", [])
        )

        logger.info(f"[graph] 生成完成，答案长度: {len(response.response)}")
        return {
            **state,
            "answer": response.response,
            "sources": response.sources,
            "confidence": response.confidence
        }

    except Exception as e:
        logger.error(f"[graph] 生成失败: {e}")
        return {**state, "error": str(e), "answer": f"抱歉，生成答案时出现错误：{str(e)}"}


def should_continue(state: AgentState) -> str:
    """
    条件边：判断是否继续循环

    Args:
        state: 当前状态

    Returns:
        str: 下一个节点名称
    """
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

    # 默认继续
    return "continue"


def create_agent_graph() -> StateGraph:
    """
    创建 Agent 图状态机

    Returns:
        StateGraph: 编译后的图
    """
    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("generate", generate_node)

    # 设置入口
    workflow.set_entry_point("route")

    # 添加边
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")

    # 添加条件边：生成后判断是否结束
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {
            "end": END,
            "continue": "route"  # 循环回到路由（用于 CRAG 重检）
        }
    )

    # 编译图
    graph = workflow.compile()
    logger.info("[graph] Agent 图状态机编译完成")

    return graph


# 全局图实例
agent_graph = None


def get_agent_graph() -> StateGraph:
    """获取 Agent 图实例（单例模式）"""
    global agent_graph
    if agent_graph is None:
        agent_graph = create_agent_graph()
    return agent_graph

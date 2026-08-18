"""简单只读知识查询的确定性快速路径。"""

from __future__ import annotations

import re
from typing import List, Dict, Optional

from backend.app.agent.models import IntentType, Plan, PlanStep, ToolName
from backend.app.core.config import settings


QUESTION_MARKERS = (
    "什么",
    "为何",
    "为什么",
    "怎么",
    "如何",
    "哪些",
    "有哪些",
    "是否",
    "能否",
    "区别",
    "优势",
    "作用",
    "流程",
    "原理",
    "介绍",
    "解释",
    "含义",
    "是什么",
)
MUTATION_MARKERS = re.compile(
    r"创建|新增|更新|修改|删除|写入|保存|记录|导出|上传|重命名|替换|清空|取消|确认|总结|归纳|生成复习|整理成"
)
COMPLEX_MARKERS = re.compile(
    r"比较|对比|分别|同时|以及|并且|结合|如果|然后|优缺点|差异|完整流程|详细说明"
)


def is_fast_path_query(
    query: str,
    context: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """判断是否为无指代、无副作用的单轮知识查询。"""
    if not settings.fast_path_enabled or context:
        return False
    normalized = re.sub(r"\s+", "", query or "")
    if not normalized or len(normalized) > settings.fast_path_max_query_chars:
        return False
    if MUTATION_MARKERS.search(normalized):
        return False
    if COMPLEX_MARKERS.search(normalized):
        return False
    return normalized.endswith(("?", "？")) or any(
        marker in normalized for marker in QUESTION_MARKERS
    )


def build_fast_search_plan(query: str) -> Plan:
    """创建不经过 Planner 的只读检索计划。"""
    return Plan(
        intent=IntentType.SEARCH,
        original_query=query,
        steps=[
            PlanStep(
                step_id=1,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                parameters={"query": query, "top_k": settings.rag_final_top_k},
                description="快速路径：直接检索本地知识库",
            )
        ],
        estimated_steps=1,
        reasoning="单轮无指代只读知识查询，跳过 Planner 和 Query Rewrite",
        goal=query,
    )

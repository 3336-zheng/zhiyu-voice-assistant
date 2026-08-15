"""能力注册表驱动的受限 Plan-and-Execute 规划器。"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from backend.app.agent.models import (
    CreateNoteParameters,
    IntentType,
    Plan,
    PlanStep,
    SearchParameters,
    ToolCapability,
    ToolName,
    UpdateNoteParameters,
)
from backend.app.agent.plan_policy import PlanPolicy, PlanValidationError
from backend.app.agent.tool_registry import AgentToolRegistry
from backend.app.core.config import settings
from backend.app.services.memory.context_assembler import ContextAssembler

logger = logging.getLogger(__name__)

PLAN_GENERATION_PROMPT = """你是本地优先 AI Wiki 的 Planner。把用户目标拆成最少且充分的工具步骤。

约束：
1. 只能使用下方“当前允许工具”中的工具，不得虚构工具或参数。
2. 用户输入可能来自语音识别；可以修正明显同音字和口语冗余，但不能改变原意。
3. 简单任务只生成一个步骤；只有确实依赖上一步结果时才拆成多步，最多 {max_steps} 步。
4. 后续步骤可用字符串 `$step_N_results` 引用上游完整结果，并必须在 depends_on 中声明 N。
5. 不得自行联网。外部 MCP 研究由用户在独立入口显式触发，不属于本计划工具。
6. 创建、更新、删除是高风险能力。仅当用户明确要求变更知识库时规划，执行前由后端统一确认。
7. 信息不足时不要猜测页面标识或正文；宁可让参数校验失败并回退，也不要编造。
8. intent 只能是 search、create_note、update_note、delete_note、list_notes、time_query、summarize、unknown。

当前允许工具：
{capabilities}

仅返回 JSON 对象：
{{
  "goal": "用户希望达成的结果",
  "intent": "上述 intent 之一",
  "steps": [
    {{
      "step_id": 1,
      "tool_name": "工具名",
      "parameters": {{}},
      "description": "面向用户的步骤说明",
      "depends_on": [],
      "expected_output": "预期产物",
      "success_criteria": "可验证的最低成功条件"
    }}
  ],
  "reasoning": "简短说明为什么选择这些步骤"
}}
"""


class Planner:
    """动态生成结构化计划，并保留确定性规则降级。"""

    def __init__(
        self,
        tool_registry: Optional[AgentToolRegistry] = None,
        validator: Optional[PlanPolicy] = None,
    ):
        """初始化规划器"""
        self._llm_service = None
        self.tool_registry = tool_registry or AgentToolRegistry()
        self.validator = validator or PlanPolicy(self.tool_registry)
        self.context_assembler = ContextAssembler(
            context_window_tokens=settings.llm_context_window_tokens,
            history_token_budget=settings.memory_context_token_budget,
            summary_token_budget=settings.memory_summary_token_budget,
        )

    @property
    def llm_service(self):
        """延迟加载 LLM 服务"""
        if self._llm_service is None:
            try:
                from backend.app.services.ai.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            except Exception as e:
                logger.warning(f"LLM 服务加载失败，将使用规则匹配: {e}")
        return self._llm_service

    def plan(
        self,
        user_query: str,
        context: Optional[List[Dict[str, str]]] = None,
        capabilities: Optional[Sequence[ToolCapability]] = None,
    ) -> Plan:
        """
        分析用户意图并生成执行计划

        Args:
            user_query: 用户查询
            context: 对话上下文（可选）

        Returns:
            Plan: 执行计划
        """
        query = user_query.strip()

        active_capabilities = list(
            capabilities if capabilities is not None else self.validator.capabilities()
        )
        allowed_tools = [item.name for item in active_capabilities]

        if self.llm_service:
            try:
                plan = self._plan_with_llm(query, context, active_capabilities)
                return self.validator.validate(plan, allowed_tools)
            except Exception as e:
                logger.warning("LLM 计划生成失败，降级到规则匹配: %s", e)

        fallback_plan = self._plan_with_rules(query)
        return self.validator.validate(fallback_plan, allowed_tools)

    def replan(
        self,
        user_query: str,
        previous_plan: Plan,
        execution_feedback: Dict[str, Any],
        *,
        context: Optional[List[Dict[str, str]]] = None,
        capabilities: Optional[Sequence[ToolCapability]] = None,
        remaining_steps: Optional[int] = None,
    ) -> Plan:
        """根据失败观察生成一次新计划；无模型时不做伪重规划。"""
        if not self.llm_service:
            raise PlanValidationError("LLM 不可用，无法进行重规划")
        active_capabilities = list(
            capabilities if capabilities is not None else self.validator.capabilities()
        )
        feedback = {
            "previous_plan": previous_plan.model_dump(mode="json"),
            "execution_feedback": execution_feedback,
            "remaining_step_budget": remaining_steps,
            "instruction": "不要重复失败的调用结构，优先减少步骤或更换可行的只读路径。",
        }
        plan = self._plan_with_llm(
            user_query,
            context,
            active_capabilities,
            replan_feedback=feedback,
        )
        validated = self.validator.validate(plan, [item.name for item in active_capabilities])
        if remaining_steps is not None and len(validated.steps) > remaining_steps:
            raise PlanValidationError(
                f"重规划步骤数 {len(validated.steps)} 超过剩余预算 {remaining_steps}"
            )
        return validated

    def _plan_with_llm(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]],
        capabilities: Sequence[ToolCapability],
        replan_feedback: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        """让 LLM 基于当前能力目录生成完整计划。"""
        now = datetime.now()
        date_info = f"当前日期: {now.strftime('%Y-%m-%d')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"
        capability_catalog = [
            capability.model_dump(mode="json")
            for capability in capabilities
        ]
        prompt = PLAN_GENERATION_PROMPT.format(
            max_steps=self.validator.max_steps,
            capabilities=json.dumps(capability_catalog, ensure_ascii=False),
        )
        system_messages = [{"role": "system", "content": f"{prompt}\n{date_info}"}]
        if replan_feedback:
            system_messages.append(
                {
                    "role": "system",
                    "content": "这是一次受限重规划。执行观察如下：\n"
                    + json.dumps(replan_feedback, ensure_ascii=False),
                }
            )
        assembled = self.context_assembler.assemble(
            system_messages=system_messages,
            history=context,
            current_messages=[{"role": "user", "content": query}],
            output_token_reserve=settings.llm_max_tokens,
        )
        messages = assembled.messages
        logger.info("[planner] 上下文装配统计: %s", assembled.stats())
        result = self.llm_service.chat_json(messages=messages, temperature=0.1)
        steps = result.get("steps")
        if not isinstance(steps, list):
            raise PlanValidationError("LLM 未返回 steps 数组")
        plan = Plan.model_validate(
            {
                "goal": result.get("goal") or query,
                "intent": result.get("intent", IntentType.UNKNOWN.value),
                "original_query": query,
                "steps": steps,
                "estimated_steps": len(steps),
                "reasoning": result.get("reasoning", "LLM 基于当前工具能力生成计划"),
            }
        )
        logger.info(
            "[planner] 生成计划: intent=%s, steps=%s, tools=%s",
            plan.intent.value,
            len(plan.steps),
            [step.tool_name.value for step in plan.steps],
        )
        return plan

    def _plan_with_rules(self, query: str) -> Plan:
        """
        使用规则匹配进行意图识别（降级方案）

        Args:
            query: 用户查询

        Returns:
            Plan: 执行计划
        """
        intent = self._recognize_intent(query)

        if intent == IntentType.SEARCH:
            return self._plan_search(query)
        elif intent == IntentType.CREATE_NOTE:
            return self._plan_create_note(query)
        elif intent == IntentType.UPDATE_NOTE:
            return self._plan_update_note(query)
        elif intent == IntentType.DELETE_NOTE:
            return self._plan_delete_note(query)
        elif intent == IntentType.LIST_NOTES:
            return self._plan_list_notes(query)
        elif intent == IntentType.TIME_QUERY:
            return self._plan_time_query(query)
        elif intent == IntentType.SUMMARIZE:
            return self._plan_summarize(query)
        else:
            return self._plan_search(query)

    def _recognize_intent(self, query: str) -> IntentType:
        """使用规则识别意图（课堂学习场景，7种核心意图）"""
        query_lower = query.lower()

        # 明确的摘要请求优先于泛化的“复习”检索请求。
        summarize_patterns = [r"总结.*", r"摘要.*", r"概括.*", r"归纳.*", r"复习卡片", r"生成卡片"]
        for pattern in summarize_patterns:
            if re.search(pattern, query_lower):
                return IntentType.SUMMARIZE

        # 检索/问答意图（优先于创建笔记，避免 "搜索会议记录" 被误判）
        search_patterns = [r"^搜索", r"^查找", r"^查一下", r"^找一下", r"^看看", r"是什么", r"有哪些", r"复习", r"什么是"]
        for pattern in search_patterns:
            if re.search(pattern, query_lower):
                return IntentType.SEARCH

        # 笔记CRUD意图
        create_patterns = [r"创建笔记", r"新建笔记", r"添加笔记", r"^记录.*", r"^记一下.*"]
        for pattern in create_patterns:
            if re.search(pattern, query_lower):
                return IntentType.CREATE_NOTE

        update_patterns = [r"更新笔记", r"修改笔记", r"编辑笔记"]
        for pattern in update_patterns:
            if re.search(pattern, query_lower):
                return IntentType.UPDATE_NOTE

        delete_patterns = [r"删除笔记", r"移除笔记", r"删掉.*"]
        for pattern in delete_patterns:
            if re.search(pattern, query_lower):
                return IntentType.DELETE_NOTE

        list_patterns = [r"列出.*笔记", r"显示.*笔记", r"所有笔记", r"笔记列表", r"有哪些笔记"]
        for pattern in list_patterns:
            if re.search(pattern, query_lower):
                return IntentType.LIST_NOTES

        # 时间查询意图
        time_patterns = [r"现在.*时间", r"今天.*日期", r"几点.*", r"星期.*"]
        for pattern in time_patterns:
            if re.search(pattern, query_lower):
                return IntentType.TIME_QUERY

        # 默认为检索/问答意图
        return IntentType.SEARCH

    def _plan_search(self, query: str) -> Plan:
        """生成检索计划"""
        params = self._parse_search_params(query)
        return Plan(
            intent=IntentType.SEARCH,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                parameters=params.dict(),
                description=f"检索知识库: '{params.query}'"
            )],
            estimated_steps=1,
            reasoning=f"用户查询'{query}'被识别为检索意图。"
        )

    def _plan_create_note(self, query: str) -> Plan:
        """生成创建笔记计划"""
        params = self._parse_create_note_params(query)
        return Plan(
            intent=IntentType.CREATE_NOTE,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.CREATE_NOTE,
                parameters=params.dict(),
                description=f"创建笔记: {params.title}"
            )],
            estimated_steps=1,
            reasoning=f"用户意图是创建新笔记，标题为'{params.title}'。"
        )

    def _plan_update_note(self, query: str) -> Plan:
        """生成更新笔记计划"""
        params = self._parse_update_note_params(query)
        return Plan(
            intent=IntentType.UPDATE_NOTE,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.UPDATE_NOTE,
                parameters=params.dict(),
                description=f"更新笔记: {params.filename}"
            )],
            estimated_steps=1,
            reasoning=f"用户意图是更新笔记 {params.filename}。"
        )

    def _plan_delete_note(self, query: str) -> Plan:
        """生成删除笔记计划"""
        filename = self._extract_filename(query)
        return Plan(
            intent=IntentType.DELETE_NOTE,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.DELETE_NOTE,
                parameters={"filename": filename},
                description=f"删除笔记: {filename}" if filename else "删除笔记"
            )],
            estimated_steps=1,
            reasoning="用户意图是删除笔记。"
        )

    def _plan_list_notes(self, query: str) -> Plan:
        """生成列出笔记计划"""
        date_from, date_to = self._extract_date_range(query)
        return Plan(
            intent=IntentType.LIST_NOTES,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.LIST_NOTES,
                parameters={
                    "date_from": date_from.isoformat() if date_from else None,
                    "date_to": date_to.isoformat() if date_to else None,
                    "limit": 20
                },
                description="列出笔记"
            )],
            estimated_steps=1,
            reasoning="用户意图是列出笔记。"
        )

    def _plan_time_query(self, query: str) -> Plan:
        """生成时间查询计划"""
        return Plan(
            intent=IntentType.TIME_QUERY,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.GET_CURRENT_TIME,
                parameters={},
                description="获取当前时间"
            )],
            estimated_steps=1,
            reasoning="用户意图是查询当前时间或日期。"
        )

    def _plan_summarize(self, query: str) -> Plan:
        """生成摘要计划"""
        topic = re.sub(r"(总结|摘要|概括|归纳|提炼)", "", query).strip()
        return Plan(
            intent=IntentType.SUMMARIZE,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                parameters={"query": topic, "top_k": 10},
                description=f"检索相关内容: '{topic}'"
            )],
            estimated_steps=2,
            reasoning=f"用户意图是总结'{topic}'相关内容。"
        )

    def _parse_search_params(self, query: str) -> SearchParameters:
        """解析检索参数"""
        date_from, date_to = self._extract_date_range(query)
        tag_filter = self._extract_tag_filter(query)

        clean_query = query
        if date_from or date_to:
            clean_query = re.sub(r"(上|本|这|下)?(周|月|年)\d{0,2}", "", clean_query)
            clean_query = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{0,2}", "", clean_query)
        if tag_filter:
            clean_query = re.sub(rf"标签[是为]?{tag_filter}", "", clean_query)

        clean_query = clean_query.strip() or query

        return SearchParameters(
            query=clean_query,
            top_k=5,
            tag_filter=tag_filter,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None
        )

    def _parse_create_note_params(self, query: str) -> CreateNoteParameters:
        """解析创建笔记参数"""
        title_match = re.search(r"标题[是为]([^，。,.]+)", query)
        title = title_match.group(1).strip() if title_match else query[:20] + ("..." if len(query) > 20 else "")

        content_match = re.search(r"内容[是为](.+)", query, re.DOTALL)
        content = content_match.group(1).strip() if content_match else query

        return CreateNoteParameters(title=title, content=content, tags=self._extract_tags(query))

    def _parse_update_note_params(self, query: str) -> UpdateNoteParameters:
        """解析更新笔记参数"""
        filename = self._extract_filename(query) or ""
        params = {"filename": filename}

        title_match = re.search(r"标题[改设置为]([^，。,.]+)", query)
        if title_match:
            params["title"] = title_match.group(1).strip()

        content_match = re.search(r"内容[改设置为](.+)", query, re.DOTALL)
        if content_match:
            params["content"] = content_match.group(1).strip()

        tags = self._extract_tags(query)
        if tags:
            params["tags"] = tags

        return UpdateNoteParameters(**params)

    def _extract_filename(self, query: str) -> Optional[str]:
        """提取笔记文件名（从用户查询中）"""
        # 匹配"笔记xxx"或"文件xxx"格式
        patterns = [
            r'(?:笔记|文件)\s*[名叫是为：:]\s*[「"“]?([^」"”，。,.]+)',
            r'(?:笔记|文件)\s+(.+?)(?:\s|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                name = match.group(1).strip()
                # 过滤掉无意义的词
                if name and name not in ("的", "了", "吗", "吧", "呢"):
                    return name if name.endswith(".md") else name
        return None

    def _extract_date_range(self, query: str):
        """提取日期范围"""
        today = datetime.now()
        query_lower = query.lower()

        if "今天" in query_lower:
            return today.replace(hour=0, minute=0, second=0, microsecond=0), \
                   today.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif "昨天" in query_lower:
            yesterday = today - timedelta(days=1)
            return yesterday.replace(hour=0, minute=0, second=0, microsecond=0), \
                   yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif "本周" in query_lower or "这星期" in query_lower:
            monday = today - timedelta(days=today.weekday())
            return monday.replace(hour=0, minute=0, second=0, microsecond=0), \
                   today.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif "上周" in query_lower or "上星期" in query_lower:
            last_monday = today - timedelta(days=today.weekday() + 7)
            last_sunday = last_monday + timedelta(days=6)
            return last_monday.replace(hour=0, minute=0, second=0, microsecond=0), \
                   last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif "本月" in query_lower or "这个月" in query_lower:
            return today.replace(day=1, hour=0, minute=0, second=0, microsecond=0), \
                   today.replace(hour=23, minute=59, second=59, microsecond=999999)

        return None, None

    def _extract_tag_filter(self, query: str) -> Optional[str]:
        """提取标签过滤"""
        patterns = [r'标签[是为]\s*["“]?([^"“\s]+)["”]?', r'#([^\s#，。,.]+)']
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        return None

    def _extract_tags(self, query: str) -> list:
        """提取标签列表"""
        tags = []
        tag_matches = re.findall(r"#([^\s#，。,.]+)", query)
        tags.extend(tag_matches)
        return list(set(tags)) if tags else None


# 全局规划器实例
planner_instance = None


def get_planner() -> Planner:
    """获取规划器实例（单例模式）"""
    global planner_instance
    if planner_instance is None:
        planner_instance = Planner()
    return planner_instance

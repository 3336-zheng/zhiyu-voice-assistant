"""
Plan-and-Execute Agent - Planner 规划器（课堂学习场景聚焦版）
支持 LLM 增强的意图识别和计划生成
"""
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from backend.app.agent.models import (
    Plan, PlanStep, ToolName, IntentType,
    SearchParameters, CreateNoteParameters, UpdateNoteParameters
)

logger = logging.getLogger(__name__)

# LLM 意图识别的系统提示（课堂学习场景聚焦版，7种意图）
INTENT_RECOGNITION_PROMPT = """你是一个课堂学习助手的意图识别模块。根据学生的查询，识别其意图并提取参数。

重要：学生的输入可能来自语音识别（ASR），包含同音字替代、错别字、口语化表达。你必须：
1. 修正语音识别错误（如"词类"→"词语"、"分快"→"分块"、"走"→"找"等同音字/近音字）
2. 去除口语化冗余词（如"帮我"、"查一下"、"看看"），提取核心检索关键词
3. 将修正后的关键词填入 query 字段

示例：
- 学生说"帮我查走RAG的词类方式分快" → query: "RAG分块方式"
- 学生说"找一下有关向量数据库的笔" → query: "向量数据库"
- 学生说"看看那个agent开发的坑" → query: "Agent开发踩坑"

支持的意图类型（7种核心意图）：
- search: 检索知识库/问答复习（如"查找关于RAG的笔记"、"什么是向量数据库"、"复习一下第3讲内容"）
- create_note: 创建笔记（如"创建笔记标题是XXX内容是YYY"、"记录一下..."）
- update_note: 更新笔记（如"更新笔记xxx的内容"、"修改笔记..."），需要 filename 参数
- delete_note: 删除笔记（如"删除笔记xxx"、"删掉这条笔记"），需要 filename 参数
- list_notes: 列出笔记（如"显示所有笔记"、"列出本周的笔记"、"有哪些笔记"）
- time_query: 时间查询（如"现在几点"、"今天星期几"）
- summarize: 摘要总结/生成复习卡片（如"总结关于RAG的内容"、"生成复习卡片"、"概括一下这节课的重点"）

重要：知识统一保存为 Wiki 页面。更新、删除时，filename 参数可以填写页面 UUID、标题或唯一别名；优先使用页面 UUID。

请以JSON格式返回：
{
    "intent": "意图类型",
    "confidence": 0.95,
    "parameters": {
        "query": "修正语音错误后的核心检索关键词（去除口语冗余）",
        "original_query_cleaned": "修正语音错误但保留完整语义的查询",
        "title": "笔记标题（如果是创建/更新）",
        "content": "笔记内容（如果是创建/更新）",
        "filename": "页面 UUID、标题或唯一别名（更新/删除时必须）",
        "tags": ["标签1", "标签2"]
    },
    "reasoning": "识别理由，说明做了哪些语音纠错"
}

注意：
1. 如果学生没有明确指定参数，不要猜测
2. 对于时间相关查询，根据当前日期推算具体日期
3. confidence 表示识别的置信度（0-1）"""


class Planner:
    """
    Agent 规划器
    支持 LLM 增强的意图识别，同时保留规则匹配作为降级方案
    """

    def __init__(self):
        """初始化规划器"""
        self._llm_service = None

    @property
    def llm_service(self):
        """延迟加载 LLM 服务"""
        if self._llm_service is None:
            try:
                from backend.app.services.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            except Exception as e:
                logger.warning(f"LLM 服务加载失败，将使用规则匹配: {e}")
        return self._llm_service

    def plan(self, user_query: str, context: List[Dict[str, str]] = None) -> Plan:
        """
        分析用户意图并生成执行计划

        Args:
            user_query: 用户查询
            context: 对话上下文（可选）

        Returns:
            Plan: 执行计划
        """
        query = user_query.strip()

        # 优先使用 LLM 进行意图识别
        if self.llm_service:
            try:
                return self._plan_with_llm(query, context)
            except Exception as e:
                logger.warning(f"LLM 意图识别失败，降级到规则匹配: {e}")

        # 降级到规则匹配
        return self._plan_with_rules(query)

    def _plan_with_llm(self, query: str, context: List[Dict[str, str]] = None) -> Plan:
        """
        使用 LLM 进行意图识别和计划生成

        Args:
            query: 用户查询
            context: 对话上下文

        Returns:
            Plan: 执行计划
        """
        # 构建消息
        now = datetime.now()
        date_info = f"当前日期: {now.strftime('%Y-%m-%d')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"

        messages = [
            {"role": "system", "content": INTENT_RECOGNITION_PROMPT + f"\n{date_info}"}
        ]

        # 添加对话上下文
        if context:
            for msg in context[-6:]:  # 最近 3 轮对话
                messages.append(msg)

        messages.append({"role": "user", "content": query})

        # 调用 LLM
        result = self.llm_service.chat_json(messages=messages, temperature=0.1)

        # 解析结果
        intent_str = result.get("intent", "search")
        parameters = result.get("parameters", {})
        reasoning = result.get("reasoning", "")
        logger.info(f"[planner] LLM 返回: intent={intent_str}, query={parameters.get('query', '')}, content_len={len(parameters.get('content', ''))}")

        # 映射意图类型（7种核心意图）
        intent_map = {
            "search": IntentType.SEARCH,
            "create_note": IntentType.CREATE_NOTE,
            "update_note": IntentType.UPDATE_NOTE,
            "delete_note": IntentType.DELETE_NOTE,
            "list_notes": IntentType.LIST_NOTES,
            "time_query": IntentType.TIME_QUERY,
            "summarize": IntentType.SUMMARIZE,
        }
        intent = intent_map.get(intent_str, IntentType.SEARCH)

        # 根据意图生成计划
        if intent == IntentType.SEARCH:
            return self._plan_search_from_llm(query, parameters, reasoning)
        elif intent == IntentType.CREATE_NOTE:
            return self._plan_create_note_from_llm(query, parameters, reasoning)
        elif intent == IntentType.UPDATE_NOTE:
            return self._plan_update_note_from_llm(query, parameters, reasoning)
        elif intent == IntentType.DELETE_NOTE:
            return self._plan_delete_note_from_llm(query, parameters, reasoning)
        elif intent == IntentType.LIST_NOTES:
            return self._plan_list_notes_from_llm(query, parameters, reasoning)
        elif intent == IntentType.TIME_QUERY:
            return self._plan_time_query(query)
        elif intent == IntentType.SUMMARIZE:
            return self._plan_summarize_from_llm(query, parameters, reasoning)
        else:
            return self._plan_search_from_llm(query, parameters, reasoning)

    def _plan_search_from_llm(self, query: str, params: Dict, reasoning: str) -> Plan:
        """从 LLM 结果生成检索计划"""
        search_params = SearchParameters(
            query=params.get("query", query),
            top_k=params.get("top_k", 5),
            tag_filter=params.get("tag_filter"),
            date_from=params.get("date_from"),
            date_to=params.get("date_to")
        )

        return Plan(
            intent=IntentType.SEARCH,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.SEARCH_KNOWLEDGE_BASE,
                parameters=search_params.dict(),
                description=f"检索知识库: '{search_params.query}'"
            )],
            estimated_steps=1,
            reasoning=reasoning or f"用户查询'{query}'被识别为检索意图。"
        )

    def _plan_create_note_from_llm(self, query: str, params: Dict, reasoning: str) -> Plan:
        """从 LLM 结果生成创建笔记计划"""
        create_params = CreateNoteParameters(
            title=params.get("title", query[:20]),
            content=params.get("content", query),
            tags=params.get("tags")
        )

        return Plan(
            intent=IntentType.CREATE_NOTE,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.CREATE_NOTE,
                parameters=create_params.dict(),
                description=f"创建笔记: {create_params.title}"
            )],
            estimated_steps=1,
            reasoning=reasoning or f"用户意图是创建新笔记。"
        )

    def _plan_update_note_from_llm(self, query: str, params: Dict, reasoning: str) -> Plan:
        """从 LLM 结果生成更新笔记计划"""
        filename = params.get("filename", "")
        update_params = UpdateNoteParameters(
            filename=filename,
            title=params.get("title"),
            content=params.get("content"),
            tags=params.get("tags")
        )

        return Plan(
            intent=IntentType.UPDATE_NOTE,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.UPDATE_NOTE,
                parameters=update_params.dict(),
                description=f"更新笔记: {filename}"
            )],
            estimated_steps=1,
            reasoning=reasoning or f"用户意图是更新笔记。"
        )

    def _plan_delete_note_from_llm(self, query: str, params: Dict, reasoning: str) -> Plan:
        """从 LLM 结果生成删除笔记计划"""
        filename = params.get("filename", "")

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
            reasoning=reasoning or "用户意图是删除笔记。"
        )

    def _plan_list_notes_from_llm(self, query: str, params: Dict, reasoning: str) -> Plan:
        """从 LLM 结果生成列出笔记计划"""
        return Plan(
            intent=IntentType.LIST_NOTES,
            original_query=query,
            steps=[PlanStep(
                step_id=1,
                tool_name=ToolName.LIST_NOTES,
                parameters={
                    "date_from": params.get("date_from"),
                    "date_to": params.get("date_to"),
                    "limit": 20
                },
                description="列出笔记"
            )],
            estimated_steps=1,
            reasoning=reasoning or "用户意图是列出笔记。"
        )

    def _plan_summarize_from_llm(self, query: str, params: Dict, reasoning: str) -> Plan:
        """从 LLM 结果生成摘要计划"""
        topic = params.get("query", query)

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
            reasoning=reasoning or f"用户意图是总结'{topic}'相关内容。"
        )

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

        # 摘要/复习卡片意图
        summarize_patterns = [r"总结.*", r"摘要.*", r"概括.*", r"归纳.*", r"复习卡片", r"生成卡片"]
        for pattern in summarize_patterns:
            if re.search(pattern, query_lower):
                return IntentType.SUMMARIZE

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

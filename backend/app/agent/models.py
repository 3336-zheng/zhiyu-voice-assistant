"""
Agent 数据模型定义（课堂学习场景聚焦版）
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class IntentType(str, Enum):
    """用户意图类型（课堂学习场景，7种核心意图）"""
    SEARCH = "search"              # 检索知识库/问答复习
    CREATE_NOTE = "create_note"    # 创建笔记
    UPDATE_NOTE = "update_note"    # 更新笔记
    DELETE_NOTE = "delete_note"    # 删除笔记
    LIST_NOTES = "list_notes"      # 列出笔记
    TIME_QUERY = "time_query"      # 时间/日期查询
    SUMMARIZE = "summarize"        # 摘要总结/生成复习卡片
    UNKNOWN = "unknown"            # 未知意图


class ToolName(str, Enum):
    """可用工具名称（课堂学习场景）"""
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    CREATE_NOTE = "create_note"
    UPDATE_NOTE = "update_note"
    DELETE_NOTE = "delete_note"
    LIST_NOTES = "list_notes"
    GET_CURRENT_TIME = "get_current_time"
    SUMMARIZE_TEXT = "summarize_text"


class PlanStep(BaseModel):
    """计划步骤"""
    step_id: int
    tool_name: ToolName
    parameters: Dict[str, Any]
    description: str
    depends_on: Optional[List[int]] = None  # 依赖的步骤ID


class Plan(BaseModel):
    """执行计划"""
    intent: IntentType
    original_query: str
    steps: List[PlanStep]
    estimated_steps: int
    reasoning: str  # 规划的理由


class ToolResult(BaseModel):
    """工具执行结果"""
    step_id: Optional[int] = None  # 对应的计划步骤ID
    tool_name: ToolName
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None


class ExecutionResult(BaseModel):
    """执行器结果"""
    plan: Plan
    results: List[ToolResult]
    completed_steps: int
    total_steps: int
    success: bool
    execution_log: List[str]
    final_data: Dict[str, Any]  # 聚合的数据


class AgentResponse(BaseModel):
    """Agent 最终响应"""
    query: str
    response: str
    session_id: Optional[str] = None  # 会话 ID（新增）
    plan: Optional[Plan] = None
    execution_result: Optional[ExecutionResult] = None
    sources: Optional[List[Dict]] = None  # 引用来源
    confirmation_required: bool = False
    pending_action_id: Optional[str] = None
    action_preview: Optional[List[Dict[str, Any]]] = None
    confidence: float = 1.0  # 置信度
    evidence_status: str = "not_applicable"
    evidence_score: Optional[float] = None
    evidence_source_count: int = 0
    evidence_reason: Optional[str] = None
    timestamp: datetime
    execution_time_ms: Optional[int] = None


class SearchParameters(BaseModel):
    """检索参数"""
    query: str
    top_k: int = 5
    tag_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class CreateNoteParameters(BaseModel):
    """创建笔记参数"""
    title: str
    content: str
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    audio_id: Optional[int] = None


class UpdateNoteParameters(BaseModel):
    """更新笔记参数（filename 兼容页面 UUID、标题或唯一别名）"""
    filename: str
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None

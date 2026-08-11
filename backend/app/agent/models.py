"""
Agent 数据模型定义（课堂学习场景聚焦版）
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class ToolRiskLevel(str, Enum):
    """工具风险级别，用于执行门禁。"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"


class ToolCapability(BaseModel):
    """暴露给 Planner 的工具能力描述，不包含可执行函数。"""

    name: ToolName
    description: str
    parameters_schema: Dict[str, Any]
    risk_level: ToolRiskLevel
    requires_confirmation: bool = False
    supports_parallel: bool = True


class PlanStep(BaseModel):
    """计划步骤"""
    step_id: int
    tool_name: ToolName
    parameters: Dict[str, Any]
    description: str
    depends_on: Optional[List[int]] = None  # 依赖的步骤ID
    expected_output: Optional[str] = None
    success_criteria: Optional[str] = None


class Plan(BaseModel):
    """执行计划"""
    intent: IntentType
    original_query: str
    steps: List[PlanStep]
    estimated_steps: int
    reasoning: str  # 规划的理由
    goal: Optional[str] = None


class ToolResult(BaseModel):
    """工具执行结果"""
    step_id: Optional[int] = None  # 对应的计划步骤ID
    tool_name: ToolName
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    context_tokens: int = 0
    context_truncated: bool = False


class ExecutionResult(BaseModel):
    """执行器结果"""
    plan: Plan
    results: List[ToolResult]
    completed_steps: int
    total_steps: int
    success: bool
    execution_log: List[str]
    final_data: Dict[str, Any]  # 聚合的数据
    context_stats: Dict[str, Any] = Field(default_factory=dict)


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
    external_research_available: bool = False
    request_id: Optional[str] = None
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_stats: Optional[Dict[str, Any]] = None
    model_usage: Optional[Dict[str, Any]] = None
    timestamp: datetime
    execution_time_ms: Optional[int] = None


class ToolParameters(BaseModel):
    """工具参数公共约束：拒绝模型生成但工具不认识的字段。"""

    model_config = ConfigDict(extra="forbid")


class SearchParameters(ToolParameters):
    """检索参数"""
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    tag_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class CreateNoteParameters(ToolParameters):
    """创建笔记参数"""
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    audio_id: Optional[int] = None
    notebook: Optional[str] = None
    research_run_id: Optional[str] = None


class UpdateNoteParameters(ToolParameters):
    """更新笔记参数（filename 兼容页面 UUID、标题或唯一别名）"""
    filename: str = Field(min_length=1)
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


class DeleteNoteParameters(ToolParameters):
    """删除页面参数。"""

    filename: str = Field(min_length=1)


class ListNotesParameters(ToolParameters):
    """页面列表参数。"""

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class CurrentTimeParameters(ToolParameters):
    """当前时间工具无需参数。"""


class SummarizeTextParameters(ToolParameters):
    """摘要工具参数，content 可引用上游步骤结果。"""

    content: Any

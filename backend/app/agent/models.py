"""
Agent 数据模型定义
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime


class IntentType(str, Enum):
    """用户意图类型"""
    SEARCH = "search"              # 检索知识库
    CREATE_NOTE = "create_note"    # 创建笔记
    UPDATE_NOTE = "update_note"    # 更新笔记
    DELETE_NOTE = "delete_note"    # 删除笔记
    LIST_NOTES = "list_notes"      # 列出笔记
    TIME_QUERY = "time_query"      # 时间/日期查询
    SUMMARIZE = "summarize"        # 摘要总结
    CREATE_MD = "create_md"        # 创建MD文件
    WRITE_MD = "write_md"          # 写入MD文件
    DATE_SEARCH = "date_search"    # 按日期搜索笔记
    NOTE_DETAIL = "note_detail"    # 查看笔记详情
    UNKNOWN = "unknown"            # 未知意图


class ToolName(str, Enum):
    """可用工具名称"""
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    CREATE_NOTE = "create_note"
    UPDATE_NOTE = "update_note"
    DELETE_NOTE = "delete_note"
    LIST_NOTES = "list_notes"
    GET_CURRENT_TIME = "get_current_time"
    SEARCH_BY_DATE_RANGE = "search_by_date_range"
    GET_NOTE_DETAIL = "get_note_detail"
    CREATE_MD_FILE = "create_md_file"
    WRITE_MD_FILE = "write_md_file"
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
    confidence: float = 1.0  # 置信度
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
    """更新笔记参数"""
    note_id: int
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


class DateRangeParameters(BaseModel):
    """日期范围参数"""
    date_from: Optional[str] = None  # ISO格式
    date_to: Optional[str] = None
    query: Optional[str] = None  # 可选的附加查询


class CreateMdParameters(BaseModel):
    """创建MD文件参数"""
    filename: str  # 文件名（不含扩展名）
    title: Optional[str] = None  # 标题（可选，作为文件首行）
    content: Optional[str] = None  # 初始内容（可选）
    directory: Optional[str] = None  # 目标目录（可选，默认 data/notes）


class WriteMdParameters(BaseModel):
    """写入MD文件参数"""
    filename: str  # 文件名（不含扩展名或含.md扩展名）
    content: str  # 要写入的内容
    mode: str = "append"  # 写入模式：append（追加）或 overwrite（覆盖）
    directory: Optional[str] = None  # 目标目录（可选，默认 data/notes）

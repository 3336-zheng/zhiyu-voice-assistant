"""Agent 运行时事件协议。"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Dict, Union

from pydantic import BaseModel, Field


class AgentEventType(StrEnum):
    """前后端共享的 Agent 运行事件类型。"""

    RUN_STARTED = "run_started"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOKEN = "token"
    RUN_COMPLETED = "run_completed"
    RUN_ERROR = "run_error"
    RUN_CANCELLED = "run_cancelled"


TERMINAL_EVENT_TYPES = {
    AgentEventType.RUN_COMPLETED,
    AgentEventType.RUN_ERROR,
    AgentEventType.RUN_CANCELLED,
}


class AgentRuntimeEvent(BaseModel):
    """可排序、可序列化和可回放的运行事件。"""

    type: AgentEventType
    run_id: str
    session_id: str
    sequence: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)


class AgentRunCancelled(RuntimeError):
    """Agent 收到协作式取消信号。"""


AgentEventCallback = Callable[[Union[AgentEventType, str], Dict[str, Any]], None]

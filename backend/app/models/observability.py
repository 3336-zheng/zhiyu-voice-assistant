"""Agent 请求运行记录。"""

from sqlalchemy import Column, Integer, JSON, String, Text
from sqlalchemy.sql import func

from ..core.database import Base
from .types import DateTimeMs


class AgentRun(Base):
    """保存 Agent 运行状态、终态快照和请求级观测信息。"""

    __tablename__ = "agent_runs"

    request_id = Column(String(128), primary_key=True)
    session_id = Column(String(64), index=True)
    query = Column(Text, nullable=False)
    intent = Column(String(50))
    status = Column(String(20), nullable=False)
    execution_time_ms = Column(Integer)
    timeline = Column(JSON)
    retrieval_stats = Column(JSON)
    model_usage = Column(JSON)
    response = Column(Text)
    error = Column(Text)
    events = Column(JSON)
    runtime_snapshot = Column(JSON)
    created_at = Column(DateTimeMs, server_default=func.now())
    updated_at = Column(DateTimeMs, server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTimeMs)
    # 执行者活着的心跳。多实例下启动恢复靠它区分「我上次跑挂的运行」和
    # 「别的实例正在跑的运行」——没有这个字段的话，任一实例启动都会把
    # 所有未完成的运行标记为失败，把别人正在跑的对话一起杀掉。
    #
    # 不能复用 updated_at：状态变更、字段回写都会刷新它，判断不出执行者是否还活着。
    heartbeat_at = Column(DateTimeMs, nullable=True, index=True)

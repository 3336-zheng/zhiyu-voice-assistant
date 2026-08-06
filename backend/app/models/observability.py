"""Agent 请求运行记录。"""

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.sql import func

from ..core.database import Base


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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))

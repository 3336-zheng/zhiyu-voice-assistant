"""
对话历史模型 - 支持多轮对话记忆
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from ..core.database import Base


class Conversation(Base):
    """对话会话表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(255))  # 默认取首条用户消息，便于检索历史会话
    summary = Column(Text)  # 会话摘要（用于长对话压缩）
    summary_message_id = Column(Integer)  # 摘要已覆盖到的原始消息 ID
    message_count = Column(Integer, default=0)  # 消息计数
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Conversation(session_id={self.session_id})>"


class ConversationMessage(Base):
    """对话消息表"""
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    intent = Column(String(50))  # 识别的意图
    extra_data = Column(JSON)  # 附加元数据（如引用来源）
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ConversationMessage(session_id={self.session_id}, role={self.role})>"

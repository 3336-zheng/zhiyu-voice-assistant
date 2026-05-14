"""
笔记模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.sql import func
from ..core.database import Base

class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    content = Column(Text)
    summary = Column(Text)  # 笔记摘要
    tags = Column(JSON)  # 标签列表
    audio_id = Column(Integer, index=True)  # 关联的音频ID
    duration = Column(Float)  # 音频时长
    language = Column(String(10))  # 语言
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Note(id={self.id}, title={self.title})>"
"""
音频模型
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from ..core.database import Base

class Audio(Base):
    __tablename__ = "audios"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, index=True)
    original_filename = Column(String(255))
    file_path = Column(String(512))
    file_size = Column(Integer)
    duration = Column(Float)  # 音频时长（秒）
    language = Column(String(10))  # 识别语言
    transcription = Column(Text)  # 转录文本
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Audio(id={self.id}, filename={self.filename})>"
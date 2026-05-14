"""
检索模型
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.sql import func
from ..core.database import Base

class Retrieval(Base):
    __tablename__ = "retrievals"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)  # 查询文本
    query_embedding = Column(Text)  # 查询向量
    results = Column(JSON)  # 检索结果（存储为JSON字符串）
    similarity_scores = Column(JSON)  # 相似度分数
    top_k = Column(Integer)  # 返回结果数量
    used_model = Column(String(100))  # 使用的模型
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Retrieval(id={self.id}, query_text={self.query_text[:20]}...)"
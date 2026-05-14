"""
检索服务（向后兼容接口）
内部使用 HybridRetrievalService (BM25 + Embedding + RRF + Reranker)
"""
import logging
from typing import List
from sqlalchemy.orm import Session
from ..core.database import SessionLocal
from ..models import Note
from .hybrid_retrieval_service import get_hybrid_retrieval_service

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    检索服务（向后兼容）
    内部委托给 HybridRetrievalService
    """

    def __init__(self):
        self.hybrid_service = get_hybrid_retrieval_service()

    def search_notes(self, query: str, top_k: int = 5) -> list:
        """
        搜索笔记（使用混合检索：BM25 + Embedding + RRF + Reranker）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            相关笔记列表（按相似度排序）
        """
        db = SessionLocal()
        try:
            results = self.hybrid_service.search_hybrid(query, top_k, db=db)

            if not results:
                logger.info("混合检索无匹配结果")
                return []

            # 从结果中提取笔记对象（兼容旧接口）
            note_ids = [r["id"] for r in results]
            notes = db.query(Note).filter(Note.id.in_(note_ids)).all()
            note_map = {note.id: note for note in notes}

            # 按混合检索结果顺序排列
            final_notes = []
            for r in results:
                note = note_map.get(r["id"])
                if note:
                    final_notes.append(note)

            logger.info(f"混合检索完成，找到 {len(final_notes)} 个相关笔记")
            return final_notes

        except Exception as e:
            logger.error(f"混合检索失败: {e}", exc_info=True)
            return []
        finally:
            db.close()


# 全局服务实例
retrieval_service_instance = None


def get_retrieval_service() -> RetrievalService:
    """获取检索服务实例（单例模式）"""
    global retrieval_service_instance
    if retrieval_service_instance is None:
        retrieval_service_instance = RetrievalService()
    return retrieval_service_instance
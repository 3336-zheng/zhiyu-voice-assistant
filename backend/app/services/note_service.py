"""
笔记服务
从音频创建笔记，同时将向量写入 ChromaDB 向量数据库并更新 BM25 索引
"""
import logging
from sqlalchemy.orm import Session
import librosa

logger = logging.getLogger(__name__)
from ..models import Note, Audio
from .whisper_service import get_asr_service
from .embedding_service import EmbeddingService
from .chroma_service import get_chroma_service
from .bm25_service import get_bm25_service


class NoteService:
    def __init__(self):
        self.asr_service = get_asr_service()
        self.embedding_service = EmbeddingService()
        self.chroma_service = get_chroma_service()
        self.bm25_service = get_bm25_service()

    def create_note_from_audio(self, audio_path: str, original_filename: str, db: Session) -> Note:
        """
        从音频文件创建笔记（写入 SQLite + ChromaDB + BM25）

        Args:
            audio_path: 音频文件路径
            original_filename: 原始文件名
            db: 数据库会话（由调用方传入）

        Returns:
            创建的笔记对象
        """
        # 1. 转录音频
        transcription_result = self.asr_service.transcribe(audio_path)
        content = transcription_result["transcription"]

        # 2. 生成嵌入向量
        embedding = self.embedding_service.encode(content)

        # 3. 创建笔记（写入 SQLite）
        try:
            audio_duration = librosa.get_duration(path=audio_path)

            note = Note(
                title=f"笔记 - {original_filename}",
                content=content,
                summary=content[:200] + "..." if len(content) > 200 else content,
                tags=["语音转录"],
                audio_id=0,
                duration=audio_duration,
                language=transcription_result["language"]
            )

            db.add(note)
            db.commit()
            db.refresh(note)

            # 4. 将向量写入 ChromaDB
            if embedding:
                self.chroma_service.add_embedding(
                    note_id=note.id,
                    embedding=embedding,
                    content=content,
                    metadata={
                        "title": note.title,
                        "tags": note.tags,
                        "language": note.language
                    }
                )

            # 5. 更新 BM25 索引
            self.bm25_service.add_document(f"note_{note.id}", content, note.title)

            logger.info(f"笔记创建成功: {note.id} (ChromaDB + BM25 已同步)")
            return note

        except Exception as e:
            db.rollback()
            logger.error(f"笔记创建失败: {e}", exc_info=True)
            raise


# 全局服务实例
note_service_instance = None


def get_note_service() -> NoteService:
    """获取笔记服务实例（单例模式）"""
    global note_service_instance
    if note_service_instance is None:
        note_service_instance = NoteService()
    return note_service_instance
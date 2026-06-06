"""
对话记忆服务
管理多轮对话历史，支持上下文窗口和摘要压缩
"""
import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.models.conversation import Conversation, ConversationMessage

logger = logging.getLogger(__name__)


class MemoryService:
    """
    对话记忆服务
    负责存储和检索对话历史，维护上下文窗口
    """

    def __init__(self):
        """初始化记忆服务"""
        self.max_history = settings.memory_max_history
        self.summary_threshold = settings.memory_summary_threshold
        self.session_ttl = settings.session_ttl_hours

    def get_or_create_session(self, session_id: str, db: Session) -> Conversation:
        """
        获取或创建对话会话

        Args:
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            Conversation: 会话对象
        """
        conversation = db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).first()

        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                message_count=0
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            logger.info(f"创建新会话: {session_id}")

        return conversation

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = None,
        metadata: Dict = None,
        db: Session = None
    ) -> ConversationMessage:
        """
        添加对话消息

        Args:
            session_id: 会话 ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            intent: 识别的意图
            metadata: 附加元数据
            db: 数据库会话

        Returns:
            ConversationMessage: 消息对象
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            # 确保会话存在
            self.get_or_create_session(session_id, db)

            # 创建消息
            message = ConversationMessage(
                session_id=session_id,
                role=role,
                content=content,
                intent=intent,
                extra_data=metadata
            )
            db.add(message)

            # 更新会话消息计数
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            if conversation:
                conversation.message_count = (conversation.message_count or 0) + 1
                conversation.updated_at = datetime.now()

            db.commit()
            db.refresh(message)
            logger.debug(f"添加消息到会话 {session_id}: {role}")
            return message

        finally:
            if should_close:
                db.close()

    def get_history(
        self,
        session_id: str,
        limit: int = None,
        db: Session = None
    ) -> List[Dict[str, str]]:
        """
        获取对话历史（LLM 格式）

        Args:
            session_id: 会话 ID
            limit: 返回消息数量限制
            db: 数据库会话

        Returns:
            List[Dict]: 消息列表，格式: [{"role": "user", "content": "..."}]
        """
        if limit is None:
            limit = self.max_history

        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            # 获取会话摘要（如果有）
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()

            messages = []

            # 如果有摘要，作为系统消息添加
            if conversation and conversation.summary:
                messages.append({
                    "role": "system",
                    "content": f"以下是之前对话的摘要：{conversation.summary}"
                })

            # 获取最近的消息
            db_messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).order_by(
                ConversationMessage.created_at.desc()
            ).limit(limit).all()

            # 反转为时间正序
            db_messages.reverse()

            for msg in db_messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            return messages

        finally:
            if should_close:
                db.close()

    def get_context_window(
        self,
        session_id: str,
        current_query: str,
        db: Session = None
    ) -> List[Dict[str, str]]:
        """
        获取上下文窗口（用于 LLM 调用）

        Args:
            session_id: 会话 ID
            current_query: 当前用户查询
            db: 数据库会话

        Returns:
            List[Dict]: 完整的对话上下文
        """
        history = self.get_history(session_id, db=db)

        # 添加当前查询
        context = history + [{"role": "user", "content": current_query}]
        return context

    def summarize_if_needed(self, session_id: str, db: Session = None):
        """
        如果消息数量超过阈值，自动进行摘要压缩

        Args:
            session_id: 会话 ID
            db: 数据库会话
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()

            if not conversation:
                return

            if (conversation.message_count or 0) < self.summary_threshold:
                return

            # 获取所有消息
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).order_by(ConversationMessage.created_at).all()

            if len(messages) < self.summary_threshold:
                return

            # 生成摘要
            try:
                from backend.app.services.llm_service import get_llm_service
                llm_service = get_llm_service()

                # 构建待摘要的对话文本
                conversation_text = "\n".join([
                    f"{msg.role}: {msg.content}"
                    for msg in messages[:-5]  # 保留最近 5 条消息
                ])

                summary_messages = [
                    {
                        "role": "system",
                        "content": "请对以下对话历史生成简洁的摘要，保留关键信息和上下文。"
                    },
                    {
                        "role": "user",
                        "content": conversation_text
                    }
                ]

                summary = llm_service.chat(summary_messages, max_tokens=300)
                conversation.summary = summary

                # 删除已摘要的旧消息（保留最近 5 条）
                old_messages = messages[:-5]
                for msg in old_messages:
                    db.delete(msg)

                conversation.message_count = len(messages) - len(old_messages) + 1
                db.commit()
                logger.info(f"会话 {session_id} 已进行摘要压缩，保留 {conversation.message_count} 条消息")

            except Exception as e:
                logger.error(f"摘要压缩失败: {e}")

        finally:
            if should_close:
                db.close()

    def clear_session(self, session_id: str, db: Session = None) -> bool:
        """
        清除会话历史

        Args:
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            bool: 是否成功
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            # 删除消息
            db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).delete()

            # 删除会话
            db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).delete()

            db.commit()
            logger.info(f"清除会话: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清除会话失败: {e}")
            db.rollback()
            return False

        finally:
            if should_close:
                db.close()

    def list_sessions(self, db: Session = None, limit: int = 50) -> List[Dict]:
        """
        列出所有会话

        Args:
            db: 数据库会话
            limit: 返回数量限制

        Returns:
            List[Dict]: 会话列表
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            conversations = db.query(Conversation).order_by(
                Conversation.updated_at.desc()
            ).limit(limit).all()

            return [
                {
                    "session_id": c.session_id,
                    "message_count": c.message_count,
                    "has_summary": bool(c.summary),
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "updated_at": c.updated_at.isoformat() if c.updated_at else None
                }
                for c in conversations
            ]

        finally:
            if should_close:
                db.close()

    def cleanup_expired_sessions(self, db: Session = None) -> Dict:
        """
        清理过期会话及其消息

        Args:
            db: 数据库会话

        Returns:
            Dict: 清理结果统计
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            from datetime import timedelta
            threshold = datetime.now() - timedelta(hours=self.session_ttl)

            expired = db.query(Conversation).filter(
                Conversation.updated_at < threshold
            ).all()

            cleaned_count = 0
            cleaned_messages = 0
            for conv in expired:
                msg_count = db.query(ConversationMessage).filter(
                    ConversationMessage.session_id == conv.session_id
                ).delete()
                cleaned_messages += msg_count
                db.delete(conv)
                cleaned_count += 1

            db.commit()
            if cleaned_count > 0:
                logger.info(f"清理过期会话: {cleaned_count} 个会话, {cleaned_messages} 条消息")
            return {
                "cleaned_sessions": cleaned_count,
                "cleaned_messages": cleaned_messages
            }

        except Exception as e:
            logger.error(f"清理过期会话失败: {e}")
            db.rollback()
            return {"cleaned_sessions": 0, "cleaned_messages": 0, "error": str(e)}

        finally:
            if should_close:
                db.close()

    def get_session_stats(self, db: Session = None) -> Dict:
        """
        获取会话统计信息

        Returns:
            Dict: 统计数据
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            from datetime import timedelta
            total_sessions = db.query(Conversation).count()
            total_messages = db.query(ConversationMessage).count()

            threshold = datetime.now() - timedelta(hours=self.session_ttl)
            expired_count = db.query(Conversation).filter(
                Conversation.updated_at < threshold
            ).count()

            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "expired_sessions": expired_count,
                "session_ttl_hours": self.session_ttl
            }

        finally:
            if should_close:
                db.close()

    def generate_session_id(self) -> str:
        """
        生成新的会话 ID

        Returns:
            str: 唯一会话 ID
        """
        return f"session_{uuid.uuid4().hex[:16]}"


# 全局服务实例
memory_service_instance = None


def get_memory_service() -> MemoryService:
    """获取记忆服务实例（单例模式）"""
    global memory_service_instance
    if memory_service_instance is None:
        memory_service_instance = MemoryService()
    return memory_service_instance

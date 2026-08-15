"""
对话记忆服务
管理多轮对话历史，支持上下文窗口和摘要压缩
"""
import uuid
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime, timezone
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.models.conversation import Conversation, ConversationMessage
from backend.app.services.retrieval.token_budget_service import estimate_tokens, truncate_text

logger = logging.getLogger(__name__)


def _conversation_title(content: str, limit: int = 80) -> str:
    """把首条用户消息整理为历史会话标题。"""
    normalized = re.sub(r"\s+", " ", content or "").strip()
    if not normalized:
        return "新对话"
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit - 3].rstrip()}..."


def _search_snippet(content: str, query: str, radius: int = 48) -> str:
    """返回命中词附近的单行消息片段。"""
    normalized = re.sub(r"\s+", " ", content or "").strip()
    index = normalized.casefold().find(query.casefold())
    if index < 0:
        return normalized[: radius * 2]
    start = max(0, index - radius)
    end = min(len(normalized), index + len(query) + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(normalized) else ""
    return f"{prefix}{normalized[start:end]}{suffix}"


def _escape_like(value: str) -> str:
    """转义 LIKE 通配符，使用户输入按字面匹配。"""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class MemoryService:
    """
    对话记忆服务
    负责存储和检索对话历史，维护上下文窗口
    """

    def __init__(self):
        """初始化记忆服务"""
        self.max_history = settings.memory_max_history
        self.summary_threshold = settings.memory_summary_threshold
        self.summary_trigger_tokens = settings.memory_summary_trigger_tokens
        self.summary_token_budget = settings.memory_summary_token_budget
        self.summary_input_token_budget = settings.memory_summary_input_token_budget
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
            conversation = self.get_or_create_session(session_id, db)

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
            if conversation:
                conversation.message_count = (conversation.message_count or 0) + 1
                conversation.updated_at = datetime.now()
                if role == "user" and not conversation.title:
                    conversation.title = _conversation_title(content)

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
            history_query = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            )
            if conversation and conversation.summary_message_id:
                history_query = history_query.filter(
                    ConversationMessage.id > conversation.summary_message_id
                )
            db_messages = history_query.order_by(
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

            # 只读取尚未纳入摘要的原始消息。
            message_query = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            )
            if conversation.summary_message_id:
                message_query = message_query.filter(
                    ConversationMessage.id > conversation.summary_message_id
                )
            messages = message_query.order_by(ConversationMessage.id).all()

            pending_text = "\n".join(
                f"{message.role}: {message.content}" for message in messages
            )
            if (
                len(messages) < self.summary_threshold
                and estimate_tokens(pending_text) < self.summary_trigger_tokens
            ):
                return

            messages_to_summarize = messages[:-5]
            if not messages_to_summarize:
                return

            previous_summary = conversation.summary or "（无）"
            previous_summary = truncate_text(
                previous_summary,
                min(
                    self.summary_token_budget,
                    max(128, self.summary_input_token_budget // 4),
                ),
            )
            summary_input_remaining = max(
                64,
                self.summary_input_token_budget - estimate_tokens(previous_summary) - 180,
            )
            selected_messages = []
            selected_text = []
            for message in messages_to_summarize:
                item = f"{message.role}: {message.content}"
                item_tokens = estimate_tokens(item)
                if item_tokens <= summary_input_remaining:
                    selected_messages.append(message)
                    selected_text.append(item)
                    summary_input_remaining -= item_tokens
                    continue
                if not selected_messages and summary_input_remaining > 64:
                    selected_messages.append(message)
                    selected_text.append(truncate_text(item, summary_input_remaining))
                break

            if not selected_messages:
                return

            # 生成摘要
            try:
                from backend.app.services.ai.llm_service import get_llm_service
                llm_service = get_llm_service()

                # 构建待摘要的对话文本
                conversation_text = "\n".join(selected_text)

                summary_messages = [
                    {
                        "role": "system",
                        "content": (
                            "请增量更新个人 Wiki 助手的对话摘要，只使用提供的内容，不要虚构。"
                            "请严格保留以下栏目：\n"
                            "## 用户目标\n## 已确认事实\n## 页面与实体\n"
                            "## 约束与偏好\n## 已完成事项\n## 未完成事项\n"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"已有摘要：\n{previous_summary}\n\n新增对话：\n{conversation_text}"
                    }
                ]

                summary = llm_service.chat(
                    summary_messages,
                    max_tokens=self.summary_token_budget,
                )
                if not summary or not str(summary).strip():
                    return
                conversation.summary = summary
                conversation.summary_message_id = selected_messages[-1].id
                db.commit()
                logger.info(
                    "会话 %s 已增量摘要到消息 %s，原始消息完整保留",
                    session_id,
                    conversation.summary_message_id,
                )

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

    def list_sessions(
        self,
        db: Session = None,
        limit: int = 50,
        query: str = None,
    ) -> List[Dict]:
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
            normalized_query = (query or "").strip()
            conversations_query = db.query(Conversation)
            if normalized_query:
                pattern = f"%{_escape_like(normalized_query)}%"
                matching_sessions = select(ConversationMessage.session_id).where(
                    ConversationMessage.content.ilike(pattern, escape="\\")
                )
                conversations_query = conversations_query.filter(
                    or_(
                        Conversation.title.ilike(pattern, escape="\\"),
                        Conversation.session_id.in_(matching_sessions),
                    )
                )

            conversations = conversations_query.order_by(
                Conversation.updated_at.desc()
            ).limit(limit).all()

            matched_messages = {}
            if normalized_query and conversations:
                session_ids = [conversation.session_id for conversation in conversations]
                matches = db.query(ConversationMessage).filter(
                    ConversationMessage.session_id.in_(session_ids),
                    ConversationMessage.content.ilike(pattern, escape="\\"),
                ).order_by(ConversationMessage.id.asc()).all()
                for message in matches:
                    matched_messages.setdefault(message.session_id, message)

            result = []
            for conversation in conversations:
                matched_message = matched_messages.get(conversation.session_id)
                result.append({
                    "session_id": conversation.session_id,
                    "title": conversation.title or "新对话",
                    "message_count": conversation.message_count,
                    "has_summary": bool(conversation.summary),
                    "match_snippet": (
                        _search_snippet(matched_message.content, normalized_query)
                        if matched_message else None
                    ),
                    "matched_message_id": matched_message.id if matched_message else None,
                    "created_at": (
                        conversation.created_at.isoformat()
                        if conversation.created_at else None
                    ),
                    "updated_at": (
                        conversation.updated_at.isoformat()
                        if conversation.updated_at else None
                    ),
                })
            return result

        finally:
            if should_close:
                db.close()

    def get_session_messages(
        self,
        session_id: str,
        db: Session = None,
    ) -> Optional[Dict]:
        """读取指定会话的完整历史消息，供前端恢复对话。"""
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            if conversation is None:
                return None
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).order_by(ConversationMessage.id.asc()).all()
            from backend.app.models.wiki import AgentPendingAction

            action_ids = {
                str((message.extra_data or {}).get("pending_action_id"))
                for message in messages
                if (message.extra_data or {}).get("pending_action_id")
            }
            pending_actions = {
                action.id: action
                for action in (
                    db.query(AgentPendingAction)
                    .filter(AgentPendingAction.id.in_(action_ids))
                    .all()
                    if action_ids else []
                )
            }

            def message_to_dict(message: ConversationMessage) -> Dict:
                metadata = dict(message.extra_data or {})
                action_id = metadata.get("pending_action_id")
                if action_id:
                    pending = pending_actions.get(action_id)
                    expires_at = pending.expires_at if pending else None
                    if expires_at and expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    is_action_pending = bool(
                        pending
                        and pending.status == "pending"
                        and expires_at
                        and expires_at > datetime.now(timezone.utc)
                    )
                    metadata["confirmation_required"] = is_action_pending
                    metadata["action_preview"] = (
                        list(pending.preview or []) if is_action_pending else []
                    )
                return {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "intent": message.intent,
                    "metadata": metadata,
                    "created_at": (
                        message.created_at.isoformat()
                        if message.created_at else None
                    ),
                }

            return {
                "session_id": conversation.session_id,
                "title": conversation.title or "新对话",
                "messages": [message_to_dict(message) for message in messages],
            }
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

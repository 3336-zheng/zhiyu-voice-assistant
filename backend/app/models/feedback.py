"""回答反馈与自动纠错闭环模型。"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String, Text, UniqueConstraint

from ..core.database import Base


def utc_now() -> datetime:
    """返回带时区的 UTC 时间。"""
    return datetime.now(timezone.utc)


class AnswerFeedback(Base):
    """保存一次回答反馈从上报到自动复测的完整状态。"""

    __tablename__ = "answer_feedbacks"
    __table_args__ = (
        UniqueConstraint("request_id", name="uq_answer_feedback_request"),
    )

    id = Column(String(36), primary_key=True)
    request_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="reported", index=True)
    question = Column(Text, nullable=False)
    answer_snapshot = Column(Text, nullable=False)
    retrieval_snapshot = Column(JSON, nullable=False, default=dict)
    user_note = Column(Text, nullable=True)
    target_page_id = Column(String(36), nullable=True, index=True)
    external_research_run_id = Column(String(36), nullable=True, index=True)
    pending_action_id = Column(String(36), nullable=True, index=True)
    draft_title = Column(String(255), nullable=True)
    draft_content = Column(Text, nullable=True)
    write_result = Column(JSON, nullable=True)
    index_result = Column(JSON, nullable=True)
    retest_request_id = Column(String(128), nullable=True, index=True)
    retest_answer = Column(Text, nullable=True)
    retest_snapshot = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

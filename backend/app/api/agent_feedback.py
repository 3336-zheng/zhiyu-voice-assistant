"""回答反馈、知识纠错和自动复测路由。"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.agent.agent import AgentActionError, get_agent
from backend.app.core.database import get_db
from backend.app.services.answer_feedback_service import (
    AnswerFeedbackConflict,
    AnswerFeedbackError,
    AnswerFeedbackNotFound,
    AnswerFeedbackValidationError,
    get_answer_feedback_service,
)
from backend.app.services.external_research_service import (
    ExternalResearchError,
    ExternalResearchUnavailable,
)

from .agent_schemas import (
    AnswerFeedbackActionRequest,
    AnswerFeedbackCreateRequest,
    AnswerFeedbackResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _raise_feedback_error(exc: Exception) -> None:
    if isinstance(exc, AnswerFeedbackNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (AnswerFeedbackConflict, AgentActionError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, AnswerFeedbackValidationError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, ExternalResearchUnavailable):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, (ExternalResearchError, AnswerFeedbackError)):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    logger.error("回答纠错闭环执行失败: %s", exc, exc_info=True)
    raise HTTPException(status_code=500, detail="回答纠错处理失败") from exc


@router.post("/feedback/", response_model=AnswerFeedbackResponse)
async def create_answer_feedback(
    request: AnswerFeedbackCreateRequest,
    db: Session = Depends(get_db),
):
    try:
        return get_answer_feedback_service().create(
            **request.model_dump(),
            db=db,
        )
    except Exception as exc:
        _raise_feedback_error(exc)


@router.get("/feedback/{feedback_id}", response_model=AnswerFeedbackResponse)
async def get_answer_feedback(
    feedback_id: str,
    session_id: str,
    db: Session = Depends(get_db),
):
    try:
        return get_answer_feedback_service().get(feedback_id, session_id, db)
    except Exception as exc:
        _raise_feedback_error(exc)


@router.post("/feedback/{feedback_id}/prepare", response_model=AnswerFeedbackResponse)
async def prepare_answer_feedback(
    feedback_id: str,
    request: AnswerFeedbackActionRequest,
    db: Session = Depends(get_db),
):
    try:
        return await get_answer_feedback_service().prepare(
            feedback_id,
            request.session_id,
            db,
            get_agent(),
        )
    except Exception as exc:
        _raise_feedback_error(exc)


@router.post("/feedback/{feedback_id}/confirm", response_model=AnswerFeedbackResponse)
async def confirm_answer_feedback(
    feedback_id: str,
    request: AnswerFeedbackActionRequest,
    db: Session = Depends(get_db),
):
    try:
        return await get_answer_feedback_service().confirm(
            feedback_id,
            request.session_id,
            db,
            get_agent(),
        )
    except Exception as exc:
        _raise_feedback_error(exc)


@router.post("/feedback/{feedback_id}/retry", response_model=AnswerFeedbackResponse)
async def retry_answer_feedback(
    feedback_id: str,
    request: AnswerFeedbackActionRequest,
    db: Session = Depends(get_db),
):
    try:
        return await get_answer_feedback_service().retry(
            feedback_id,
            request.session_id,
            db,
        )
    except Exception as exc:
        _raise_feedback_error(exc)


@router.post("/feedback/{feedback_id}/cancel", response_model=AnswerFeedbackResponse)
async def cancel_answer_feedback(
    feedback_id: str,
    request: AnswerFeedbackActionRequest,
    db: Session = Depends(get_db),
):
    try:
        return get_answer_feedback_service().cancel(
            feedback_id,
            request.session_id,
            db,
            get_agent(),
        )
    except Exception as exc:
        _raise_feedback_error(exc)

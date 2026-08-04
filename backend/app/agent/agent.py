"""
Plan-and-Execute Agent 主类（LangGraph 版本）
整合 Planner、Executor、Responder，支持多轮对话记忆
使用 LangGraph 状态机替代手写 if-else 编排
"""
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy.orm import Session

from backend.app.agent.models import AgentResponse, IntentType, Plan
from backend.app.agent.planner import get_planner
from backend.app.agent.executor import get_executor
from backend.app.agent.responder import get_responder
from backend.app.core.database import SessionLocal
from backend.app.core.config import settings
from backend.app.core.observability import timed_stage
from backend.app.models.wiki import AgentPendingAction

logger = logging.getLogger(__name__)

WRITE_INTENTS = {
    IntentType.CREATE_NOTE,
    IntentType.UPDATE_NOTE,
    IntentType.DELETE_NOTE,
}


class AgentActionError(Exception):
    """待确认操作不存在、已失效或不属于当前会话。"""


class PlanExecuteAgent:
    """
    Plan-and-Execute Agent
    主执行流程：Plan → Execute → Respond
    支持多轮对话记忆
    """

    def __init__(self):
        """初始化 Agent"""
        self.planner = get_planner()
        self.executor = get_executor()
        self.responder = get_responder()
        self._memory_service = None

    @property
    def memory_service(self):
        """延迟加载记忆服务"""
        if self._memory_service is None:
            try:
                from backend.app.services.memory_service import get_memory_service
                self._memory_service = get_memory_service()
            except Exception as e:
                logger.warning(f"记忆服务加载失败: {e}")
        return self._memory_service

    def _get_session_id(self, session_id: Optional[str] = None) -> str:
        """
        获取或生成会话 ID

        Args:
            session_id: 可选的会话 ID

        Returns:
            str: 会话 ID
        """
        if session_id:
            return session_id
        return f"session_{uuid.uuid4().hex[:16]}"

    async def run(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        db: Session = None
    ) -> AgentResponse:
        """
        执行 Agent 主流程（使用 LangGraph 图状态机）

        Args:
            user_query: 用户输入
            session_id: 会话 ID（可选，用于多轮对话）
            db: 数据库会话

        Returns:
            AgentResponse: Agent 响应
        """
        start_time = time.time()
        session_id = self._get_session_id(session_id)

        logger.info(f"Agent 开始处理查询: '{user_query[:50]}...'" if len(user_query) > 50 else f"Agent 开始处理查询: '{user_query}'")
        logger.info(f"会话 ID: {session_id}")

        # 管理数据库会话
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False

        try:
            # 获取对话上下文
            context = []
            if self.memory_service:
                try:
                    context = self.memory_service.get_history(session_id, db=db)
                except Exception as e:
                    logger.warning(f"获取对话历史失败: {e}")

            with timed_stage("agent.plan"):
                plan = self.planner.plan(user_query, context)
            if plan.intent in WRITE_INTENTS:
                response = self._create_pending_response(
                    user_query,
                    session_id,
                    plan,
                    db,
                    start_time,
                )
            elif plan.intent in {IntentType.SEARCH, IntentType.SUMMARIZE}:
                response = self._run_retrieval_graph(
                    user_query,
                    session_id,
                    context,
                    plan,
                    start_time,
                )
            else:
                execution_result = await self.executor.execute(plan, db)
                with timed_stage("agent.generation"):
                    response = self.responder.generate_response(
                        user_query,
                        plan,
                        execution_result,
                        context,
                    )
                response.session_id = session_id
                response.execution_time_ms = int((time.time() - start_time) * 1000)

            self._save_conversation(user_query, response, db)

            logger.info(f"Agent 处理完成，总耗时 {response.execution_time_ms}ms")
            return response

        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            return AgentResponse(
                query=user_query,
                response=f"抱歉，处理您的请求时出现错误：{str(e)}",
                session_id=session_id,
                confidence=0.0,
                timestamp=datetime.now(),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        finally:
            if should_close:
                db.close()

    def _run_retrieval_graph(
        self,
        user_query: str,
        session_id: str,
        context: List[Dict[str, str]],
        plan: Plan,
        start_time: float,
    ) -> AgentResponse:
        """执行只读的 Agentic RAG 图。"""
        from backend.app.agent.graph import get_agent_graph
        from backend.app.services.evidence_service import assess_evidence

        initial_state = {
            "query": user_query,
            "session_id": session_id,
            "context": context,
            "plan": plan,
            "rewritten_queries": [],
            "search_results": None,
            "execution_results": None,
            "relevance_grade": None,
            "refined_content": None,
            "answer": None,
            "sources": None,
            "confidence": 0.0,
            "evidence_status": "not_applicable",
            "evidence_score": None,
            "evidence_source_count": 0,
            "evidence_reason": None,
            "iter_count": 0,
            "max_iterations": settings.agent_max_iterations,
            "error": None,
        }
        final_state = get_agent_graph().invoke(initial_state)
        with timed_stage("agent.evidence"):
            evidence = assess_evidence(final_state.get("search_results"))
        if final_state.get("error"):
            logger.warning("图执行过程中出现错误: %s", final_state["error"])
        answer = final_state.get("answer")
        if not answer and final_state.get("relevance_grade") == "incorrect":
            answer = "现有 Wiki 中没有找到足以支持回答的证据，请补充相关页面或缩小查询范围。"
        return AgentResponse(
            query=user_query,
            response=answer or "现有 Wiki 证据不足，暂时无法回答该问题。",
            session_id=session_id,
            plan=plan,
            sources=final_state.get("sources", []),
            confidence=final_state.get("confidence", 0.0),
            evidence_status=final_state.get("evidence_status", evidence.status),
            evidence_score=final_state.get("evidence_score", evidence.score),
            evidence_source_count=final_state.get("evidence_source_count", evidence.source_count),
            evidence_reason=final_state.get("evidence_reason", evidence.reason),
            timestamp=datetime.now(),
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    def _create_pending_response(
        self,
        user_query: str,
        session_id: str,
        plan: Plan,
        db: Session,
        start_time: float,
    ) -> AgentResponse:
        """持久化写入计划，只返回预览而不执行工具。"""
        preview = [
            {
                "step_id": step.step_id,
                "operation": step.tool_name.value,
                "description": step.description,
                "parameters": step.parameters,
            }
            for step in plan.steps
        ]
        pending = AgentPendingAction(
            id=str(uuid.uuid4()),
            session_id=session_id,
            query=user_query,
            plan_data=plan.model_dump(mode="json"),
            preview=preview,
            status="pending",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        db.add(pending)
        db.commit()
        descriptions = "\n".join(f"- {item['description']}" for item in preview)
        return AgentResponse(
            query=user_query,
            response=f"以下知识变更等待确认：\n\n{descriptions}",
            session_id=session_id,
            plan=plan,
            confirmation_required=True,
            pending_action_id=pending.id,
            action_preview=preview,
            confidence=1.0,
            timestamp=datetime.now(),
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    async def confirm_action(
        self,
        action_id: str,
        session_id: str,
        db: Session = None,
    ) -> AgentResponse:
        """执行已保存的写入计划；重复确认返回第一次的结果。"""
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False
        try:
            pending = db.get(AgentPendingAction, action_id)
            self._validate_pending_action(pending, session_id, db)
            if pending.status == "completed" and pending.result_data:
                return AgentResponse.model_validate(pending.result_data)

            plan = Plan.model_validate(pending.plan_data)
            start_time = time.time()
            execution_result = await self.executor.execute(plan, db)
            response = self.responder.generate_response(
                pending.query,
                plan,
                execution_result,
                [],
            )
            response.session_id = session_id
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            if execution_result.success:
                pending.status = "completed"
                pending.completed_at = datetime.now(timezone.utc)
                pending.result_data = response.model_dump(mode="json")
                pending.error = None
            else:
                pending.status = "failed"
                pending.error = next(
                    (
                        result.error_message
                        for result in execution_result.results
                        if not result.success and result.error_message
                    ),
                    "写入计划执行失败",
                )
            db.commit()
            if execution_result.success:
                self._save_assistant_message(response, db)
            return response
        finally:
            if should_close:
                db.close()

    def cancel_action(self, action_id: str, session_id: str, db: Session) -> Dict[str, Any]:
        """取消仍在等待确认的写入计划。"""
        pending = db.get(AgentPendingAction, action_id)
        self._validate_pending_action(pending, session_id, db)
        if pending.status == "completed":
            raise AgentActionError("已完成的操作不能取消")
        pending.status = "cancelled"
        db.commit()
        return {"action_id": action_id, "status": "cancelled"}

    @staticmethod
    def _validate_pending_action(
        pending: Optional[AgentPendingAction],
        session_id: str,
        db: Session,
    ) -> None:
        if pending is None:
            raise AgentActionError("待确认操作不存在")
        if pending.session_id != session_id:
            raise AgentActionError("待确认操作不属于当前会话")
        if pending.status in {"cancelled", "failed", "expired"}:
            raise AgentActionError(f"待确认操作当前状态不可执行: {pending.status}")
        expires_at = pending.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if pending.status == "pending" and expires_at <= datetime.now(timezone.utc):
            pending.status = "expired"
            db.commit()
            raise AgentActionError("待确认操作已过期")

    def _save_conversation(
        self,
        user_query: str,
        response: AgentResponse,
        db: Session,
    ) -> None:
        """保存一轮对话及其结构化来源和确认状态。"""
        if not self.memory_service:
            return
        try:
            self.memory_service.add_message(
                session_id=response.session_id,
                role="user",
                content=user_query,
                db=db,
            )
            self._save_assistant_message(response, db)
            self.memory_service.summarize_if_needed(response.session_id, db=db)
        except Exception as exc:
            logger.warning("保存对话历史失败: %s", exc)

    def _save_assistant_message(self, response: AgentResponse, db: Session) -> None:
        if not self.memory_service:
            return
        self.memory_service.add_message(
            session_id=response.session_id,
            role="assistant",
            content=response.response,
            metadata={
                "sources": response.sources,
                "confirmation_required": response.confirmation_required,
                "pending_action_id": response.pending_action_id,
                "evidence_status": response.evidence_status,
                "evidence_score": response.evidence_score,
                "evidence_source_count": response.evidence_source_count,
            },
            db=db,
        )

    async def chat(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        db: Session = None
    ) -> str:
        """
        简化版对话接口，只返回回复文本

        Args:
            user_query: 用户输入
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            str: 回复文本
        """
        response = await self.run(user_query, session_id, db)
        return response.response

    def clear_session(self, session_id: str, db: Session = None) -> bool:
        """
        清除会话历史

        Args:
            session_id: 会话 ID
            db: 数据库会话

        Returns:
            bool: 是否成功
        """
        if self.memory_service:
            return self.memory_service.clear_session(session_id, db)
        return False

    def list_sessions(self, db: Session = None) -> List[Dict]:
        """
        列出所有会话

        Args:
            db: 数据库会话

        Returns:
            List[Dict]: 会话列表
        """
        if self.memory_service:
            return self.memory_service.list_sessions(db)
        return []


# 全局 Agent 实例
agent_instance = None


def get_agent() -> PlanExecuteAgent:
    """获取 Agent 实例（单例模式）"""
    global agent_instance
    if agent_instance is None:
        agent_instance = PlanExecuteAgent()
    return agent_instance

"""受限 Plan-and-Execute Agent 主流程。"""
import asyncio
import time
import logging
from typing import Callable, Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import uuid

from sqlalchemy.orm import Session

from backend.app.agent.models import AgentResponse, Plan, ToolName
from backend.app.agent.fast_path import build_fast_search_plan, is_fast_path_query
from backend.app.agent.planner import get_planner
from backend.app.agent.executor import get_executor
from backend.app.agent.plan_policy import PlanPolicy
from backend.app.agent.responder import get_responder
from backend.app.agent.tool_registry import AgentToolRegistry
from backend.app.core.database import SessionLocal
from backend.app.core.config import settings
from backend.app.core.observability import (
    get_execution_timeline,
    get_context_usage,
    get_model_usage,
    get_request_id,
    record_timing,
    store_current_trace,
    timed_stage,
)
from backend.app.models.observability import AgentRun
from backend.app.models.wiki import AgentPendingAction
from backend.app.agent.events import AgentEventType, AgentRunCancelled

logger = logging.getLogger(__name__)

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
        self.executor = get_executor()
        self.plan_policy = PlanPolicy(
            self.executor.tool_registry,
            max_steps=settings.agent_plan_max_steps,
        )
        self.planner = get_planner()
        self.planner.tool_registry = self.executor.tool_registry
        self.planner.validator = self.plan_policy
        self.responder = get_responder()
        self._memory_service = None

    @property
    def memory_service(self):
        """延迟加载记忆服务"""
        if self._memory_service is None:
            try:
                from backend.app.services.memory.memory_service import get_memory_service
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
        db: Session = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        raise_errors: bool = False,
        allow_external_research: bool = False,
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

            self._raise_if_cancelled(cancel_check)
            policy = self._get_plan_policy()
            fast_path = is_fast_path_query(user_query, context)
            if fast_path:
                self._emit_event(
                    event_callback,
                    AgentEventType.STAGE_STARTED,
                    {"stage": "agent.fast_path"},
                )
                with timed_stage("agent.fast_path"):
                    plan = policy.validate(
                        build_fast_search_plan(user_query),
                        [ToolName.SEARCH_KNOWLEDGE_BASE],
                    )
                self._emit_event(
                    event_callback,
                    AgentEventType.STAGE_COMPLETED,
                    {"stage": "agent.fast_path", "status": "completed"},
                )
            else:
                self._emit_event(
                    event_callback,
                    AgentEventType.STAGE_STARTED,
                    {"stage": "agent.plan"},
                )
                plan_status = "completed"
                try:
                    with timed_stage("agent.plan"):
                        capabilities = policy.capabilities()
                        plan = await asyncio.to_thread(
                            self.planner.plan,
                            user_query,
                            context,
                            capabilities,
                        )
                        plan = policy.validate(
                            plan,
                            [capability.name for capability in capabilities],
                        )
                except AgentRunCancelled:
                    plan_status = "cancelled"
                    raise
                except Exception:
                    plan_status = "failed"
                    raise
                finally:
                    self._emit_event(
                        event_callback,
                        AgentEventType.STAGE_COMPLETED,
                        {"stage": "agent.plan", "status": plan_status},
                    )
            self._raise_if_cancelled(cancel_check)
            decision = policy.decide(plan)
            if decision.requires_confirmation:
                response = self._create_pending_response(
                    user_query,
                    session_id,
                    plan,
                    db,
                    start_time,
                )
            elif decision.is_retrieval_plan:
                response = await asyncio.to_thread(
                    self._run_retrieval_graph,
                    user_query,
                    session_id,
                    context,
                    plan,
                    start_time,
                    event_callback,
                    cancel_check,
                    fast_path=fast_path,
                    allow_external_research=allow_external_research,
                )
            else:
                response = await self._run_tool_plan(
                    user_query,
                    session_id,
                    plan,
                    db,
                    context,
                    start_time,
                    event_callback,
                    cancel_check,
                    allow_external_research=allow_external_research,
                )

            self._raise_if_cancelled(cancel_check)
            self._save_conversation(user_query, response, db)
            response.execution_time_ms = int((time.time() - start_time) * 1000)
            record_timing("agent.total", response.execution_time_ms)
            self._attach_observability(response)
            self._save_agent_run(response, db)
            self._store_agent_trace(response)

            logger.info(f"Agent 处理完成，总耗时 {response.execution_time_ms}ms")
            return response

        except AgentRunCancelled:
            db.rollback()
            raise
        except Exception as e:
            if raise_errors:
                db.rollback()
                raise
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            response = AgentResponse(
                query=user_query,
                response=f"抱歉，处理您的请求时出现错误：{str(e)}",
                session_id=session_id,
                confidence=0.0,
                timestamp=datetime.now(),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            record_timing("agent.total", response.execution_time_ms or 0)
            self._attach_observability(response)
            self._save_agent_run(response, db)
            self._store_agent_trace(response)
            return response

        finally:
            if should_close:
                db.close()

    def _get_plan_policy(self) -> PlanPolicy:
        """兼容测试构造方式，并保证确认执行时仍会重新校验计划。"""
        policy = getattr(self, "plan_policy", None)
        if policy is not None:
            return policy
        registry = getattr(getattr(self, "executor", None), "tool_registry", None)
        policy = PlanPolicy(
            registry or AgentToolRegistry(),
            max_steps=settings.agent_plan_max_steps,
        )
        self.plan_policy = policy
        return policy

    async def _run_tool_plan(
        self,
        user_query: str,
        session_id: str,
        plan: Plan,
        db: Session,
        context: List[Dict[str, str]],
        start_time: float,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        cancel_check: Optional[Callable[[], bool]],
        allow_external_research: bool = False,
    ) -> AgentResponse:
        """执行只读工具计划，并在失败或空结果时进行有限重规划。"""
        policy = self._get_plan_policy()
        current_plan = policy.validate(plan)
        signatures = {policy.signature(current_plan)}
        execution_result = None

        for replan_count in range(max(0, settings.agent_max_replans) + 1):
            execution_result = await self.executor.execute(
                current_plan,
                db,
                event_callback=event_callback,
                cancel_check=cancel_check,
            )
            self._raise_if_cancelled(cancel_check)
            self._emit_event(
                event_callback,
                AgentEventType.STAGE_STARTED,
                {"stage": "agent.evaluate"},
            )
            with timed_stage("agent.evaluate"):
                evaluation = policy.evaluate(execution_result)
            self._emit_event(
                event_callback,
                AgentEventType.STAGE_COMPLETED,
                {
                    "stage": "agent.evaluate",
                    "status": "completed",
                    "successful": evaluation.successful,
                },
            )
            if evaluation.successful or replan_count >= settings.agent_max_replans:
                break

            self._emit_event(
                event_callback,
                AgentEventType.STAGE_STARTED,
                {"stage": "agent.replan", "attempt": replan_count + 1},
            )
            replan_status = "completed"
            try:
                with timed_stage("agent.replan"):
                    capabilities = policy.capabilities()
                    next_plan = await asyncio.to_thread(
                        self.planner.replan,
                        user_query,
                        current_plan,
                        evaluation.as_feedback(),
                        context=context,
                        capabilities=capabilities,
                        remaining_steps=max(
                            1,
                            settings.agent_plan_max_steps - execution_result.completed_steps,
                        ),
                    )
                    next_plan = policy.validate(
                        next_plan,
                        [capability.name for capability in capabilities],
                    )
            except AgentRunCancelled:
                replan_status = "cancelled"
                raise
            except Exception as exc:
                replan_status = "failed"
                logger.warning("Agent 重规划失败，保留首次执行结果: %s", exc)
                break
            finally:
                self._emit_event(
                    event_callback,
                    AgentEventType.STAGE_COMPLETED,
                    {"stage": "agent.replan", "status": replan_status},
                )

            signature = policy.signature(next_plan)
            if signature in signatures:
                logger.warning("重规划返回了已执行计划，阻止重复调用")
                break
            signatures.add(signature)
            next_decision = policy.decide(next_plan)
            if next_decision.requires_confirmation:
                return self._create_pending_response(
                    user_query,
                    session_id,
                    next_plan,
                    db,
                    start_time,
                )
            if next_decision.is_retrieval_plan:
                return await asyncio.to_thread(
                    self._run_retrieval_graph,
                    user_query,
                    session_id,
                    context,
                    next_plan,
                    start_time,
                    event_callback,
                    cancel_check,
                    allow_external_research=allow_external_research,
                )
            current_plan = next_plan

        return await self._generate_tool_response(
            user_query,
            session_id,
            current_plan,
            execution_result,
            context,
            start_time,
            event_callback,
        )

    async def _generate_tool_response(
        self,
        user_query: str,
        session_id: str,
        plan: Plan,
        execution_result: Any,
        context: List[Dict[str, str]],
        start_time: float,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> AgentResponse:
        """统一执行非 RAG 工具计划的最终回答阶段。"""
        self._emit_event(
            event_callback,
            AgentEventType.STAGE_STARTED,
            {"stage": "agent.generation"},
        )
        generation_status = "completed"
        try:
            with timed_stage("agent.generation"):
                response_kwargs = {}
                token_callback = self._token_callback(event_callback)
                if token_callback:
                    response_kwargs["token_callback"] = token_callback
                response = await asyncio.to_thread(
                    self.responder.generate_response,
                    user_query,
                    plan,
                    execution_result,
                    context,
                    **response_kwargs,
                )
        except AgentRunCancelled:
            generation_status = "cancelled"
            raise
        except Exception:
            generation_status = "failed"
            raise
        finally:
            self._emit_event(
                event_callback,
                AgentEventType.STAGE_COMPLETED,
                {"stage": "agent.generation", "status": generation_status},
            )
        response.session_id = session_id
        response.execution_time_ms = int((time.time() - start_time) * 1000)
        return response

    def _run_retrieval_graph(
        self,
        user_query: str,
        session_id: str,
        context: List[Dict[str, str]],
        plan: Plan,
        start_time: float,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        fast_path: bool = False,
        allow_external_research: bool = False,
    ) -> AgentResponse:
        """执行只读的 Agentic RAG 图。"""
        from backend.app.agent.graph import get_agent_graph
        from backend.app.services.retrieval.evidence_service import assess_evidence

        initial_state = {
            "query": user_query,
            "session_id": session_id,
            "context": context,
            "plan": plan,
            "fast_path": fast_path,
            "skip_query_rewrite": fast_path,
            "retrieval_recovery_attempted": False,
            "retrieval_recovery_pending": False,
            "crag_grading_failed": False,
            "retrieval_query": None,
            "rewritten_queries": [],
            "search_results": None,
            "execution_results": None,
            "relevance_grade": None,
            "crag_max_score": None,
            "crag_upper_threshold": settings.crag_upper_threshold,
            "crag_lower_threshold": settings.crag_lower_threshold,
            "crag_coverage": None,
            "crag_support_count": 0,
            "crag_limited_support_count": 0,
            "crag_incorrect_count": 0,
            "refined_content": None,
            "retrieval_stats": None,
            "answer": None,
            "sources": None,
            "confidence": 0.0,
            # 检索流程默认处于“证据不足”，只有 generate_node 通过门禁后才改为 sufficient。
            "evidence_status": "insufficient",
            "evidence_score": None,
            "evidence_source_count": 0,
            "evidence_reason": None,
            "iter_count": 0,
            "max_iterations": settings.agent_max_iterations,
            "error": None,
            "event_callback": event_callback,
            "cancel_check": cancel_check,
            "token_callback": self._token_callback(event_callback),
        }
        final_state = get_agent_graph().invoke(initial_state)
        with timed_stage("agent.evidence"):
            evidence = assess_evidence(final_state.get("search_results"))
        if final_state.get("error"):
            logger.warning("图执行过程中出现错误: %s", final_state["error"])
        answer = final_state.get("answer")
        if not answer and final_state.get("relevance_grade") == "incorrect":
            answer = "现有 Wiki 中没有找到足以支持回答的证据，请补充相关页面或缩小查询范围。"
        evidence_status = final_state.get("evidence_status") or evidence.status
        if final_state.get("relevance_grade") == "incorrect":
            evidence_status = "insufficient"
        return AgentResponse(
            query=user_query,
            response=answer or "现有 Wiki 证据不足，暂时无法回答该问题。",
            session_id=session_id,
            plan=plan,
            sources=final_state.get("sources", []),
            confidence=final_state.get("confidence", 0.0),
            evidence_status=evidence_status,
            evidence_score=final_state.get("evidence_score", evidence.score),
            evidence_source_count=final_state.get("evidence_source_count", evidence.source_count),
            evidence_reason=final_state.get("evidence_reason", evidence.reason),
            external_research_available=(
                allow_external_research
                and evidence_status == "insufficient"
                and settings.mcp_research_available()
            ),
            retrieval_stats=final_state.get("retrieval_stats"),
            timestamp=datetime.now(),
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    @staticmethod
    def _attach_observability(response: AgentResponse) -> None:
        """把请求级追踪快照附加到响应，供前端和持久化使用。"""
        response.request_id = get_request_id()
        if settings.observability_enabled:
            # 先冻结本次模型调用快照，再把调用结果映射到时间线节点。
            response.model_usage = get_model_usage()
            response.token_budget = PlanExecuteAgent._build_token_budget(
                response,
                get_context_usage(),
            )
            response.timeline = PlanExecuteAgent._enrich_timeline(
                get_execution_timeline(),
                response,
            )

    @staticmethod
    def _build_token_budget(
        response: AgentResponse,
        context_usage: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """汇总本次请求的模型预算、RAG 上下文预算和工具上下文预算。"""
        usage = response.model_usage or {}
        retrieval = response.retrieval_stats or {}
        calls = list(usage.get("calls") or [])
        output_budget = int(usage.get("output_token_budget", 0) or 0)
        output_used = int(usage.get("completion_tokens", 0) or 0)
        input_budget = int(usage.get("input_token_budget", 0) or 0)
        input_used = int(usage.get("prompt_tokens", 0) or 0)

        tool_used = 0
        execution = response.execution_result
        tool_steps = []
        response_summary = None
        if execution:
            tool_used = sum(int(item.context_tokens or 0) for item in execution.results)
            tool_steps = [
                {
                    "step_id": item.step_id,
                    "tool": item.tool_name.value,
                    "budget": settings.agent_tool_context_token_budget,
                    "used": int(item.context_tokens or 0),
                    "remaining": max(
                        0,
                        settings.agent_tool_context_token_budget - int(item.context_tokens or 0),
                    ),
                    "truncated": bool(item.context_truncated),
                }
                for item in execution.results
            ]
            response_summary = (execution.context_stats or {}).get("response_summary")
        tool_budget = (
            settings.agent_tool_context_token_budget * len(execution.results)
            if execution
            else 0
        )

        rag_budget = retrieval.get("token_budget")
        rag_used = retrieval.get("context_tokens")
        model_contexts = {}
        for stage, stats in (context_usage or {}).items():
            total_budget = int(stats.get("total_budget", 0) or 0)
            used = int(stats.get("used_tokens", 0) or 0)
            input_budget_for_stage = int(stats.get("input_budget", 0) or 0)
            model_contexts[stage] = {
                **stats,
                "remaining": max(0, input_budget_for_stage - used),
                "budget_remaining": max(0, total_budget - used),
            }
        return {
            "definition": "模型输入/输出预算分别累计；RAG 和工具上下文预算独立统计，不将不同上下文窗口相加。",
            "output": {
                "budget": output_budget,
                "used": output_used,
                "remaining": max(0, output_budget - output_used),
            },
            "input": {
                "budget": input_budget,
                "used": input_used,
                "remaining": max(0, input_budget - input_used),
            },
            "model": {
                "total_used": int(usage.get("total_tokens", 0) or 0),
                "call_count": len(calls),
                "context_window": int(usage.get("context_window_tokens", 0) or 0),
            },
            "contexts": {
                "rag": {
                    "budget": int(rag_budget) if rag_budget is not None else None,
                    "used": int(rag_used) if rag_used is not None else None,
                    "remaining": (
                        max(0, int(rag_budget) - int(rag_used))
                        if rag_budget is not None and rag_used is not None
                        else None
                    ),
                    "truncated": bool(retrieval.get("context_truncated", False)),
                    "selected_results": int(retrieval.get("selected_results", 0) or 0),
                },
                "tool": {
                    "budget": tool_budget,
                    "used": tool_used,
                    "remaining": max(0, tool_budget - tool_used),
                    "per_step": settings.agent_tool_context_token_budget if execution else 0,
                    "steps": tool_steps,
                },
                "model": model_contexts,
                "response_summary": response_summary,
                "memory": {
                    "history_budget": settings.memory_context_token_budget,
                    "summary_budget": settings.memory_summary_token_budget,
                    "summary_trigger": settings.memory_summary_trigger_tokens,
                    "summary_input_budget": settings.memory_summary_input_token_budget,
                },
            },
            "calls": [
                {
                    "stage": call.get("operation") or "llm",
                    "budget": int(call.get("token_budget", 0) or 0),
                    "used": int(call.get("completion_tokens", 0) or 0),
                    "total_used": int(call.get("total_tokens", 0) or 0),
                    "input_tokens": int(call.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(call.get("completion_tokens", 0) or 0),
                    "remaining": int(call.get("token_remaining", 0) or 0),
                }
                for call in calls
            ],
        }

    @staticmethod
    def _enrich_timeline(
        timeline: List[Dict[str, Any]],
        response: AgentResponse,
    ) -> List[Dict[str, Any]]:
        """为耗时节点补充安全的结构化结果，供运行追踪展开查看。"""
        stats = response.retrieval_stats or {}
        crag = stats.get("crag") or {}
        quality = PlanExecuteAgent._answer_quality(response)
        model_calls = list((response.model_usage or {}).get("calls") or [])
        model_index = 0

        def model_result_for(stage: str) -> Dict[str, Any]:
            """按 operation 找到模型调用，补齐流式生成等没有 llm 时间线节点的阶段。"""
            call = next(
                (
                    item
                    for item in model_calls
                    if item.get("operation") == stage
                    or str(item.get("operation") or "").startswith(f"{stage}.")
                ),
                None,
            )
            if not call:
                return {}
            return {
                key: call.get(key)
                for key in (
                    "model",
                    "total_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "token_budget",
                    "token_remaining",
                    "input_budget",
                    "input_remaining",
                )
                if call.get(key) is not None
            }

        results: List[Dict[str, Any]] = []
        for item in timeline:
            stage = item.get("stage")
            result: Optional[Dict[str, Any]] = None
            if stage == "agent.plan":
                plan = response.plan
                result = {
                    "intent": plan.intent.value if plan else None,
                    "goal": plan.goal if plan else None,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "tool": step.tool_name.value,
                            "description": step.description,
                        }
                        for step in (plan.steps if plan else [])
                    ],
                }
                result.update(model_result_for("agent.plan"))
            elif stage == "agent.fast_path":
                result = {"mode": "fast_path", "planner_skipped": True}
            elif stage == "agent.replan":
                result = model_result_for("agent.replan")
            elif stage == "agent.query_rewrite":
                result = {
                    "retrieval_query": stats.get("retrieval_query"),
                    "rewritten_queries": stats.get("rewritten_queries", []),
                }
                result.update(model_result_for("agent.query_rewrite"))
            elif stage in {"agent.retrieve", "retrieval.recall"}:
                result = {
                    key: stats.get(key, 0)
                    for key in (
                        "query_count",
                        "bm25_hits",
                        "embedding_hits",
                        "fused_candidates",
                        "reranked_candidates",
                        "selected_results",
                        "context_tokens",
                        "token_budget",
                    )
                }
            elif stage == "retrieval.embedding":
                result = {"hits": stats.get("embedding_hits", 0)}
            elif stage == "retrieval.bm25":
                result = {"hits": stats.get("bm25_hits", 0)}
            elif stage == "retrieval.fusion":
                result = {"fused_candidates": stats.get("fused_candidates", 0)}
            elif stage == "retrieval.fetch_chunks":
                result = {"fetched_candidates": stats.get("reranked_candidates", 0)}
            elif stage == "retrieval.rerank":
                selected_documents = []
                for source in response.sources or []:
                    if not isinstance(source, dict):
                        continue
                    selected_documents.append(
                        {
                            "title": source.get("title")
                            or source.get("filename")
                            or source.get("page_id")
                            or "知识库页面",
                            "score": source.get("rerank_score", source.get("score")),
                            "rrf_score": source.get("rrf_score"),
                            "section_title": source.get("section_title"),
                        }
                    )
                result = {
                    "candidates": stats.get("reranked_candidates", 0),
                    "selected": stats.get("quality_filtered_candidates", 0)
                    or stats.get("selected_results", 0),
                    "selected_documents": selected_documents,
                }
            elif stage == "agent.crag_grade":
                result = {
                    key: crag.get(key)
                    for key in (
                        "model_grade",
                        "coverage",
                        "max_score",
                        "upper_threshold",
                        "lower_threshold",
                        "support_count",
                        "limited_support_count",
                        "incorrect_count",
                        "accepted_count",
                        "grading_failed",
                        "skipped",
                    )
                    if key in crag
                }
                result.update(model_result_for("agent.crag_grade"))
            elif stage == "agent.evidence":
                result = quality
            elif stage == "agent.evaluate":
                execution = response.execution_result
                tool_results = (execution.context_stats or {}).get("tool_results", []) if execution else []
                tool_used = sum(int(item.get("context_tokens", 0) or 0) for item in tool_results)
                result = {
                    "tool_context_tokens": tool_used,
                    "tool_context_budget": settings.agent_tool_context_token_budget * len(tool_results),
                    "tool_steps": len(tool_results),
                }
            elif stage == "agent.generation":
                result = {
                    "answer_generated": bool(response.response),
                    "source_count": len(response.sources or []),
                    "evidence_status": response.evidence_status,
                }
                result.update(model_result_for("agent.generation"))
            elif stage.startswith("llm."):
                if model_index < len(model_calls):
                    call = model_calls[model_index]
                    model_index += 1
                    result = {
                        key: call.get(key)
                        for key in (
                            "operation",
                            "model",
                            "total_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                            "duration_ms",
                            "success",
                            "finish_reason",
                            "token_budget",
                            "token_remaining",
                            "input_budget",
                            "input_remaining",
                        )
                        if call.get(key) is not None
                    }
            elif stage == "agent.total":
                result = {"total_ms": response.execution_time_ms or 0}

            enriched = dict(item)
            enriched["result"] = result or {
                "status": item.get("status", "completed")
            }
            results.append(enriched)
        return results

    @staticmethod
    def _answer_quality(response: AgentResponse) -> Dict[str, Any]:
        """提取可展示的查询质量，不保存答案正文。"""
        return {
            "confidence": response.confidence,
            "evidence_status": response.evidence_status,
            "evidence_score": response.evidence_score,
            "evidence_source_count": response.evidence_source_count,
            "evidence_reason": response.evidence_reason,
        }

    @classmethod
    def _store_agent_trace(cls, response: AgentResponse) -> None:
        """保存 Agent 查询追踪，避免把 HTTP 请求混入运行追踪。"""
        if not settings.observability_enabled or not response.request_id:
            return
        store_current_trace(
            request_id=response.request_id,
            method="AGENT",
            path="agent.run",
            status_code=200 if response.confidence > 0 else 500,
            total_ms=response.execution_time_ms or 0,
            status="completed" if response.confidence > 0 else "failed",
            trace_type="agent",
            query=response.query,
            answer_quality=cls._answer_quality(response),
            retrieval_stats=response.retrieval_stats,
            timeline=response.timeline,
            model_usage=response.model_usage,
            token_budget=response.token_budget,
        )

    @staticmethod
    def _save_agent_run(response: AgentResponse, db: Session) -> None:
        """持久化运行统计；失败不影响主请求。"""
        if not settings.observability_enabled or not response.request_id:
            return
        try:
            run = db.get(AgentRun, response.request_id)
            if run is None:
                run = AgentRun(request_id=response.request_id, query=response.query)
                db.add(run)
            run.session_id = response.session_id
            run.query = response.query
            run.intent = response.plan.intent.value if response.plan else None
            # “证据不足但已安全拒答”是正常完成，不应被追踪接口误报为 500。
            run.status = (
                "completed"
                if response.confidence > 0
                or response.evidence_status in {"sufficient", "insufficient"}
                else "failed"
            )
            run.execution_time_ms = response.execution_time_ms
            run.timeline = response.timeline
            run.retrieval_stats = response.retrieval_stats
            run.model_usage = response.model_usage
            runtime_snapshot = dict(run.runtime_snapshot or {})
            runtime_snapshot["answer_quality"] = PlanExecuteAgent._answer_quality(response)
            runtime_snapshot["token_budget"] = response.token_budget or {}
            run.runtime_snapshot = runtime_snapshot
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("保存 Agent 运行统计失败: %s", exc)

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

            plan = self._get_plan_policy().validate(
                Plan.model_validate(pending.plan_data)
            )
            if not self._get_plan_policy().decide(plan).requires_confirmation:
                raise AgentActionError("待确认计划不包含需要确认的知识变更")
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
                self._reset_research_save_state(pending, db)
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
        self._reset_research_save_state(pending, db)
        db.commit()
        return {"action_id": action_id, "status": "cancelled"}

    @staticmethod
    def _reset_research_save_state(pending: AgentPendingAction, db: Session) -> None:
        """取消或失败后允许同一研究草稿重新发起保存确认。"""
        from backend.app.models.wiki import ExternalResearchRun

        try:
            plan = Plan.model_validate(pending.plan_data)
        except Exception:
            return
        for step in plan.steps:
            run_id = step.parameters.get("research_run_id")
            if not run_id:
                continue
            run = db.get(ExternalResearchRun, run_id)
            if run is not None and run.status == "save_pending":
                run.status = "completed"

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
            PlanExecuteAgent._reset_research_save_state(pending, db)
            db.commit()
            raise AgentActionError("待确认操作已过期")

    def _save_conversation(
        self,
        user_query: str,
        response: AgentResponse,
        db: Session,
        summarize: bool = True,
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
            if summarize:
                self.memory_service.summarize_if_needed(response.session_id, db=db)
        except Exception as exc:
            logger.warning("保存对话历史失败: %s", exc)

    @staticmethod
    def _emit_event(
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        event_type: AgentEventType,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if event_callback:
            event_callback(event_type.value, data or {})

    @staticmethod
    def _token_callback(
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> Optional[Callable[[str], None]]:
        if event_callback is None:
            return None

        def emit(chunk: str) -> None:
            event_callback(AgentEventType.TOKEN.value, {"content": chunk})

        return emit

    @staticmethod
    def _raise_if_cancelled(
        cancel_check: Optional[Callable[[], bool]],
    ) -> None:
        if cancel_check and cancel_check():
            raise AgentRunCancelled("Agent 运行已取消")

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
                "evidence_reason": response.evidence_reason,
                "external_research_available": response.external_research_available,
                "action_preview": response.action_preview,
                "request_id": response.request_id,
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

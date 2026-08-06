"""
Plan-and-Execute Agent - Executor 执行器（课堂学习场景聚焦版）
执行计划中的工具调用，支持无依赖步骤并行执行
"""
import asyncio
import time
from typing import Callable, Dict, Any, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor as ThreadExecutor
import logging

from sqlalchemy.orm import Session

from backend.app.agent.models import (
    Plan, PlanStep, ToolName,
    ToolResult, ExecutionResult,
)
from backend.app.core.database import SessionLocal
from backend.app.core.config import settings
from backend.app.agent.events import AgentEventType, AgentRunCancelled
from backend.app.agent.tool_registry import AgentToolRegistry
from backend.app.services.token_budget_service import limit_context

logger = logging.getLogger(__name__)


def result_to_log(step: PlanStep, result: ToolResult) -> str:
    """将步骤结果转为日志字符串"""
    status = "成功" if result.success else f"失败: {result.error_message}"
    ms = f" ({result.execution_time_ms}ms)" if result.execution_time_ms else ""
    return f"步骤 {step.step_id}: {step.description} → {status}{ms}"


class Executor:
    """
    Agent 执行器
    负责执行计划中的工具调用
    """

    def __init__(self, tool_registry: Optional[AgentToolRegistry] = None):
        """初始化执行器"""
        self._thread_pool = ThreadExecutor(max_workers=4)
        self.tool_registry = tool_registry or AgentToolRegistry()
        self.tools = dict(self.tool_registry.handlers)

    @property
    def hybrid_retrieval(self):
        """延迟加载混合检索，避免非检索意图依赖本地模型目录。"""
        return self.tool_registry.hybrid_retrieval

    def _build_waves(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """
        拓扑排序：将步骤按依赖关系分层（wave），同层步骤可并行执行

        Args:
            steps: 计划步骤列表

        Returns:
            List[List[PlanStep]]: 分层后的步骤列表
        """
        step_map = {s.step_id: s for s in steps}
        completed: Set[int] = set()
        remaining = set(step_map.keys())
        waves = []

        while remaining:
            # 找出所有依赖已满足的步骤
            wave = []
            for sid in list(remaining):
                step = step_map[sid]
                deps = step.depends_on or []
                if all(d in completed for d in deps):
                    wave.append(step)

            if not wave:
                # 剩余步骤有循环依赖，强制逐个执行
                logger.warning(f"检测到循环依赖或无法满足的依赖，剩余 {len(remaining)} 步骤串行执行")
                for sid in list(remaining):
                    waves.append([step_map[sid]])
                break

            waves.append(wave)
            for s in wave:
                completed.add(s.step_id)
                remaining.discard(s.step_id)

        return waves

    async def execute(
        self,
        plan: Plan,
        db: Session = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ExecutionResult:
        """
        执行计划（无依赖步骤并行执行）

        Args:
            plan: 执行计划
            db: 数据库会话

        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()
        results = []
        execution_log = []
        final_data = {}

        # 管理数据库会话
        if db is None:
            db = SessionLocal()
            should_close = True
        else:
            should_close = False

        try:
            waves = self._build_waves(plan.steps)
            logger.info(f"计划共 {len(plan.steps)} 步，分为 {len(waves)} 个执行波次")

            for wave_idx, wave in enumerate(waves):
                self._raise_if_cancelled(cancel_check)
                if len(wave) == 1:
                    # 单步骤直接执行
                    result = self._execute_step(
                        wave[0],
                        db,
                        results,
                        plan.original_query,
                        event_callback,
                        cancel_check,
                    )
                    results.append(result)
                    execution_log.append(result_to_log(wave[0], result))
                    if result.success:
                        self._aggregate_data(final_data, wave[0].tool_name, result.result)
                else:
                    # 多步骤并行执行（每个线程使用独立 db session）
                    logger.info(f"波次 {wave_idx + 1}: {len(wave)} 个步骤并行执行")
                    loop = asyncio.get_event_loop()
                    futures = []
                    for step in wave:
                        futures.append(
                            loop.run_in_executor(
                                self._thread_pool,
                                self._execute_step_concurrent,
                                step,
                                results,
                                plan.original_query,
                                event_callback,
                                cancel_check,
                            )
                        )
                    wave_results = await asyncio.gather(*futures)
                    for result, step in zip(wave_results, wave):
                        results.append(result)
                        execution_log.append(result_to_log(step, result))
                        if result.success:
                            self._aggregate_data(final_data, step.tool_name, result.result)

            total_time = int((time.time() - start_time) * 1000)
            success = all(r.success for r in results)
            completed_steps = sum(1 for r in results if r.success)

            return ExecutionResult(
                plan=plan,
                results=results,
                completed_steps=completed_steps,
                total_steps=len(plan.steps),
                success=success,
                execution_log=execution_log,
                final_data=final_data,
                context_stats={
                    "tool_results": [
                        {
                            "step_id": item.step_id,
                            "tool_name": item.tool_name.value,
                            "context_tokens": item.context_tokens,
                            "truncated": item.context_truncated,
                        }
                        for item in results
                    ]
                },
            )

        finally:
            if should_close:
                db.close()

    def _resolve_step_references(self, parameters: Dict, prev_results: List[ToolResult]) -> Dict:
        """
        解析参数中的步骤结果引用（如 $step_1_results）

        Args:
            parameters: 步骤参数
            prev_results: 已执行步骤的结果列表

        Returns:
            Dict: 解析后的参数
        """
        import copy
        resolved = copy.deepcopy(parameters)
        for key, value in resolved.items():
            if isinstance(value, str) and value.startswith("$step_") and value.endswith("_results"):
                # 提取步骤ID: $step_1_results → 1
                try:
                    ref_step_id = int(value.split("_")[1])
                    dep_result = next(
                        (r for r in prev_results if r.step_id == ref_step_id and r.success),
                        None
                    )
                    if dep_result and dep_result.result:
                        # 将检索结果格式化为文本
                        formatted = self._format_search_results(dep_result.result)
                        limited = limit_context(
                            formatted,
                            settings.agent_tool_context_token_budget,
                        )
                        resolved[key] = limited.text
                        logger.info(f"解析引用 {value} → 获取到 {len(resolved[key])} 字符, 前100字: {resolved[key][:100]}")
                    else:
                        resolved[key] = ""
                        logger.warning(f"引用 {value} 对应的步骤未成功执行")
                except (ValueError, IndexError):
                    logger.warning(f"无法解析引用: {value}")
        return resolved

    def _format_search_results(self, search_result: Any) -> str:
        """将检索结果格式化为可写入的文本"""
        if isinstance(search_result, dict):
            # summarize_text 工具返回的格式：{"success": True, "summary": "..."}
            if "summary" in search_result:
                return search_result["summary"]
            items = search_result.get("results", search_result.get("items", []))
            if isinstance(items, list):
                parts = []
                for i, item in enumerate(items, 1):
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        content = item.get("content", item.get("text", ""))
                        score = item.get("score", "")
                        parts.append(f"### {i}. {title}" if title else f"### {i}")
                        if content:
                            parts.append(content)
                        if score:
                            parts.append(f"*相关度: {score}*")
                        parts.append("")
                return "\n".join(parts)
            # 如果是简单字典，直接取内容
            return search_result.get("content", search_result.get("text", str(search_result)))
        if isinstance(search_result, list):
            parts = []
            for i, item in enumerate(search_result, 1):
                if isinstance(item, dict):
                    content = item.get("content", item.get("text", str(item)))
                    parts.append(f"### {i}\n{content}")
                else:
                    parts.append(f"### {i}\n{item}")
            return "\n\n".join(parts)
        return str(search_result)

    def _execute_step(
        self,
        step: PlanStep,
        db: Session,
        prev_results: List[ToolResult],
        original_query: str = "",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ToolResult:
        """
        执行单个步骤（同步，在当前线程）

        Args:
            step: 计划步骤
            db: 数据库会话
            prev_results: 已有结果列表（用于依赖检查）
            original_query: 原始用户查询（传递给 summarize_text）

        Returns:
            ToolResult: 执行结果
        """
        step_start = time.time()
        logger.info(f"执行步骤 {step.step_id}: {step.tool_name.value}")
        self._raise_if_cancelled(cancel_check)
        self._emit_tool_event(event_callback, AgentEventType.TOOL_STARTED, step)

        # 检查依赖
        if step.depends_on:
            for dep_id in step.depends_on:
                dep_result = next((r for r in prev_results if r.step_id == dep_id and r.success), None)
                if not dep_result:
                    tool_result = ToolResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        success=False,
                        result=None,
                        error_message=f"依赖步骤 {dep_id} 未成功执行",
                        execution_time_ms=int((time.time() - step_start) * 1000),
                    )
                    self._emit_tool_event(
                        event_callback,
                        AgentEventType.TOOL_COMPLETED,
                        step,
                        tool_result,
                    )
                    return tool_result

        try:
            tool_func = self.tools.get(step.tool_name)
            if not tool_func:
                tool_result = ToolResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    result=None,
                    error_message=f"未知工具: {step.tool_name.value}",
                    execution_time_ms=int((time.time() - step_start) * 1000),
                )
                self._emit_tool_event(
                    event_callback,
                    AgentEventType.TOOL_COMPLETED,
                    step,
                    tool_result,
                )
                return tool_result

            # 解析参数中的步骤结果引用（如 $step_1_results）
            resolved_params = self._resolve_step_references(step.parameters, prev_results)
            # summarize_text 需要原始查询用于相关性过滤
            if step.tool_name == ToolName.SUMMARIZE_TEXT:
                result = tool_func(resolved_params, db, query=original_query)
            else:
                result = tool_func(resolved_params, db)
            step_time = int((time.time() - step_start) * 1000)
            context = limit_context(result, settings.agent_tool_context_token_budget)
            tool_result = ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=True,
                result=result,
                execution_time_ms=step_time,
                context_tokens=context.used_tokens,
                context_truncated=context.truncated,
            )
            self._emit_tool_event(
                event_callback,
                AgentEventType.TOOL_COMPLETED,
                step,
                tool_result,
            )
            return tool_result
        except AgentRunCancelled:
            raise
        except Exception as e:
            step_time = int((time.time() - step_start) * 1000)
            logger.error(f"工具执行失败: {step.tool_name.value}, 错误: {e}")
            tool_result = ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=step_time
            )
            self._emit_tool_event(
                event_callback,
                AgentEventType.TOOL_COMPLETED,
                step,
                tool_result,
            )
            return tool_result

    def _execute_step_concurrent(
        self,
        step: PlanStep,
        prev_results: List[ToolResult],
        original_query: str = "",
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ToolResult:
        """
        并发执行单个步骤（独立线程，使用独立 db session）
        """
        step_start = time.time()
        logger.info(f"[并发] 执行步骤 {step.step_id}: {step.tool_name.value}")
        self._raise_if_cancelled(cancel_check)
        self._emit_tool_event(event_callback, AgentEventType.TOOL_STARTED, step)

        # 检查依赖
        if step.depends_on:
            for dep_id in step.depends_on:
                dep_result = next((r for r in prev_results if r.step_id == dep_id and r.success), None)
                if not dep_result:
                    tool_result = ToolResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        success=False,
                        result=None,
                        error_message=f"依赖步骤 {dep_id} 未成功执行",
                        execution_time_ms=int((time.time() - step_start) * 1000),
                    )
                    self._emit_tool_event(
                        event_callback,
                        AgentEventType.TOOL_COMPLETED,
                        step,
                        tool_result,
                    )
                    return tool_result

        db = SessionLocal()
        try:
            tool_func = self.tools.get(step.tool_name)
            if not tool_func:
                tool_result = ToolResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=False,
                    result=None,
                    error_message=f"未知工具: {step.tool_name.value}",
                    execution_time_ms=int((time.time() - step_start) * 1000),
                )
                self._emit_tool_event(
                    event_callback,
                    AgentEventType.TOOL_COMPLETED,
                    step,
                    tool_result,
                )
                return tool_result

            # 解析参数中的步骤结果引用（如 $step_1_results）
            resolved_params = self._resolve_step_references(step.parameters, prev_results)
            # summarize_text 需要原始查询用于相关性过滤
            if step.tool_name == ToolName.SUMMARIZE_TEXT:
                result = tool_func(resolved_params, db, query=original_query)
            else:
                result = tool_func(resolved_params, db)
            step_time = int((time.time() - step_start) * 1000)
            context = limit_context(result, settings.agent_tool_context_token_budget)
            tool_result = ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=True,
                result=result,
                execution_time_ms=step_time,
                context_tokens=context.used_tokens,
                context_truncated=context.truncated,
            )
            self._emit_tool_event(
                event_callback,
                AgentEventType.TOOL_COMPLETED,
                step,
                tool_result,
            )
            return tool_result
        except AgentRunCancelled:
            raise
        except Exception as e:
            step_time = int((time.time() - step_start) * 1000)
            logger.error(f"[并发] 工具执行失败: {step.tool_name.value}, 错误: {e}")
            tool_result = ToolResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=step_time
            )
            self._emit_tool_event(
                event_callback,
                AgentEventType.TOOL_COMPLETED,
                step,
                tool_result,
            )
            return tool_result
        finally:
            db.close()

    @staticmethod
    def _raise_if_cancelled(cancel_check: Optional[Callable[[], bool]]) -> None:
        if cancel_check and cancel_check():
            raise AgentRunCancelled("Agent 运行已取消")

    @staticmethod
    def _emit_tool_event(
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        event_type: AgentEventType,
        step: PlanStep,
        result: Optional[ToolResult] = None,
    ) -> None:
        if event_callback is None:
            return
        data = {
            "step_id": step.step_id,
            "tool_name": step.tool_name.value,
            "description": step.description,
        }
        if result is not None:
            data.update(
                {
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error_message,
                    "context_tokens": result.context_tokens,
                    "context_truncated": result.context_truncated,
                }
            )
        event_callback(event_type.value, data)

    def _aggregate_data(self, final_data: Dict, tool_name: ToolName, result: Any):
        """
        聚合工具执行结果到 final_data

        Args:
            final_data: 最终数据字典
            tool_name: 工具名称
            result: 工具返回结果
        """
        if tool_name == ToolName.SEARCH_KNOWLEDGE_BASE:
            final_data["search_results"] = result
        elif tool_name == ToolName.LIST_NOTES:
            final_data["notes_list"] = result
        elif tool_name == ToolName.GET_CURRENT_TIME:
            final_data["current_time"] = result
        elif tool_name == ToolName.CREATE_NOTE:
            final_data["created_note"] = result
        elif tool_name == ToolName.UPDATE_NOTE:
            final_data["updated_note"] = result
        elif tool_name == ToolName.DELETE_NOTE:
            final_data["deleted_note"] = result
        elif tool_name == ToolName.SUMMARIZE_TEXT:
            final_data["summarized_text"] = result

    def search_knowledge_base(self, parameters: Dict, db: Session) -> List[Dict]:
        """兼容旧调用入口，具体实现由工具注册表负责。"""
        return self.tool_registry.search_knowledge_base(parameters, db)

    def create_note(self, parameters: Dict, db: Session) -> Dict:
        """兼容旧调用入口。"""
        return self.tool_registry.create_note(parameters, db)

    def update_note(self, parameters: Dict, db: Session) -> Dict:
        """兼容旧调用入口。"""
        return self.tool_registry.update_note(parameters, db)

    def delete_note(self, parameters: Dict, db: Session) -> Dict:
        """兼容旧调用入口。"""
        return self.tool_registry.delete_note(parameters, db)

    def list_notes(self, parameters: Dict, db: Session) -> List[Dict]:
        """兼容旧调用入口。"""
        return self.tool_registry.list_notes(parameters, db)

    def get_current_time(self, parameters: Dict, db: Session = None) -> Dict:
        """兼容旧调用入口。"""
        return self.tool_registry.get_current_time(parameters, db)

    def summarize_text(self, parameters: Dict, db: Session = None, query: str = "") -> Dict:
        """兼容旧调用入口。"""
        return self.tool_registry.summarize_text(parameters, db, query)


# 全局执行器实例
executor_instance = None


def get_executor() -> Executor:
    """获取执行器实例（单例模式）"""
    global executor_instance
    if executor_instance is None:
        executor_instance = Executor()
    return executor_instance

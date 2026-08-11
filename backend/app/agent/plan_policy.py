"""LLM 计划的确定性校验、风险判断与执行结果评估。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from backend.app.agent.models import (
    ExecutionResult,
    Plan,
    PlanStep,
    ToolCapability,
    ToolName,
    ToolRiskLevel,
)
from backend.app.agent.tool_registry import AgentToolRegistry

STEP_REFERENCE_PATTERN = re.compile(r"^\$step_(\d+)_results$")
RISK_ORDER = {
    ToolRiskLevel.READ: 0,
    ToolRiskLevel.WRITE: 1,
    ToolRiskLevel.DELETE: 2,
}


class PlanValidationError(ValueError):
    """计划未通过工具、参数、依赖或权限校验。"""


@dataclass(frozen=True)
class PlanPolicyDecision:
    """计划执行前的确定性策略结果。"""

    requires_confirmation: bool
    highest_risk: ToolRiskLevel
    is_retrieval_plan: bool
    is_read_only: bool


@dataclass(frozen=True)
class PlanEvaluation:
    """执行结果是否满足再次规划的最低条件。"""

    successful: bool
    reasons: tuple[str, ...]
    failed_step_ids: tuple[int, ...]
    empty_step_ids: tuple[int, ...]

    def as_feedback(self) -> dict[str, Any]:
        return {
            "reasons": list(self.reasons),
            "failed_step_ids": list(self.failed_step_ids),
            "empty_step_ids": list(self.empty_step_ids),
        }


class PlanPolicy:
    """把不可信的模型计划收敛为可执行的受限 DAG。"""

    def __init__(self, registry: AgentToolRegistry, max_steps: int = 6) -> None:
        self.registry = registry
        self.max_steps = max_steps

    def capabilities(
        self,
        allowed_tools: Optional[Iterable[ToolName]] = None,
    ) -> list[ToolCapability]:
        tools = list(allowed_tools) if allowed_tools is not None else None
        return self.registry.get_capabilities(tools)

    def validate(
        self,
        plan: Plan,
        allowed_tools: Optional[Iterable[ToolName]] = None,
    ) -> Plan:
        """校验并标准化计划；任何失败都禁止进入 Executor。"""
        if not plan.steps:
            raise PlanValidationError("计划至少需要一个步骤")
        if len(plan.steps) > self.max_steps:
            raise PlanValidationError(f"计划步骤数超过上限 {self.max_steps}")

        allowed = (
            set(allowed_tools)
            if allowed_tools is not None
            else {capability.name for capability in self.capabilities()}
        )
        step_ids = [step.step_id for step in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            raise PlanValidationError("计划步骤 ID 必须唯一")
        if any(step_id <= 0 for step_id in step_ids):
            raise PlanValidationError("计划步骤 ID 必须为正整数")

        normalized_steps = []
        step_id_set = set(step_ids)
        for step in plan.steps:
            if step.tool_name not in allowed:
                raise PlanValidationError(f"工具不在当前允许列表: {step.tool_name.value}")
            dependencies = set(step.depends_on or [])
            if step.step_id in dependencies:
                raise PlanValidationError(f"步骤 {step.step_id} 不能依赖自身")
            missing_dependencies = dependencies - step_id_set
            if missing_dependencies:
                missing = ", ".join(str(item) for item in sorted(missing_dependencies))
                raise PlanValidationError(f"步骤 {step.step_id} 引用了不存在的依赖: {missing}")
            self._validate_references(step, dependencies)
            try:
                params = self.registry.get_parameter_model(step.tool_name).model_validate(
                    step.parameters
                )
            except (KeyError, ValidationError) as exc:
                raise PlanValidationError(
                    f"步骤 {step.step_id} 的 {step.tool_name.value} 参数无效: {exc}"
                ) from exc
            normalized_steps.append(
                step.model_copy(
                    update={"parameters": params.model_dump(exclude_none=True)}
                )
            )

        self._validate_acyclic(normalized_steps)
        return plan.model_copy(
            update={
                "steps": normalized_steps,
                "estimated_steps": len(normalized_steps),
                "goal": plan.goal or plan.original_query,
            }
        )

    def decide(self, plan: Plan) -> PlanPolicyDecision:
        """从工具元数据计算风险与路由，不信任 LLM 声明的 Intent。"""
        capabilities = {item.name: item for item in self.capabilities()}
        selected = [capabilities[step.tool_name] for step in plan.steps]
        highest_risk = max(
            (item.risk_level for item in selected),
            key=RISK_ORDER.__getitem__,
        )
        tool_names = {step.tool_name for step in plan.steps}
        retrieval_tools = {
            ToolName.SEARCH_KNOWLEDGE_BASE,
            ToolName.SUMMARIZE_TEXT,
        }
        return PlanPolicyDecision(
            requires_confirmation=any(item.requires_confirmation for item in selected),
            highest_risk=highest_risk,
            is_retrieval_plan=(
                ToolName.SEARCH_KNOWLEDGE_BASE in tool_names
                and tool_names.issubset(retrieval_tools)
            ),
            is_read_only=highest_risk == ToolRiskLevel.READ,
        )

    @staticmethod
    def signature(plan: Plan) -> str:
        """为实际调用结构生成稳定签名，用于阻止重复 Replan。"""
        payload = [
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name.value,
                "parameters": step.parameters,
                "depends_on": sorted(step.depends_on or []),
            }
            for step in plan.steps
        ]
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def evaluate(execution_result: ExecutionResult) -> PlanEvaluation:
        """识别失败、空结果和工具内部失败信号。"""
        failed_ids: list[int] = []
        empty_ids: list[int] = []
        reasons: list[str] = []
        for result in execution_result.results:
            step_id = result.step_id or 0
            internal_failure = (
                isinstance(result.result, dict)
                and result.result.get("success") is False
            )
            if not result.success or internal_failure:
                failed_ids.append(step_id)
                reasons.append(
                    result.error_message
                    or (
                        result.result.get("error", "工具返回失败状态")
                        if isinstance(result.result, dict)
                        else "工具执行失败"
                    )
                )
                continue
            if PlanPolicy._is_empty_result(result.result):
                empty_ids.append(step_id)
                reasons.append(f"步骤 {step_id} 未返回有效结果")

        successful = (
            execution_result.success
            and not failed_ids
            and not empty_ids
            and execution_result.completed_steps == execution_result.total_steps
        )
        return PlanEvaluation(
            successful=successful,
            reasons=tuple(reasons),
            failed_step_ids=tuple(failed_ids),
            empty_step_ids=tuple(empty_ids),
        )

    @staticmethod
    def _validate_references(step: PlanStep, dependencies: set[int]) -> None:
        for value in PlanPolicy._walk_values(step.parameters):
            if not isinstance(value, str):
                continue
            match = STEP_REFERENCE_PATTERN.fullmatch(value)
            if not match:
                continue
            referenced_step = int(match.group(1))
            if referenced_step not in dependencies:
                raise PlanValidationError(
                    f"步骤 {step.step_id} 引用步骤 {referenced_step} 的结果，"
                    "但未声明对应 depends_on"
                )

    @staticmethod
    def _walk_values(value: Any):
        if isinstance(value, dict):
            for item in value.values():
                yield from PlanPolicy._walk_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from PlanPolicy._walk_values(item)
        else:
            yield value

    @staticmethod
    def _validate_acyclic(steps: list[PlanStep]) -> None:
        dependencies = {
            step.step_id: set(step.depends_on or [])
            for step in steps
        }
        completed: set[int] = set()
        while len(completed) < len(steps):
            ready = {
                step_id
                for step_id, deps in dependencies.items()
                if step_id not in completed and deps.issubset(completed)
            }
            if not ready:
                raise PlanValidationError("计划存在循环依赖")
            completed.update(ready)

    @staticmethod
    def _is_empty_result(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set)):
            return len(value) == 0
        if isinstance(value, dict):
            if not value:
                return True
            if "results" in value:
                return PlanPolicy._is_empty_result(value["results"])
            if "items" in value:
                return PlanPolicy._is_empty_result(value["items"])
        return False

"""能力注册表、计划校验与有限重规划测试。"""

import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.agent.agent import PlanExecuteAgent
from backend.app.agent.models import (
    AgentResponse,
    ExecutionResult,
    IntentType,
    Plan,
    PlanStep,
    ToolName,
    ToolResult,
)
from backend.app.agent.plan_policy import PlanPolicy, PlanValidationError
from backend.app.agent.planner import Planner
from backend.app.agent.tool_registry import AgentToolRegistry
from backend.app.core.database import Base


def make_plan(*steps: PlanStep, intent: IntentType = IntentType.UNKNOWN) -> Plan:
    return Plan(
        goal="测试目标",
        intent=intent,
        original_query="测试查询",
        steps=list(steps),
        estimated_steps=len(steps),
        reasoning="测试计划",
    )


def make_step(
    step_id: int,
    tool_name: ToolName,
    parameters: dict,
    depends_on=None,
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        tool_name=tool_name,
        parameters=parameters,
        description=f"执行 {tool_name.value}",
        depends_on=depends_on,
    )


class FakeJSONLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def chat_json(self, **kwargs):
        return self.payload


class AgentPlanPolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.registry = AgentToolRegistry()
        self.policy = PlanPolicy(self.registry, max_steps=6)

    def test_capability_catalog_contains_json_schema_and_risk_metadata(self):
        capabilities = {item.name: item for item in self.policy.capabilities()}

        search = capabilities[ToolName.SEARCH_KNOWLEDGE_BASE]
        create = capabilities[ToolName.CREATE_NOTE]
        self.assertIn("query", search.parameters_schema["properties"])
        self.assertEqual(search.risk_level.value, "read")
        self.assertEqual(create.risk_level.value, "write")
        self.assertTrue(create.requires_confirmation)

    def test_validator_rejects_tool_outside_current_phase(self):
        plan = make_plan(make_step(1, ToolName.LIST_NOTES, {}))

        with self.assertRaisesRegex(PlanValidationError, "允许列表"):
            self.policy.validate(plan, [ToolName.SEARCH_KNOWLEDGE_BASE])

    def test_validator_rejects_invalid_parameters(self):
        plan = make_plan(
            make_step(
                1,
                ToolName.SEARCH_KNOWLEDGE_BASE,
                {"query": "RAG", "top_k": 0},
            )
        )

        with self.assertRaisesRegex(PlanValidationError, "参数无效"):
            self.policy.validate(plan)

    def test_validator_rejects_cycle_and_undeclared_reference(self):
        cycle = make_plan(
            make_step(1, ToolName.GET_CURRENT_TIME, {}, depends_on=[2]),
            make_step(2, ToolName.GET_CURRENT_TIME, {}, depends_on=[1]),
        )
        with self.assertRaisesRegex(PlanValidationError, "循环依赖"):
            self.policy.validate(cycle)

        invalid_reference = make_plan(
            make_step(1, ToolName.LIST_NOTES, {}),
            make_step(
                2,
                ToolName.SUMMARIZE_TEXT,
                {"content": "$step_1_results"},
            ),
        )
        with self.assertRaisesRegex(PlanValidationError, "depends_on"):
            self.policy.validate(invalid_reference)

    def test_policy_uses_tools_instead_of_declared_intent_for_confirmation(self):
        write_plan_with_read_intent = make_plan(
            make_step(
                1,
                ToolName.CREATE_NOTE,
                {"title": "Agent", "content": "正文"},
            ),
            intent=IntentType.SEARCH,
        )
        read_plan_with_write_intent = make_plan(
            make_step(1, ToolName.GET_CURRENT_TIME, {}),
            intent=IntentType.CREATE_NOTE,
        )

        self.assertTrue(
            self.policy.decide(self.policy.validate(write_plan_with_read_intent)).requires_confirmation
        )
        self.assertFalse(
            self.policy.decide(self.policy.validate(read_plan_with_write_intent)).requires_confirmation
        )

    def test_llm_can_generate_valid_multi_step_plan(self):
        planner = Planner(self.registry, self.policy)
        planner._llm_service = FakeJSONLLM(
            {
                "goal": "总结 RAG 笔记",
                "intent": "summarize",
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "search_knowledge_base",
                        "parameters": {"query": "RAG", "top_k": 8},
                        "description": "检索 RAG",
                        "depends_on": [],
                    },
                    {
                        "step_id": 2,
                        "tool_name": "summarize_text",
                        "parameters": {"content": "$step_1_results"},
                        "description": "总结检索结果",
                        "depends_on": [1],
                    },
                ],
                "reasoning": "先检索，再总结",
            }
        )

        plan = planner.plan("总结 RAG 笔记")

        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[1].depends_on, [1])
        self.assertTrue(self.policy.decide(plan).is_retrieval_plan)

    def test_empty_capability_set_does_not_fall_back_to_all_tools(self):
        planner = Planner(self.registry, self.policy)
        planner._llm_service = FakeJSONLLM(
            {
                "intent": "time_query",
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "get_current_time",
                        "parameters": {},
                        "description": "读取时间",
                    }
                ],
            }
        )

        with self.assertRaises(PlanValidationError):
            planner.plan("现在几点", capabilities=[])

    def test_replan_cannot_exceed_remaining_step_budget(self):
        planner = Planner(self.registry, self.policy)
        planner._llm_service = FakeJSONLLM(
            {
                "intent": "list_notes",
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "get_current_time",
                        "parameters": {},
                        "description": "读取时间",
                    },
                    {
                        "step_id": 2,
                        "tool_name": "list_notes",
                        "parameters": {},
                        "description": "列出页面",
                    },
                ],
            }
        )
        previous = make_plan(make_step(1, ToolName.LIST_NOTES, {}))

        with self.assertRaisesRegex(PlanValidationError, "剩余预算"):
            planner.replan(
                "换一种方式",
                previous,
                {"reasons": ["空结果"]},
                remaining_steps=1,
            )


class FakePlanExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.call_count = 0

    async def execute(self, plan, db, **kwargs):
        result_value = self.results[min(self.call_count, len(self.results) - 1)]
        self.call_count += 1
        tool_result = ToolResult(
            step_id=plan.steps[0].step_id,
            tool_name=plan.steps[0].tool_name,
            success=True,
            result=result_value,
        )
        return ExecutionResult(
            plan=plan,
            results=[tool_result],
            completed_steps=1,
            total_steps=1,
            success=True,
            execution_log=[],
            final_data={"result": result_value},
        )


class FakeReplanner:
    def __init__(self, plan):
        self.plan = plan
        self.call_count = 0

    def replan(self, *args, **kwargs):
        self.call_count += 1
        return self.plan


class FakePlanResponder:
    @staticmethod
    def generate_response(query, plan, execution_result, context, **kwargs):
        return AgentResponse(
            query=query,
            response="执行完成",
            plan=plan,
            execution_result=execution_result,
            confidence=1.0,
            timestamp=datetime.now(),
        )


class AgentLimitedReplanTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "planning-test.db"
        engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        self.registry = AgentToolRegistry()
        self.policy = PlanPolicy(self.registry, max_steps=6)
        self.initial_plan = make_plan(make_step(1, ToolName.LIST_NOTES, {}))

    async def asyncTearDown(self):
        self.session.close()
        self.temp_dir.cleanup()

    def make_agent(self, executor, replanner):
        agent = PlanExecuteAgent.__new__(PlanExecuteAgent)
        agent.executor = executor
        agent.planner = replanner
        agent.responder = FakePlanResponder()
        agent.plan_policy = self.policy
        agent._memory_service = False
        return agent

    async def test_empty_result_triggers_one_replan_and_executes_new_plan(self):
        next_plan = make_plan(make_step(1, ToolName.GET_CURRENT_TIME, {}))
        executor = FakePlanExecutor([[], {"time": "12:00"}])
        replanner = FakeReplanner(next_plan)
        agent = self.make_agent(executor, replanner)

        response = await agent._run_tool_plan(
            "列出笔记",
            "session-replan",
            self.initial_plan,
            self.session,
            [],
            time.time(),
            None,
            None,
        )

        self.assertEqual(executor.call_count, 2)
        self.assertEqual(replanner.call_count, 1)
        self.assertEqual(response.plan.steps[0].tool_name, ToolName.GET_CURRENT_TIME)

    async def test_identical_replan_is_not_executed_twice(self):
        executor = FakePlanExecutor([[]])
        replanner = FakeReplanner(self.initial_plan)
        agent = self.make_agent(executor, replanner)

        await agent._run_tool_plan(
            "列出笔记",
            "session-repeat",
            self.initial_plan,
            self.session,
            [],
            time.time(),
            None,
            None,
        )

        self.assertEqual(executor.call_count, 1)
        self.assertEqual(replanner.call_count, 1)

    async def test_replan_with_write_tool_pauses_for_confirmation(self):
        write_plan = make_plan(
            make_step(
                1,
                ToolName.CREATE_NOTE,
                {"title": "研究结果", "content": "正文"},
            )
        )
        executor = FakePlanExecutor([[]])
        replanner = FakeReplanner(write_plan)
        agent = self.make_agent(executor, replanner)

        response = await agent._run_tool_plan(
            "整理研究结果",
            "session-confirm",
            self.initial_plan,
            self.session,
            [],
            time.time(),
            None,
            None,
        )

        self.assertTrue(response.confirmation_required)
        self.assertIsNotNone(response.pending_action_id)
        self.assertEqual(executor.call_count, 1)


if __name__ == "__main__":
    unittest.main()

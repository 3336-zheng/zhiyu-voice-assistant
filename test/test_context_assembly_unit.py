"""分层上下文装配、长期摘要保留和 Token 预算测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.agent.models import (
    ExecutionResult,
    IntentType,
    Plan,
    PlanStep,
    ToolName,
)
from backend.app.agent.planner import Planner
from backend.app.agent.responder import Responder
from backend.app.agent.tool_registry import AgentToolRegistry
from backend.app.core.database import Base
from backend.app.services.context_assembler import (
    ContextAssembler,
)
from backend.app.services.memory_service import MemoryService
from backend.app.agent.plan_policy import PlanPolicy


class ContextAssemblerTestCase(unittest.TestCase):
    def setUp(self):
        self.assembler = ContextAssembler(
            context_window_tokens=900,
            history_token_budget=80,
            summary_token_budget=80,
        )

    def test_summary_is_preserved_when_recent_history_is_long(self):
        history = [
            {"role": "system", "content": "长期目标：准备智语 Agent 面试，保留页面 ID。"},
            *[
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"近期消息 {index}"}
                for index in range(20)
            ],
        ]

        result = self.assembler.assemble(
            system_messages=[{"role": "system", "content": "你是知识助手。"}],
            history=history,
            current_messages=[{"role": "user", "content": "继续刚才的面试整理"}],
            output_token_reserve=200,
        )

        self.assertTrue(any("长期目标" in item["content"] for item in result.messages))
        self.assertEqual(result.messages[-1]["content"], "继续刚才的面试整理")
        self.assertGreater(result.dropped_recent_messages, 0)
        self.assertLessEqual(result.used_tokens, result.input_budget)

    def test_large_current_task_is_truncated_without_exceeding_budget(self):
        result = ContextAssembler(
            context_window_tokens=300,
            history_token_budget=50,
            summary_token_budget=40,
        ).assemble(
            system_messages=[{"role": "system", "content": "系统规则"}],
            history=[{"role": "system", "content": "长期目标"}],
            current_messages=[{"role": "user", "content": "当前证据 " * 1_000}],
            output_token_reserve=80,
        )

        self.assertTrue(result.truncated)
        self.assertLessEqual(result.used_tokens, result.input_budget)
        self.assertEqual(result.output_reserved_tokens, 80)


class CapturedJSONLLM:
    def __init__(self):
        self.messages = None

    def chat_json(self, messages, **kwargs):
        self.messages = messages
        return {
            "intent": "time_query",
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "get_current_time",
                    "parameters": {},
                    "description": "查询时间",
                }
            ],
        }


class CapturedChatLLM:
    def __init__(self):
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return "生成结果"


def make_time_plan():
    return Plan(
        intent=IntentType.TIME_QUERY,
        original_query="现在几点",
        steps=[
            PlanStep(
                step_id=1,
                tool_name=ToolName.GET_CURRENT_TIME,
                parameters={},
                description="查询时间",
            )
        ],
        estimated_steps=1,
        reasoning="测试",
    )


class ModelContextIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.registry = AgentToolRegistry()
        self.policy = PlanPolicy(self.registry, max_steps=6)
        self.history = [
            {"role": "system", "content": "长期摘要：用户正在准备 Agent 面试，页面 ID 为 page-123。"},
            *[
                {"role": "user" if index % 2 == 0 else "assistant", "content": f"对话 {index}"}
                for index in range(12)
            ],
        ]

    def test_planner_keeps_long_term_summary(self):
        planner = Planner(self.registry, self.policy)
        planner.context_assembler = ContextAssembler(
            context_window_tokens=5_000,
            history_token_budget=100,
            summary_token_budget=80,
        )
        llm = CapturedJSONLLM()
        planner._llm_service = llm

        planner.plan("现在几点", self.history)

        self.assertTrue(any("长期摘要" in item["content"] for item in llm.messages))

    def test_responder_keeps_long_term_summary(self):
        responder = Responder()
        responder.context_assembler = ContextAssembler(
            context_window_tokens=3_000,
            history_token_budget=100,
            summary_token_budget=80,
        )
        llm = CapturedChatLLM()
        responder._llm_service = llm
        execution = ExecutionResult(
            plan=make_time_plan(),
            results=[],
            completed_steps=1,
            total_steps=1,
            success=True,
            execution_log=[],
            final_data={"current_time": {"time": "12:00"}},
        )

        responder.generate_response("现在几点", make_time_plan(), execution, self.history)

        self.assertTrue(any("长期摘要" in item["content"] for item in llm.messages))
        self.assertEqual(execution.context_stats["model_context"]["output_reserved_tokens"], 1_000)


class MemoryTokenTriggerTestCase(unittest.TestCase):
    def test_token_trigger_and_cursor_only_advance_summarized_messages(self):
        with tempfile.TemporaryDirectory() as root:
            engine = create_engine(f"sqlite:///{Path(root) / 'memory.db'}")
            Base.metadata.create_all(engine)
            session = sessionmaker(bind=engine)()
            service = MemoryService()
            service.summary_threshold = 100
            service.summary_trigger_tokens = 1
            service.summary_input_token_budget = 120

            class FakeLLM:
                @staticmethod
                def chat(messages, max_tokens=600):
                    return "## 用户目标\n准备面试\n## 未完成事项\n继续整理"

            try:
                for index in range(8):
                    service.add_message(
                        "token-summary-session",
                        "user" if index % 2 == 0 else "assistant",
                        f"一条消息 {index}",
                        db=session,
                    )
                with patch(
                    "backend.app.services.llm_service.get_llm_service",
                    return_value=FakeLLM(),
                ):
                    service.summarize_if_needed("token-summary-session", db=session)

                from backend.app.models.conversation import Conversation

                conversation = session.query(Conversation).filter_by(
                    session_id="token-summary-session"
                ).one()
                self.assertIsNotNone(conversation.summary)
                self.assertEqual(conversation.summary_message_id, 3)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()

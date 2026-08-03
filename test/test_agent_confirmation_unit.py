"""Agent 写入确认与幂等测试。"""

import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.agent.agent import AgentActionError, PlanExecuteAgent
from backend.app.agent.models import (
    AgentResponse,
    ExecutionResult,
    IntentType,
    Plan,
    PlanStep,
    ToolName,
    ToolResult,
)
from backend.app.core.database import Base


class FakeExecutor:
    def __init__(self):
        self.call_count = 0

    async def execute(self, plan, db):
        self.call_count += 1
        result = {"id": "page-1", "title": "确认测试", "filename": "page-1.md"}
        return ExecutionResult(
            plan=plan,
            results=[
                ToolResult(
                    step_id=1,
                    tool_name=ToolName.CREATE_NOTE,
                    success=True,
                    result=result,
                )
            ],
            completed_steps=1,
            total_steps=1,
            success=True,
            execution_log=[],
            final_data={"created_note": result},
        )


class FakeResponder:
    @staticmethod
    def generate_response(query, plan, execution_result, context):
        return AgentResponse(
            query=query,
            response="页面创建成功",
            plan=plan,
            execution_result=execution_result,
            confidence=1.0,
            timestamp=datetime.now(),
        )


class AgentConfirmationTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "agent-test.db"
        engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(engine)
        self.session = sessionmaker(bind=engine)()
        self.agent = PlanExecuteAgent.__new__(PlanExecuteAgent)
        self.agent.executor = FakeExecutor()
        self.agent.responder = FakeResponder()
        self.agent._memory_service = False
        self.plan = Plan(
            intent=IntentType.CREATE_NOTE,
            original_query="创建一条确认测试笔记",
            steps=[
                PlanStep(
                    step_id=1,
                    tool_name=ToolName.CREATE_NOTE,
                    parameters={"title": "确认测试", "content": "正文"},
                    description="创建页面：确认测试",
                )
            ],
            estimated_steps=1,
            reasoning="测试",
        )

    async def asyncTearDown(self):
        self.session.close()
        self.temp_dir.cleanup()

    async def test_confirmation_is_required_and_idempotent(self):
        preview = self.agent._create_pending_response(
            self.plan.original_query,
            "session-test",
            self.plan,
            self.session,
            time.time(),
        )
        self.assertTrue(preview.confirmation_required)
        self.assertEqual(self.agent.executor.call_count, 0)

        first = await self.agent.confirm_action(
            preview.pending_action_id,
            "session-test",
            self.session,
        )
        second = await self.agent.confirm_action(
            preview.pending_action_id,
            "session-test",
            self.session,
        )
        self.assertEqual(first.response, "页面创建成功")
        self.assertEqual(second.response, first.response)
        self.assertEqual(self.agent.executor.call_count, 1)

        with self.assertRaises(AgentActionError):
            await self.agent.confirm_action(
                preview.pending_action_id,
                "another-session",
                self.session,
            )


if __name__ == "__main__":
    unittest.main()


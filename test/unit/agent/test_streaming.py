"""Agent 单次流式生成和工具事件闭合测试。"""

import unittest

from backend.app.agent.events import AgentEventType
from backend.app.agent.executor import Executor
from backend.app.agent.models import (
    ExecutionResult,
    IntentType,
    Plan,
    PlanStep,
    ToolName,
)
from backend.app.agent.responder import Responder


def make_plan(step: PlanStep) -> Plan:
    return Plan(
        intent=IntentType.TIME_QUERY,
        original_query="现在几点",
        steps=[step],
        estimated_steps=1,
        reasoning="查询当前时间",
    )


class FakeStreamingLLM:
    def __init__(self):
        self.chat_calls = 0
        self.stream_calls = 0

    def chat(self, **kwargs):
        self.chat_calls += 1
        return "不应调用"

    def stream_chat(self, **kwargs):
        self.stream_calls += 1
        yield "当前"
        yield "时间"


class AgentStreamingTestCase(unittest.TestCase):
    def test_responder_uses_one_stream_call_as_final_response(self):
        step = PlanStep(
            step_id=1,
            tool_name=ToolName.GET_CURRENT_TIME,
            parameters={},
            description="查询时间",
        )
        plan = make_plan(step)
        execution = ExecutionResult(
            plan=plan,
            results=[],
            completed_steps=1,
            total_steps=1,
            success=True,
            execution_log=[],
            final_data={"current_time": {"date": "2026-08-06", "time": "12:00"}},
        )
        llm = FakeStreamingLLM()
        responder = Responder()
        responder._llm_service = llm
        chunks = []

        response = responder.generate_response(
            "现在几点",
            plan,
            execution,
            token_callback=chunks.append,
        )

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(llm.chat_calls, 0)
        self.assertEqual(chunks, ["当前", "时间"])
        self.assertEqual(response.response, "当前时间")

    def test_dependency_failure_and_unknown_tool_emit_completed_event(self):
        executor = Executor()
        events = []
        callback = lambda event_type, data: events.append((event_type, data))
        try:
            dependency_step = PlanStep(
                step_id=2,
                tool_name=ToolName.GET_CURRENT_TIME,
                parameters={},
                description="依赖失败",
                depends_on=[1],
            )
            result = executor._execute_step(
                dependency_step,
                db=None,
                prev_results=[],
                event_callback=callback,
            )
            self.assertFalse(result.success)
            self.assertEqual(
                [event[0] for event in events],
                [AgentEventType.TOOL_STARTED.value, AgentEventType.TOOL_COMPLETED.value],
            )
            self.assertFalse(events[-1][1]["success"])

            events.clear()
            unknown_step = PlanStep(
                step_id=1,
                tool_name=ToolName.GET_CURRENT_TIME,
                parameters={},
                description="未知工具",
            )
            executor.tools.pop(ToolName.GET_CURRENT_TIME)
            result = executor._execute_step(
                unknown_step,
                db=None,
                prev_results=[],
                event_callback=callback,
            )
            self.assertFalse(result.success)
            self.assertEqual(events[-1][0], AgentEventType.TOOL_COMPLETED.value)
            self.assertIn("未知工具", events[-1][1]["error"])
        finally:
            executor._thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()

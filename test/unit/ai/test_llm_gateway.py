"""模型故障转移、Token 与成本记录测试。"""

import unittest
from types import SimpleNamespace

from backend.app.core.config import settings
from backend.app.core.observability import get_model_usage, reset_request, start_request
from backend.app.services.ai.llm_service import LLMService


class APITimeoutError(Exception):
    pass


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result):
        self.chat = SimpleNamespace(completions=FakeCompletions(result))


def response(content="备用答案", prompt_tokens=100, completion_tokens=20, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class LLMGatewayTestCase(unittest.TestCase):
    def setUp(self):
        self.previous_rates = (
            settings.llm_fallback_input_cost_per_million,
            settings.llm_fallback_output_cost_per_million,
        )
        settings.llm_fallback_input_cost_per_million = 2.0
        settings.llm_fallback_output_cost_per_million = 4.0
        (
            self.request_id,
            self.request_token,
            self.timings_token,
            self.timeline_token,
            self.usage_token,
            self.context_token,
        ) = start_request("llm-fallback-test")

    def tearDown(self):
        reset_request(
            self.request_token,
            self.timings_token,
            self.timeline_token,
            self.usage_token,
            self.context_token,
        )
        (
            settings.llm_fallback_input_cost_per_million,
            settings.llm_fallback_output_cost_per_million,
        ) = self.previous_rates

    def make_service(self, primary, fallback):
        service = LLMService.__new__(LLMService)
        service.client = FakeClient(primary)
        service.model = "primary-model"
        service.fallback_client = FakeClient(fallback) if fallback is not None else None
        service.fallback_model = "fallback-model"
        service.max_tokens = 100
        service.temperature = 0.1
        service._langfuse = None
        service._json_response_format_supported = None
        return service

    def test_retryable_primary_failure_uses_fallback_and_records_cost(self):
        service = self.make_service(APITimeoutError("timeout"), response())
        content = service.chat([{"role": "user", "content": "问题"}])
        self.assertEqual(content, "备用答案")
        usage = get_model_usage()
        self.assertEqual(usage["call_count"], 2)
        self.assertTrue(usage["fallback_used"])
        self.assertEqual(usage["total_tokens"], 120)
        self.assertAlmostEqual(usage["estimated_cost"], 0.00028)

    def test_non_retryable_failure_does_not_use_fallback(self):
        service = self.make_service(ValueError("bad request"), response())
        with self.assertRaisesRegex(RuntimeError, "bad request"):
            service.chat([{"role": "user", "content": "问题"}])
        self.assertEqual(len(service.fallback_client.chat.completions.calls), 0)

    def test_function_call_returns_arguments(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="submit", arguments='{"value": 3}')
        )
        service = self.make_service(response(tool_calls=[tool_call]), None)

        result = service.call_function(
            [{"role": "user", "content": "提交"}],
            name="submit",
            description="提交结果",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        )

        self.assertEqual(result, {"value": 3})
        call = service.client.chat.completions.calls[0]
        self.assertEqual(call["tool_choice"]["function"]["name"], "submit")

    def test_chat_json_downgrades_when_gateway_rejects_response_format(self):
        service = self.make_service(response('{"intent":"search"}'), None)

        class ResponseFormatRejectingCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if "response_format" in kwargs:
                    raise RuntimeError(
                        "LLM 服务调用失败: Error code: 400 "
                        "response_format invalid_request_error"
                    )
                return response('{"intent":"search"}')

        completions = ResponseFormatRejectingCompletions()
        service.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        self.assertEqual(service.chat_json([]), {"intent": "search"})
        self.assertFalse(service._json_response_format_supported)
        self.assertEqual(service.chat_json([]), {"intent": "search"})
        self.assertEqual(len(completions.calls), 3)
        self.assertIn("response_format", completions.calls[0])
        self.assertNotIn("response_format", completions.calls[1])
        self.assertNotIn("response_format", completions.calls[2])

    def test_function_call_rejects_invalid_json_arguments(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="submit", arguments="{invalid")
        )
        service = self.make_service(response(tool_calls=[tool_call]), None)

        with self.assertRaisesRegex(RuntimeError, "无效 JSON"):
            service.call_function(
                [{"role": "user", "content": "提交"}],
                name="submit",
                description="提交结果",
                parameters={"type": "object"},
            )


if __name__ == "__main__":
    unittest.main()

"""LLM 服务接入并发闸门的测试：获取与释放必须成对。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.app.services.ai.llm_service import LLMService


class RetryableError(Exception):
    """带 429 状态的异常，用于触发主备切换。"""

    status_code = 429


def _fake_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class LLMGateTest(unittest.TestCase):
    def setUp(self):
        with patch("backend.app.services.ai.llm_service.OpenAI"):
            self.service = LLMService()

    def test_chat_获取并释放许可(self):
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        self.service.client.chat.completions.create.return_value = _fake_response()
        with patch(
            "backend.app.services.ai.llm_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            self.assertEqual(
                self.service.chat([{"role": "user", "content": "hi"}]),
                "ok",
            )
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[0], "llm")
        gate.__exit__.assert_called_once()

    def test_流式生成被关闭时释放许可(self):
        def _chunk(content=None, finish_reason=None):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )]
            )

        self.service.client.chat.completions.create.return_value = [
            _chunk("a"), _chunk("b"),
        ]
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.llm_service.acquire_model_slot",
            return_value=gate,
        ):
            stream = self.service.stream_chat([{"role": "user", "content": "hi"}])
            self.assertEqual(next(stream), "a")
            gate.__exit__.assert_not_called()
            stream.close()  # 模拟客户端断线 / 取消
        gate.__exit__.assert_called_once()

        fallback = MagicMock()
        self.service.fallback_client = fallback
        self.service.fallback_model = "fallback-model"
        self.service.client.chat.completions.create.side_effect = RetryableError("boom")
        fallback.chat.completions.create.return_value = _fake_response("fallback")
        gate = MagicMock()
        gate.__enter__ = MagicMock(return_value=gate)
        gate.__exit__ = MagicMock(return_value=False)
        with patch(
            "backend.app.services.ai.llm_service.acquire_model_slot",
            return_value=gate,
        ) as factory:
            self.assertEqual(
                self.service.chat([{"role": "user", "content": "hi"}]),
                "fallback",
            )
        self.assertEqual(factory.call_count, 2)

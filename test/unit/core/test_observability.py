"""结构化日志、脱敏和请求追踪测试。"""

import json
import logging
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from backend.app.api.system.observability import _require_local_access
from backend.app.core.config import settings
from backend.app.core.errors import ErrorCode, sanitize_text
from backend.app.core.logging_config import JsonFormatter, RequestContextFilter
from backend.app.core.observability import (
    get_recent_trace,
    reset_request,
    start_request,
    store_current_trace,
    timed_stage,
)


class ObservabilityTestCase(unittest.TestCase):
    @staticmethod
    def _request(origin=None):
        headers = [] if origin is None else [(b"origin", origin.encode("utf-8"))]
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/observability/requests",
                "headers": headers,
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8337),
                "scheme": "http",
            }
        )

    def test_trace_api_rejects_non_local_browser_origin(self):
        with (
            patch.object(settings, "observability_enabled", True),
            patch.object(settings, "observability_trace_api_enabled", True),
            patch.object(settings, "observability_trace_allow_remote", False),
        ):
            _require_local_access(self._request("http://localhost:5173"))
            _require_local_access(self._request())
            with self.assertRaises(HTTPException) as raised:
                _require_local_access(self._request("https://untrusted.example"))
        self.assertEqual(raised.exception.status_code, 403)

    def test_json_log_contains_request_context_and_redacts_credentials(self):
        (
            _,
            request_token,
            timings_token,
            timeline_token,
            usage_token,
            context_token,
        ) = start_request("observability-log-001")
        try:
            record = logging.LogRecord(
                name="test.provider",
                level=logging.ERROR,
                pathname=__file__,
                lineno=20,
                msg=(
                    "调用失败 api_key=sk-sensitive "
                    "Authorization: Bearer bearer-sensitive "
                    "url=https://example.test?a=1&token=query-sensitive"
                ),
                args=(),
                exc_info=None,
            )
            record.component = "llm"
            record.error_code = ErrorCode.LLM_PROVIDER_ERROR.value
            RequestContextFilter().filter(record)
            payload = json.loads(JsonFormatter().format(record))

            self.assertEqual(payload["request_id"], "observability-log-001")
            self.assertEqual(payload["error_code"], "LLM_PROVIDER_ERROR")
            rendered = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("sk-sensitive", rendered)
            self.assertNotIn("bearer-sensitive", rendered)
            self.assertNotIn("query-sensitive", rendered)
            self.assertIn("[REDACTED]", rendered)
        finally:
            reset_request(
                request_token,
                timings_token,
                timeline_token,
                usage_token,
                context_token,
            )

    def test_failed_stage_is_available_by_request_id(self):
        (
            request_id,
            request_token,
            timings_token,
            timeline_token,
            usage_token,
            context_token,
        ) = start_request("observability-trace-001")
        try:
            with self.assertRaises(TimeoutError):
                with timed_stage("retrieval.rerank"):
                    raise TimeoutError("Authorization: Bearer private-token")
            store_current_trace(
                request_id=request_id,
                method="POST",
                path="/agent/runs",
                status_code=500,
                total_ms=12.5,
            )
        finally:
            reset_request(
                request_token,
                timings_token,
                timeline_token,
                usage_token,
                context_token,
            )

        trace = get_recent_trace(request_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["error"]["error_code"], "RERANK_PROVIDER_ERROR")
        self.assertTrue(trace["error"]["retryable"])
        self.assertNotIn("private-token", trace["error"]["message"])
        self.assertEqual(trace["timeline"][-1]["stage"], "retrieval.rerank")
        self.assertEqual(trace["timeline"][-1]["status"], "failed")

    def test_sanitize_text_limits_large_messages(self):
        result = sanitize_text("x" * 200, max_length=20)
        self.assertTrue(result.startswith("x" * 20))
        self.assertTrue(result.endswith("[TRUNCATED]"))


if __name__ == "__main__":
    unittest.main()

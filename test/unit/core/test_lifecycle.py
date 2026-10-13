"""应用生命周期与请求观测上下文测试。"""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.core import lifecycle
from backend.app.core.observability import (
    get_execution_timeline,
    get_model_usage,
    get_request_id,
    get_stage_timings,
    reset_request,
    start_request,
    timed_stage,
)


class LifecycleTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_starts_and_stops_background_tasks(self):
        cleanup_started = asyncio.Event()
        dispatcher_ready = asyncio.Event()

        async def fake_cleanup():
            cleanup_started.set()
            await asyncio.Event().wait()

        async def fake_init_dispatcher():
            dispatcher_ready.set()

        async def fake_close_dispatcher():
            pass

        app = SimpleNamespace(state=SimpleNamespace())
        with (
            patch.object(lifecycle.settings, "sync_legacy_docs_on_startup", False),
            patch.object(lifecycle, "_initialize_database"),
            patch.object(lifecycle, "_migrate_relative_paths"),
            patch.object(lifecycle, "_sync_document_index"),
            patch.object(lifecycle, "_rebuild_bm25_index") as rebuild_bm25,
            patch.object(lifecycle, "_recover_wiki_index_tasks"),
            patch.object(lifecycle, "_periodic_cleanup", fake_cleanup),
            patch(
                "backend.app.tasks.queue.init_dispatcher", fake_init_dispatcher
            ),
            patch(
                "backend.app.tasks.queue.close_dispatcher", fake_close_dispatcher
            ),
        ):
            async with lifecycle.lifespan(app):
                await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                # 索引消费已经交给独立的 arq worker 进程，进程内只剩会话清理。
                self.assertEqual(len(app.state.background_tasks), 1)
                self.assertTrue(all(not task.done() for task in app.state.background_tasks))
                rebuild_bm25.assert_called_once_with()

        self.assertTrue(dispatcher_ready.is_set())
        self.assertEqual(app.state.background_tasks, [])

    async def test_observability_context_is_reset(self):
        (
            request_id,
            request_token,
            timings_token,
            timeline_token,
            usage_token,
            context_token,
        ) = start_request("unit-request-001")
        self.assertEqual(request_id, "unit-request-001")
        with timed_stage("agent.evidence"):
            time.sleep(0.001)
        timings = get_stage_timings()
        self.assertEqual(timings["agent.evidence"]["count"], 1)
        self.assertGreater(timings["agent.evidence"]["total_ms"], 0)
        self.assertEqual(get_execution_timeline()[0]["stage"], "agent.evidence")
        self.assertEqual(get_model_usage()["call_count"], 0)
        reset_request(
            request_token,
            timings_token,
            timeline_token,
            usage_token,
            context_token,
        )
        self.assertIsNone(get_request_id())
        self.assertEqual(get_stage_timings(), {})


if __name__ == "__main__":
    unittest.main()

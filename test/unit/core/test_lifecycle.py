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
        worker_started = asyncio.Event()

        async def fake_worker():
            worker_started.set()
            await asyncio.Event().wait()

        app = SimpleNamespace(state=SimpleNamespace())
        with (
            patch.object(lifecycle, "_initialize_database"),
            patch.object(lifecycle, "_migrate_relative_paths"),
            patch.object(lifecycle, "_sync_document_index"),
            patch.object(lifecycle, "_recover_wiki_index_tasks"),
            patch(
                "backend.app.services.wiki.wiki_index_worker.run_wiki_index_worker",
                fake_worker,
            ),
        ):
            async with lifecycle.lifespan(app):
                await asyncio.wait_for(worker_started.wait(), timeout=1)
                self.assertEqual(len(app.state.background_tasks), 2)
                self.assertTrue(all(not task.done() for task in app.state.background_tasks))

        self.assertEqual(app.state.background_tasks, [])

    async def test_observability_context_is_reset(self):
        (
            request_id,
            request_token,
            timings_token,
            timeline_token,
            usage_token,
        ) = start_request("unit-request-001")
        self.assertEqual(request_id, "unit-request-001")
        with timed_stage("agent.evidence"):
            time.sleep(0.001)
        timings = get_stage_timings()
        self.assertEqual(timings["agent.evidence"]["count"], 1)
        self.assertGreater(timings["agent.evidence"]["total_ms"], 0)
        self.assertEqual(get_execution_timeline()[0]["stage"], "agent.evidence")
        self.assertEqual(get_model_usage()["call_count"], 0)
        reset_request(request_token, timings_token, timeline_token, usage_token)
        self.assertIsNone(get_request_id())
        self.assertEqual(get_stage_timings(), {})


if __name__ == "__main__":
    unittest.main()

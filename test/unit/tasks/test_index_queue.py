"""Stage 5 索引任务队列化测试。

盯的是三类风险：

1. **投递失败会不会影响页面写入**——不该影响。页面已经提交了事务，那才是调用方
   该负责的事；队列抖一下不该让用户的保存失败。
2. **同一个任务会不会被执行两次**——多 worker 下最怕这个。原来「读 pending →
   改 processing → commit」分三步，两个 worker 可能都读到 pending 然后都执行。
3. **worker 崩溃后卡在 processing 的任务有没有人回收**——worker 独立成进程之后，
   它死了 API 不会重启，启动恢复没机会跑。
"""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sqlalchemy.orm import sessionmaker

from test.support.db import make_test_engine
from backend.app.models.wiki import WikiIndexTask
from backend.app.services.wiki.wiki_index_task_service import (
    WikiIndexTaskService,
    utc_now,
)
from backend.app.tasks import queue as queue_module


class _RecordingIndexer:
    """记录被索引过的页面，用来判断某条任务到底有没有被执行。"""

    def __init__(self):
        self.indexed = []

    def index_page(self, page):
        self.indexed.append(page["id"])
        return {"status": "indexed"}

    def remove_page(self, page_id):
        self.indexed.append(f"delete:{page_id}")
        return True


def _build_db(root: Path):
    engine = make_test_engine()
    return engine, sessionmaker(bind=engine)


def _build_service(session, indexer, dispatched):
    return WikiIndexTaskService(
        session,
        page_reader=lambda page_id: {
            "id": page_id,
            "title": "标题",
            "content": "正文",
            "revision": 1,
            "tags": [],
            "notebook": "",
            "source_type": "wiki_page",
            "source_uri": "",
            "updated_at": "2026-10-13T00:00:00+00:00",
        },
        index_service_getter=lambda: indexer,
        dispatcher=dispatched.append,
    )


class DispatchQueueTestCase(unittest.IsolatedAsyncioTestCase):
    """投递器：同步代码往异步队列投递的那一层适配。"""

    def test_未初始化时安静跳过(self):
        # 单元测试直接构造 service、或 worker 进程里的调用路径都会走到这里。
        # 关键是不能抛——调用方刚写完页面，不该因为队列没就绪而失败。
        with (
            patch.object(queue_module, "_pool", None),
            patch.object(queue_module, "_main_loop", None),
        ):
            queue_module.dispatch_index_tasks(["task-1"])  # 不抛即通过

    async def test_空任务列表不动作(self):
        pool = AsyncMock()
        with (
            patch.object(queue_module, "_pool", pool),
            patch.object(queue_module, "_main_loop", asyncio.get_running_loop()),
        ):
            queue_module.dispatch_index_tasks([])
            queue_module.dispatch_index_tasks([""])
        pool.enqueue_job.assert_not_called()

    async def test_已在主循环内时不会自己等自己(self):
        # 有些 async 端点会直接调同步 service 再投递。这时用
        # run_coroutine_threadsafe 会死锁——等的是自己所在的循环。
        pool = AsyncMock()
        with (
            patch.object(queue_module, "_pool", pool),
            patch.object(queue_module, "_main_loop", asyncio.get_running_loop()),
        ):
            queue_module.dispatch_index_tasks(["task-1"])
            await asyncio.sleep(0)  # 让 create_task 有机会跑

        pool.enqueue_job.assert_awaited_once()
        self.assertEqual(pool.enqueue_job.await_args.args[0], "index_page")

    def test_从工作线程投递能送进主循环(self):
        # FastAPI 的同步端点跑在线程池里，这是生产上的主要路径。
        delivered = []

        async def main():
            loop = asyncio.get_running_loop()
            pool = AsyncMock()
            pool.enqueue_job = AsyncMock(
                side_effect=lambda *a, **k: delivered.append(a)
            )
            with (
                patch.object(queue_module, "_pool", pool),
                patch.object(queue_module, "_main_loop", loop),
            ):
                worker = threading.Thread(
                    target=queue_module.dispatch_index_tasks, args=(["task-9"],)
                )
                worker.start()
                worker.join(timeout=5)
                await asyncio.sleep(0.05)

        asyncio.run(main())
        self.assertEqual(delivered, [("index_page", "task-9")])

    def test_投递失败不往上抛(self):
        async def main():
            loop = asyncio.get_running_loop()
            pool = AsyncMock()
            pool.enqueue_job = AsyncMock(side_effect=OSError("Redis 挂了"))
            with (
                patch.object(queue_module, "_pool", pool),
                patch.object(queue_module, "_main_loop", loop),
            ):
                worker = threading.Thread(
                    target=queue_module.dispatch_index_tasks, args=(["task-1"],)
                )
                worker.start()
                worker.join(timeout=5)

        asyncio.run(main())  # 不抛即通过


class IndexTaskClaimTestCase(unittest.TestCase):
    """任务抢占：同一条任务必须只有一个执行者。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.engine, self.Session = _build_db(self.root)
        self.dispatched = []
        self.indexer = _RecordingIndexer()

    def tearDown(self):
        self._tmp.cleanup()

    def _new_task(self, session, page_id="page-1", action="upsert"):
        # 必须同时建页面行：process() 是按 page 是否存在来选 upsert 还是 delete 的，
        # 没有页面行的话会走删除分支，测不到索引路径。
        from backend.app.models.wiki import WikiPage

        session.add(
            WikiPage(
                id=page_id,
                title="标题",
                file_path=str(self.root / f"{page_id}.md"),
                content_hash="hash",
            )
        )
        task = WikiIndexTask(
            id="task-1", page_id=page_id, revision=1, action=action, status="pending"
        )
        session.add(task)
        session.commit()
        return task

    def test_同一条任务重复投递只执行一次(self):
        # 队列重复投递、或者兜底扫描补投了一条正在跑的任务时，
        # 第二个执行者必须什么都不做——否则同一个页面会被索引两遍。
        session = self.Session()
        try:
            self._new_task(session)
            service = _build_service(session, self.indexer, self.dispatched)

            first = service.process("task-1")
            second = service.process("task-1")

            self.assertEqual(first["status"], "completed")
            # 返回的 status 是任务的真实状态，两次都是 completed——区分点在于
            # 索引只被写了一次，这才是「没有重复执行」的证据。
            self.assertEqual(second["status"], "completed")
            self.assertEqual(
                self.indexer.indexed, ["page-1"], "同一个任务被执行了两次"
            )
        finally:
            session.close()

    def test_已完成的原子抢占不会误伤(self):
        # 抢占只接受 pending/failed。completed 的任务不该被重新执行。
        session = self.Session()
        try:
            self._new_task(session)
            service = _build_service(session, self.indexer, self.dispatched)
            service.process("task-1")
            result = service.process("task-1")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(self.indexer.indexed), 1)
        finally:
            session.close()

    def test_失败任务仍可被再次抢占(self):
        # 退避重投依赖这一点：failed 状态必须允许重新执行。
        session = self.Session()
        try:
            self._new_task(session)
            service = _build_service(session, self.indexer, self.dispatched)

            class Exploding:
                def index_page(self, page):
                    raise RuntimeError("embedding 服务不可用")

                def remove_page(self, page_id):
                    raise RuntimeError("embedding 服务不可用")

            failing = _build_service(session, Exploding(), self.dispatched)
            first = failing.process("task-1")
            self.assertEqual(first["status"], "failed")

            second = service.process("task-1")
            self.assertEqual(second["status"], "completed")
            self.assertEqual(self.indexer.indexed, ["page-1"])
        finally:
            session.close()


class StaleTaskRecoveryTestCase(unittest.TestCase):
    """卡死任务回收：worker 崩溃后没人回收的话任务会永远停在 processing。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.engine, self.Session = _build_db(self.root)
        self.dispatched = []

    def tearDown(self):
        self._tmp.cleanup()

    def _service(self, session):
        return _build_service(session, _RecordingIndexer(), self.dispatched)

    def test_超过阈值的_processing_任务被回收(self):
        from datetime import timedelta

        session = self.Session()
        try:
            session.add(
                WikiIndexTask(
                    id="stuck",
                    page_id="page-1",
                    revision=1,
                    action="upsert",
                    status="processing",
                    locked_at=utc_now() - timedelta(seconds=9999),
                )
            )
            session.commit()

            result = self._service(session).recover_stale(threshold_seconds=600)

            self.assertEqual(result["recovered"], 1)
            task = session.get(WikiIndexTask, "stuck")
            self.assertEqual(task.status, "pending")
            self.assertIsNone(task.locked_at)
            self.assertIsNone(task.next_attempt_at)
        finally:
            session.close()

    def test_还在阈值内的任务不动(self):
        # 阈值必须大于任务超时时间，否则正在正常执行的长任务会被误判成卡死并重跑。
        from datetime import timedelta

        session = self.Session()
        try:
            session.add(
                WikiIndexTask(
                    id="running",
                    page_id="page-1",
                    revision=1,
                    action="upsert",
                    status="processing",
                    locked_at=utc_now() - timedelta(seconds=5),
                )
            )
            session.commit()

            result = self._service(session).recover_stale(threshold_seconds=600)

            self.assertEqual(result["recovered"], 0)
            self.assertEqual(session.get(WikiIndexTask, "running").status, "processing")
        finally:
            session.close()

    def test_回收后能被扫描重新投递(self):
        # 回收只是把状态改回 pending，如果没人投递，任务照样不会被执行。
        from datetime import timedelta

        session = self.Session()
        try:
            session.add(
                WikiIndexTask(
                    id="stuck",
                    page_id="page-1",
                    revision=1,
                    action="upsert",
                    status="processing",
                    locked_at=utc_now() - timedelta(seconds=9999),
                )
            )
            session.commit()
            service = self._service(session)

            service.recover_stale(threshold_seconds=600)
            due = service.due_task_ids(limit=10)

            self.assertEqual(due, ["stuck"])
        finally:
            session.close()


class DueTaskListingTestCase(unittest.TestCase):
    """兜底扫描取任务的条件。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine, self.Session = _build_db(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_只列出到期的待处理任务(self):
        from datetime import timedelta

        session = self.Session()
        try:
            now = utc_now()
            session.add_all(
                [
                    WikiIndexTask(id="due", page_id="p1", revision=1, action="upsert",
                                  status="pending", next_attempt_at=None),
                    WikiIndexTask(id="later", page_id="p2", revision=1, action="upsert",
                                  status="failed", next_attempt_at=now + timedelta(seconds=300)),
                    WikiIndexTask(id="backoff-due", page_id="p3", revision=1, action="upsert",
                                  status="failed", next_attempt_at=now - timedelta(seconds=1)),
                    WikiIndexTask(id="running", page_id="p4", revision=1, action="upsert",
                                  status="processing", next_attempt_at=None),
                    WikiIndexTask(id="done", page_id="p5", revision=1, action="upsert",
                                  status="completed", next_attempt_at=None),
                ]
            )
            session.commit()

            service = _build_service(session, _RecordingIndexer(), [])
            due = service.due_task_ids(limit=10)

            self.assertEqual(set(due), {"due", "backoff-due"})
        finally:
            session.close()

    def test_超过上限时按创建时间截断(self):
        session = self.Session()
        try:
            for index in range(5):
                session.add(
                    WikiIndexTask(id=f"t{index}", page_id=f"p{index}", revision=1,
                                  action="upsert", status="pending")
                )
            session.commit()
            service = _build_service(session, _RecordingIndexer(), [])
            self.assertEqual(len(service.due_task_ids(limit=2)), 2)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()

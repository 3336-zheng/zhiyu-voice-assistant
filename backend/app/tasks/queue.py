"""arq 任务投递器：让同步代码能把任务送进异步队列。

写页面的整条链路是同步的——FastAPI 的同步端点跑在线程池里，service 层用的是
同步的 SQLAlchemy Session——而 arq 的 enqueue_job 是 async 的。把全链路改成异步
的改动面远大于在这里做一层适配，所以维护一份主事件循环的引用，
同步侧用 run_coroutine_threadsafe 把投递请求送过去。

投递是 best-effort：失败只记日志，不往上抛。页面写入不该因为队列抖一下而失败；
漏掉的任务由 worker 的兜底扫描捡回来——**表才是任务的真相来源，
队列只是让它更快被执行的通道**。
"""

import asyncio
import logging
from typing import Iterable, List, Optional

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# 投递是 best-effort，不该让同步调用方等太久。Redis 不可用时宁可这单放过，
# 交给兜底扫描——否则一次页面保存会被一个挂掉的队列拖住。
DISPATCH_TIMEOUT_SECONDS = 1.0

_pool: Optional[ArqRedis] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None


async def create_arq_pool() -> ArqRedis:
    """创建 arq 连接池。API 进程与 worker 进程各自持有自己的池。"""
    return await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def ensure_pool() -> ArqRedis:
    """async 调用方取连接池，未初始化时补建一次。

    API 的 lifespan 会初始化它，正常路径走不到补建。兜底是为了测试和脚本——
    它们不跑 lifespan，但也需要能投递任务。
    """
    global _pool, _main_loop
    if _pool is None:
        _main_loop = asyncio.get_running_loop()
        _pool = await create_arq_pool()
    return _pool


async def init_dispatcher() -> None:
    """在应用启动时记录主事件循环并建立连接池。

    必须由 lifespan 调用：同步侧要投递就得知道往哪个循环上送，
    而这个循环只有应用启动时才知道是哪一个。
    """
    global _pool, _main_loop
    _main_loop = asyncio.get_running_loop()
    _pool = await create_arq_pool()
    logger.info("任务投递器已就绪，队列: %s", settings.arq_index_queue)


async def close_dispatcher() -> None:
    """释放连接池并清掉循环引用，避免关闭后仍有投递尝试。"""
    global _pool, _main_loop
    _main_loop = None
    if _pool is not None:
        await _pool.aclose()
        _pool = None


def dispatch_index_tasks(task_ids: Iterable[str]) -> None:
    """同步投递入口，供 page_service 这类同步代码调用。

    投递不成功时**不抛异常**：调用方刚写完页面并提交了事务，那才是它该负责的事；
    队列的事由兜底扫描兜住。
    """
    ids: List[str] = [task_id for task_id in task_ids if task_id]
    if not ids:
        return

    loop, pool = _main_loop, _pool
    if loop is None or pool is None or loop.is_closed():
        # 投递器没初始化——单元测试直接构造 service、或 worker 进程里的调用路径。
        # 这些场景下任务留在表里，由兜底扫描负责。
        logger.debug("投递器未就绪，%s 个索引任务留给兜底扫描", len(ids))
        return

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is loop:
        # 已经在主循环内部（async 端点直接调同步 service 的情况）。
        # 这时用 run_coroutine_threadsafe 会自己等自己，直接建任务即可。
        for task_id in ids:
            loop.create_task(_enqueue_one(pool, task_id))
        return

    try:
        future = asyncio.run_coroutine_threadsafe(_enqueue_many(pool, ids), loop)
        future.result(timeout=DISPATCH_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 — 投递失败不该影响页面写入
        logger.warning("索引任务投递失败，将由兜底扫描补投: %s", exc)


async def _enqueue_one(pool: ArqRedis, task_id: str) -> None:
    await pool.enqueue_job(
        "index_page", task_id, _queue_name=settings.arq_index_queue
    )


async def enqueue_agent_run(run_id: str) -> bool:
    """把一次 Agent 运行投给 worker 执行。返回 False 表示这次运行已经在队列里。

    执行发生在 worker 进程里，API 只负责创建运行、订阅事件和转发取消——
    这样 API 重启不会打断正在跑的运行，API 实例也可以任意扩缩。

    job_id 用运行 id：同一条运行重复投递会被 arq 挡掉，不会被执行两次。
    """
    pool = await ensure_pool()
    job = await pool.enqueue_job(
        "run_agent",
        run_id,
        _job_id=f"agent:{run_id}",
        _queue_name=settings.arq_agent_queue,
    )
    return job is not None


async def _enqueue_many(pool: ArqRedis, task_ids: List[str]) -> None:
    for task_id in task_ids:
        await _enqueue_one(pool, task_id)

"""arq worker 的启动配置与消费者定义。

启动方式：
    arq backend.app.tasks.worker.WorkerSettings

worker 是独立进程，不经过 FastAPI 的 lifespan，所以日志与遥测要在这里自己初始化，
否则 worker 的日志会和 API 的格式不一致，排查时对不上时间线。
"""

import logging
from typing import Any, Dict

from arq import cron
from arq.connections import RedisSettings

from backend.app.core.config import settings
from backend.app.core.logging_config import configure_logging
from .index_tasks import index_page, sweep_pending_tasks

logger = logging.getLogger(__name__)


async def startup(ctx: Dict[str, Any]) -> None:
    configure_logging(settings)
    from backend.app.core.telemetry import configure_telemetry

    configure_telemetry()
    logger.info(
        "arq worker 已启动，队列: %s，并发: %s",
        settings.arq_index_queue,
        settings.arq_index_max_jobs,
    )


async def shutdown(ctx: Dict[str, Any]) -> None:
    logger.info("arq worker 已停止")


class WorkerSettings:
    functions = [index_page]
    # 兜底扫描兼做两件事：补投漏掉的任务，回收卡死的 processing 任务。
    # run_at_startup 让 worker 一起来就把上一轮遗留的问题处理掉，不用等一分钟。
    cron_jobs = [cron(sweep_pending_tasks, run_at_startup=True)]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    queue_name = settings.arq_index_queue
    max_jobs = settings.arq_index_max_jobs
    job_timeout = settings.arq_job_timeout_seconds
    # 不让 arq 自己重试：退避由 WikiIndexTask.next_attempt_at 统一决定，
    # 两套重试并存会让实际间隔说不清。
    max_tries = 1
    # 调用方不读 job 结果，保留一小会儿够调试即可。
    keep_result = 60
    on_startup = startup
    on_shutdown = shutdown

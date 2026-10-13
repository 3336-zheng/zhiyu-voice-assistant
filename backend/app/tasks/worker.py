"""arq worker 的启动配置与消费者定义。

启动方式：
    arq backend.app.tasks.worker.WorkerSettings       # 索引队列
    arq backend.app.tasks.worker.AsrWorkerSettings    # ASR 队列

两个队列各起一个进程：索引任务是几百毫秒的短活，转写可能跑满 180 秒，
混在同一个 worker 里会让短任务排在长任务后面。

worker 是独立进程，不经过 FastAPI 的 lifespan，所以日志与遥测要在这里自己初始化，
否则 worker 的日志会和 API 的格式不一致，排查时对不上时间线。
"""

import logging
from typing import Any, Dict

from arq import cron
from arq.connections import RedisSettings

from backend.app.core.config import settings
from backend.app.core.logging_config import configure_logging
from .asr_tasks import transcribe_audio
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


async def asr_startup(ctx: Dict[str, Any]) -> None:
    configure_logging(settings)
    from backend.app.core.telemetry import configure_telemetry

    configure_telemetry()
    logger.info(
        "ASR worker 已启动，队列: %s，并发: %s",
        settings.arq_asr_queue,
        settings.arq_asr_max_jobs,
    )


async def asr_shutdown(ctx: Dict[str, Any]) -> None:
    logger.info("ASR worker 已停止")


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


class AsrWorkerSettings:
    """ASR 专用 worker。

    `job_timeout` 必须 ≥ API 侧的等待超时（asr_timeout_seconds），否则会出现
    "worker 还在跑、API 已经超时返回"的错配：用户看到失败，但结果其实落库了。
    这里留一点余量让 worker 有机会正常收尾。

    `max_tries=1`：转写失败要么是音频本身有问题，要么是上游 API 故障，
    两者都不是立刻重试能解决的；重复调用还会重复计费。
    """

    functions = [transcribe_audio]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    queue_name = settings.arq_asr_queue
    max_jobs = settings.arq_asr_max_jobs
    job_timeout = int(settings.asr_timeout_seconds) + 30
    max_tries = 1
    # 结果由 API 通过 job.result() 取走，这边不需要长期保留。
    # 保留太久还会有个副作用：同一个音频在这个窗口内无法重新转写。
    keep_result = 60
    # 队列积压时任务不该无限期等待——转写超过 10 分钟才被执行已经没有意义了。
    expires_extra_ms = 600_000
    on_startup = asr_startup
    on_shutdown = asr_shutdown

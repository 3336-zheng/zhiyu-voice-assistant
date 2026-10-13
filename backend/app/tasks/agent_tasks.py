"""Agent 运行任务。

执行从 API 进程搬到 worker：一次运行可能跑满 180 秒，放在 API 里意味着
重启就打断正在跑的对话；搬过来之后 API 只负责创建运行、订阅事件、转发取消，
可以任意扩缩，运行也不受 API 重启影响。

运行状态、事件流、取消信号都在 Redis 上（Stage 6a/6b 已就位），所以 worker
只要拿 run_id 就能接着跑——API 和 worker 之间不需要传别的上下文。
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def run_agent(ctx: Dict[str, Any], run_id: str) -> None:
    """arq 任务入口：执行一次 Agent 运行。"""
    from backend.app.services.runtime.agent_runtime_service import (
        get_agent_runtime_service,
    )

    service = get_agent_runtime_service()
    logger.info("worker 开始执行 Agent 运行: run_id=%s", run_id)
    try:
        await service.execute_run(run_id)
    except Exception:  # noqa: BLE001 — 运行自身的失败已在服务内落成终态
        logger.exception("Agent 运行执行异常: run_id=%s", run_id)
        raise

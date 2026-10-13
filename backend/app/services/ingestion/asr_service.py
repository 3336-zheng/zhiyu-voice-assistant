"""ASR Provider 注册与转写调用。

执行调度（去重、超时、并发）已交给 arq 队列，见 tasks/asr_tasks.py。
这里只剩「有哪些 Provider、怎么解析、怎么调」。
"""

from __future__ import annotations

import logging
from typing import Protocol

from backend.app.core.config import settings


logger = logging.getLogger(__name__)


class ASRService(Protocol):
    def transcribe(self, audio_path: str, language: str = "auto") -> dict: ...


class ASRError(RuntimeError):
    """ASR 对外错误基类。

    原先还有 AlreadyRunning / Timeout / Execution 三个子类，现在不再需要：
    去重由 arq 的 job_id 承担（重复提交直接被拒），超时由调用方的
    asyncio.wait_for 承担，执行失败由 worker 返回的 code=execution 承担。
    这三件事都不再需要用异常类型来表达。
    """


class ASRConfigurationError(ASRError):
    pass


class ASRProviderNotFoundError(ASRError):
    pass


class ASRInputError(ASRError):
    pass


def _provider_definitions() -> dict[str, dict]:
    return {
        "dashscope": {
            "name": f"百炼 DashScope ({settings.dashscope_asr_model})",
            "available": bool(settings.dashscope_asr_api_key),
        },
        "mimo": {
            "name": f"小米 MiMo ({settings.mimo_asr_model})",
            "available": bool(settings.mimo_asr_api_key),
        },
    }


def resolve_provider(provider: str | None) -> str:
    provider_id = (provider or settings.asr_provider).strip().lower()
    definitions = _provider_definitions()
    if provider_id not in definitions:
        raise ASRProviderNotFoundError(f"不支持的 ASR Provider: {provider_id}")
    if not definitions[provider_id]["available"]:
        raise ASRConfigurationError(f"ASR Provider 尚未配置: {provider_id}")
    return provider_id


def get_asr_service(provider_id: str) -> ASRService:
    if provider_id == "dashscope":
        from .dashscope_asr_service import get_dashscope_asr_service

        return get_dashscope_asr_service()
    if provider_id == "mimo":
        from .mimo_asr_service import get_mimo_asr_service

        return get_mimo_asr_service()
    raise ASRProviderNotFoundError(f"不支持的 ASR Provider: {provider_id}")


def get_available_asr_providers() -> list[dict]:
    return [
        {"id": provider_id, **definition}
        for provider_id, definition in _provider_definitions().items()
    ]


def transcribe_sync(provider_id: str, audio_path: str, language: str) -> dict:
    """同步执行一次转写。

    Provider 都是同步实现（SDK 或 requests），调用方负责把它放进线程或 worker。
    这里不做并发控制：去重与超时现在由 arq 的任务队列负责（job_id 相同的任务
    不会被重复投递），进程内的信号量在多实例下本来也拦不住。
    """
    return get_asr_service(provider_id).transcribe(audio_path, language)

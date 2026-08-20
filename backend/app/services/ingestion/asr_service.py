"""ASR Provider 注册与非阻塞执行控制。"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import Protocol

from backend.app.core.config import settings


logger = logging.getLogger(__name__)


class ASRService(Protocol):
    def transcribe(self, audio_path: str, language: str = "auto") -> dict: ...


class ASRError(RuntimeError):
    """ASR 对外错误基类。"""


class ASRAlreadyRunningError(ASRError):
    pass


class ASRConfigurationError(ASRError):
    pass


class ASRProviderNotFoundError(ASRError):
    pass


class ASRInputError(ASRError):
    pass


class ASRTimeoutError(ASRError):
    pass


class ASRExecutionError(ASRError):
    pass


def _provider_definitions() -> dict[str, dict]:
    return {
        "whisper": {
            "name": "本地 Whisper",
            "available": bool(
                settings.whisper_model_path
                and Path(settings.whisper_model_path).expanduser().exists()
            ),
        },
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
    if provider_id == "whisper":
        from .whisper_service import get_whisper_service

        return get_whisper_service()
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


def _transcribe_sync(provider_id: str, audio_path: str, language: str) -> dict:
    return get_asr_service(provider_id).transcribe(audio_path, language)


class ASRExecutionService:
    """将同步 Provider 放入线程，并管理去重、超时和本地单并发。"""

    def __init__(self) -> None:
        self._running_audio_ids: set[int] = set()
        self._running_lock = asyncio.Lock()
        self._whisper_slot = asyncio.Semaphore(1)

    async def transcribe(
        self,
        audio_id: int,
        audio_path: str,
        provider: str | None = None,
        language: str = "auto",
    ) -> dict:
        provider_id = resolve_provider(provider)
        await self._claim(audio_id)
        task = asyncio.create_task(self._run(provider_id, audio_path, language))
        release_deferred = False

        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=settings.asr_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            release_deferred = True
            task.add_done_callback(partial(self._release_background, audio_id))
            raise ASRTimeoutError("语音转写超时") from exc
        except asyncio.CancelledError:
            release_deferred = True
            task.add_done_callback(partial(self._release_background, audio_id))
            raise
        except ASRError:
            raise
        except Exception as exc:
            logger.exception("ASR Provider 执行失败: provider=%s audio_id=%s", provider_id, audio_id)
            raise ASRExecutionError("语音转写服务暂时不可用") from exc
        finally:
            if not release_deferred:
                await self._release(audio_id)

    async def _run(self, provider_id: str, audio_path: str, language: str) -> dict:
        if provider_id == "whisper":
            async with self._whisper_slot:
                return await asyncio.to_thread(
                    _transcribe_sync, provider_id, audio_path, language
                )
        return await asyncio.to_thread(_transcribe_sync, provider_id, audio_path, language)

    async def _claim(self, audio_id: int) -> None:
        async with self._running_lock:
            if audio_id in self._running_audio_ids:
                raise ASRAlreadyRunningError("该音频正在转写")
            self._running_audio_ids.add(audio_id)

    async def _release(self, audio_id: int) -> None:
        async with self._running_lock:
            self._running_audio_ids.discard(audio_id)

    def _release_background(self, audio_id: int, task: asyncio.Task) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass
        asyncio.create_task(self._release(audio_id))


_asr_execution_service: ASRExecutionService | None = None


def get_asr_execution_service() -> ASRExecutionService:
    global _asr_execution_service
    if _asr_execution_service is None:
        _asr_execution_service = ASRExecutionService()
    return _asr_execution_service

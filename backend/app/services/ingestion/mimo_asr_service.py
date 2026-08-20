"""小米 MiMo OpenAI 兼容 ASR Provider。"""

from __future__ import annotations

import base64
import logging
import os
import time

import requests

from backend.app.core.config import settings
from backend.app.services.ingestion.asr_service import ASRInputError
from backend.app.services.ingestion.audio_processing import probe_duration


logger = logging.getLogger(__name__)


class MiMoASRService:
    def __init__(self) -> None:
        if not settings.mimo_asr_api_key:
            raise ValueError("MIMO_ASR_API_KEY 未配置")
        base_url = settings.mimo_asr_api_url.rstrip("/")
        self.endpoint = (
            base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        )
        self.model = settings.mimo_asr_model

    def transcribe(self, audio_path: str, language: str = "auto") -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError("音频文件不存在")

        duration = probe_duration(audio_path, settings.audio_probe_timeout_seconds)
        with open(audio_path, "rb") as audio_file:
            encoded = base64.b64encode(audio_file.read()).decode("ascii")
        data_url = f"data:audio/wav;base64,{encoded}"
        if len(data_url.encode("ascii")) > settings.mimo_asr_max_base64_bytes:
            raise ASRInputError("规范化音频超过 MiMo Base64 10 MB 限制")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data_url}}
                    ],
                }
            ],
            "asr_options": {"language": language if language in {"zh", "en"} else "auto"},
        }
        started_at = time.perf_counter()
        response = requests.post(
            self.endpoint,
            headers={
                "api-key": settings.mimo_asr_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.mimo_asr_request_timeout_seconds,
        )
        elapsed = time.perf_counter() - started_at
        if not response.ok:
            logger.warning("MiMo ASR 请求失败: status=%s", response.status_code)
            raise RuntimeError(f"MiMo ASR 请求失败，HTTP {response.status_code}")

        try:
            transcription = response.json()["choices"][0]["message"]["content"].strip()
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("MiMo ASR 返回格式无效") from exc
        if not transcription:
            raise RuntimeError("MiMo ASR 未返回转写文本")

        return {
            "transcription": transcription,
            "segments": [],
            "language": language if language in {"zh", "en"} else "auto",
            "language_probability": None,
            "duration": duration,
            "transcribe_time": elapsed,
            "rtf": elapsed / duration,
        }


_mimo_asr_service: MiMoASRService | None = None


def get_mimo_asr_service() -> MiMoASRService:
    global _mimo_asr_service
    if _mimo_asr_service is None:
        _mimo_asr_service = MiMoASRService()
    return _mimo_asr_service

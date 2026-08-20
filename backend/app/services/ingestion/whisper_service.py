"""本地 faster-whisper 语音识别服务。"""

import os
import time
import logging
import threading

from faster_whisper import WhisperModel

from backend.app.core.config import settings
from backend.app.services.ingestion.audio_processing import probe_duration

logger = logging.getLogger(__name__)

class WhisperService:
    def __init__(self):
        """初始化Whisper服务"""
        model_path = settings.whisper_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Whisper模型路径不存在: {model_path}\n"
                f"请检查 .env 文件中的 WHISPER_MODEL_PATH 配置，"
                f"或在服务器上下载并部署模型到该路径。"
            )
        logger.info("加载 Whisper 模型...")
        start_load = time.time()
        self.model = WhisperModel(
            model_path,
            device="cuda" if self._check_cuda() else "cpu",
            compute_type="int8"
        )
        self._transcribe_lock = threading.Lock()
        load_time = time.time() - start_load
        logger.info(f"模型加载耗时: {load_time:.2f}秒")

    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def transcribe(self, audio_path: str, language: str = "auto") -> dict:
        """串行执行本地 Whisper，并返回统一的 ASR 结果。"""
        with self._transcribe_lock:
            return self._transcribe(audio_path, language)

    def _transcribe(self, audio_path: str, language: str) -> dict:
        logger.info(f"开始转录: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError("音频文件不存在")
        start_trans = time.time()
        audio_duration = probe_duration(audio_path, settings.audio_probe_timeout_seconds)
        whisper_language = None if language == "auto" else language

        try:
            return self._decode(
                audio_path, whisper_language, audio_duration, start_trans, use_vad=True
            )
        except Exception as vad_error:
            logger.warning("VAD 转录失败，尝试禁用 VAD: %s", vad_error)
            try:
                return self._decode(
                    audio_path, whisper_language, audio_duration, start_trans, use_vad=False
                )
            except Exception as fallback_error:
                raise RuntimeError("本地 Whisper 转录失败") from fallback_error

    def _decode(
        self,
        audio_path: str,
        language: str | None,
        audio_duration: float,
        started_at: float,
        use_vad: bool,
    ) -> dict:
        options = {"vad_filter": use_vad}
        if use_vad:
            options["vad_parameters"] = {"min_silence_duration_ms": 500}
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            **options,
        )
        segments_list = list(segments)
        elapsed = time.time() - started_at
        transcription = " ".join(segment.text for segment in segments_list)
        logger.info("Whisper 转录完成: elapsed=%.2fs text_length=%s", elapsed, len(transcription))
        return {
            "transcription": transcription,
            "segments": [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in segments_list
            ],
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": audio_duration,
            "transcribe_time": elapsed,
            "rtf": elapsed / audio_duration,
        }


# 全局服务实例
whisper_service_instance = None


def get_whisper_service() -> WhisperService:
    """获取 Whisper 服务实例（单例模式）"""
    global whisper_service_instance
    if whisper_service_instance is None:
        whisper_service_instance = WhisperService()
    return whisper_service_instance

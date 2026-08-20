"""
百炼 DashScope ASR 服务
通过 dashscope SDK 调用 paraformer 模型进行语音识别
"""
import os
import time
import logging

from backend.app.core.config import settings
from backend.app.services.ingestion.audio_processing import probe_duration

logger = logging.getLogger(__name__)


class _ASRCallback:
    """Recognition 回调，收集识别结果"""

    def __init__(self):
        self.sentences = []

    def on_open(self):
        pass

    def on_complete(self):
        pass

    def on_close(self):
        pass

    def on_error(self, result):
        logger.error(f"[DashScope ASR] 回调错误: {result}")

    def on_event(self, result):
        sentence = result.get_sentence()
        if sentence:
            if isinstance(sentence, list):
                self.sentences.extend(sentence)
            else:
                self.sentences.append(sentence)


class DashScopeASRService:
    """百炼 DashScope ASR 服务（使用官方 SDK）"""

    def __init__(self):
        import dashscope

        api_key = settings.dashscope_asr_api_key
        if not api_key:
            raise ValueError(
                "DashScope ASR API Key 未配置，请在 .env 中设置 DASHSCOPE_ASR_API_KEY"
            )

        dashscope.api_key = api_key
        self.model = settings.dashscope_asr_model
        logger.info(f"DashScope ASR 服务初始化完成，模型: {self.model}")

    def transcribe(self, audio_path: str, language: str = "auto") -> dict:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 目标语言

        Returns:
            与 WhisperService 相同格式的结果字典
        """
        from dashscope.audio.asr import Recognition

        logger.info(f"[DashScope ASR] 开始转录: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        audio_duration = probe_duration(audio_path, settings.audio_probe_timeout_seconds)

        # 获取音频格式和采样率
        audio_format = self._detect_format(audio_path)
        sample_rate = self._get_sample_rate(audio_path)

        start_time = time.time()
        callback = _ASRCallback()

        try:
            recognition = Recognition(
                model=self.model,
                callback=callback,
                format=audio_format,
                sample_rate=sample_rate,
            )

            result = recognition.call(audio_path)
            transcribe_time = time.time() - start_time

            # 检查返回状态
            if result.status_code != 200:
                error_msg = result.message or f"状态码: {result.status_code}"
                logger.error(f"[DashScope ASR] API 错误: {error_msg}")
                raise RuntimeError(f"DashScope ASR 错误 ({result.status_code}): {error_msg}")

            # 解析识别结果
            sentences = result.get_sentence() or []
            if not sentences and callback.sentences:
                sentences = callback.sentences

            # 提取文本和分段信息
            full_text_parts = []
            segments = []
            for sent in sentences:
                if isinstance(sent, dict):
                    text = sent.get("text", "")
                    full_text_parts.append(text)
                    segments.append({
                        "start": (sent.get("begin_time", 0) or 0) / 1000,
                        "end": (sent.get("end_time", 0) or 0) / 1000,
                        "text": text,
                    })
                elif isinstance(sent, str):
                    full_text_parts.append(sent)

            transcription = "".join(full_text_parts)

            output = {
                "transcription": transcription,
                "segments": segments,
                "language": language,
                "language_probability": 1.0,
                "duration": audio_duration,
                "transcribe_time": transcribe_time,
                "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0,
            }

            logger.info(f"[DashScope ASR] 转录完成，耗时: {transcribe_time:.2f}秒")
            logger.info("[DashScope ASR] 转录结果长度: %s", len(transcription))
            return output

        except RuntimeError:
            raise
        except Exception as e:
            transcribe_time = time.time() - start_time
            logger.error("[DashScope ASR] 转录异常 (%.1f秒)", transcribe_time, exc_info=True)
            raise RuntimeError("DashScope ASR 转录失败") from e

    def _detect_format(self, audio_path: str) -> str:
        """根据文件扩展名检测音频格式"""
        ext = os.path.splitext(audio_path)[1].lower().strip(".")
        format_map = {
            "wav": "wav", "mp3": "mp3", "flac": "flac",
            "ogg": "ogg", "webm": "webm", "m4a": "m4a",
            "aac": "aac", "opus": "opus",
        }
        return format_map.get(ext, "wav")

    def _get_sample_rate(self, audio_path: str) -> int:
        """获取音频采样率"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.samplerate
        except Exception:
            pass
        logger.warning("[DashScope ASR] 无法获取采样率，使用默认 16000")
        return 16000


# 全局服务实例
dashscope_asr_service_instance = None


def get_dashscope_asr_service() -> DashScopeASRService:
    """获取 DashScope ASR 服务实例（单例模式）"""
    global dashscope_asr_service_instance
    if dashscope_asr_service_instance is None:
        dashscope_asr_service_instance = DashScopeASRService()
    return dashscope_asr_service_instance

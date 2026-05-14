"""
Whisper语音识别服务
"""
import os
import time
import logging
import librosa
from faster_whisper import WhisperModel
from ..core.config import settings
from ..models import Audio

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
        load_time = time.time() - start_load
        logger.info(f"模型加载耗时: {load_time:.2f}秒")

    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def transcribe(self, audio_path: str, language: str = "zh") -> dict:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 目标语言

        Returns:
            包含转录结果的字典
        """
        logger.info(f"开始转录: {audio_path}")
        logger.debug(f"文件存在: {os.path.exists(audio_path)}")
        if os.path.exists(audio_path):
            logger.debug(f"文件大小: {os.path.getsize(audio_path)} 字节")
        start_trans = time.time()

        # 获取音频时长
        try:
            audio_duration = librosa.get_duration(path=audio_path)
            logger.info(f"音频时长: {audio_duration:.2f}秒")
        except Exception as e:
            logger.error(f"获取音频时长失败: {e}", exc_info=True)
            raise RuntimeError(f"无法读取音频文件，请检查文件格式是否正确: {audio_path}") from e

        try:
            # 执行转录
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )

            # 强制迭代生成器，获取实际解码时间
            segments_list = list(segments)
            transcribe_time = time.time() - start_trans

            # 收集结果
            result = {
                "transcription": " ".join([seg.text for seg in segments_list]),
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    } for seg in segments_list
                ],
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "transcribe_time": transcribe_time,
                "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0
            }

            logger.info(f"转录完成，耗时: {transcribe_time:.2f}秒")
            logger.info(f"转录结果: {result['transcription'][:100]}")
            return result

        except Exception as e:
            logger.warning(f"VAD 转录失败，尝试禁用 VAD: {e}", exc_info=True)
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    task="transcribe"
                )
                segments_list = list(segments)
                transcribe_time = time.time() - start_trans

                result = {
                    "transcription": " ".join([seg.text for seg in segments_list]),
                    "segments": [
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text
                        } for seg in segments_list
                    ],
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "transcribe_time": transcribe_time,
                    "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0
                }
                logger.info(f"禁用 VAD 转录完成，耗时: {transcribe_time:.2f}秒")
                return result
            except Exception as e2:
                logger.error(f"禁用 VAD 转录也失败: {e2}", exc_info=True)
                raise RuntimeError(f"转录失败: {str(e2)}") from e2


# 全局服务实例
whisper_service_instance = None


def get_whisper_service() -> WhisperService:
    """获取 Whisper 服务实例（单例模式）"""
    global whisper_service_instance
    if whisper_service_instance is None:
        whisper_service_instance = WhisperService()
    return whisper_service_instance


def get_asr_service(provider: str = None):
    """
    ASR 工厂函数，根据配置或参数返回对应的 ASR 服务实例

    Args:
        provider: ASR 引擎名称，可选 "whisper" 或 "dashscope"。
                  为 None 时使用配置文件中的 asr_provider。

    Returns:
        具有 transcribe(audio_path, language) 方法的 ASR 服务实例
    """
    engine = (provider or settings.asr_provider).lower().strip()

    if engine == "dashscope":
        from .dashscope_asr_service import get_dashscope_asr_service
        return get_dashscope_asr_service()
    elif engine == "whisper":
        return get_whisper_service()
    else:
        logger.warning(f"未知的 ASR 引擎 '{engine}'，回退到本地 Whisper")
        return get_whisper_service()


def get_available_asr_providers() -> list:
    """获取当前可用的 ASR 引擎列表"""
    providers = [{"id": "whisper", "name": "本地 Whisper", "available": True}]

    # 检查 DashScope 是否配置
    dashscope_configured = bool(settings.dashscope_asr_api_key)
    providers.append({
        "id": "dashscope",
        "name": f"百炼 DashScope ({settings.dashscope_asr_model})",
        "available": dashscope_configured,
    })

    return providers
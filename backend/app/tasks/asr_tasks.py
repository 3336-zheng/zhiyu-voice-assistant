"""ASR 转写任务。

转写从 API 进程搬到 worker：一次远程调用可能跑满 180 秒的超时，占着 API 的
线程没有意义；搬过来之后 API 重启不会打断已经在跑的转写，ASR 也能按自己的
节奏单独扩缩（长任务混在索引 worker 里会占住并发槽）。

**落库放在 worker 侧**：转写和写 audio 表是同一件事，拆开的话 API 超时那一刻
结果就丢了，而任务其实还在跑。放这边则无论 API 是否还在等，结果都会落库。
"""

import asyncio
import logging
import os
from typing import Any, Dict

from backend.app.core.config import settings
from backend.app.core.database import SessionLocal

logger = logging.getLogger(__name__)


# 任务失败时返回的 code 全集。API 侧要把它映射成 HTTP 状态码，
# 这里定义成常量是为了让「worker 新增了一种失败」这件事有个单一出处——
# 否则 API 那边会静默落进默认分支，所有新错误都变成含糊的 502。
FAILURE_CODES = (
    "not_found",          # 音频记录或文件不存在
    "provider_not_found",  # 引擎名不认识
    "not_configured",      # 引擎没有配 key
    "input",               # 音频本身不合规（如超过大小限制）
    "execution",           # 上游调用失败，兜底
)


def _transcribe_and_persist(
    audio_id: int, provider_id: str, language: str
) -> Dict[str, Any]:
    """执行一次转写并写回 audio 表，返回结构化结果。

    返回 dict 而不是抛异常：任务在另一个进程里跑，异常要跨进程序列化才能传回
    API，而自定义异常类在两端版本不一致时会变成难查的解析错误。用结构化结果
    把「成功」和「哪一类失败」编码进去，API 侧只做映射。
    """
    from backend.app.models import Audio
    from backend.app.services.ingestion.asr_service import (
        ASRConfigurationError,
        ASRInputError,
        ASRProviderNotFoundError,
        transcribe_sync,
    )

    db = SessionLocal()
    try:
        audio = db.query(Audio).filter(Audio.id == audio_id).first()
        if audio is None:
            return {"ok": False, "code": "not_found", "message": "音频文件未找到"}

        file_path = audio.file_path
        if file_path and not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
            audio.file_path = file_path
            db.commit()

        if not file_path or not os.path.exists(file_path):
            return {"ok": False, "code": "not_found", "message": "音频文件不存在"}

        try:
            result = transcribe_sync(provider_id, file_path, language)
        except ASRProviderNotFoundError as exc:
            return {"ok": False, "code": "provider_not_found", "message": str(exc)}
        except ASRConfigurationError as exc:
            return {"ok": False, "code": "not_configured", "message": str(exc)}
        except ASRInputError as exc:
            return {"ok": False, "code": "input", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 — 细节留在 worker 日志里
            logger.exception(
                "ASR 转写失败: provider=%s audio_id=%s", provider_id, audio_id
            )
            return {
                "ok": False,
                "code": "execution",
                "message": "语音转写服务暂时不可用",
                "detail": str(exc)[:200],
            }

        audio.transcription = result["transcription"]
        audio.transcription_segments = result.get("segments", [])
        audio.language = result["language"]
        audio.duration = result.get("duration", audio.duration)
        db.commit()

        return {
            "ok": True,
            "result": {
                "transcription": result["transcription"],
                "segments": result.get("segments", []),
                "language": result["language"],
                "duration": result.get("duration"),
                "transcribe_time": result.get("transcribe_time"),
                "rtf": result.get("rtf"),
            },
        }
    finally:
        db.close()


async def transcribe_audio(
    ctx: Dict[str, Any], audio_id: int, provider_id: str, language: str
) -> Dict[str, Any]:
    """arq 任务入口：把同步的转写与落库挪到线程里执行。"""
    outcome = await asyncio.to_thread(
        _transcribe_and_persist, audio_id, provider_id, language
    )
    if not outcome.get("ok"):
        logger.warning(
            "ASR 任务未成功: audio_id=%s code=%s message=%s",
            audio_id, outcome.get("code"), outcome.get("message"),
        )
    return outcome


def asr_job_id(audio_id: int) -> str:
    """同一个音频同一时刻只允许一个转写任务，靠 arq 的 job_id 唯一性实现。

    进程内的 `_running_audio_ids` 只在单进程里有效，多实例下每个进程各有一份，
    拦不住别的实例；job_id 存在 Redis 上，所有实例共用一个判断。
    """
    return f"asr:{audio_id}"


async def release_asr_job(pool: Any, job_id: str) -> None:
    """释放 job_id，让同一个音频可以立刻换引擎重转。

    **必须删两个键**：arq 判断重复投递时看的是
    `exists(job_key, result_key_prefix + job_id)`。任务完成后 job 键 arq 自己
    会删掉，但结果键会按 keep_result 保留一段时间——只删 job 键是空操作，
    重投照样被拒，表现为「刚转完就不让再转了」。

    不释放的话要等 keep_result 过期（60 秒），用户会觉得莫名其妙。
    """
    await pool.delete(f"arq:job:{job_id}", f"arq:result:{job_id}")

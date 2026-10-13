"""音频上传、转写与回放 API。"""

import asyncio
import os
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.services.ingestion.asr_service import (
    ASRConfigurationError,
    ASRProviderNotFoundError,
    get_available_asr_providers,
    resolve_provider,
)
from backend.app.tasks.asr_tasks import asr_job_id, release_asr_job
from backend.app.tasks.queue import ensure_pool

# 转写任务的失败码到 HTTP 状态的映射。测试会验证它覆盖了
# asr_tasks.FAILURE_CODES 里的每一种，防止新增失败类型后静默落进 502。
ASR_FAILURE_STATUS = {
    "not_found": 404,
    "provider_not_found": 422,
    "not_configured": 503,
    "input": 413,
}
from backend.app.services.ingestion.audio_processing import (
    AudioProcessingError,
    AudioTooLargeError,
    normalize_upload,
)
from backend.app.services.ai.llm_service import get_llm_service
from backend.app.models import Audio


logger = logging.getLogger(__name__)
router = APIRouter()


async def _read_upload(file: UploadFile) -> bytes:
    chunks = []
    total_size = 0
    while chunk := await file.read(1024 * 1024):
        total_size += len(chunk)
        if total_size > settings.max_file_size:
            raise AudioTooLargeError("原始音频超过大小限制")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/upload/")
async def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """校验上传内容，并统一保存为标准 WAV。"""
    logger.info("收到音频上传请求: filename=%s", file.filename)
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名为空，请确保文件有正确的扩展名")
        original_name = Path(file.filename.replace("\\", "/")).name
        if not original_name or original_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="文件名不合法")

        allowed_extensions = settings.get_allowed_extensions()
        if Path(original_name).suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，允许: {settings.allowed_extensions}",
            )
        stem = Path(original_name).stem
        if not stem or stem in {".", ".."}:
            raise HTTPException(status_code=400, detail="文件名不合法")
        content = await _read_upload(file)
        artifact = await asyncio.to_thread(
            normalize_upload,
            original_name,
            content,
            settings.get_upload_dir(),
            settings.max_file_size,
            settings.audio_normalize_timeout_seconds,
            settings.audio_probe_timeout_seconds,
        )

        compatible_names = {f"{stem}{extension}" for extension in allowed_extensions}
        compatible_names.add(artifact.filename)
        existing_audio = db.query(Audio).filter(
            Audio.filename.in_(compatible_names)
        ).first()
        if existing_audio:
            existing_audio.filename = artifact.filename
            existing_audio.original_filename = original_name
            existing_audio.file_path = artifact.file_path
            existing_audio.file_size = artifact.file_size
            existing_audio.transcription = None
            existing_audio.transcription_segments = []
            existing_audio.language = None
            existing_audio.duration = artifact.duration
            db.commit()
            db.refresh(existing_audio)
            return {
                "message": "文件更新成功",
                "audio_id": existing_audio.id,
                "filename": artifact.filename,
                "duration": artifact.duration,
            }

        audio = Audio(
            filename=artifact.filename,
            original_filename=original_name,
            file_path=artifact.file_path,
            file_size=artifact.file_size,
            duration=artifact.duration,
        )
        db.add(audio)
        db.commit()
        db.refresh(audio)
        return {
            "message": "文件上传成功",
            "audio_id": audio.id,
            "filename": artifact.filename,
            "duration": artifact.duration,
        }
    except AudioTooLargeError as exc:
        max_mb = settings.max_file_size // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过限制（规范化前后均不得超过 {max_mb}MB）",
        ) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("音频上传失败")
        raise HTTPException(status_code=500, detail="音频上传失败") from exc


@router.delete("/{audio_id}")
async def delete_audio(audio_id: int, db: Session = Depends(get_db)):
    """删除音频文件及数据库记录"""
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="音频文件未找到")

    # 删除磁盘文件
    file_path = audio.file_path
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info("已删除音频文件: audio_id=%s", audio_id)
        except Exception as e:
            logger.exception("删除音频文件失败: audio_id=%s", audio_id)
            raise HTTPException(status_code=500, detail="文件删除失败") from e

    # 删除数据库记录
    db.delete(audio)
    db.commit()

    return {"message": "音频删除成功", "audio_id": audio_id}


@router.post("/transcribe/{audio_id}")
async def transcribe_audio(
    audio_id: int,
    provider: str | None = Query(None, description="ASR Provider，不传则使用默认配置"),
    db: Session = Depends(get_db),
):
    """把转写交给 ASR worker 执行，同步等它返回。

    接口形态没变（提交后等结果），变的是执行位置：转写与落库都在 worker 进程里，
    API 只负责提交和等待。这样 API 重启不会打断已在跑的转写，API 进程也不必
    为一次 180 秒的调用占住线程。
    """
    # 先做本地能做的校验，避免为一个明显无效的请求去占用队列。
    try:
        provider_id = resolve_provider(provider)
    except ASRProviderNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ASRConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="音频文件未找到")

    file_path = audio.file_path
    if file_path and not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
        audio.file_path = file_path
        db.commit()
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")

    # job_id 固定为音频维度：同一个音频同时只允许一个转写任务。
    # arq 对已存在的 job_id 会返回 None，这就是跨实例的重复提交拦截。
    job_id = asr_job_id(audio_id)
    pool = await ensure_pool()
    job = await pool.enqueue_job(
        "transcribe_audio",
        audio_id,
        provider_id,
        "auto",
        _job_id=job_id,
        _queue_name=settings.arq_asr_queue,
    )
    if job is None:
        raise HTTPException(status_code=409, detail="该音频正在转写，请勿重复提交")

    try:
        outcome = await asyncio.wait_for(
            job.result(), timeout=settings.asr_timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        # 任务可能还在 worker 里跑完并落库，所以不删 job 键——
        # 留着才能继续拦住重复提交。
        raise HTTPException(status_code=504, detail="语音转写超时，请稍后重试") from exc
    except Exception as exc:  # noqa: BLE001 — 取结果失败（Redis 中断、worker 被杀）
        logger.exception("获取转写结果失败: audio_id=%s", audio_id)
        raise HTTPException(status_code=502, detail="语音转写服务暂时不可用") from exc

    # 拿到结果就释放 job_id，让同一个音频可以立刻换引擎重转；
    # 不释放的话要等 keep_result 过期，用户会觉得「刚转完就不让再转了」。
    await release_asr_job(pool, job_id)

    if not outcome or not outcome.get("ok"):
        code = (outcome or {}).get("code", "execution")
        message = (outcome or {}).get("message", "语音转写服务暂时不可用")
        status = ASR_FAILURE_STATUS.get(code, 502)
        raise HTTPException(status_code=status, detail=message)

    result = outcome["result"]
    return {
        "message": "转录成功",
        "audio_id": audio_id,
        "transcription": result["transcription"],
        "segments": result.get("segments", []),
        "duration": result.get("duration"),
        "transcribe_time": result.get("transcribe_time"),
        "rtf": result.get("rtf"),
    }


@router.get("/{audio_id}/transcript")
async def get_transcript(
    audio_id: int,
    start: float = Query(0, ge=0),
    end: float = Query(None, gt=0),
    db: Session = Depends(get_db),
):
    """返回带时间戳的转录片段，可按时间范围过滤。"""
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="音频文件未找到")
    segments = audio.transcription_segments or []
    if end is not None and end < start:
        raise HTTPException(status_code=422, detail="end 必须大于或等于 start")
    filtered = [
        segment
        for segment in segments
        if float(segment.get("end", 0)) >= start
        and (end is None or float(segment.get("start", 0)) <= end)
    ]
    return {
        "audio_id": audio.id,
        "filename": audio.filename,
        "duration": audio.duration,
        "transcription": audio.transcription,
        "segments": filtered,
        "audio_url": f"/audio/{audio.id}/file",
    }


@router.get("/{audio_id}/file")
async def stream_audio(audio_id: int, db: Session = Depends(get_db)):
    """返回原始音频，支持浏览器媒体时间片定位。"""
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="音频文件未找到")
    upload_root = Path(settings.get_upload_dir()).resolve()
    path = Path(audio.file_path or "").resolve()
    if not path.is_relative_to(upload_root):
        raise HTTPException(status_code=403, detail="音频路径不在允许目录内")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=audio.filename,
        content_disposition_type="inline",
    )


@router.get("/asr-providers")
async def list_asr_providers():
    """获取可用的 ASR 引擎列表及当前默认配置"""
    providers = get_available_asr_providers()
    return {
        "default": settings.asr_provider,
        "providers": providers,
    }


@router.post("/polish/{audio_id}")
async def polish_transcription(audio_id: int, db: Session = Depends(get_db)):
    """
    对已转录的文本做口语清理（去口头禅、补标点、修正明显错误）
    不改变原意，不增删实质内容
    """
    logger.info("收到转录润色请求: audio_id=%s", audio_id)
    try:
        audio = db.query(Audio).filter(Audio.id == audio_id).first()
        if not audio:
            raise HTTPException(status_code=404, detail="音频记录未找到")

        raw_text = audio.transcription
        if not raw_text:
            raise HTTPException(status_code=400, detail="该音频尚未转录，无法润色")

        llm = get_llm_service()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个语音转录文本清理助手。请对以下语音转录文本做清理：\n"
                    "1. 去除无意义的口头禅和填充词（嗯、呃、那个、就是说、然后呢、对吧、你知道吗、怎么说呢、额、啊等）\n"
                    "2. 补全标点符号（句号、逗号、问号、感叹号等）\n"
                    "3. 修正语音识别错误：\n"
                    "   - 同音字/近音字错误（如「在见」→「再见」、「人事」→「认识」）\n"
                    "   - 明显的断词错误（如「机 器学习」→「机器学习」）\n"
                    "   - 乱码或无意义字符直接删除\n"
                    "4. 合理分段\n\n"
                    "严格要求：\n"
                    "- 不要改变原文的含义和观点\n"
                    "- 不要增删实质内容（口头禅和识别错误除外）\n"
                    "- 不要重组句子结构\n"
                    "- 不要添加原文没有的信息\n"
                    "- 遇到不确定的专有名词，保留原文不做修改\n"
                    "- 直接返回清理后的文本，不要加任何解释或前缀"
                )
            },
            {
                "role": "user",
                "content": raw_text
            }
        ]

        polished = llm.chat(messages=messages, max_tokens=2000, temperature=0.3)
        logger.info(
            "转录润色完成: audio_id=%s raw_length=%s polished_length=%s",
            audio_id,
            len(raw_text),
            len(polished),
        )

        return {
            "success": True,
            "raw": raw_text,
            "polished": polished
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("转录润色失败: audio_id=%s", audio_id)
        # 润色失败时返回原始文本，不影响使用
        return {
            "success": False,
            "raw": audio.transcription if audio else "",
            "polished": audio.transcription if audio else "",
            "detail": f"润色失败，已返回原始文本: {str(e)}"
        }

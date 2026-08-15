"""
音频管理API
"""
import os
import sys
import uuid
import logging
import shutil
import subprocess
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.services.ingestion.whisper_service import get_asr_service, get_available_asr_providers
from backend.app.services.ai.llm_service import get_llm_service
from backend.app.models import Audio


def _find_ffmpeg() -> str:
    """查找 ffmpeg 可执行文件路径"""
    # 优先从 PATH 中查找
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    # 尝试当前 Python 环境的常见位置
    python_dir = Path(sys.executable).parent
    candidates = [
        python_dir / "ffmpeg.exe",
        python_dir / "Library" / "bin" / "ffmpeg.exe",
        python_dir.parent / "Library" / "bin" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "找不到 ffmpeg，请确保已安装并在 PATH 中，"
        "或在当前 conda 环境中执行: conda install -c conda-forge ffmpeg"
    )

def log(msg):
    logger.info(msg)

def get_absolute_upload_path(filename: str) -> str:
    """获取文件的绝对路径"""
    upload_dir = Path(settings.get_upload_dir())
    return str(upload_dir / filename)


def is_wav_file(file_path: str) -> bool:
    """通过文件头检测是否为真正的 WAV 文件"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
            # WAV 文件头: RIFF....WAVE
            return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except Exception:
        return False


def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    使用 ffmpeg 将音频文件转换为 16-bit PCM WAV（16kHz 单声道）。
    直接输出到 output_path，不原地替换，避免 Windows 文件锁定问题。
    """
    try:
        ffmpeg_bin = _find_ffmpeg()
        cmd = [
            ffmpeg_bin, "-y", "-i", input_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=60
        )
        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace")[:300]
            log(f"[Convert] ffmpeg 转换失败: {stderr_text}")
            raise RuntimeError(f"ffmpeg 转换失败: {stderr_text[:200]}")

        log(f"[Convert] 已转换为标准 WAV: {output_path}")
        return output_path
    except subprocess.TimeoutExpired:
        log(f"[Convert] ffmpeg 转换超时")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError("音频转换超时")
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

router = APIRouter()

@router.post("/upload/")
async def upload_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    上传音频文件
    """
    log(f"[Upload] 收到上传请求: filename={file.filename}, content_type={file.content_type}")
    try:
        # 检查文件名是否有效
        if not file.filename:
            log(f"[Upload] 文件名为空")
            raise HTTPException(status_code=400, detail="文件名为空，请确保文件有正确的扩展名")
        original_name = Path(file.filename.replace("\\", "/")).name
        if not original_name or original_name in {".", ".."}:
            raise HTTPException(status_code=400, detail="文件名不合法")

        # 检查文件类型
        allowed_extensions = settings.get_allowed_extensions()
        log(f"[Upload] 允许的扩展名: {allowed_extensions}")
        if not any(original_name.lower().endswith(ext) for ext in allowed_extensions):
            log(f"[Upload] 文件格式不支持: {original_name}")
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {original_name}，允许: {settings.allowed_extensions}")

        # 获取绝对路径（最终 WAV 路径，统一用 .wav 后缀）
        upload_dir = Path(settings.get_upload_dir())
        stem = Path(original_name).stem
        if not stem or stem in {".", ".."}:
            raise HTTPException(status_code=400, detail="文件名不合法")
        final_path = str(upload_dir / f"{stem}.wav")
        # 原始文件路径（保留原始扩展名，用于保存未转换的文件）
        raw_path = str(upload_dir / original_name)
        log(f"[Upload] 最终路径: {final_path}")

        # 读取文件内容
        content = await file.read()
        log(f"[Upload] 文件大小: {len(content)} 字节")

        # 检查文件是否为空
        if len(content) == 0:
            log(f"[Upload] 文件内容为空")
            raise HTTPException(status_code=400, detail="上传的文件内容为空")

        # 检查文件大小
        if len(content) > settings.max_file_size:
            max_mb = settings.max_file_size // (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {max_mb}MB）")

        # 保存原始文件到磁盘
        with open(raw_path, "wb") as buffer:
            buffer.write(content)
        log(f"[Upload] 原始文件已保存: {raw_path}")

        # 检测并转换音频格式（确保下游 Whisper 拿到标准 WAV）
        if is_wav_file(raw_path):
            # 已是 WAV，直接使用
            file_path = final_path
            if raw_path != final_path:
                # 如果原始文件名不是 .wav（理论上不会），重命名
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(raw_path, final_path)
            else:
                # 文件名就是 .wav，覆盖旧文件
                pass
        else:
            log(f"[Upload] 检测到非 WAV 格式，开始 ffmpeg 转换...")
            # 删除旧的最终文件（如果存在）
            if os.path.exists(final_path) and final_path != raw_path:
                try:
                    os.remove(final_path)
                except PermissionError:
                    log(f"[Upload] 旧 WAV 文件被占用")
            # 转换：原始文件 -> 最终 WAV 路径
            file_path = convert_to_wav(raw_path, final_path)
            # 删除原始非 WAV 文件
            if raw_path != final_path and os.path.exists(raw_path):
                try:
                    os.remove(raw_path)
                except Exception:
                    pass  # 临时文件删除失败不影响主流程

        # 读取最终文件大小
        with open(file_path, "rb") as f:
            content = f.read()
        log(f"[Upload] 最终文件大小: {len(content)} 字节")

        # 最终文件名（统一为 .wav）
        final_filename = f"{stem}.wav"

        # 检查是否已存在同名文件记录（按 stem 匹配，兼容不同扩展名上传）
        existing_audio = db.query(Audio).filter(
            Audio.filename.like(f"{stem}.%")
        ).first()
        log(f"[Upload] 数据库已有记录: {existing_audio is not None}")

        if existing_audio:
            # 更新已有记录
            existing_audio.filename = final_filename
            existing_audio.original_filename = original_name
            existing_audio.file_path = file_path
            existing_audio.file_size = len(content)
            existing_audio.transcription = None
            existing_audio.transcription_segments = []
            existing_audio.language = None
            existing_audio.duration = None
            log(f"[Upload] 更新记录 id={existing_audio.id}")
            db.commit()
            db.refresh(existing_audio)
            log(f"[Upload] 更新成功")
            return {"message": "文件更新成功", "audio_id": existing_audio.id, "filename": final_filename}
        else:
            # 创建新记录
            audio = Audio(
                filename=final_filename,
                original_filename=original_name,
                file_path=file_path,
                file_size=len(content)
            )
            db.add(audio)
            log(f"[Upload] 创建新记录")
            db.commit()
            db.refresh(audio)
            log(f"[Upload] 创建成功 id={audio.id}")
            return {"message": "文件上传成功", "audio_id": audio.id, "filename": final_filename}
    except HTTPException:
        log(f"[Upload] HTTP异常")
        raise
    except Exception as e:
        log(f"[Upload] 未知异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"[上传失败] {type(e).__name__}: {str(e)}")

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
            log(f"[Delete] 已删除音频文件: {file_path}")
        except Exception as e:
            log(f"[Delete] 删除文件失败: {e}")
            raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")

    # 删除数据库记录
    db.delete(audio)
    db.commit()

    log(f"[Delete] 音频删除成功: id={audio_id}")
    return {"message": "音频删除成功", "audio_id": audio_id}


@router.post("/transcribe/{audio_id}")
async def transcribe_audio(
    audio_id: int,
    provider: str = Query(None, description="ASR 引擎: whisper 或 dashscope，不传则使用默认配置"),
    db: Session = Depends(get_db),
):
    """
    转录音频文件

    Args:
        provider: 可选的 ASR 引擎选择（whisper/dashscope）
    """
    log(f"[Transcribe] 收到转录请求: audio_id={audio_id}, provider={provider}")
    try:
        # 获取音频记录
        audio = db.query(Audio).filter(Audio.id == audio_id).first()
        if not audio:
            log(f"[Transcribe] 音频记录不存在: id={audio_id}")
            raise HTTPException(status_code=404, detail="音频文件未找到")

        log(f"[Transcribe] 音频记录: filename={audio.filename}, file_path={audio.file_path}")

        # 确保文件路径是绝对路径（兼容旧的相对路径记录）
        file_path = audio.file_path
        if file_path and not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
            log(f"[Transcribe] 相对路径转绝对路径: {audio.file_path} -> {file_path}")
            # 更新数据库中的路径
            audio.file_path = file_path
            db.commit()

        if not file_path or not os.path.exists(file_path):
            log(f"[Transcribe] 音频文件不存在: {file_path}")
            raise HTTPException(status_code=404, detail=f"音频文件不存在: {file_path}")

        # 使用工厂函数获取 ASR 服务
        asr_service = get_asr_service(provider)
        engine_name = type(asr_service).__name__
        log(f"[Transcribe] 使用 ASR 引擎: {engine_name}, 开始转录...")
        result = asr_service.transcribe(file_path)
        log(f"[Transcribe] 转录完成: {result['transcription'][:50]}...")

        # 更新音频记录
        audio.transcription = result["transcription"]
        audio.transcription_segments = result.get("segments", [])
        audio.language = result["language"]
        audio.duration = result.get("duration", audio.duration)
        db.commit()
        log(f"[Transcribe] 数据库已更新")

        return {
            "message": "转录成功",
            "audio_id": audio.id,
            "transcription": result["transcription"],
            "segments": result.get("segments", []),
            "duration": result.get("duration"),
            "transcribe_time": result["transcribe_time"],
            "rtf": result["rtf"]
        }
    except HTTPException:
        log(f"[Transcribe] HTTP异常")
        raise
    except Exception as e:
        log(f"[Transcribe] 未知异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"转录失败: {str(e)}")


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
    log(f"[Polish] 收到润色请求: audio_id={audio_id}")
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
        log(f"[Polish] 润色完成，原文 {len(raw_text)} 字 -> 润色后 {len(polished)} 字")

        return {
            "success": True,
            "raw": raw_text,
            "polished": polished
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"[Polish] 润色异常: {type(e).__name__}: {e}")
        # 润色失败时返回原始文本，不影响使用
        return {
            "success": False,
            "raw": audio.transcription if audio else "",
            "polished": audio.transcription if audio else "",
            "detail": f"润色失败，已返回原始文本: {str(e)}"
        }

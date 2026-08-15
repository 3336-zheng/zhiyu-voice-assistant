"""
课堂笔记模块 API
生成课堂笔记 → 预览 → 用户确认 → 通过 PageService 保存为 Wiki 页面
"""
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models import Audio
from backend.app.services.ai.llm_service import get_llm_service
from backend.app.services.wiki.page_service import PageValidationError, get_page_service

logger = logging.getLogger(__name__)

router = APIRouter()


class TranscriptSegment(BaseModel):
    """ASR 转录时间片。"""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class SummaryGenerateRequest(BaseModel):
    """生成课堂笔记请求"""
    content: str           # 转录文字
    title: str = None      # 可选标题（课程名称）
    segments: List[TranscriptSegment] = Field(default_factory=list)


class SummarySaveRequest(BaseModel):
    """保存纪要请求"""
    content: str           # 纪要内容（经过用户确认/编辑）
    filename: str          # 文件名
    title: str = None      # 标题
    audio_id: Optional[int] = None


def _format_timecode(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, second = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{second:02d}"
    return f"{minutes:02d}:{second:02d}"


def _timestamped_transcript(req: SummaryGenerateRequest) -> str:
    if not req.segments:
        return req.content
    return "\n".join(
        f"[{_format_timecode(segment.start)}-{_format_timecode(segment.end)}] "
        f"{segment.text.strip()}"
        for segment in req.segments
        if segment.text.strip()
    ) or req.content


@router.post("/generate")
async def generate_summary(req: SummaryGenerateRequest):
    """
    生成课堂笔记（仅预览，不存储）
    输出固定四段结构：① 知识点提纲 ② 重点概念/公式 ③ 课后疑问 ④ 复习卡片
    """
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="转录内容不能为空")

    try:
        llm = get_llm_service()

        system_prompt = (
            "你是一个课堂笔记整理助手。请根据以下课堂转录内容，生成结构化的课堂笔记。\n\n"
            "输出格式要求（严格按以下四段结构输出）：\n\n"
            "## 📚 知识点提纲\n"
            "按授课顺序列出本节课的主要知识点，使用层级列表（一、二、三...下辖 1. 2. 3...）。\n"
            "每个知识点用一句话概括核心内容。\n\n"
            "## ⭐ 重点概念与公式\n"
            "提取本节课的关键概念、定理、公式、算法步骤等，逐条列出。\n"
            "每个概念给出简明定义或解释，公式使用 LaTeX 格式。\n\n"
            "## ❓ 课后疑问\n"
            "列出学生可能存在的疑问点，包括：\n"
            "1. 转录中明确提到「不太懂」「为什么」「没听清」的地方\n"
            "2. 概念跳跃较大、逻辑链条断裂的地方\n"
            "3. 需要进一步查阅资料才能理解的内容\n\n"
            "## 🎴 复习卡片（Q&A）\n"
            "生成 5-10 张复习卡片，每张格式如下：\n"
            "**Q:** [问题]\n"
            "**A:** [答案]\n\n"
            "要求：\n"
            "1. 只使用转录中提到的内容，不要编造\n"
            "2. 语言简洁，适合快速复习\n"
            "3. 使用 Markdown 格式\n"
            "4. 保留转录中的关键术语、公式、示例"
        )
        if req.segments:
            system_prompt += (
                "\n5. 每个知识点、概念和复习卡片都要保留最相关的原始时间标记，"
                "格式必须为 [MM:SS-MM:SS]，不得修改或编造时间。"
            )

        # 如果有标题，作为课程名称提示
        title_hint = f"课程名称：{req.title}\n\n" if req.title else ""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{title_hint}请基于以下课堂转录内容生成结构化笔记：\n\n"
                    f"{_timestamped_transcript(req)}"
                ),
            }
        ]

        summary = llm.chat(messages=messages, temperature=0.3, max_tokens=3000)

        return {
            "success": True,
            "summary": summary,
            "original_length": len(req.content),
            "summary_length": len(summary),
            "segments_used": len(req.segments),
        }

    except Exception as e:
        logger.error(f"课堂笔记生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"课堂笔记生成失败: {str(e)}")


@router.post("/save")
async def save_summary(req: SummarySaveRequest, db: Session = Depends(get_db)):
    """
    用户确认后将课堂笔记保存为统一 Wiki 页面并建立索引
    """
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="纪要内容不能为空")

    if not req.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    try:
        # 确保文件名有 .md 后缀
        filename = req.filename
        if not filename.endswith(".md"):
            filename += ".md"

        # 清理文件名中的非法字符
        illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in illegal_chars:
            filename = filename.replace(char, '_')

        title = req.title or Path(filename).stem
        if req.audio_id is not None:
            audio = db.get(Audio, req.audio_id)
            if audio is None:
                raise HTTPException(status_code=404, detail="关联音频不存在")
            source_uri = f"audio:{req.audio_id}#summary:{filename}"
        else:
            source_uri = f"summary:{filename}"
        page = get_page_service(db).upsert_page_by_source(
            title=title,
            content=req.content,
            source_type="class_audio",
            source_uri=source_uri,
            change_summary="用户确认并保存课堂笔记",
        )
        logger.info(f"课堂笔记已保存为 Wiki 页面: {page['id']}")

        return {
            "success": True,
            "page_id": page["id"],
            "revision": page["revision"],
            "filename": page["filename"],
            "file_path": page["file_path"],
            "deduplicated": page["deduplicated"],
            "status": page["index_status"],
            "index_error": page["index_error"],
            "source_uri": page["source_uri"],
        }

    except HTTPException:
        raise
    except PageValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"纪要保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"纪要保存失败: {str(e)}")

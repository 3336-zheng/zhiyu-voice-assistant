"""
纪要总结模块 API
生成会议纪要 → 预览 → 用户确认 → 保存到 data/docs/ + 索引到 ChromaDB + BM25
"""
import os
import logging
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.llm_service import get_llm_service
from ..services.doc_index_service import get_doc_index_service

logger = logging.getLogger(__name__)

router = APIRouter()


class SummaryGenerateRequest(BaseModel):
    """生成纪要请求"""
    content: str           # 转录文字
    title: str = None      # 可选标题
    style: str = "meeting" # 纪要风格: meeting(会议纪要) / lecture(课堂笔记) / general(通用)


class SummarySaveRequest(BaseModel):
    """保存纪要请求"""
    content: str           # 纪要内容（经过用户确认/编辑）
    filename: str          # 文件名
    title: str = None      # 标题


@router.post("/generate")
async def generate_summary(req: SummaryGenerateRequest):
    """
    生成纪要（仅预览，不存储）
    """
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="转录内容不能为空")

    try:
        llm = get_llm_service()

        style_prompts = {
            "meeting": (
                "你是一个专业的会议纪要助手。请根据以下转录内容生成结构化的会议纪要。\n"
                "要求：\n"
                "1. 使用 Markdown 格式\n"
                "2. 包含：会议主题、参会人（如有提及）、讨论要点、决议事项、待办事项\n"
                "3. 只使用转录中提到的内容，不要编造\n"
                "4. 语言简洁专业"
            ),
            "lecture": (
                "你是一个课堂笔记整理助手。请根据以下转录内容生成结构化的课堂笔记。\n"
                "要求：\n"
                "1. 使用 Markdown 格式\n"
                "2. 按知识点分章节，使用标题层级\n"
                "3. 保留关键概念、公式、示例\n"
                "4. 只使用转录中提到的内容"
            ),
            "general": (
                "你是一个文本整理助手。请根据以下转录内容生成结构化的笔记。\n"
                "要求：\n"
                "1. 使用 Markdown 格式\n"
                "2. 按主题分类整理，去除口语化表达\n"
                "3. 保留关键信息\n"
                "4. 只使用转录中提到的内容"
            ),
        }

        system_prompt = style_prompts.get(req.style, style_prompts["general"])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请基于以下转录内容生成纪要：\n\n{req.content}"}
        ]

        summary = llm.chat(messages=messages, temperature=0.3, max_tokens=3000)

        return {
            "success": True,
            "summary": summary,
            "original_length": len(req.content),
            "summary_length": len(summary)
        }

    except Exception as e:
        logger.error(f"纪要生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"纪要生成失败: {str(e)}")


@router.post("/save")
async def save_summary(req: SummarySaveRequest):
    """
    保存纪要到 data/docs/ 并索引到 ChromaDB + BM25
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

        # 保存到 data/docs/
        docs_dir = "data/docs"
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, filename)

        # 构建文件内容
        title = req.title or Path(filename).stem
        file_content = f"# {title}\n\n{req.content}"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        logger.info(f"纪要已保存: {file_path}")

        # 立即索引到 ChromaDB + BM25
        doc_index_service = get_doc_index_service()
        index_result = doc_index_service.index_doc(file_path)

        logger.info(f"纪要已索引: {index_result}")

        return {
            "success": True,
            "filename": filename,
            "file_path": file_path,
            "chunks": index_result.get("chunks", 0),
            "status": index_result.get("status", "unknown")
        }

    except Exception as e:
        logger.error(f"纪要保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"纪要保存失败: {str(e)}")

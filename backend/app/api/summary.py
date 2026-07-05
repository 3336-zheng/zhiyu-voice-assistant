"""
课堂笔记模块 API
生成课堂笔记 → 预览 → 用户确认 → 保存到 data/docs/ + 索引到 ChromaDB + BM25
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
    """生成课堂笔记请求"""
    content: str           # 转录文字
    title: str = None      # 可选标题（课程名称）


class SummarySaveRequest(BaseModel):
    """保存纪要请求"""
    content: str           # 纪要内容（经过用户确认/编辑）
    filename: str          # 文件名
    title: str = None      # 标题


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

        # 如果有标题，作为课程名称提示
        title_hint = f"课程名称：{req.title}\n\n" if req.title else ""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{title_hint}请基于以下课堂转录内容生成结构化笔记：\n\n{req.content}"}
        ]

        summary = llm.chat(messages=messages, temperature=0.3, max_tokens=3000)

        return {
            "success": True,
            "summary": summary,
            "original_length": len(req.content),
            "summary_length": len(summary)
        }

    except Exception as e:
        logger.error(f"课堂笔记生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"课堂笔记生成失败: {str(e)}")


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

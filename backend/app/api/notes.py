"""
笔记管理API（笔记以 md 文件形式存储在 data/notes/ 目录下）
"""
import os
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from ..agent.markdown_agent import get_markdown_agent

logger = logging.getLogger(__name__)

router = APIRouter()

NOTES_DIR = "data/notes"


# ==================== 请求模型 ====================

class NoteCreateRequest(BaseModel):
    """创建笔记请求（已废弃）"""
    title: str
    content: str
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


class NoteUpdateRequest(BaseModel):
    """编辑笔记请求"""
    title: Optional[str] = None
    content: Optional[str] = None


class NoteCreateFileRequest(BaseModel):
    """直接创建笔记文件请求（写入 data/notes/）"""
    title: str
    content: str


# ==================== 已废弃接口 ====================

@router.post("/create")
async def create_note_independent(req: NoteCreateRequest):
    """独立创建笔记（已废弃）"""
    raise HTTPException(
        status_code=410,
        detail="此接口已废弃。笔记现在以 md 文件形式存储在 data/notes/ 目录下，请使用 POST /notes/create-file"
    )


@router.post("/create/")
async def create_note_from_audio(audio_id: int):
    """从音频创建笔记（已废弃）"""
    raise HTTPException(
        status_code=410,
        detail="此接口已废弃。录音转录不再自动存为笔记，请使用纪要总结模块（/summary/generate + /summary/save）"
    )


# ==================== 笔记文件 CRUD ====================

@router.post("/create-file")
async def create_note_file(req: NoteCreateFileRequest):
    """创建笔记 md 文件到 data/notes/"""
    from datetime import datetime
    agent = get_markdown_agent()

    filename = req.title or f"笔记_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result = agent.create_md_file(
        filename=filename,
        title=req.title,
        content=req.content,
        directory=NOTES_DIR
    )

    if result.get("success"):
        logger.info(f"笔记文件创建成功: {result.get('file_path')}")
        return {
            "success": True,
            "message": "笔记创建成功",
            "filename": result.get("filename"),
            "file_path": result.get("file_path"),
            "title": req.title
        }
    else:
        raise HTTPException(status_code=500, detail=f"创建笔记文件失败: {result.get('error')}")


@router.get("/list")
async def list_notes(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    """分页列出笔记（读取 data/notes/ 目录）"""
    agent = get_markdown_agent()
    result = agent.list_md_files(directory=NOTES_DIR)
    files = result.get("files", [])

    total = len(files)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = files[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {
                "filename": f["filename"],
                "title": f["filename"].replace(".md", ""),
                "size": f.get("size", 0),
                "modified_at": f.get("modified_at"),
            }
            for f in paginated
        ]
    }


@router.get("/{filename}")
async def get_note(filename: str):
    """获取笔记详情（读取 data/notes/ 下的 md 文件）"""
    agent = get_markdown_agent()
    result = agent.read_md_file(filename, directory=NOTES_DIR)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=f"笔记不存在: {filename}")

    return {
        "filename": filename,
        "title": filename.replace(".md", ""),
        "content": result.get("content", ""),
        "content_length": result.get("content_length", 0),
        "file_path": result.get("file_path"),
    }


@router.put("/{filename}")
async def update_note(filename: str, req: NoteUpdateRequest):
    """编辑笔记（覆写 data/notes/ 下的 md 文件）"""
    agent = get_markdown_agent()

    # 先检查文件是否存在
    read_result = agent.read_md_file(filename, directory=NOTES_DIR)
    if not read_result.get("success"):
        raise HTTPException(status_code=404, detail=f"笔记不存在: {filename}")

    # 构建新内容
    new_content = req.content if req.content is not None else read_result.get("content", "")
    if req.title:
        new_content = f"# {req.title}\n\n{new_content}"

    result = agent.write_md_file(
        filename=filename,
        content=new_content,
        mode="overwrite",
        directory=NOTES_DIR
    )

    if result.get("success"):
        logger.info(f"笔记更新成功: {filename}")
        return {
            "success": True,
            "message": "笔记更新成功",
            "filename": filename,
            "title": req.title or filename.replace(".md", ""),
        }
    else:
        raise HTTPException(status_code=500, detail=f"更新笔记失败: {result.get('error')}")


@router.delete("/{filename}")
async def delete_note(filename: str):
    """删除笔记（删除 data/notes/ 下的 md 文件）"""
    agent = get_markdown_agent()
    file_path = agent._get_file_path(filename, directory=NOTES_DIR)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"笔记不存在: {filename}")

    os.remove(file_path)

    logger.info(f"笔记文件已删除: {file_path}")
    return {"success": True, "message": "笔记删除成功", "filename": filename}


# ==================== 搜索 ====================

@router.get("/search/")
async def search_notes(
    query: str,
    top_k: int = 5,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    """检索相关笔记（混合检索 ChromaDB + BM25）"""
    if not query:
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    from ..services.retrieval_service import RetrievalService
    retrieval_service = RetrievalService()
    notes = retrieval_service.search_notes(query, top_k)

    total = len(notes)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = notes[start:end]

    return {
        "query": query,
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {
                "note_id": n.get("id"),
                "title": n.get("title", ""),
                "summary": n.get("summary", ""),
                "content": n.get("content", "")[:200] + "..." if len(n.get("content", "")) > 200 else n.get("content", "")
            }
            for n in paginated
        ]
    }
"""笔记兼容 API，内部统一委托给 PageService。"""
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from .compatibility import mark_legacy_response
from backend.app.services.wiki.page_service import (
    AmbiguousPageError,
    PageConflictError,
    PageNotFoundError,
    PageValidationError,
    get_page_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    deprecated=True,
    dependencies=[Depends(mark_legacy_response)],
)

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
    """通过兼容入口创建 Wiki 页面。"""
    title: str
    content: str


# ==================== 已废弃接口 ====================

@router.post("/create")
async def create_note_independent(req: NoteCreateRequest):
    """独立创建笔记（已废弃）"""
    raise HTTPException(
        status_code=410,
        detail="此接口已废弃。请使用 POST /notes/create-file 或 POST /api/pages"
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
async def create_note_file(req: NoteCreateFileRequest, db: Session = Depends(get_db)):
    """通过 PageService 创建统一 Wiki 页面。"""
    try:
        result = get_page_service(db).create_page(
            title=req.title,
            content=req.content,
            source_type="note",
            change_summary="通过兼容笔记接口创建",
        )
        logger.info(f"笔记页面创建成功: {result['id']}")
        return {
            "success": True,
            "message": "笔记创建成功",
            "page_id": result["id"],
            "revision": result["revision"],
            "filename": result["filename"],
            "file_path": result["file_path"],
            "title": result["title"],
            "index_status": result["index_status"],
        }
    except PageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/list")
async def list_notes(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db),
):
    """分页列出通过新旧笔记入口沉淀的 Wiki 页面。"""
    result = get_page_service(db).list_pages(offset=0, limit=1_000_000)
    files = [
        item for item in result["items"]
        if item["source_type"] in {"note", "legacy_note"}
    ]
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
                "page_id": f["id"],
                "filename": f["filename"],
                "title": f["title"],
                "revision": f["revision"],
                "size": 0,
                "modified_at": f.get("updated_at"),
            }
            for f in paginated
        ]
    }


# ==================== 搜索 ====================

@router.get("/search/")
async def search_notes(
    query: str,
    top_k: int = 5,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    """检索相关笔记（混合检索 ChromaDB + BM25）。"""
    if not query:
        raise HTTPException(status_code=400, detail="查询文本不能为空")

    from backend.app.services.retrieval.retrieval_service import RetrievalService

    notes = RetrievalService().search_notes(query, top_k)
    total = len(notes)
    start = (page - 1) * page_size
    paginated = notes[start:start + page_size]
    return {
        "query": query,
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {
                "note_id": item.get("page_id") or item.get("id"),
                "chunk_id": item.get("chunk_id") or item.get("id"),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "content": item.get("content", "")[:200],
                "page_revision": item.get("page_revision"),
                "section_title": item.get("section_title"),
                "source_url": item.get("source_url"),
            }
            for item in paginated
        ],
    }


@router.get("/{filename}")
async def get_note(filename: str, db: Session = Depends(get_db)):
    """按稳定 ID、标题或别名读取笔记页面。"""
    try:
        result = get_page_service(db).find_page(filename)
        return {
            "page_id": result["id"],
            "revision": result["revision"],
            "filename": result["filename"],
            "title": result["title"],
            "content": result["content"],
            "content_length": len(result["content"]),
            "file_path": result["file_path"],
        }
    except PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AmbiguousPageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put("/{filename}")
async def update_note(filename: str, req: NoteUpdateRequest, db: Session = Depends(get_db)):
    """通过 PageService 更新笔记并保存历史版本。"""
    service = get_page_service(db)
    try:
        current = service.find_page(filename)
        result = service.update_page(
            current["id"],
            expected_revision=current["revision"],
            title=req.title,
            content=req.content,
            change_summary="通过兼容笔记接口更新",
        )
        logger.info(f"笔记页面更新成功: {result['id']}")
        return {
            "success": True,
            "message": "笔记更新成功",
            "page_id": result["id"],
            "revision": result["revision"],
            "filename": result["filename"],
            "title": result["title"],
            "index_status": result["index_status"],
        }
    except PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (PageConflictError, AmbiguousPageError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/{filename}")
async def delete_note(filename: str, db: Session = Depends(get_db)):
    """通过 PageService 删除笔记及派生索引。"""
    service = get_page_service(db)
    try:
        current = service.find_page(filename)
        result = service.delete_page(
            current["id"],
            expected_revision=current["revision"],
        )
        logger.info(f"笔记页面已删除: {result['id']}")
        return {
            "success": True,
            "message": "笔记删除成功",
            "page_id": result["id"],
            "filename": result["filename"],
        }
    except PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (PageConflictError, AmbiguousPageError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

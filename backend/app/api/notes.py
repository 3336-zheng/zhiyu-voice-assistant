"""
笔记管理API
"""
import os
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..services.note_service import NoteService
from ..services.retrieval_service import RetrievalService
from ..services.chroma_service import get_chroma_service
from ..services.bm25_service import get_bm25_service
from ..services.embedding_service import get_embedding_service
from ..models import Note, Audio

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== 请求模型 ====================

class NoteCreateRequest(BaseModel):
    """独立创建笔记请求"""
    title: str
    content: str
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


class NoteUpdateRequest(BaseModel):
    """编辑笔记请求"""
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


# ==================== 笔记 CRUD ====================

@router.post("/create")
async def create_note_independent(req: NoteCreateRequest, db: Session = Depends(get_db)):
    """独立创建笔记（不依赖音频）"""
    # 自动生成摘要
    summary = req.summary
    if not summary and req.content:
        summary = req.content[:100] + "..." if len(req.content) > 100 else req.content

    note = Note(
        title=req.title,
        content=req.content,
        summary=summary,
        tags=req.tags or [],
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    # 同步到 ChromaDB + BM25
    try:
        embedding = get_embedding_service().encode(req.content)
        get_chroma_service().add_embedding(
            note_id=note.id,
            embedding=embedding,
            content=req.content,
            metadata={"title": req.title, "tags": req.tags or []}
        )
        get_bm25_service().add_document(f"note_{note.id}", req.content, req.title)
    except Exception as e:
        logger.warning(f"笔记 {note.id} 向量索引失败（不影响创建）: {e}")

    logger.info(f"笔记创建成功: id={note.id}, title={note.title}")
    return {
        "message": "笔记创建成功",
        "note_id": note.id,
        "title": note.title,
        "summary": note.summary
    }


@router.post("/create/")
async def create_note_from_audio(audio_id: int, db: Session = Depends(get_db)):
    """从音频创建笔记"""
    audio = db.query(Audio).filter(Audio.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="音频文件未找到")

    if not audio.file_path or not os.path.exists(audio.file_path):
        raise HTTPException(status_code=404, detail="音频文件不存在")

    note_service = NoteService()
    note = note_service.create_note_from_audio(audio.file_path, audio.original_filename, db)

    return {
        "message": "笔记创建成功",
        "note_id": note.id,
        "title": note.title,
        "summary": note.summary
    }


@router.get("/list")
async def list_notes(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db)
):
    """分页列出笔记"""
    total = db.query(Note).count()
    notes = (
        db.query(Note)
        .order_by(Note.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": [
            {
                "id": n.id,
                "title": n.title,
                "summary": n.summary,
                "tags": n.tags,
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in notes
        ]
    }


@router.get("/{note_id}")
async def get_note(note_id: int, db: Session = Depends(get_db)):
    """获取笔记详情"""
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="笔记不存在")
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "summary": note.summary,
        "tags": note.tags,
        "audio_id": note.audio_id,
        "duration": note.duration,
        "created_at": note.created_at.isoformat() if note.created_at else None,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }


@router.put("/{note_id}")
async def update_note(note_id: int, req: NoteUpdateRequest, db: Session = Depends(get_db)):
    """编辑笔记"""
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="笔记不存在")

    if req.title is not None:
        note.title = req.title
    if req.content is not None:
        note.content = req.content
        if not req.summary:
            note.summary = req.content[:100] + "..." if len(req.content) > 100 else req.content
    if req.tags is not None:
        note.tags = req.tags
    if req.summary is not None:
        note.summary = req.summary

    db.commit()
    db.refresh(note)

    # 同步更新 ChromaDB + BM25
    if req.content is not None:
        try:
            embedding = get_embedding_service().encode(note.content)
            get_chroma_service().update_embedding(
                note_id=note.id,
                embedding=embedding,
                content=note.content,
                metadata={"title": note.title, "tags": note.tags}
            )
            get_bm25_service().update_document(note.id, note.content, note.title)
        except Exception as e:
            logger.warning(f"笔记 {note.id} 向量更新失败: {e}")

    logger.info(f"笔记更新成功: id={note.id}")
    return {
        "message": "笔记更新成功",
        "note_id": note.id,
        "title": note.title,
        "summary": note.summary
    }


@router.delete("/{note_id}")
async def delete_note(note_id: int, db: Session = Depends(get_db)):
    """删除笔记（同步清理向量索引）"""
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="笔记不存在")

    db.delete(note)
    db.commit()

    # 清理 ChromaDB + BM25
    try:
        get_chroma_service().delete_by_note_id(note_id)
        get_bm25_service().remove_document(note_id)
    except Exception as e:
        logger.warning(f"笔记 {note_id} 向量清理失败: {e}")

    logger.info(f"笔记删除成功: id={note_id}")
    return {"message": "笔记删除成功", "note_id": note_id}


# ==================== 搜索 ====================

@router.get("/search/")
async def search_notes(
    query: str,
    top_k: int = 5,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    db: Session = Depends(get_db)
):
    """检索相关笔记（分页）"""
    if not query:
        raise HTTPException(status_code=400, detail="查询文本不能为空")

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
                "note_id": n.id,
                "title": n.title,
                "summary": n.summary,
                "content": n.content[:200] + "..." if len(n.content) > 200 else n.content
            }
            for n in paginated
        ]
    }
"""文档管理兼容 API，上传内容统一沉淀为 Wiki 页面。"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from .legacy import mark_legacy_response
from ..services.page_service import (
    AmbiguousPageError,
    PageNotFoundError,
    PageValidationError,
    get_page_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(
    deprecated=True,
    dependencies=[Depends(mark_legacy_response)],
)


def _document_pages(service):
    listed = service.list_pages(offset=0, limit=1_000_000)
    return [
        item for item in listed["items"]
        if item["source_type"] in {"document", "legacy_document"}
    ]


@router.get("/list")
async def list_docs(db: Session = Depends(get_db)):
    """获取由文档入口沉淀的 Wiki 页面列表。"""
    service = get_page_service(db)
    files = []
    for item in _document_pages(service):
        path = Path(service.get_page(item["id"])["file_path"])
        files.append(
            {
                "name": item["filename"],
                "title": item["title"],
                "page_id": item["id"],
                "revision": item["revision"],
                "size": path.stat().st_size if path.exists() else 0,
                "date": item["updated_at"],
                "index_status": item["index_status"],
            }
        )
    return {"files": files}


@router.post("/upload")
async def upload_doc(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """上传并转换文档，然后通过 PageService 幂等写入 Wiki。"""
    supported_extensions = (".md", ".txt", ".pdf", ".docx")
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {original_name}，支持: {', '.join(supported_extensions)}",
        )

    raw = await file.read()
    if len(raw) > settings.max_file_size:
        max_mb = settings.max_file_size // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {max_mb}MB）")

    converted = suffix not in {".md", ".txt"}
    if converted:
        from ..services.doc_convert_service import get_converter

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(raw)
                temporary_path = handle.name
            converter = get_converter(original_name)
            if converter is None:
                raise HTTPException(status_code=400, detail=f"不支持的文件格式: {original_name}")
            content = converter(temporary_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)
    else:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="文本文件必须使用 UTF-8 编码") from exc

    if not content or not content.strip():
        raise HTTPException(status_code=422, detail="文件内容为空")
    title = Path(original_name).stem
    try:
        page = get_page_service(db).upsert_page_by_source(
            title=title,
            content=content,
            source_type="document",
            source_uri=f"upload:{original_name}",
            change_summary=f"上传文档 {original_name}",
        )
    except PageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "message": "内容未变化，已跳过重复导入" if page["deduplicated"] else "上传成功",
        "filename": page["filename"],
        "original_filename": original_name,
        "page_id": page["id"],
        "revision": page["revision"],
        "converted": converted,
        "deduplicated": page["deduplicated"],
        "index_status": page["index_status"],
        "index_error": page["index_error"],
    }


@router.get("/view/{filename}")
async def view_doc(filename: str, db: Session = Depends(get_db)):
    """按页面 ID、标题或别名查看文档内容。"""
    try:
        page = get_page_service(db).find_page(filename)
        return {
            "filename": page["filename"],
            "page_id": page["id"],
            "title": page["title"],
            "revision": page["revision"],
            "content": page["content"],
        }
    except PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AmbiguousPageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/delete/{filename}")
async def delete_doc(filename: str, db: Session = Depends(get_db)):
    """删除文档页面并清理派生索引。"""
    service = get_page_service(db)
    try:
        current = service.find_page(filename)
        deleted = service.delete_page(
            current["id"],
            expected_revision=current["revision"],
        )
        return {
            "message": "删除成功",
            "filename": deleted["filename"],
            "page_id": deleted["id"],
        }
    except PageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AmbiguousPageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/reindex")
async def reindex_docs(db: Session = Depends(get_db)):
    """从 Wiki 主数据生成全量索引任务，由后台 worker 执行。"""
    try:
        result = get_page_service(db).queue_reindex()
        return {"message": "重建索引任务已入队", "result": result}
    except Exception as exc:
        logger.error("重建 Wiki 索引失败: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc

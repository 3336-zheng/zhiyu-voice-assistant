"""文档管理兼容 API，上传内容统一沉淀为 Wiki 页面。"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from .compatibility import mark_legacy_response
from backend.app.services.wiki.page_service import (
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
    """上传并转换文档，然后通过 PageService 幂等写入 Wiki。

    只接受 Markdown 和纯文本。PDF / Word 不走这里——它们要先经过格式转换，
    转换质量不确定、通常还要重排结构，那是另一个入口的事。混在一起会让
    「上传后内容变了」和「上传后内容原样」两种语义共用一条路径。
    """
    supported_extensions = (".md", ".txt")
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件格式: {original_name}。"
                f"这里只接受 {', '.join(supported_extensions)}；"
                "PDF / Word 请先用文档转换功能转成 Markdown 再上传。"
            ),
        )

    raw = await file.read()
    if len(raw) > settings.max_file_size:
        max_mb = settings.max_file_size // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {max_mb}MB）")

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

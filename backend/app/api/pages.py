"""统一 Wiki 页面 API。"""

import difflib
import io
import json
import zipfile
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..services.page_service import (
    AmbiguousPageError,
    PageConflictError,
    PageNotFoundError,
    PageService,
    PageValidationError,
    get_page_service,
)

router = APIRouter()


class PageCreateRequest(BaseModel):
    """创建 Wiki 页面。"""

    title: str
    content: str = ""
    notebook: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    source_type: str = "manual"
    source_uri: Optional[str] = None


class PageUpdateRequest(BaseModel):
    """更新 Wiki 页面，expected_revision 用于防止覆盖并发修改。"""

    expected_revision: int = Field(ge=1)
    title: Optional[str] = None
    content: Optional[str] = None
    notebook: Optional[str] = None
    tags: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    status: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    change_summary: Optional[str] = None


class PageRenameRequest(BaseModel):
    """重命名页面。"""

    title: str
    expected_revision: int = Field(ge=1)


class PageRollbackRequest(BaseModel):
    """回滚历史版本。"""

    target_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=1)


class LegacyImportRequest(BaseModel):
    """导入固定的旧知识目录，避免开放任意文件系统路径。"""

    source: Literal["notes", "documents"]
    notebook: Optional[str] = None
    sync_index: bool = False


def page_service(db: Session = Depends(get_db)) -> PageService:
    """获取绑定请求数据库会话的页面服务。"""
    return get_page_service(db)


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, PageNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (PageConflictError, AmbiguousPageError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PageValidationError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.post("")
@router.post("/")
def create_page(request: PageCreateRequest, service: PageService = Depends(page_service)):
    """创建页面并立即尝试建立派生索引。"""
    try:
        return service.create_page(**request.model_dump())
    except Exception as exc:
        _raise_http_error(exc)


@router.get("")
@router.get("/")
def list_pages(
    notebook: Optional[str] = None,
    tag: Optional[str] = None,
    query: Optional[str] = None,
    source_type: Optional[str] = None,
    include_deleted: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: PageService = Depends(page_service),
):
    """分页列出 Wiki 页面。"""
    result = service.list_pages(
        notebook=notebook,
        tag=tag,
        query=query,
        source_type=source_type,
        include_deleted=include_deleted,
        offset=(page - 1) * page_size,
        limit=page_size,
    )
    return {
        "total": result["total"],
        "page": page,
        "page_size": page_size,
        "results": result["items"],
    }


@router.post("/import-legacy")
def import_legacy(request: LegacyImportRequest, service: PageService = Depends(page_service)):
    """显式、幂等地导入旧笔记或文档目录。"""
    directory = "data/notes" if request.source == "notes" else "data/docs"
    source_type = "legacy_note" if request.source == "notes" else "legacy_document"
    return service.import_legacy_directory(
        directory,
        source_type=source_type,
        notebook=request.notebook,
        sync_index=request.sync_index,
    )


@router.post("/reindex")
def rebuild_index(service: PageService = Depends(page_service)):
    """从 Markdown 主数据生成全量索引任务，由后台 worker 执行。"""
    return service.queue_reindex()


@router.post("/index-tasks/retry")
def retry_index_tasks(
    limit: int = Query(100, ge=1, le=1000),
    service: PageService = Depends(page_service),
):
    """重试待处理或失败的索引任务。"""
    return service.retry_index_tasks(limit=limit)


@router.get("/export")
def export_pages(service: PageService = Depends(page_service)):
    """将所有活动页面和清单导出为 ZIP。"""
    listed = service.list_pages(offset=0, limit=1_000_000)
    manifest = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for summary in listed["items"]:
            page = service.get_page(summary["id"])
            with open(page["file_path"], "rb") as handle:
                archive.writestr(f"pages/{page['filename']}", handle.read())
            manifest.append({key: value for key, value in summary.items() if key != "file_path"})
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    export_name = "zhiyu-wiki-export.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{export_name}"'},
    )


@router.get("/{page_id}")
def get_page(page_id: str, service: PageService = Depends(page_service)):
    """读取指定页面。"""
    try:
        return service.get_page(page_id)
    except Exception as exc:
        _raise_http_error(exc)


@router.put("/{page_id}")
def update_page(
    page_id: str,
    request: PageUpdateRequest,
    service: PageService = Depends(page_service),
):
    """更新页面并生成新版本。"""
    try:
        return service.update_page(page_id, **request.model_dump())
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/{page_id}/rename")
def rename_page(
    page_id: str,
    request: PageRenameRequest,
    service: PageService = Depends(page_service),
):
    """重命名页面，旧标题自动成为别名。"""
    try:
        return service.rename_page(page_id, **request.model_dump())
    except Exception as exc:
        _raise_http_error(exc)


@router.delete("/{page_id}")
def delete_page(
    page_id: str,
    expected_revision: int = Query(..., ge=1),
    service: PageService = Depends(page_service),
):
    """删除当前 Markdown 与派生索引，保留历史版本。"""
    try:
        return service.delete_page(page_id, expected_revision=expected_revision)
    except Exception as exc:
        _raise_http_error(exc)


@router.get("/{page_id}/links")
def get_page_links(page_id: str, service: PageService = Depends(page_service)):
    """读取页面出链和反向链接。"""
    try:
        return service.get_links(page_id)
    except Exception as exc:
        _raise_http_error(exc)


@router.get("/{page_id}/revisions")
def list_page_revisions(page_id: str, service: PageService = Depends(page_service)):
    """列出页面版本。"""
    try:
        return {"page_id": page_id, "revisions": service.list_revisions(page_id)}
    except Exception as exc:
        _raise_http_error(exc)


@router.get("/{page_id}/revisions/{revision}")
def get_page_revision(
    page_id: str,
    revision: int,
    service: PageService = Depends(page_service),
):
    """读取指定版本快照。"""
    try:
        return service.get_revision(page_id, revision)
    except Exception as exc:
        _raise_http_error(exc)


@router.get("/{page_id}/diff")
def diff_page_revisions(
    page_id: str,
    from_revision: int = Query(..., ge=1),
    to_revision: int = Query(..., ge=1),
    service: PageService = Depends(page_service),
):
    """生成两个历史版本之间的统一差异。"""
    try:
        before = service.get_revision(page_id, from_revision)
        after = service.get_revision(page_id, to_revision)
        diff = "".join(
            difflib.unified_diff(
                before["content"].splitlines(keepends=True),
                after["content"].splitlines(keepends=True),
                fromfile=f"revision-{from_revision}",
                tofile=f"revision-{to_revision}",
            )
        )
        return {
            "page_id": page_id,
            "from_revision": from_revision,
            "to_revision": to_revision,
            "diff": diff,
        }
    except Exception as exc:
        _raise_http_error(exc)


@router.post("/{page_id}/rollback")
def rollback_page(
    page_id: str,
    request: PageRollbackRequest,
    service: PageService = Depends(page_service),
):
    """恢复历史内容并创建一个新版本。"""
    try:
        return service.rollback_page(page_id, **request.model_dump())
    except Exception as exc:
        _raise_http_error(exc)

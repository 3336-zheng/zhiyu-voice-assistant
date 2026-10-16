"""文档转换与结构重排接口。

**这三个接口都不写库**：转换和重排只返回文本给前端预览，用户在对比之后
点「写入知识库」才真正落库（走 commit）。这样用户能在入库前看到模型的
改动，不满意就改或放弃，不会污染知识库。

流程上分三段，按扩展名决定前两段走几段：

    PDF/DOCX  上传 → convert → restructure → commit
    MD/TXT    上传 →           restructure → commit

MD/TXT 的 convert 也会被调用（见 convert 里的分支），只是直接解码返回原文，
不经过 doc_convert_service。
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.services.wiki.page_service import (
    PageService,
    PageValidationError,
    get_page_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def page_service(db: Session = Depends(get_db)) -> PageService:
    """获取绑定请求数据库会话的页面服务。

    必须走依赖注入而不是在函数里直接调 get_page_service(db)：测试用
    `dependency_overrides` 替换的是这个依赖，直接调用会绕过替换，
    测试写进真实的 data/wiki/pages 目录。这个坑实际踩过——集成测试
    跑完，生产目录里多了几个「转换结果」孤儿文件。
    """
    return get_page_service(db)

# 走 doc_convert_service 的格式，其余按纯文本读。
CONVERT_EXTENSIONS = (".pdf", ".docx")
PLAIN_EXTENSIONS = (".md", ".txt")
MAX_EXTENSIONS = CONVERT_EXTENSIONS + PLAIN_EXTENSIONS


class RestructureRequest(BaseModel):
    content: str = Field(min_length=1, description="待重排的 Markdown 正文")


class CommitRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    notebook: str | None = Field(default=None, max_length=255)
    source_uri: str | None = Field(default=None, max_length=1024)


@router.post("/convert")
async def convert_document(file: UploadFile = File(...)):
    """上传文件，返回其 Markdown 文本。不写库，供前端预览。

    PDF/DOCX 按字号猜标题层级，转出来的结构通常很粗糙——这正是下一步
    「结构重排」要处理的。MD/TXT 直接解码，内容原样返回。
    """
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in MAX_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件格式: {original_name}。"
                f"支持: {', '.join(MAX_EXTENSIONS)}"
            ),
        )

    raw = await file.read()
    if len(raw) > settings.max_file_size:
        max_mb = settings.max_file_size // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {max_mb}MB）")

    if suffix in CONVERT_EXTENSIONS:
        content = await _convert_binary(raw, suffix, original_name)
    else:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="文本文件必须使用 UTF-8 编码") from exc

    if not content or not content.strip():
        raise HTTPException(status_code=422, detail="文件内容为空")

    return {
        "filename": original_name,
        "title": Path(original_name).stem,
        # 前端用它决定要不要显示「先转换」那一步的说明。
        "converted": suffix in CONVERT_EXTENSIONS,
        "content": content,
        "char_count": len(content),
    }


async def _convert_binary(raw: bytes, suffix: str, original_name: str) -> str:
    """把二进制文档转成 Markdown。临时文件用完即删。"""
    import asyncio
    import os
    import tempfile

    from backend.app.services.ingestion.doc_convert_service import get_converter

    converter = get_converter(original_name)
    if converter is None:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {original_name}")

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(raw)
            temporary_path = handle.name
        # 转换是纯 CPU/IO 的同步操作，放线程池里别卡住事件循环。
        return await asyncio.to_thread(converter, temporary_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — 转换失败要说清楚是哪个文件
        logger.exception("文档转换失败: %s", original_name)
        raise HTTPException(status_code=422, detail=f"文档转换失败: {exc}") from exc
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@router.post("/restructure")
async def restructure_document(request: RestructureRequest):
    """用 LLM 给正文补结构。不写库，返回结果供前端做前后对比。"""
    import asyncio

    from backend.app.services.ingestion.doc_restructure_service import (
        get_doc_restructure_service,
    )

    try:
        # LLM 调用是同步阻塞的，放线程池。
        result = await asyncio.to_thread(
            get_doc_restructure_service().restructure, request.content
        )
    except ValueError as exc:
        # 空内容、超长——都是用户能自己修的，422 而不是 500。
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档重排失败")
        raise HTTPException(status_code=502, detail=f"重排失败: {exc}") from exc

    return {
        "content": result["content"],
        "original": request.content,
        "model": result["model"],
        "original_chars": len(request.content),
        "restructured_chars": len(result["content"]),
        # 数字被改动过。不拦截，由前端在对比界面标出来让用户自己判断——
        # 提示词里写了不许改内容，但模型仍会顺手改数字。
        "numbers_changed": result["numbers_changed"],
        "number_diff": result["number_diff"],
    }


@router.post("/commit")
def commit_document(
    request: CommitRequest,
    service: PageService = Depends(page_service),
):
    """把确认后的内容写入知识库并建索引。"""
    try:
        page = service.upsert_page_by_source(
            title=request.title,
            content=request.content,
            notebook=request.notebook,
            source_type="document",
            source_uri=request.source_uri or f"transform:{request.title}",
            change_summary=f"文档转换并重排后写入: {request.title}",
        )
    except PageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "message": "内容未变化，已跳过重复导入" if page["deduplicated"] else "已写入知识库",
        "page_id": page["id"],
        "title": page["title"],
        "revision": page["revision"],
        "deduplicated": page["deduplicated"],
        "index_status": page["index_status"],
    }

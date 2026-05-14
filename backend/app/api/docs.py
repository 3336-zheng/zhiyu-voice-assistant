"""
文档管理 API
支持文档的上传、查看、删除，以及自动索引到向量库
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..core.config import settings
from ..services.doc_index_service import get_doc_index_service

logger = logging.getLogger(__name__)

router = APIRouter()

# 文档存储目录
DOCS_DIR = os.path.abspath("data/docs")


def ensure_docs_dir():
    """确保文档目录存在"""
    os.makedirs(DOCS_DIR, exist_ok=True)


def safe_path(filename: str) -> str:
    """校验文件名，防止路径遍历"""
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    filepath = os.path.abspath(os.path.join(DOCS_DIR, filename))
    if not filepath.startswith(DOCS_DIR):
        raise HTTPException(status_code=400, detail="非法文件路径")
    return filepath


@router.get("/list")
async def list_docs():
    """获取所有文档列表"""
    ensure_docs_dir()

    files = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(('.md', '.txt')):
            filepath = os.path.join(DOCS_DIR, filename)
            stat = os.stat(filepath)
            files.append({
                "name": filename,
                "size": stat.st_size,
                "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            })

    # 按修改时间倒序
    files.sort(key=lambda x: x["date"], reverse=True)
    return {"files": files}


@router.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    """上传文档并自动索引到向量库。支持 .md、.txt、.pdf、.docx、.doc 格式。"""
    supported_extensions = ('.md', '.txt', '.pdf', '.docx')
    if not any(file.filename.lower().endswith(ext) for ext in supported_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file.filename}，支持: {', '.join(supported_extensions)}"
        )

    ensure_docs_dir()

    content = await file.read()

    # 检查文件大小
    from ..core.config import settings as _settings
    if len(content) > _settings.max_file_size:
        max_mb = _settings.max_file_size // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {max_mb}MB）")

    # 判断是否需要格式转换
    is_markdown = file.filename.lower().endswith(('.md', '.txt'))

    if is_markdown:
        # 直接保存为原文件
        filepath = safe_path(file.filename)
        with open(filepath, "wb") as f:
            f.write(content)
        index_filename = file.filename
    else:
        # PDF/Word → 先转换为 Markdown，再保存为 .md
        from ..services.doc_convert_service import get_converter

        # 保存原始文件到临时路径
        import tempfile
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            converter = get_converter(file.filename)
            if converter is None:
                raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file.filename}")

            logger.info(f"开始转换文件: {file.filename}")
            md_content = converter(tmp_path)

            if not md_content or not md_content.strip():
                raise HTTPException(status_code=422, detail="文件转换后内容为空")

            # 生成 .md 文件名，保存到 data/docs/
            stem = Path(file.filename).stem
            md_filename = f"{stem}.md"
            filepath = safe_path(md_filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            index_filename = md_filename
            logger.info(f"文件转换完成: {file.filename} -> {md_filename} ({len(md_content)} 字符)")
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 自动索引到向量库
    try:
        doc_index = get_doc_index_service()
        result = doc_index.index_doc(filepath)
        return {
            "message": "上传成功" + ("（已自动转换为 Markdown）" if not is_markdown else ""),
            "filename": index_filename,
            "original_filename": file.filename if not is_markdown else None,
            "index_result": result
        }
    except Exception as e:
        # 索引失败不影响上传成功
        return {
            "message": "上传成功（索引失败）",
            "filename": index_filename,
            "original_filename": file.filename if not is_markdown else None,
            "index_error": str(e)
        }


@router.get("/view/{filename}")
async def view_doc(filename: str):
    """查看文档内容"""
    filepath = safe_path(filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文档不存在")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    return {"filename": filename, "content": content}


@router.delete("/delete/{filename}")
async def delete_doc(filename: str):
    """删除文档并清理向量库索引"""
    filepath = safe_path(filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文档不存在")

    os.remove(filepath)

    # 清理向量库索引
    try:
        doc_index = get_doc_index_service()
        doc_index.remove_doc(filename)
    except Exception as e:
        logger.warning(f"清理索引失败: {e}", exc_info=True)

    return {"message": "删除成功", "filename": filename}


@router.post("/reindex")
async def reindex_docs():
    """全量重建文档索引"""
    try:
        doc_index = get_doc_index_service()
        result = doc_index.reindex_all()
        return {"message": "重建索引完成", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {str(e)}")

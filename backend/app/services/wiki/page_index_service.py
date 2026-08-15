"""Wiki 页面派生索引服务。"""

import logging
import re
from typing import Any, Dict, List

from backend.app.core.config import settings
from backend.app.services.ai.embedding_service import get_embedding_service
from backend.app.services.ingestion.doc_index_service import (
    clean_markdown_for_chunking,
    split_markdown_by_headers,
)
from backend.app.services.retrieval.bm25_service import get_bm25_service
from backend.app.services.retrieval.chroma_service import get_chroma_service

logger = logging.getLogger(__name__)


def split_parent_into_children(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    """将父块切成更适合召回的子块，并保留少量上下文重叠。"""
    max_chars = max(1, max_chars)
    overlap_chars = min(max(0, overlap_chars), max_chars - 1)
    normalized = text.strip()
    if not normalized or len(normalized) <= max_chars:
        return []

    segments = [item.strip() for item in re.split(r"\n\s*\n", normalized) if item.strip()]
    if len(segments) == 1:
        segments = [item.strip() for item in re.split(r"(?<=[。！？.!?])", normalized) if item.strip()]

    children: List[str] = []
    current = ""
    for segment in segments:
        if not current:
            current = segment
            continue
        separator = "\n\n" if "\n" in normalized else ""
        if len(current) + len(separator) + len(segment) <= max_chars:
            current += separator + segment
            continue
        children.append(current)
        overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
        current = f"{overlap}{separator}{segment}".strip()

    if current:
        children.append(current)

    # 对无法按段落或句子切开的超长片段使用固定窗口兜底。
    step = max(1, max_chars - overlap_chars)
    normalized_children: List[str] = []
    for child in children:
        if len(child) <= max_chars:
            normalized_children.append(child)
            continue
        normalized_children.extend(
            child[start:start + max_chars]
            for start in range(0, len(child), step)
        )
    return normalized_children


class PageIndexService:
    """将 Wiki 页面分块后同步到 ChromaDB 与 BM25。"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.chroma_service = get_chroma_service()
        self.bm25_service = get_bm25_service()

    def index_page(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """索引页面的当前版本，旧版本分块会先被清理。"""
        page_id = page["id"]
        revision = int(page["revision"])
        content = clean_markdown_for_chunking(page.get("content", ""))
        self.remove_page(page_id)
        if not content.strip():
            return {
                "page_id": page_id,
                "revision": revision,
                "chunks": 0,
                "status": "skipped_empty",
            }

        chunks = split_markdown_by_headers(content, f"{page_id}.md")
        if not chunks:
            return {
                "page_id": page_id,
                "revision": revision,
                "chunks": 0,
                "status": "skipped_no_chunks",
            }

        parent_ids = [
            f"page:{page_id}:revision:{revision}:chunk:{index}"
            for index in range(len(chunks))
        ]
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            section_title = chunk.get("section_title") or page["title"]
            parent_id = parent_ids[index]
            base_metadata = {
                "source_type": "wiki_page",
                "page_id": page_id,
                "page_revision": revision,
                "page_title": page["title"],
                "filename": page.get("filename", f"{page_id}.md"),
                "section_title": section_title,
                "section_path": section_title,
                "chunk_index": index,
                "notebook": page.get("notebook") or "",
                "tags": ",".join(page.get("tags") or []),
                "source_uri": page.get("source_uri") or "",
                "updated_at": page.get("updated_at") or "",
                "parent_chunk_id": parent_id,
            }
            ids.append(parent_id)
            texts.append(chunk["text"])
            metadatas.append({**base_metadata, "chunk_level": "parent", "child_index": -1})

            if settings.rag_parent_child_enabled:
                child_size = max(80, settings.rag_child_chunk_chars)
                children = split_parent_into_children(
                    chunk["text"],
                    max_chars=child_size,
                    overlap_chars=min(
                        max(0, settings.rag_child_chunk_overlap_chars),
                        child_size - 1,
                    ),
                )
                for child_index, child_text in enumerate(children):
                    child_id = f"{parent_id}:child:{child_index}"
                    ids.append(child_id)
                    texts.append(child_text)
                    metadatas.append(
                        {
                            **base_metadata,
                            "chunk_level": "child",
                            "child_index": child_index,
                        }
                    )

        embeddings = self.embedding_service.encode_documents(texts)

        self.chroma_service.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        for chunk_id, text, metadata in zip(ids, texts, metadatas):
            self.bm25_service.add_document(
                chunk_id,
                text,
                f"{page['title']} {metadata['section_title']}",
            )
        child_count = len(ids) - len(chunks)
        logger.info(
            "Wiki 页面索引完成: %s@%s，父块 %s，子块 %s",
            page_id,
            revision,
            len(chunks),
            child_count,
        )
        return {
            "page_id": page_id,
            "revision": revision,
            "chunks": len(chunks),
            "child_chunks": child_count,
            "index_units": len(ids),
            "status": "indexed",
        }

    def remove_page(self, page_id: str) -> bool:
        """删除页面的全部历史分块。"""
        if not self.chroma_service.delete_by_filter({"page_id": page_id}):
            raise RuntimeError(f"ChromaDB 页面索引删除失败: {page_id}")
        prefix = f"page:{page_id}:revision:"
        chunk_ids = [
            doc_id
            for doc_id in list(self.bm25_service.corpus)
            if doc_id.startswith(prefix)
        ]
        for chunk_id in chunk_ids:
            self.bm25_service.remove_document(chunk_id)
        logger.info("Wiki 页面索引已删除: %s，共 %s 块", page_id, len(chunk_ids))
        return True

    def clear_page_index(self) -> bool:
        """清理全部 Wiki 页面派生索引。"""
        if not self.chroma_service.delete_by_filter({"source_type": "wiki_page"}):
            raise RuntimeError("ChromaDB Wiki 页面索引清理失败")
        chunk_ids = [
            doc_id
            for doc_id in list(self.bm25_service.corpus)
            if doc_id.startswith("page:")
        ]
        for chunk_id in chunk_ids:
            self.bm25_service.remove_document(chunk_id)
        return True


page_index_service_instance = None


def get_page_index_service() -> PageIndexService:
    """获取页面索引服务单例。"""
    global page_index_service_instance
    if page_index_service_instance is None:
        page_index_service_instance = PageIndexService()
    return page_index_service_instance

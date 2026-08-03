"""Wiki 页面派生索引服务。"""

import logging
from typing import Any, Dict, List

from .bm25_service import get_bm25_service
from .chroma_service import get_chroma_service
from .doc_index_service import clean_markdown_for_chunking, split_markdown_by_headers
from .embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


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

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.encode_documents(texts)
        ids = [
            f"page:{page_id}:revision:{revision}:chunk:{index}"
            for index in range(len(chunks))
        ]
        metadatas: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            section_title = chunk.get("section_title") or page["title"]
            metadatas.append(
                {
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
                }
            )

        self.chroma_service.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        for chunk_id, chunk, metadata in zip(ids, chunks, metadatas):
            self.bm25_service.add_document(
                chunk_id,
                chunk["text"],
                f"{page['title']} {metadata['section_title']}",
            )
        logger.info("Wiki 页面索引完成: %s@%s，共 %s 块", page_id, revision, len(chunks))
        return {
            "page_id": page_id,
            "revision": revision,
            "chunks": len(chunks),
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

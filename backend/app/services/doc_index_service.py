"""
文档索引服务
将 data/docs/ 下的 Markdown 文档分块、嵌入、存入 ChromaDB 和 BM25
分块策略：基于 Markdown 文档结构（按标题层级切分）

启动时采用增量同步策略：
- ChromaDB 持久化数据不丢失，无需重建
- BM25 是内存索引，每次启动需从 ChromaDB doc chunks 重建
- 仅对磁盘上新增/修改的文件重新嵌入入库，已有的跳过
- 清理磁盘上已删除文件的残留 chunks
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import logging

from backend.app.core.config import settings
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.chroma_service import get_chroma_service
from backend.app.services.bm25_service import get_bm25_service

logger = logging.getLogger(__name__)

def clean_markdown_for_chunking(content: str) -> str:
    """
    对 Markdown 文本进行数据清洗，优化分块和检索效果。
    适用于所有入库的 .md 文件（不论来源是手动编写还是 PDF/Word 转换）。

    清洗内容：
    1. 去除控制字符（\\x00 等）
    2. 去除页眉页脚残留（页码模式、PDF 提取的重复短行）
    3. 标题层级规范化（不跳级，适配 split_markdown_by_headers）
    4. 合并连续空行（超过 2 个换行压缩为 2 个）
    5. 去除行首行尾多余空格
    """
    # 1. 去除控制字符（保留换行和制表符）
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)

    # 2. 去除页码模式（如 "第 1 页"、"Page 1"、"- 1 -"、"1/10" 等独立行）
    content = re.sub(
        r"^\s*(第\s*\d+\s*页|Page\s*\d+|\d+\s*/\s*\d+|\-\s*\d+\s*\-)\s*$",
        "",
        content,
        flags=re.MULTILINE
    )

    # 3. 去除连续重复的短行（PDF 转换常见页眉页脚残留）
    lines = content.split("\n")
    cleaned_lines = _remove_repeated_short_lines(lines)
    content = "\n".join(cleaned_lines)

    # 4. 标题层级规范化：确保标题不跳级（如 # 之后直接 ### 补为 ##）
    content = _normalize_header_levels(content)

    # 5. 合并连续空行（最多保留 1 个空行）
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 6. 去除每行首尾多余空格
    lines = content.split("\n")
    lines = [line.rstrip() for line in lines]
    content = "\n".join(lines)

    return content.strip()


def _remove_repeated_short_lines(lines: list) -> list:
    """
    去除连续重复出现的短行（通常是 PDF 转换的页眉页脚残留）。
    如果某行（长度 <= 30 字符）在上下文中出现 3 次以上且间隔相近，则视为页眉页脚。
    """
    # 统计每行出现次数（仅统计短行）
    short_line_counts = Counter()
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) <= 30 and not stripped.startswith("#"):
            short_line_counts[stripped] += 1

    # 出现 3 次以上的短行视为疑似页眉页脚
    repeated_lines = {line for line, count in short_line_counts.items() if count >= 3}

    if not repeated_lines:
        return lines

    result = []
    for line in lines:
        stripped = line.strip()
        if stripped in repeated_lines:
            # 检查是否是独立行（前后为空行或边界），才去除
            # 如果是内容中间的行则保留（避免误删正文中的重复短句）
            result.append("")  # 替换为空行
        else:
            result.append(line)

    return result


def _normalize_header_levels(content: str) -> str:
    """
    规范化标题层级，确保不跳级。
    例如：# 一级 → # 一级，### 三级（无二级）→ ## 三级
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    lines = content.split("\n")
    result = []
    last_level = 0

    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # 跳级修正：新标题级别不能比上一级大超过 1
            if level > last_level + 1 and last_level > 0:
                level = last_level + 1

            last_level = level
            result.append(f"{'#' * level} {title}")
        else:
            result.append(line)

    return "\n".join(result)


def split_markdown_by_headers(
    content: str,
    filename: str,
    max_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    按 Markdown 标题层级切分文档，返回带元数据的块列表。
    超长块按段落递归细分。

    Args:
        content: Markdown 文档内容
        filename: 文件名

    Returns:
        List[Dict]: [{text, metadata}, ...]
    """
    del filename  # 保留参数以兼容既有调用方。
    chunk_chars = max_chars or settings.rag_parent_chunk_chars
    chunk_chars = max(1, chunk_chars)
    chunk_overlap = (
        settings.rag_parent_chunk_overlap_chars
        if overlap_chars is None
        else overlap_chars
    )
    chunk_overlap = min(max(0, chunk_overlap), chunk_chars - 1)
    lines = content.split("\n")
    chunks = []
    current_headers = {}  # {level: header_text}
    current_lines = []
    current_section_title = ""

    def flush_chunk():
        """将当前缓冲区的内容作为一个块保存"""
        text = "\n".join(current_lines).strip()
        if not text:
            return
        # 超长块按段落细分
        sub_chunks = _split_long_text(text, chunk_chars, chunk_overlap)
        for sub_text in sub_chunks:
            chunks.append({
                "text": sub_text,
                "section_title": current_section_title,
                "headers": dict(current_headers)
            })

    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            # 遇到新标题，先保存之前的块
            flush_chunk()

            level = len(match.group(1))
            title_text = match.group(2).strip()

            # 更新标题层级（清除同级和子级）
            current_headers = {k: v for k, v in current_headers.items() if k < level}
            current_headers[level] = title_text

            # 构建完整的章节标题路径
            current_section_title = " > ".join(
                current_headers[k] for k in sorted(current_headers.keys())
            )

            # 开始新块，标题行也包含在块中
            current_lines = [line]
        else:
            current_lines.append(line)

    # 保存最后一个块
    flush_chunk()

    return chunks


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    将超长文本按段落切分，带重叠。

    Args:
        text: 原始文本
        max_chars: 单块最大字符数
        overlap: 重叠字符数

    Returns:
        List[str]: 切分后的文本块
    """
    if len(text) <= max_chars:
        return [text]

    # 按段落（双换行）切分
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        # 没有段落分隔，按单换行切分
        paragraphs = [line for line in text.split("\n") if line.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += "\n\n" + para
        else:
            # 当前块已满
            chunks.append(current_chunk)
            # 重叠：取上一块末尾
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class DocIndexService:
    """文档索引服务"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.chroma_service = get_chroma_service()
        self.bm25_service = get_bm25_service()
        self.docs_dir = "data/docs"

    def index_doc(self, filepath: str) -> Dict[str, Any]:
        """
        索引单个文档：读取 → 分块 → 嵌入 → 存入 ChromaDB + BM25

        Args:
            filepath: 文件路径

        Returns:
            Dict: 索引结果
        """
        filepath = str(filepath)
        filename = os.path.basename(filepath)

        # 读取文件
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logger.warning(f"文档内容为空，跳过索引: {filename}")
            return {"filename": filename, "chunks": 0, "status": "skipped_empty"}

        # 先删除旧索引
        self.remove_doc(filename)

        # 数据清洗（对所有 md 文件统一处理）
        content = clean_markdown_for_chunking(content)
        logger.info(f"文档 {filename} 清洗完成，清洗后 {len(content)} 字符")

        # 分块
        chunks = split_markdown_by_headers(content, filename)
        if not chunks:
            logger.warning(f"文档分块为空: {filename}")
            return {"filename": filename, "chunks": 0, "status": "skipped_no_chunks"}

        logger.info(f"文档 {filename} 分块完成: {len(chunks)} 块")

        # 批量嵌入
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.encode_documents(texts)

        # 写入 ChromaDB
        file_mtime = os.path.getmtime(filepath)
        doc_ids = []
        for i, chunk in enumerate(chunks):
            doc_id = f"doc_{Path(filename).stem}_{i}"
            doc_ids.append(doc_id)
            metadata = {
                "source_type": "doc",
                "filename": filename,
                "section_title": chunk["section_title"],
                "chunk_idx": i,
                "file_mtime": file_mtime,
            }
            self.chroma_service.collection.add(
                ids=[doc_id],
                embeddings=[embeddings[i]],
                documents=[chunk["text"]],
                metadatas=[metadata]
            )

        # 写入 BM25
        for i, chunk in enumerate(chunks):
            doc_id = doc_ids[i]
            self.bm25_service.add_document(
                doc_id, chunk["text"], chunk["section_title"]
            )

        logger.info(f"文档索引完成: {filename}, {len(chunks)} 块已入库")
        return {"filename": filename, "chunks": len(chunks), "status": "indexed"}

    def remove_doc(self, filename: str) -> bool:
        """
        删除指定文档的所有索引

        Args:
            filename: 文件名

        Returns:
            bool: 是否成功
        """
        # 从 ChromaDB 删除
        self.chroma_service.delete_by_source(filename)

        # 从 BM25 删除（需要找到所有属于该文件的 doc_id）
        doc_ids_to_remove = [
            doc_id for doc_id in self.bm25_service.corpus.keys()
            if doc_id.startswith(f"doc_{Path(filename).stem}_")
        ]
        for doc_id in doc_ids_to_remove:
            self.bm25_service.remove_document(doc_id)

        logger.info(f"已删除文档索引: {filename} ({len(doc_ids_to_remove)} 块)")
        return True

    def reindex_all(self) -> Dict[str, Any]:
        """
        全量重建 data/docs/ 下所有 md 文件的索引

        Returns:
            Dict: 重建结果统计
        """
        if not os.path.exists(self.docs_dir):
            logger.warning(f"文档目录不存在: {self.docs_dir}")
            return {"total": 0, "indexed": 0, "skipped": 0}

        # 先删除所有 doc 类型的旧索引
        self._clear_all_doc_index()

        results = {"total": 0, "indexed": 0, "skipped": 0, "errors": []}

        for filename in os.listdir(self.docs_dir):
            if not filename.endswith((".md", ".txt")):
                continue

            results["total"] += 1
            filepath = os.path.join(self.docs_dir, filename)

            try:
                result = self.index_doc(filepath)
                if result["status"] == "indexed":
                    results["indexed"] += 1
                else:
                    results["skipped"] += 1
            except Exception as e:
                logger.error(f"索引文档失败 {filename}: {e}")
                results["errors"].append(f"{filename}: {str(e)}")

        logger.info(
            f"全量重建完成: 共 {results['total']} 个文件, "
            f"成功 {results['indexed']}, 跳过 {results['skipped']}, "
            f"失败 {len(results['errors'])}"
        )
        return results

    def sync_docs(self) -> Dict[str, Any]:
        """
        增量同步 data/docs/ 下的文档索引（启动时调用）

        策略：
        1. 从 ChromaDB 已有 doc chunks 重建 BM25（BM25 是内存索引）
        2. 扫描磁盘文件，与 ChromaDB 中已存的 mtime 对比
        3. 仅对新增/修改的文件重新嵌入入库
        4. 清理磁盘上已删除文件的残留 chunks

        Returns:
            Dict: 同步结果统计
        """
        if not os.path.exists(self.docs_dir):
            logger.warning(f"文档目录不存在: {self.docs_dir}")
            os.makedirs(self.docs_dir, exist_ok=True)
            return {"total": 0, "reindexed": 0, "skipped": 0, "cleaned": 0}

        # ---- 第一步：重建 BM25 ----
        self._rebuild_bm25_from_persistent()

        # ---- 第二步：扫描磁盘文件，增量同步 ChromaDB ----
        disk_files: Dict[str, float] = {}  # {filename: mtime}
        for filename in os.listdir(self.docs_dir):
            if filename.endswith((".md", ".txt")):
                filepath = os.path.join(self.docs_dir, filename)
                disk_files[filename] = os.path.getmtime(filepath)

        # 获取 ChromaDB 中已有的 doc chunks，按 filename 分组，取最新 mtime
        chroma_mtimes: Dict[str, float] = self._get_indexed_mtimes()

        reindexed = 0
        skipped = 0

        for filename, disk_mtime in disk_files.items():
            stored_mtime = chroma_mtimes.get(filename)
            if stored_mtime is not None and stored_mtime >= disk_mtime:
                # 文件未修改，跳过嵌入（BM25 已在第一步重建）
                skipped += 1
                continue

            # 新文件或已修改 → 重新索引（index_doc 内部会先 remove 旧的）
            try:
                result = self.index_doc(os.path.join(self.docs_dir, filename))
                if result["status"] == "indexed":
                    reindexed += 1
                    logger.info(f"增量索引: {filename}")
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"增量索引失败 {filename}: {e}")

        # ---- 第三步：清理已删除文件的残留 chunks ----
        cleaned = 0
        chroma_filenames: Set[str] = set(chroma_mtimes.keys())
        stale_filenames = chroma_filenames - set(disk_files.keys())
        for stale_name in stale_filenames:
            try:
                self.remove_doc(stale_name)
                cleaned += 1
                logger.info(f"清理残留索引: {stale_name}")
            except Exception as e:
                logger.error(f"清理残留索引失败 {stale_name}: {e}")

        result = {
            "total": len(disk_files),
            "reindexed": reindexed,
            "skipped": skipped,
            "cleaned": cleaned,
        }
        logger.info(
            f"增量同步完成: 磁盘 {result['total']} 个文件, "
            f"重新索引 {reindexed}, 跳过 {skipped}, 清理 {cleaned}"
        )
        return result

    def _rebuild_bm25_from_persistent(self):
        """
        从 ChromaDB 已有 doc chunks 重建 BM25 索引。
        BM25 是纯内存索引，每次启动必须重建。
        先收集所有文档，再一次性构建索引，避免逐条 add 的 O(n²) 开销。
        """
        from rank_bm25 import BM25Okapi

        bm25 = self.bm25_service
        bm25.clear_index()

        # 收集所有待索引文档: [(doc_id, content, title)]
        entries: List[tuple] = []

        # 从 ChromaDB 加载旧文档与统一 Wiki 页面分块
        try:
            results = self.chroma_service.collection.get(
                include=["documents", "metadatas"]
            )
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    content = results["documents"][i] if results["documents"] else ""
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    if metadata.get("source_type") in {"doc", "wiki_page"}:
                        section_title = metadata.get("section_title", "")
                        page_title = metadata.get("page_title", "")
                        entries.append((doc_id, content, f"{page_title} {section_title}".strip()))
        except Exception as e:
            logger.error(f"从 ChromaDB 加载 doc chunks 失败: {e}")

        # 一次性构建 BM25 索引
        for doc_id, content, title in entries:
            bm25.corpus[doc_id] = content
            bm25.doc_id_list.append(doc_id)
            text = f"{title} {content}"
            tokens = bm25._tokenize(text)
            bm25.tokenized_corpus.append(tokens)

        if bm25.tokenized_corpus:
            bm25.bm25 = BM25Okapi(
                bm25.tokenized_corpus,
                k1=bm25.k1,
                b=bm25.b
            )

        logger.info(f"BM25 索引重建完成，文档与 Wiki 分块数: {bm25.get_document_count()}")

    def _get_indexed_mtimes(self) -> Dict[str, float]:
        """
        获取 ChromaDB 中已索引文档的文件修改时间。
        每个文件取其 chunks 中最新的 mtime。

        Returns:
            Dict[str, float]: {filename: mtime}
        """
        try:
            results = self.chroma_service.collection.get(
                where={"source_type": "doc"},
                include=["metadatas"]
            )
            mtimes: Dict[str, float] = {}
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    fname = metadata.get("filename", "")
                    mtime = metadata.get("file_mtime", 0)
                    if fname and mtime > mtimes.get(fname, 0):
                        mtimes[fname] = mtime
            return mtimes
        except Exception as e:
            logger.error(f"获取已索引 mtime 失败: {e}")
            return {}

    def _clear_all_doc_index(self):
        """清除所有 doc 类型的索引"""
        # ChromaDB
        self.chroma_service.delete_by_filter(where={"source_type": "doc"})

        # BM25
        doc_ids_to_remove = [
            doc_id for doc_id in self.bm25_service.corpus.keys()
            if doc_id.startswith("doc_")
        ]
        for doc_id in doc_ids_to_remove:
            self.bm25_service.remove_document(doc_id)

        logger.info(f"已清除所有文档索引 ({len(doc_ids_to_remove)} 块)")


# 全局服务实例
doc_index_service_instance = None


def get_doc_index_service() -> DocIndexService:
    """获取文档索引服务实例（单例模式）"""
    global doc_index_service_instance
    if doc_index_service_instance is None:
        doc_index_service_instance = DocIndexService()
    return doc_index_service_instance

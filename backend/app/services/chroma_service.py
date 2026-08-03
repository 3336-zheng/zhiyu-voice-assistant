"""
ChromaDB 向量数据库服务
支持元数据过滤和混合检索
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Tuple, Optional, Dict, Any
import logging

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class ChromaService:
    """
    ChromaDB 向量数据库服务
    负责向量存储、检索和元数据过滤
    """

    def __init__(self, persist_directory: str = None):
        """
        初始化 ChromaDB 服务

        Args:
            persist_directory: ChromaDB 持久化目录，默认使用配置中的路径
        """
        self.persist_directory = persist_directory or settings.chroma_persist_path
        self.collection_name = settings.chroma_collection_name

        # 初始化 ChromaDB 客户端（持久化模式）
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,  # 禁用匿名遥测
            )
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

        logger.info(f"ChromaDB 服务初始化完成，集合: {self.collection_name}")

    def add_embedding(
        self,
        note_id: int,
        embedding: List[float],
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加向量到 ChromaDB

        Args:
            note_id: 笔记ID
            embedding: 1024维向量
            content: 笔记内容（用于文档存储）
            metadata: 元数据字典（可包含标题、标签、时间等）

        Returns:
            bool: 是否成功
        """
        try:
            # 准备元数据
            if metadata is None:
                metadata = {}
            metadata["note_id"] = note_id

            # 使用 note_id 作为文档ID
            doc_id = f"note_{note_id}"

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            logger.debug(f"成功添加向量: note_id={note_id}")
            return True
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return False

    def add_embeddings_batch(
        self,
        note_ids: List[int],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        批量添加向量

        Args:
            note_ids: 笔记ID列表
            embeddings: 向量列表
            contents: 内容列表
            metadatas: 元数据列表

        Returns:
            bool: 是否成功
        """
        try:
            if metadatas is None:
                metadatas = [{} for _ in note_ids]

            # 添加 note_id 到元数据
            for i, note_id in enumerate(note_ids):
                metadatas[i]["note_id"] = note_id

            doc_ids = [f"note_{nid}" for nid in note_ids]

            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

            logger.info(f"成功批量添加 {len(note_ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"批量添加向量失败: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        向量相似度检索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            where: 元数据过滤条件，如 {"tag": "会议"}

        Returns:
            List[Tuple[str, float]]: [(doc_id, score), ...]，按分数降序
            doc_id 格式: "note_1" 或 "doc_xxx_0"
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["metadatas", "distances", "documents"]
            )

            # 提取结果
            doc_scores = []
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else []

                for i, chroma_id in enumerate(ids):
                    # 使用 ChromaDB 的 doc_id（即存储时的 id）
                    source_type = metadatas[i].get("source_type", "note")
                    if source_type in {"doc", "wiki_page"}:
                        # 文档块和 Wiki 页面块直接使用 ChromaDB 的稳定 ID
                        doc_id = chroma_id
                    else:
                        # 笔记：转换为 "note_{note_id}" 格式
                        note_id = metadatas[i].get("note_id")
                        if note_id is not None:
                            doc_id = f"note_{note_id}"
                        else:
                            continue
                    # ChromaDB 返回的是距离（越小越相似），转换为相似度分数
                    distance = distances[i]
                    similarity = 1.0 - distance
                    doc_scores.append((doc_id, similarity))

            return doc_scores
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        """
        文本检索（使用 ChromaDB 内置的向量化，需要配置嵌入函数）

        Note: 当前项目使用外部 BGE-M3 模型，此方法不推荐使用
        """
        logger.warning("search_by_text 需要配置嵌入函数，请使用 search() 方法")
        return []

    def delete_by_note_id(self, note_id: int) -> bool:
        """
        删除指定笔记的向量

        Args:
            note_id: 笔记ID

        Returns:
            bool: 是否成功
        """
        try:
            doc_id = f"note_{note_id}"
            self.collection.delete(ids=[doc_id])
            logger.debug(f"成功删除向量: note_id={note_id}")
            return True
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    def delete_by_filter(self, where: Dict[str, Any]) -> bool:
        """
        根据元数据过滤条件删除向量

        Args:
            where: 过滤条件，如 {"tag": "测试"}

        Returns:
            bool: 是否成功
        """
        try:
            self.collection.delete(where=where)
            logger.info(f"成功删除符合条件的向量: {where}")
            return True
        except Exception as e:
            logger.error(f"按条件删除向量失败: {e}")
            return False

    def delete_by_source(self, filename: str) -> bool:
        """
        删除指定来源文件的所有文档块

        Args:
            filename: 文件名

        Returns:
            bool: 是否成功
        """
        try:
            self.collection.delete(where={
                "$and": [
                    {"source_type": "doc"},
                    {"filename": filename}
                ]
            })
            logger.info(f"成功删除文档索引: {filename}")
            return True
        except Exception as e:
            logger.error(f"删除文档索引失败: {e}")
            return False

    def get_doc_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """
        获取指定文件的所有文档块

        Args:
            filename: 文件名

        Returns:
            List[Dict]: 文档块列表
        """
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"source_type": "doc"},
                        {"filename": filename}
                    ]
                },
                include=["documents", "metadatas"]
            )
            chunks = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    chunks.append({
                        "id": doc_id,
                        "content": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    })
            return chunks
        except Exception as e:
            logger.error(f"获取文档块失败: {e}")
            return []

    def get_by_note_id(self, note_id: int) -> Optional[Dict[str, Any]]:
        """
        获取指定笔记的向量数据

        Args:
            note_id: 笔记ID

        Returns:
            Dict: 包含向量、元数据、内容的字典，或 None
        """
        try:
            doc_id = f"note_{note_id}"
            result = self.collection.get(
                ids=[doc_id],
                include=["embeddings", "metadatas", "documents"]
            )

            if result["ids"] and len(result["ids"]) > 0:
                return {
                    "note_id": note_id,
                    "embedding": result["embeddings"][0] if result["embeddings"] else None,
                    "metadata": result["metadatas"][0] if result["metadatas"] else None,
                    "content": result["documents"][0] if result["documents"] else None
                }
            return None
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return None

    def get_count(self) -> int:
        """
        获取集合中的文档数量

        Returns:
            int: 文档数量
        """
        return self.collection.count()

    def update_embedding(
        self,
        note_id: int,
        embedding: List[float],
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新指定笔记的向量

        Args:
            note_id: 笔记ID
            embedding: 新向量
            content: 新内容
            metadata: 新元数据

        Returns:
            bool: 是否成功
        """
        try:
            # 准备元数据
            if metadata is None:
                metadata = {}
            metadata["note_id"] = note_id

            doc_id = f"note_{note_id}"

            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            logger.debug(f"成功更新向量: note_id={note_id}")
            return True
        except Exception as e:
            logger.error(f"更新向量失败: {e}")
            return False

    def upsert_embedding(
        self,
        note_id: int,
        embedding: List[float],
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        插入或更新向量（如果不存在则插入，存在则更新）

        Args:
            note_id: 笔记ID
            embedding: 向量
            content: 内容
            metadata: 元数据

        Returns:
            bool: 是否成功
        """
        try:
            # 准备元数据
            if metadata is None:
                metadata = {}
            metadata["note_id"] = note_id

            doc_id = f"note_{note_id}"

            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            logger.debug(f"成功 upsert 向量: note_id={note_id}")
            return True
        except Exception as e:
            logger.error(f"upsert 向量失败: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        清空整个集合（危险操作）

        Returns:
            bool: 是否成功
        """
        try:
            # 删除集合并重新创建
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning(f"已清空集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False


# 全局服务实例
chroma_service = None


def get_chroma_service() -> ChromaService:
    """获取 ChromaDB 服务实例（单例模式）"""
    global chroma_service
    if chroma_service is None:
        chroma_service = ChromaService()
    return chroma_service

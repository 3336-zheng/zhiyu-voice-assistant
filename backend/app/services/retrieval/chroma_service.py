"""
ChromaDB 向量数据库服务
支持元数据过滤和混合检索
"""
import hashlib
import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Tuple, Optional, Dict, Any
import logging

from backend.app.core import index_version
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def resolve_embedding_collection_name(
    base_name: str,
    *,
    provider: str,
    api_url: str = "",
    model: str = "",
    dimensions: int = 0,
) -> str:
    """为在线 Embedding 配置生成稳定且不包含凭证的集合名。"""
    if provider == "local":
        return base_name
    profile = "|".join(
        [provider.strip(), api_url.strip().rstrip("/"), model.strip(), str(dimensions or 0)]
    )
    digest = hashlib.sha256(profile.encode("utf-8")).hexdigest()[:12]
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "-", base_name).strip("._-") or "notes"
    safe_base = safe_base[:49].rstrip("._-") or "notes"
    return f"{safe_base}-{digest}"


def _create_client(persist_directory: str):
    """按 chroma_mode 创建客户端：embedded 直连本地目录，server 连独立容器。

    抽成函数而不是在 __init__ 里写 if/else，是因为 client 的构造差异是本模块的
    内部细节——全仓没有任何外部代码引用 chroma_service.client，把差异钉死在这一个
    函数体里，其余代码只管「拿到一个 client」。也没有必要为两种模式抽基类和子类：
    两者的 Collection API 完全一致，子类之间唯一的差别就是这几行构造代码。

    server 模式下这里会主动 heartbeat 一次。多花一个来回，换的是「连不上就启动失败」
    而不是「启动正常、第一次检索才发现召回是 0」——后者正是本 Stage 要消灭的那类
    静默失败。
    """
    if settings.chroma_mode == "server":
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            client.heartbeat()
        except Exception as exc:
            # 本机开着系统级代理时，这里是最容易卡住的一步：chromadb 内部用 httpx，
            # 而 httpx 默认 trust_env=True，在 macOS 上会去读系统网络偏好里的代理，
            # 于是连本机容器的请求也被送进代理，返回 502。curl 不读系统代理设置，
            # 所以会出现「curl 通、应用连不上」这种看起来自相矛盾的现象。
            # 解法是让 NO_PROXY 覆盖 chroma_host（环境变量优先级高于系统设置）。
            raise RuntimeError(
                f"连接 Chroma server 失败（{settings.chroma_host}:{settings.chroma_port}）: {exc}。"
                f"若本机开启了系统代理，需要把 NO_PROXY 设为包含 {settings.chroma_host}。"
            ) from exc
        logger.info(
            "Chroma 以 server 模式连接: %s:%s",
            settings.chroma_host,
            settings.chroma_port,
        )
        return client

    logger.info("Chroma 以 embedded 模式运行，数据目录: %s", persist_directory)
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaSettings(
            anonymized_telemetry=False,  # 禁用匿名遥测
        ),
    )


class ChromaService:
    """
    ChromaDB 向量数据库服务
    负责向量存储、检索和元数据过滤
    """

    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """
        初始化 ChromaDB 服务

        Args:
            persist_directory: ChromaDB 持久化目录，默认使用配置中的路径
        """
        self.persist_directory = persist_directory or settings.chroma_persist_path
        self.base_collection_name = collection_name or settings.chroma_collection_name
        self.collection_name = (
            self.base_collection_name
            if collection_name
            else resolve_embedding_collection_name(
                self.base_collection_name,
                provider=settings.embedding_provider,
                api_url=settings.embedding_api_url,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        )

        # 初始化 ChromaDB 客户端（embedded 或 server，由配置决定）
        self.client = _create_client(self.persist_directory)

        # 获取或创建集合。属性名带下划线是有意的：本服务之外不该有人拿到原生
        # Collection 再直接调 Chroma 的 API，那样会绕过 mark_index_changed()，
        # 索引变了而版本号没涨，其他实例就不会重建——一个不报错的错。
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        # 索引版本号用来识别“文档数量不变但正文已更新”的缓存失效场景。
        # 它存在 Redis 上而不是进程内，否则多实例时别人改了索引本进程不会知道。

        logger.info(
            "ChromaDB 服务初始化完成，集合: %s，Embedding Provider: %s",
            self.collection_name,
            settings.embedding_provider,
        )

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
            embedding: 当前 Embedding 模型生成的向量
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

            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            logger.debug(f"成功添加向量: note_id={note_id}")
            self.mark_index_changed()
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

            self._collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

            logger.info(f"成功批量添加 {len(note_ids)} 个向量")
            self.mark_index_changed()
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
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["metadatas", "distances", "documents"]
            )

            # 提取结果。历史索引可能包含空 metadata 或不完整的并行数组，
            # 单条脏数据不能让整个向量召回失败。
            doc_scores = []
            skipped_metadata = 0
            ids_groups = results.get("ids") or []
            if ids_groups and ids_groups[0]:
                ids = ids_groups[0]
                distance_groups = results.get("distances") or []
                metadata_groups = results.get("metadatas") or []
                distances = distance_groups[0] if distance_groups else []
                metadatas = metadata_groups[0] if metadata_groups else []

                for i, chroma_id in enumerate(ids):
                    metadata = metadatas[i] if i < len(metadatas) else None
                    distance = distances[i] if i < len(distances) else None
                    if not isinstance(metadata, dict) or not isinstance(distance, (int, float)):
                        skipped_metadata += 1
                        continue

                    # 使用 ChromaDB 的 doc_id（即存储时的 id）
                    source_type = metadata.get("source_type", "note")
                    if source_type in {"doc", "wiki_page"}:
                        # 文档块和 Wiki 页面块直接使用 ChromaDB 的稳定 ID
                        doc_id = chroma_id
                    else:
                        # 笔记：转换为 "note_{note_id}" 格式
                        note_id = metadata.get("note_id")
                        if note_id is not None:
                            doc_id = f"note_{note_id}"
                        else:
                            skipped_metadata += 1
                            continue
                    # ChromaDB 返回的是距离（越小越相似），转换为相似度分数
                    similarity = 1.0 - distance
                    doc_scores.append((doc_id, similarity))

            if skipped_metadata:
                logger.warning(
                    "向量检索跳过无效结果: skipped=%s, requested_top_k=%s",
                    skipped_metadata,
                    top_k,
                )

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
            self._collection.delete(ids=[doc_id])
            logger.debug(f"成功删除向量: note_id={note_id}")
            self.mark_index_changed()
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
            self._collection.delete(where=where)
            logger.info(f"成功删除符合条件的向量: {where}")
            self.mark_index_changed()
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
            self._collection.delete(where={
                "$and": [
                    {"source_type": "doc"},
                    {"filename": filename}
                ]
            })
            logger.info(f"成功删除文档索引: {filename}")
            self.mark_index_changed()
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
            results = self._collection.get(
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
            result = self._collection.get(
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
        return self._collection.count()

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> bool:
        """批量写入分块，文档索引与 Wiki 页面索引共用。

        不复用 add_embeddings_batch：那个方法的 ID 是它自己按 note_id 拼出来的
        （note_{id}），而这里的 ID 由调用方决定（doc_{stem}_{i}、page:{id}:...），
        两者的主键规则不同，硬凑到一起只会让 ID 从哪来变得难以追踪。

        一次提交而不是循环里逐条 add：embedded 模式下两者只差几次函数调用，
        server 模式下是 N 次 HTTP 往返和 1 次的区别。
        """
        if not ids:
            return True
        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            # 写入点自己负责标记版本变化，调用方不需要（也不应该）再记得调一次。
            self.mark_index_changed()
            return True
        except Exception as e:
            logger.error(f"批量写入分块失败: {e}")
            return False

    def get_all_chunks(self, source_types: List[str]) -> List[Dict[str, Any]]:
        """按 source_type 拉取全部分块，供 BM25 全量重建使用。

        过滤条件下推到 Chroma 而不是拉回来再在 Python 里筛：embedded 模式下多拉的
        那部分几乎免费，server 模式下它要先序列化、过网络、再反序列化，最后才被丢掉。
        另外这里只 include documents 和 metadatas，不要 embeddings——向量的体积比
        正文大一个量级，而 BM25 重建根本用不到它。

        **本方法故意不捕获异常，失败时直接往上抛，不要「顺手」加 try/except 返回空列表。**
        调用方 rebuild_bm25_from_persistent 拿到空列表会把 BM25 索引清空并标记成
        「已同步到当前版本」，于是 Chroma 的一次临时故障会让检索永久退化成零结果，
        而且再也不会触发重建。让异常抛出去，调用方才有机会保住旧索引、下次重试。

        Returns:
            List[Dict]: [{"id", "content", "metadata"}]
        """
        results = self._collection.get(
            where={"source_type": {"$in": source_types}},
            include=["documents", "metadatas"],
        )
        chunks: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(results["ids"] or []):
            chunks.append({
                "id": doc_id,
                "content": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })
        return chunks

    def get_doc_metadatas(self) -> List[Dict[str, Any]]:
        """拉取所有文档类分块的元数据，用于比对文件修改时间。

        不 include documents：这个用途只看 filename 和 file_mtime，把正文一起拉回来
        在 server 模式下纯属浪费带宽。
        """
        try:
            results = self._collection.get(
                where={"source_type": "doc"},
                include=["metadatas"],
            )
            return list(results["metadatas"] or [])
        except Exception as e:
            logger.error(f"获取文档元数据失败: {e}")
            return []

    def get_chunks_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """按 ID 批量取分块详情。

        Returns:
            Dict: {doc_id: {"content": ..., "metadata": ...}}，取不到的 ID 不会出现在结果里
        """
        if not ids:
            return {}
        try:
            results = self._collection.get(
                ids=ids,
                include=["documents", "metadatas"],
            )
            chunks: Dict[str, Dict[str, Any]] = {}
            for i, doc_id in enumerate(results["ids"] or []):
                chunks[doc_id] = {
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
            return chunks
        except Exception as e:
            logger.error(f"获取文档块失败: {e}")
            return {}

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

            self._collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            logger.debug(f"成功更新向量: note_id={note_id}")
            self.mark_index_changed()
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

            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )

            logger.debug(f"成功 upsert 向量: note_id={note_id}")
            self.mark_index_changed()
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
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.mark_index_changed()
            logger.warning(f"已清空集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False

    def mark_index_changed(self) -> None:
        """标记索引变更。

        本服务内部的每个写操作都会调它，外部绕过本服务直接写 Chroma 时也要手动调。
        换句话说这里是所有索引变更的唯一汇聚点，所以全局版本号在这里递增一次就够了，
        不需要去各个写入点分别埋点。
        """
        index_version.bump()


# 全局服务实例
chroma_service = None


def get_chroma_service() -> ChromaService:
    """获取 ChromaDB 服务实例（单例模式）"""
    global chroma_service
    if chroma_service is None:
        chroma_service = ChromaService()
    return chroma_service

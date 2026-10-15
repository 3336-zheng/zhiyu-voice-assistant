"""知识图谱：把页面的语义关联算出来，供前端可视化。

**为什么用向量相似度而不是只靠 Wiki Link**：链接是用户手写的，智语现在的
23 个页面里一条真实链接都没有（导入的教材章节之间本来就不互相引用），
只画链接等于画一堆散点。而每个页面在 Chroma 里都已经有分块向量，算语义
关联不需要任何新的模型调用。

两种边分开返回，前端能区分「用户明确连的」和「系统算出来的」：

    link  来自 wiki_page_links，用户显式写的 [[...]]
    semantic 来自向量相似度，系统算的

**页面向量取分块向量的平均**。更准的做法是单独给「标题 + 摘要」编码，
但那要额外花 Embedding 调用（实测这条网关 90ms/条，23 个页面约 2 秒），
先用平均——零额外成本，效果不够再说。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.wiki import WikiPage, WikiPageLink
from backend.app.services.retrieval.chroma_service import get_chroma_service

logger = logging.getLogger(__name__)

# 页面分块所属的 source_type。与 page_index_service 写入时保持一致。
WIKI_SOURCE_TYPE = "wiki_page"


def _mean_vector(vectors: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """分块向量求平均。空输入返回 None。"""
    if not vectors:
        return None
    dimension = len(vectors[0])
    if dimension == 0:
        return None
    total = [0.0] * dimension
    counted = 0
    for vector in vectors:
        if len(vector) != dimension:
            # 维度不一致说明混了不同模型写入的数据，跳过而不是让整张图失真。
            continue
        for index, value in enumerate(vector):
            total[index] += value
        counted += 1
    if counted == 0:
        return None
    return [value / counted for value in total]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度。任一边是零向量时返回 0。"""
    if len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class GraphService:
    """计算页面之间的关联，产出 {nodes, edges}。"""

    def __init__(self, db: Session):
        self.db = db
        self.chroma = get_chroma_service()

    def build(self, *, threshold: Optional[float] = None) -> Dict[str, Any]:
        """构建知识图谱。

        threshold 不传时用配置默认值。前端可以传一个来试不同的紧密度——
        23 个页面的规模下，阈值高一点图就散了、低一点就成一团，必须能调。
        """
        threshold = settings.graph_similarity_threshold if threshold is None else threshold
        pages = (
            self.db.query(WikiPage)
            .filter(WikiPage.status != "deleted")
            .order_by(WikiPage.title)
            .all()
        )
        page_ids = [page.id for page in pages]

        vectors, missing = self._page_vectors(page_ids)
        nodes = [self._node(page, degree=0) for page in pages]
        edges = self._semantic_edges(page_ids, vectors, threshold)
        edges.extend(self._link_edges(page_ids))

        # 度 = 连到这个节点的边数，前端用它决定节点大小。
        degree: Dict[str, int] = {page_id: 0 for page_id in page_ids}
        for edge in edges:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1
        for node in nodes:
            node["degree"] = degree.get(node["id"], 0)

        return {
            "nodes": nodes,
            "edges": edges,
            "threshold": threshold,
            # 没有向量的页面画在图上但连不出边。报出来，否则用户会以为
            # 「这个页面跟谁都不像」，实际是它还没索引。
            "pages_without_vectors": missing,
        }

    def _page_vectors(
        self, page_ids: List[str]
    ) -> tuple[Dict[str, List[float]], List[str]]:
        """取每个页面的平均向量。返回 (向量表, 缺向量的页面 id)。"""
        if not page_ids:
            return {}, []
        try:
            result = self.chroma._collection.get(
                # 多个条件是平级的，Chroma 要求显式包一层 $and——平铺的字典
                # 会被当成「只能有一个操作符」而直接报错。
                where={
                    "$and": [
                        {"source_type": WIKI_SOURCE_TYPE},
                        {"page_id": {"$in": page_ids}},
                    ]
                },
                include=["embeddings", "metadatas"],
            )
        except Exception:
            # 读失败不等于图谱没意义。向量只是「边」的来源，页面本身
            # 和用户手写的 Wiki Link 都在 MySQL 里。降级成一张没有语义边的图，
            # 前端按「空心 = 尚未索引」显示，比整个接口 503 有用。
            #
            # 实际触发过一次：嵌入模式下 Chroma 是进程内的本地索引文件，
            # 另一个进程（worker 或手工脚本）写入后，本进程的句柄会报
            # "Error finding id"——重启即恢复，索引文件本身没坏。
            logger.warning("读取页面向量失败，本次图谱不计算语义边", exc_info=True)
            return {}, list(page_ids)
        grouped: Dict[str, List[List[float]]] = {}
        for index, metadata in enumerate(result["metadatas"] or []):
            page_id = (metadata or {}).get("page_id")
            embeddings = result["embeddings"]
            if not page_id or embeddings is None:
                continue
            vector = embeddings[index]
            if vector is None:
                continue
            # 只取父块：子块是父块的切分，算页面向量时重复计入会让
            # 长页面被切得越碎权重越大。
            if (metadata or {}).get("chunk_level") == "child":
                continue
            grouped.setdefault(page_id, []).append(list(vector))

        vectors: Dict[str, List[float]] = {}
        for page_id in page_ids:
            mean = _mean_vector(grouped.get(page_id, []))
            if mean is not None:
                vectors[page_id] = mean
        missing = [page_id for page_id in page_ids if page_id not in vectors]
        return vectors, missing

    def _semantic_edges(
        self,
        page_ids: List[str],
        vectors: Dict[str, List[float]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """两两算相似度，超过阈值就连边。O(n²)，23 个页面完全够用。"""
        with_vectors = [page_id for page_id in page_ids if page_id in vectors]
        edges: List[Dict[str, Any]] = []
        for i, left_id in enumerate(with_vectors):
            for right_id in with_vectors[i + 1:]:
                score = _cosine(vectors[left_id], vectors[right_id])
                if score >= threshold:
                    edges.append({
                        "source": left_id,
                        "target": right_id,
                        "type": "semantic",
                        "weight": round(score, 4),
                    })
        return edges

    def _link_edges(self, page_ids: List[str]) -> List[Dict[str, Any]]:
        """用户手写的 Wiki Link。只取两端都存在的（悬空链接没地方连）。"""
        allowed = set(page_ids)
        rows = (
            self.db.query(WikiPageLink)
            .filter(WikiPageLink.target_page_id.isnot(None))
            .all()
        )
        return [
            {
                "source": row.source_page_id,
                "target": row.target_page_id,
                "type": "link",
                "weight": 1.0,
            }
            for row in rows
            if row.source_page_id in allowed and row.target_page_id in allowed
        ]

    @staticmethod
    def _node(page: WikiPage, *, degree: int) -> Dict[str, Any]:
        return {
            "id": page.id,
            "title": page.title,
            "notebook": page.notebook or "",
            "tags": list(page.tags or []),
            # 索引没完成时页面还没有向量，图上有节点但连不出边。
            "indexed": page.index_status == "ready",
            "degree": degree,
        }

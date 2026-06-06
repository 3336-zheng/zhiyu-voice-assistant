"""
BM25 关键词检索服务
使用 rank_bm25 库实现 BM25 算法，支持中文分词
"""
import jieba
import re
from typing import List, Tuple, Optional, Dict
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


class BM25Service:
    """
    BM25 关键词检索服务
    支持中文分词和增量索引更新
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 服务

        Args:
            k1: BM25 参数，控制词频饱和度（通常 1.2-2.0）
            b: BM25 参数，控制文档长度归一化（通常 0.75）
        """
        self.k1 = k1
        self.b = b
        self.corpus: Dict[str, str] = {}  # {doc_id: content}
        self.tokenized_corpus: List[List[str]] = []  # 分词后的语料
        self.doc_id_list: List[str] = []  # 对应的 doc_id 列表（如 "note_1" 或 "doc_xxx_0"）
        self.bm25: Optional[BM25Okapi] = None
        self._dirty = False  # 标记索引是否需要重建

        # 加载 jieba 自定义词典（如果有）
        self._init_jieba()

    def _init_jieba(self):
        """初始化 jieba 分词器"""
        try:
            # 可以在这里加载自定义词典
            # jieba.load_userdict("custom_dict.txt")
            logger.info("jieba 分词器初始化完成")
        except Exception as e:
            logger.warning(f"jieba 初始化警告: {e}")

    def _rebuild_if_dirty(self):
        """仅在索引标记为 dirty 时重建，避免每次增删都 O(n) 重建"""
        if not self._dirty:
            return
        if len(self.tokenized_corpus) > 0:
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
        else:
            self.bm25 = None
        self._dirty = False
        logger.debug(f"BM25 索引已重建，文档数: {len(self.tokenized_corpus)}")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词

        Args:
            text: 原始文本

        Returns:
            List[str]: 分词后的词列表
        """
        if not text or not isinstance(text, str):
            return []

        # 清理文本
        text = self._clean_text(text)

        # 使用 jieba 分词
        tokens = list(jieba.cut_for_search(text))

        # 过滤停用词和短词
        tokens = self._filter_tokens(tokens)

        return tokens

    def _clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        # 移除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        过滤停用词和短词

        Args:
            tokens: 原始词列表

        Returns:
            List[str]: 过滤后的词列表
        """
        # 基础停用词
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
            '个', '我们', '可以', '就', '把', '来', '用', '能', '对',
            '及', '等', '与', '为', '或', '而', '但', '如果', '则', '因为',
            '所以', '虽然', '但是', '而且', '或者', '还是', '只是',
            # 标点符号
            ' ', '', '\n', '\t', ',', '.', '，', '。', '！', '？', '：', '；',
            '"', '"', ''', ''', '（', '）', '【', '】', '[', ']'
        }

        filtered = []
        for token in tokens:
            token = token.strip()
            if len(token) >= 1 and token not in stop_words:
                filtered.append(token)

        return filtered

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 关键词检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float]]: [(doc_id, bm25_score), ...]，按分数降序
        """
        self._rebuild_if_dirty()

        if self.bm25 is None:
            logger.error("BM25 索引未构建，请先调用 build_index()")
            return []

        try:
            # 对查询进行分词
            query_tokens = self._tokenize(query)

            if not query_tokens:
                logger.warning("查询分词后为空")
                return []

            # 计算 BM25 分数
            doc_scores = self.bm25.get_scores(query_tokens)

            # 获取 top-k 结果
            import numpy as np
            top_indices = np.argsort(doc_scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if doc_scores[idx] > 0:  # 只返回分数大于0的结果
                    doc_id = self.doc_id_list[idx]
                    score = float(doc_scores[idx])
                    results.append((doc_id, score))

            logger.debug(f"BM25 检索完成，查询: '{query}'，返回 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.error(f"BM25 检索失败: {e}")
            return []

    def add_document(self, doc_id: str, content: str, title: str = "") -> bool:
        """
        增量添加文档（延迟重建，搜索时才重建索引）

        Args:
            doc_id: 文档ID（如 "note_1" 或 "doc_xxx_0"）
            content: 内容
            title: 标题

        Returns:
            bool: 是否成功
        """
        try:
            if doc_id in self.corpus:
                return self.update_document(doc_id, content, title)

            self.corpus[doc_id] = content
            self.doc_id_list.append(doc_id)
            text = f"{title} {content}"
            self.tokenized_corpus.append(self._tokenize(text))
            self._dirty = True

            logger.debug(f"BM25 添加文档: doc_id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"BM25 添加文档失败: {e}")
            return False

    def update_document(self, doc_id: str, content: str, title: str = "") -> bool:
        """
        更新文档（延迟重建，搜索时才重建索引）

        Args:
            doc_id: 文档ID
            content: 内容
            title: 标题

        Returns:
            bool: 是否成功
        """
        try:
            if doc_id not in self.corpus:
                return self.add_document(doc_id, content, title)

            idx = self.doc_id_list.index(doc_id)
            self.corpus[doc_id] = content
            text = f"{title} {content}"
            self.tokenized_corpus[idx] = self._tokenize(text)
            self._dirty = True

            logger.debug(f"BM25 更新文档: doc_id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"BM25 更新文档失败: {e}")
            return False

    def remove_document(self, doc_id: str) -> bool:
        """
        删除文档（延迟重建，搜索时才重建索引）

        Args:
            doc_id: 文档ID

        Returns:
            bool: 是否成功
        """
        try:
            if doc_id not in self.corpus:
                logger.warning(f"要删除的文档不存在: doc_id={doc_id}")
                return True

            idx = self.doc_id_list.index(doc_id)
            del self.corpus[doc_id]
            del self.doc_id_list[idx]
            del self.tokenized_corpus[idx]
            self._dirty = True

            logger.debug(f"BM25 删除文档: doc_id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"BM25 删除文档失败: {e}")
            return False

    def get_document_count(self) -> int:
        """
        获取索引中的文档数量

        Returns:
            int: 文档数量
        """
        return len(self.corpus)

    def get_stats(self) -> Dict:
        """
        获取索引统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            "document_count": len(self.corpus),
            "avg_doc_length": sum(len(tokens) for tokens in self.tokenized_corpus) / len(self.tokenized_corpus) if self.tokenized_corpus else 0,
            "k1": self.k1,
            "b": self.b
        }

    def clear_index(self) -> bool:
        """
        清空索引

        Returns:
            bool: 是否成功
        """
        try:
            self.corpus = {}
            self.tokenized_corpus = []
            self.doc_id_list = []
            self.bm25 = None
            logger.info("BM25 索引已清空")
            return True
        except Exception as e:
            logger.error(f"清空 BM25 索引失败: {e}")
            return False


# 全局服务实例
bm25_service = None


def get_bm25_service() -> BM25Service:
    """获取 BM25 服务实例（单例模式）"""
    global bm25_service
    if bm25_service is None:
        bm25_service = BM25Service()
    return bm25_service

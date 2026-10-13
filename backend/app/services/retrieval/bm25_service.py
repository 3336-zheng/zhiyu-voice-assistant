"""
BM25 关键词检索服务
使用 rank_bm25 库实现 BM25 算法，支持中文分词
"""
import jieba
import re
from threading import Lock, RLock
from typing import Callable, List, Tuple, Optional, Dict
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

        # doc_id_list 和 tokenized_corpus 是靠下标一一对应的两个列表，删除文档要同时
        # 改动它们。这在多线程下必须是原子的：只删掉一半就会让两个列表错位，之后
        # BM25 算出「第 5 篇最相关」，去 doc_id_list 取第 5 个却拿到别的文档——
        # 不会抛异常，只会安静地返回错误结果。
        # 用可重入锁是因为 add_document 会转调 update_document，同一线程要能重复获取。
        self._lock = RLock()

        # 重建专用的锁，和 _lock 分开。全量重建要花几百毫秒，如果整段都占着 _lock，
        # 并发的检索会全部堵住；分成两把锁之后，重建的读取和分词阶段检索照常用旧索引，
        # 只有最后替换数据的那一小段才互斥。
        # 取锁顺序固定为 _rebuild_lock → _lock，检索只取 _lock，不存在反向路径。
        self._rebuild_lock = Lock()

        # 本实例已经同步到的全局索引版本号。None 表示尚未同步过。
        self._synced_version: Optional[str] = None

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
        """仅在索引标记为 dirty 时重建，避免每次增删都 O(n) 重建。

        调用方必须已持有 self._lock。
        """
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

    def mark_synced(self, version: str) -> None:
        """记录本实例的索引已经同步到哪个全局版本。

        传进来的 version 必须是**重建开始之前**读到的值，不能是重建完成后再读的。
        重建读的是那一刻的数据，若记成结束时的版本号，重建期间发生的变更就会被
        误认为已经包含在内，从此再也不会被重建——那是会永久漏数据的错误。
        反过来记成开始时的版本最多让下次多重建一遍，只浪费一点时间。
        """
        with self._lock:
            self._synced_version = version

    def ensure_fresh(self, rebuild: Callable[[], None]) -> bool:
        """检查全局索引版本，落后就重建。返回是否真的重建了。

        rebuild 由调用方传入，而不是由本服务自己去 import 重建逻辑：重建要从
        Chroma 拉全量数据，那是 doc_index_service 的事，反过来依赖会形成循环导入。

        已知的取舍：写入的那个实例自己也会被版本号带动重建一次，哪怕它刚刚已经
        用 add_document 增量更新过。本可以让写入方顺手把自己标成已同步，但那要求
        精确区分「这次版本变化是我造成的」和「别的实例也同时改了」，判断错一次就是
        永久漏数据。相比之下多花的那次重建只有几百毫秒，而写入本身（embedding +
        Chroma 落盘）比它慢得多，不值得为此引入一个会静默出错的判断。
        """
        from ...core import index_version

        version = index_version.current()
        with self._lock:
            if self._synced_version == version:
                return False

        with self._rebuild_lock:
            # 拿到重建锁后再看一次版本：刚才排在前面的线程很可能已经重建完了。
            # 没有这层复查，一批并发检索会各自把同一次重建重复跑一遍。
            with self._lock:
                if self._synced_version == version:
                    return False
            logger.info(
                "BM25 索引版本落后，开始重建: 本地=%s 全局=%s",
                self._synced_version,
                version,
            )
            # rebuild 内部会自己取 _lock 保护替换阶段，并调 mark_synced 记下它
            # 开始前读到的版本——那个版本只会比这里的 version 更早或相同，不会更新。
            rebuild()
            return True

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 关键词检索

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            List[Tuple[str, float]]: [(doc_id, bm25_score), ...]，按分数降序
        """
        # 只在锁内取一份「彼此一致」的快照就放锁：BM25Okapi 对象在重建时是整个替换的，
        # 所以拿到的引用不会被原地改动；doc_id_list 则浅拷贝一份（973 篇也只是复制
        # 一千个指针，微秒级）。这样算分和排序都在锁外进行，并发检索不会互相阻塞，
        # 同时手里这两份数据的下标对应关系又是同一时刻的，不会错位。
        with self._lock:
            self._rebuild_if_dirty()
            bm25 = self.bm25
            doc_ids = list(self.doc_id_list)

        if bm25 is None:
            logger.error("BM25 索引未构建，请先调用 build_index()")
            return []

        try:
            # 对查询进行分词
            query_tokens = self._tokenize(query)

            if not query_tokens:
                logger.warning("查询分词后为空")
                return []

            # 计算 BM25 分数
            doc_scores = bm25.get_scores(query_tokens)

            # 获取 top-k 结果
            import numpy as np
            top_indices = np.argsort(doc_scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if doc_scores[idx] > 0:  # 只返回分数大于0的结果
                    doc_id = doc_ids[idx]
                    score = float(doc_scores[idx])
                    results.append((doc_id, score))

            logger.debug("BM25 检索完成，查询长度=%s，返回=%s", len(query), len(results))
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
            # 分词放在锁外：jieba 只读全局词典，不碰本实例状态，而它是这里最慢的一步，
            # 持锁做分词会让并发写入排长队。
            tokens = self._tokenize(f"{title} {content}")
            with self._lock:
                if doc_id in self.corpus:
                    return self.update_document(doc_id, content, title)

                self.corpus[doc_id] = content
                self.doc_id_list.append(doc_id)
                self.tokenized_corpus.append(tokens)
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
            tokens = self._tokenize(f"{title} {content}")
            with self._lock:
                if doc_id not in self.corpus:
                    return self.add_document(doc_id, content, title)

                idx = self.doc_id_list.index(doc_id)
                self.corpus[doc_id] = content
                self.tokenized_corpus[idx] = tokens
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
            # 这三个 del 必须在同一个锁区间里：它们维护的是「doc_id_list[i] 对应
            # tokenized_corpus[i]」这条不变式，中途被别的线程插进来就会永久错位。
            with self._lock:
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
            with self._lock:
                self.corpus = {}
                self.tokenized_corpus = []
                self.doc_id_list = []
                self.bm25 = None
                self._dirty = False
                self._synced_version = None
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

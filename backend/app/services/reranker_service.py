"""
重排序服务
使用 BGE-reranker-v2-m3 模型对检索结果进行重新排序
"""
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 禁止联网，只用本地模型

import logging
from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..core.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self):
        """初始化重排序服务"""
        logger.info("加载重排序模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.reranker_model_path,
            local_files_only=True  # 只用本地文件，不联网
        )
        # GPU 时使用 fp16 节省显存，CPU 时使用 fp32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.reranker_model_path,
                local_files_only=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.reranker_model_path,
                local_files_only=True
            )
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"重排序模型加载完成，使用设备: {self.device}")

    def rerank(self, query: str, documents: List[str], top_k: int = 8) -> List[dict]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前 top_k 个结果

        Returns:
            排序后的结果列表，每项包含 index 和 score
        """
        if not documents:
            return []

        # 创建查询-文档对
        pairs = [[query, doc] for doc in documents]

        # 使用 tokenizer 编码
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 计算分数
        with torch.no_grad():
            logits = self.model(**inputs, return_dict=True).logits.view(-1,)
            # 将 logits 转换为分数，使用 sigmoid
            scores = torch.sigmoid(logits).cpu().tolist()

        # 创建带索引的结果列表
        indexed_results = [{"index": i, "score": score} for i, score in enumerate(scores)]

        # 按分数排序
        indexed_results.sort(key=lambda x: x["score"], reverse=True)

        return indexed_results[:top_k]


# 全局服务实例
reranker_service_instance = None


def get_reranker_service() -> RerankerService:
    """获取重排序服务实例（单例模式）"""
    global reranker_service_instance
    if reranker_service_instance is None:
        reranker_service_instance = RerankerService()
    return reranker_service_instance

"""
BGE嵌入服务
使用 transformers 原生 API 加载 BGE-M3 模型
支持 GPU 加速、批量处理、自动处理空文本
输出 numpy 数组供向量数据库使用
"""
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 禁止联网，只用本地模型

import logging
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        """初始化嵌入服务"""
        logger.info("加载 BGE-M3 嵌入模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.embedding_model_path,
            local_files_only=True
        )
        # GPU 时使用 fp16 节省显存，CPU 时使用 fp32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model = AutoModel.from_pretrained(
                settings.embedding_model_path,
                local_files_only=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModel.from_pretrained(
                settings.embedding_model_path,
                local_files_only=True
            )
        self.model.to(self.device)
        logger.info(f"BGE-M3 模型加载完成，使用设备: {self.device}")

    def encode(self, text: str) -> List[float]:
        """
        生成单个文本的向量

        Args:
            text: 输入文本

        Returns:
            文本向量（列表）
        """
        text = text.strip()
        if not text:
            return []

        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().tolist()[0]

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行批量 embedding

        Args:
            texts: 输入文本列表

        Returns:
            文本向量列表
        """
        batch_size = 32
        all_embeddings = []
        valid_texts = []
        text_indices = []

        # 分离有效文本和空文本
        for i, t in enumerate(texts):
            t = t.strip()
            if t:
                valid_texts.append(t)
                text_indices.append(i)
            else:
                text_indices.append(-1)

        if not valid_texts:
            embedding_dim = 1024
            return [[0.0] * embedding_dim for _ in texts]

        # 分批处理有效文本
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        # 合并结果并重新排列
        emb_list = []
        for batch in all_embeddings:
            for emb in batch:
                if len(emb.shape) == 1:
                    emb_list.append(emb.tolist())
                else:
                    emb_list.append(emb[0].tolist())

        if emb_list:
            embedding_dim = len(emb_list[0])
        else:
            embedding_dim = 1024
        zero_vector = [0.0] * embedding_dim

        result = []
        valid_idx = 0
        for idx in text_indices:
            if idx == -1:
                result.append(zero_vector)
            else:
                if valid_idx < len(emb_list):
                    result.append(emb_list[valid_idx])
                    valid_idx += 1
                else:
                    result.append(zero_vector)

        return result


# 全局服务实例
embedding_service_instance = None


def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例（单例模式）"""
    global embedding_service_instance
    if embedding_service_instance is None:
        embedding_service_instance = EmbeddingService()
    return embedding_service_instance

"""可配置的本地与 OpenAI 兼容文本嵌入服务。"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Protocol
from urllib.parse import urlsplit

from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProviderError(RuntimeError):
    """Embedding 配置、调用或响应不符合服务契约。"""


class EmbeddingBackend(Protocol):
    provider_name: str
    model_name: str
    dimension: Optional[int]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """将一批非空文本转换为同维度向量。"""


def _validate_api_url(value: str) -> str:
    """校验部署者配置的在线端点，禁止凭证嵌入 URL。"""
    normalized = (value or "").strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise EmbeddingProviderError("EMBEDDING_API_URL 必须是有效的 HTTP(S) 地址")
    if parsed.username or parsed.password:
        raise EmbeddingProviderError("EMBEDDING_API_URL 不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise EmbeddingProviderError("EMBEDDING_API_URL 不能包含查询参数或片段")
    return normalized


class LocalEmbeddingBackend:
    """使用本地 Hugging Face 模型生成文本向量。"""

    provider_name = "local"

    def __init__(self, model_path: str) -> None:
        if not (model_path or "").strip():
            raise EmbeddingProviderError("本地 Embedding 模型路径不能为空")
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = Path(model_path).name or "local-embedding"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("加载本地 Embedding 模型: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model_kwargs = {"local_files_only": True}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model.to(self.device)
        self.dimension = int(getattr(self.model.config, "hidden_size", 0) or 0) or None
        logger.info("本地 Embedding 模型加载完成，设备: %s", self.device)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            vectors.extend(embeddings.cpu().tolist())
        return vectors


class OpenAICompatibleEmbeddingBackend:
    """通过 OpenAI Embeddings 协议调用在线模型或模型网关。"""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        model: str,
        dimensions: int = 0,
        batch_size: int = 32,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: Any = None,
    ) -> None:
        self.api_url = _validate_api_url(api_url)
        self.model_name = (model or "").strip()
        if not self.model_name:
            raise EmbeddingProviderError("EMBEDDING_MODEL 不能为空")
        self.dimension = int(dimensions) or None
        self.batch_size = max(1, int(batch_size))
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=(api_key or "not-required").strip(),
                base_url=self.api_url,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
        self.client = client
        self.model = None
        self.device = "remote"

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            params = {"model": self.model_name, "input": batch}
            if self.dimension:
                params["dimensions"] = self.dimension
            try:
                response = self.client.embeddings.create(**params)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                logger.error(
                    "在线 Embedding 调用失败 provider=%s model=%s status=%s error=%s",
                    self.provider_name,
                    self.model_name,
                    status,
                    type(exc).__name__,
                )
                suffix = f"（HTTP {status}）" if isinstance(status, int) else ""
                raise EmbeddingProviderError(f"在线 Embedding 调用失败{suffix}") from None

            raw_items = list(response.data)
            indices = [getattr(item, "index", None) for item in raw_items]
            if (
                any(not isinstance(index, int) for index in indices)
                or sorted(indices) != list(range(len(batch)))
            ):
                raise EmbeddingProviderError("在线 Embedding 返回了重复或缺失的输入索引")
            items = sorted(raw_items, key=lambda item: item.index)
            batch_vectors = [list(item.embedding) for item in items]
            self._validate_vectors(batch_vectors, len(batch))
            vectors.extend(batch_vectors)
        return vectors

    def _validate_vectors(self, vectors: List[List[float]], expected: int) -> None:
        if len(vectors) != expected or not vectors or not vectors[0]:
            raise EmbeddingProviderError("在线 Embedding 返回的向量数量或格式不正确")
        actual_dimension = len(vectors[0])
        if any(len(vector) != actual_dimension for vector in vectors):
            raise EmbeddingProviderError("在线 Embedding 返回了不一致的向量维度")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for vector in vectors
            for value in vector
        ):
            raise EmbeddingProviderError("在线 Embedding 返回了无效或非有限数值")
        if self.dimension and actual_dimension != self.dimension:
            raise EmbeddingProviderError(
                f"在线 Embedding 返回维度 {actual_dimension}，与配置 {self.dimension} 不一致"
            )
        self.dimension = actual_dimension


class EmbeddingService:
    """提供稳定接口，并将具体实现委托给配置选中的后端。"""

    def __init__(self, backend: Optional[EmbeddingBackend] = None) -> None:
        self.backend = backend or self._build_backend()

    @staticmethod
    def _build_backend() -> EmbeddingBackend:
        if settings.embedding_provider == "openai_compatible":
            return OpenAICompatibleEmbeddingBackend(
                api_key=settings.embedding_api_key,
                api_url=settings.embedding_api_url,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                batch_size=settings.embedding_batch_size,
                timeout_seconds=settings.embedding_timeout_seconds,
                max_retries=settings.embedding_max_retries,
            )
        return LocalEmbeddingBackend(settings.embedding_model_path)

    @property
    def provider_name(self) -> str:
        return self.backend.provider_name

    @property
    def model_name(self) -> str:
        return self.backend.model_name

    @property
    def dimension(self) -> Optional[int]:
        return self.backend.dimension

    @property
    def model(self):
        return getattr(self.backend, "model", None)

    @property
    def device(self) -> str:
        return getattr(self.backend, "device", "unknown")

    def describe(self) -> dict:
        return {
            "status": "ready",
            "provider": self.provider_name,
            "model": self.model_name,
            "dimensions": self.dimension,
            "device": self.device,
        }

    def encode(self, text: str) -> List[float]:
        normalized = (text or "").strip()
        if not normalized:
            return []
        return self._validated(self.backend.encode_batch([normalized]), 1)[0]

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        normalized = [(text or "").strip() for text in texts]
        positions = [index for index, text in enumerate(normalized) if text]
        if not positions:
            dimension = self.dimension or 0
            return [[0.0] * dimension for _ in normalized]

        vectors = self._validated(
            self.backend.encode_batch([normalized[index] for index in positions]),
            len(positions),
        )
        dimension = len(vectors[0])
        result = [[0.0] * dimension for _ in normalized]
        for position, vector in zip(positions, vectors):
            result[position] = vector
        return result

    @staticmethod
    def _validated(vectors: List[List[float]], expected: int) -> List[List[float]]:
        if len(vectors) != expected or not vectors or not vectors[0]:
            raise EmbeddingProviderError("Embedding 返回的向量数量或格式不正确")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise EmbeddingProviderError("Embedding 返回了不一致的向量维度")
        return vectors


embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """获取进程级 Embedding 服务。"""
    global embedding_service_instance
    if embedding_service_instance is None:
        embedding_service_instance = EmbeddingService()
    return embedding_service_instance

"""可配置的本地与 Cohere/Jina 兼容重排序服务。"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Protocol
from urllib.parse import urlsplit

import requests

from ..core.config import settings

logger = logging.getLogger(__name__)


class RerankerProviderError(RuntimeError):
    """Rerank 配置、调用或响应不符合服务契约。"""


class RerankerBackend(Protocol):
    provider_name: str
    model_name: str

    def rerank(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        """返回包含原文档 index 与相关性 score 的结果。"""


def _validate_api_url(value: str) -> str:
    normalized = (value or "").strip()
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RerankerProviderError("RERANKER_API_URL 必须是完整的 HTTP(S) 地址")
    if parsed.username or parsed.password:
        raise RerankerProviderError("RERANKER_API_URL 不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise RerankerProviderError("RERANKER_API_URL 不能包含查询参数或片段")
    return normalized


class LocalRerankerBackend:
    """使用本地 Hugging Face Sequence Classification 模型精排。"""

    provider_name = "local"

    def __init__(self, model_path: str) -> None:
        if not (model_path or "").strip():
            raise RerankerProviderError("本地 Rerank 模型路径不能为空")
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = Path(model_path).name or "local-reranker"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("加载本地 Rerank 模型: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model_kwargs = {"local_files_only": True}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.model.eval()
        self.model.to(self.device)
        logger.info("本地 Rerank 模型加载完成，设备: %s", self.device)

    def rerank(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        pairs = [[query, document] for document in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs, return_dict=True).logits.view(-1)
            scores = self._torch.sigmoid(logits).cpu().tolist()
        results = [
            {"index": index, "score": float(score)}
            for index, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


class RerankCompatibleBackend:
    """调用 Vercel、Jina、SiliconFlow 等兼容 Rerank HTTP 接口。"""

    provider_name = "rerank_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        session: Any = None,
    ) -> None:
        self.api_url = _validate_api_url(api_url)
        self.api_key = (api_key or "").strip()
        self.model_name = (model or "").strip()
        if not self.model_name:
            raise RerankerProviderError("RERANKER_MODEL 不能为空")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.model = None
        self.device = "remote"

    def rerank(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "return_documents": False,
        }
        try:
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.error(
                "在线 Rerank 网络请求失败 provider=%s model=%s error=%s",
                self.provider_name,
                self.model_name,
                type(exc).__name__,
            )
            raise RerankerProviderError("在线 Rerank 网络请求失败") from None

        if response.status_code != 200:
            logger.error(
                "在线 Rerank 调用失败 provider=%s model=%s status=%s",
                self.provider_name,
                self.model_name,
                response.status_code,
            )
            raise RerankerProviderError(
                f"在线 Rerank 调用失败（HTTP {response.status_code}）"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RerankerProviderError("在线 Rerank 返回了无效 JSON") from None
        return self._normalize_results(body, len(documents), top_k)

    @staticmethod
    def _normalize_results(body: Any, document_count: int, top_k: int) -> List[dict]:
        raw_results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(raw_results, list):
            raise RerankerProviderError("在线 Rerank 返回结果缺少 results 数组")
        if document_count and not raw_results:
            raise RerankerProviderError("在线 Rerank 返回了空结果")
        results = []
        seen = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise RerankerProviderError("在线 Rerank 返回了无效结果项")
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not 0 <= index < document_count:
                raise RerankerProviderError("在线 Rerank 返回了越界的文档索引")
            if (
                index in seen
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RerankerProviderError("在线 Rerank 返回了重复索引或无效分数")
            seen.add(index)
            results.append({"index": index, "score": float(score)})
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


class RerankerService:
    """提供稳定精排接口，并委托给配置选中的后端。"""

    def __init__(self, backend: Optional[RerankerBackend] = None) -> None:
        self.backend = backend or self._build_backend()

    @staticmethod
    def _build_backend() -> RerankerBackend:
        if settings.reranker_provider == "rerank_compatible":
            return RerankCompatibleBackend(
                api_key=settings.reranker_api_key,
                api_url=settings.reranker_api_url,
                model=settings.reranker_model,
                timeout_seconds=settings.reranker_timeout_seconds,
            )
        return LocalRerankerBackend(settings.reranker_model_path)

    @property
    def provider_name(self) -> str:
        return self.backend.provider_name

    @property
    def model_name(self) -> str:
        return self.backend.model_name

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
            "device": self.device,
        }

    def rerank(self, query: str, documents: List[str], top_k: int = 8) -> List[dict]:
        normalized_query = (query or "").strip()
        if not normalized_query or not documents or top_k <= 0:
            return []
        normalized_documents = [str(document or "") for document in documents]
        return self.backend.rerank(
            normalized_query,
            normalized_documents,
            min(top_k, len(normalized_documents)),
        )


reranker_service_instance: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """获取进程级 Rerank 服务。"""
    global reranker_service_instance
    if reranker_service_instance is None:
        reranker_service_instance = RerankerService()
    return reranker_service_instance

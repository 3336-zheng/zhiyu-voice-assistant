"""
健康检查API
"""
import logging
from fastapi import APIRouter
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("")
@router.get("/")
@router.get("/health", include_in_schema=False)
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "app_name": "智语端侧智能语音笔记助手",
        "version": "1.0.0"
    }


@router.get("/models")
async def check_models():
    """检查模型实际加载状态"""
    models = {}

    # 当前 ASR 引擎配置
    models["asr_provider"] = settings.asr_provider

    # Whisper（仅当使用本地 Whisper 时加载）
    if settings.asr_provider == "whisper":
        try:
            from backend.app.services.ingestion.whisper_service import get_whisper_service
            svc = get_whisper_service()
            models["whisper"] = "loaded" if hasattr(svc, "model") and svc.model is not None else "not_loaded"
        except Exception as e:
            models["whisper"] = f"error: {str(e)[:100]}"
    else:
        models["whisper"] = "skipped (使用 API 引擎)"

    # DashScope ASR
    if settings.asr_provider == "dashscope":
        try:
            from backend.app.services.ingestion.dashscope_asr_service import get_dashscope_asr_service
            svc = get_dashscope_asr_service()
            models["dashscope_asr"] = "ready" if svc.api_key else "no_api_key"
        except Exception as e:
            models["dashscope_asr"] = f"error: {str(e)[:100]}"
    else:
        models["dashscope_asr"] = "skipped (使用本地引擎)"

    # Embedding
    try:
        from backend.app.services.ai.embedding_service import get_embedding_service
        svc = get_embedding_service()
        models["embedding"] = svc.describe()
    except Exception as e:
        logger.error("Embedding 健康检查失败: %s", type(e).__name__)
        models["embedding"] = {"status": "error", "error_type": type(e).__name__}

    # Reranker
    try:
        from backend.app.services.ai.reranker_service import get_reranker_service
        svc = get_reranker_service()
        models["reranker"] = svc.describe()
    except Exception as e:
        logger.error("Rerank 健康检查失败: %s", type(e).__name__)
        models["reranker"] = {"status": "error", "error_type": type(e).__name__}

    # ChromaDB
    try:
        from backend.app.services.retrieval.chroma_service import get_chroma_service
        svc = get_chroma_service()
        count = svc.collection.count()
        models["chromadb"] = {
            "status": "ready",
            "collection": svc.collection_name,
            "vectors": count,
        }
    except Exception as e:
        logger.error("ChromaDB 健康检查失败: %s", type(e).__name__)
        models["chromadb"] = {"status": "error", "error_type": type(e).__name__}

    # 整体状态
    required = [models.get("embedding"), models.get("reranker"), models.get("chromadb")]
    all_ok = all(
        isinstance(value, dict) and value.get("status") == "ready"
        for value in required
    )

    return {
        "status": "ok" if all_ok else "degraded",
        "models": models
    }

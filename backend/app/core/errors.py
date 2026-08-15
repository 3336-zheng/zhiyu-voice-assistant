"""可观测错误分类与安全摘要。"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Optional


class ErrorCode(StrEnum):
    """对外稳定的错误分类，不直接暴露底层异常信息。"""

    INTERNAL_ERROR = "INTERNAL_ERROR"
    API_VALIDATION_ERROR = "API_VALIDATION_ERROR"
    DB_OPERATION_ERROR = "DB_OPERATION_ERROR"
    PAGE_VERSION_CONFLICT = "PAGE_VERSION_CONFLICT"
    WIKI_WRITE_ERROR = "WIKI_WRITE_ERROR"
    INDEX_PROVIDER_ERROR = "INDEX_PROVIDER_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    EMBEDDING_PROVIDER_ERROR = "EMBEDDING_PROVIDER_ERROR"
    RERANK_PROVIDER_ERROR = "RERANK_PROVIDER_ERROR"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    MCP_CONNECTION_ERROR = "MCP_CONNECTION_ERROR"
    MCP_FETCH_ERROR = "MCP_FETCH_ERROR"
    AGENT_PLAN_ERROR = "AGENT_PLAN_ERROR"
    AGENT_TOOL_ERROR = "AGENT_TOOL_ERROR"
    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"


_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|access[_-]?token|secret|password)"
        r"(\s*[=:]\s*)((?:Bearer\s+)?[^\s,;\]\}\)]+)"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)([?&](?:api[_-]?key|token|secret)=)[^&#\s]+"),
)


def sanitize_text(value: object, max_length: int = 4_000) -> str:
    """脱敏并限制日志文本长度，避免凭证和大段正文进入日志。"""
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)\\bBearer"):
            text = pattern.sub("Bearer [REDACTED]", text)
        elif "[?&]" in pattern.pattern:
            text = pattern.sub(r"\1[REDACTED]", text)
        else:
            text = pattern.sub(r"\1\2[REDACTED]", text)
    if len(text) > max_length:
        return f"{text[:max_length]}...[TRUNCATED]"
    return text


def safe_error_message(error: object, max_length: int = 500) -> str:
    """生成可用于本地诊断界面的异常摘要。"""
    return sanitize_text(error, max_length=max_length)


def classify_stage_error(stage: str) -> tuple[ErrorCode, str, str, bool]:
    """根据受控阶段名映射稳定错误码、组件、操作和重试属性。"""
    normalized = (stage or "unknown").strip().lower()
    if normalized == "retrieval.embedding":
        return ErrorCode.EMBEDDING_PROVIDER_ERROR, "retrieval", "embedding", True
    if normalized == "retrieval.rerank":
        return ErrorCode.RERANK_PROVIDER_ERROR, "retrieval", "rerank", True
    if normalized.startswith("retrieval."):
        return ErrorCode.RETRIEVAL_ERROR, "retrieval", normalized.rsplit(".", 1)[-1], True
    if normalized.startswith("llm."):
        return ErrorCode.LLM_PROVIDER_ERROR, "llm", normalized.rsplit(".", 1)[-1], True
    if normalized in {"agent.plan", "agent.replan"}:
        return ErrorCode.AGENT_PLAN_ERROR, "agent", normalized.rsplit(".", 1)[-1], True
    if normalized == "agent.external_retrieval":
        return ErrorCode.MCP_FETCH_ERROR, "mcp", "external_retrieval", True
    if normalized.startswith("agent.external_"):
        return ErrorCode.AGENT_EXECUTION_ERROR, "agent", normalized.rsplit(".", 1)[-1], True
    if normalized.startswith("agent."):
        return ErrorCode.AGENT_EXECUTION_ERROR, "agent", normalized.rsplit(".", 1)[-1], False
    return ErrorCode.INTERNAL_ERROR, "application", normalized, False


def normalize_error_code(value: ErrorCode | str | None) -> Optional[str]:
    if value is None:
        return None
    return value.value if isinstance(value, ErrorCode) else str(value)

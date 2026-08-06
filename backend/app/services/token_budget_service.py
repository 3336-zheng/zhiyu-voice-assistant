"""模型上下文使用的轻量 Token 预算工具。"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any


TRUNCATION_MARKER = "\n...[内容已按上下文预算截断]"


def estimate_tokens(text: str) -> int:
    """使用适合中英文混合文本的保守规则估算 Token 数。"""
    value = text or ""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
    return cjk_count + math.ceil(max(0, len(value) - cjk_count) / 4)


def truncate_text(text: str, token_budget: int) -> str:
    """在不依赖具体模型 tokenizer 的情况下按预算截断文本。"""
    value = text or ""
    if token_budget <= 0:
        return ""
    if estimate_tokens(value) <= token_budget:
        return value

    marker = TRUNCATION_MARKER
    marker_tokens = estimate_tokens(marker)
    if marker_tokens > token_budget:
        marker = "..."
        marker_tokens = estimate_tokens(marker)

    content_budget = max(0, token_budget - marker_tokens)
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(value[:middle]) <= content_budget:
            low = middle
        else:
            high = middle - 1
    return value[:low].rstrip() + marker


def serialize_context(value: Any) -> str:
    """稳定序列化工具结果，无法 JSON 编码时退回字符串。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class ContextBudgetResult:
    """一次上下文限制的结果。"""

    text: str
    estimated_tokens: int
    used_tokens: int
    token_budget: int
    truncated: bool


def limit_context(value: Any, token_budget: int) -> ContextBudgetResult:
    """将任意工具结果转换为受预算保护的模型上下文文本。"""
    text = serialize_context(value)
    estimated = estimate_tokens(text)
    limited = truncate_text(text, token_budget)
    return ContextBudgetResult(
        text=limited,
        estimated_tokens=estimated,
        used_tokens=estimate_tokens(limited),
        token_budget=max(0, token_budget),
        truncated=limited != text,
    )

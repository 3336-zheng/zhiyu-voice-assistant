"""按优先级统一装配模型上下文，并为输出预留 Token 空间。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from backend.app.services.token_budget_service import estimate_tokens, truncate_text

MESSAGE_OVERHEAD_TOKENS = 4


def estimate_message_tokens(message: Dict[str, str]) -> int:
    """估算单条 Chat Completion 消息占用的 Token。"""
    return (
        MESSAGE_OVERHEAD_TOKENS
        + estimate_tokens(str(message.get("role", "")))
        + estimate_tokens(str(message.get("content", "")))
    )


def estimate_messages_tokens(messages: Iterable[Dict[str, str]]) -> int:
    """估算一组消息占用的 Token。"""
    return sum(estimate_message_tokens(message) for message in messages)


@dataclass(frozen=True)
class ContextAssemblyResult:
    """一次上下文装配结果。"""

    messages: List[Dict[str, str]]
    total_budget: int
    input_budget: int
    output_reserved_tokens: int
    used_tokens: int
    system_tokens: int
    summary_tokens: int
    recent_tokens: int
    current_tokens: int
    dropped_recent_messages: int
    truncated: bool

    def stats(self) -> Dict[str, int | bool]:
        """返回不包含正文的可观测统计。"""
        values = asdict(self)
        values.pop("messages", None)
        return values


class ContextAssembler:
    """按保留优先级装配上下文，并保持历史在当前任务之前。"""

    def __init__(
        self,
        *,
        context_window_tokens: int,
        history_token_budget: int,
        summary_token_budget: int,
    ) -> None:
        self.context_window_tokens = max(1, context_window_tokens)
        self.history_token_budget = max(0, history_token_budget)
        self.summary_token_budget = max(1, summary_token_budget)

    def assemble(
        self,
        *,
        system_messages: Iterable[Dict[str, str]],
        history: Optional[Iterable[Dict[str, str]]] = None,
        current_messages: Optional[Iterable[Dict[str, str]]] = None,
        output_token_reserve: int,
    ) -> ContextAssemblyResult:
        """按系统、摘要、当前、近期的保留优先级控制输入预算。"""
        output_reserved = min(
            max(0, output_token_reserve),
            self.context_window_tokens,
        )
        input_budget = self.context_window_tokens - output_reserved
        remaining = input_budget
        truncated = False

        selected_system, remaining, system_truncated = self._fit_ordered(
            system_messages,
            remaining,
        )
        truncated = truncated or system_truncated

        normalized_history = [self._normalize(message) for message in history or []]
        durable_messages = [
            message for message in normalized_history if message["role"] == "system"
        ]
        recent_messages = [
            message for message in normalized_history if message["role"] != "system"
        ]

        selected_summary: List[Dict[str, str]] = []
        summary_budget = min(self.summary_token_budget, remaining)
        if durable_messages and summary_budget > 0:
            summary_content = "\n\n".join(
                message["content"] for message in durable_messages if message["content"]
            )
            fitted_summary = self._fit_message(
                {"role": "system", "content": summary_content},
                summary_budget,
            )
            if fitted_summary is not None:
                selected_summary.append(fitted_summary)
                used = estimate_message_tokens(fitted_summary)
                remaining -= used
                truncated = truncated or fitted_summary["content"] != summary_content

        selected_current, remaining, current_truncated = self._fit_ordered(
            current_messages or [],
            remaining,
        )
        truncated = truncated or current_truncated

        recent_budget = min(self.history_token_budget, remaining)
        selected_recent, recent_truncated = self._select_recent(
            recent_messages,
            recent_budget,
        )
        recent_tokens = estimate_messages_tokens(selected_recent)
        remaining -= recent_tokens
        dropped_recent = len(recent_messages) - len(selected_recent)
        truncated = truncated or dropped_recent > 0 or recent_truncated

        messages = [
            *selected_system,
            *selected_summary,
            *selected_recent,
            *selected_current,
        ]
        used_tokens = estimate_messages_tokens(messages)
        return ContextAssemblyResult(
            messages=messages,
            total_budget=self.context_window_tokens,
            input_budget=input_budget,
            output_reserved_tokens=output_reserved,
            used_tokens=used_tokens,
            system_tokens=estimate_messages_tokens(selected_system),
            summary_tokens=estimate_messages_tokens(selected_summary),
            recent_tokens=recent_tokens,
            current_tokens=estimate_messages_tokens(selected_current),
            dropped_recent_messages=dropped_recent,
            truncated=truncated,
        )

    @staticmethod
    def _normalize(message: Dict[str, str]) -> Dict[str, str]:
        return {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }

    def _fit_ordered(
        self,
        messages: Iterable[Dict[str, str]],
        token_budget: int,
    ) -> tuple[List[Dict[str, str]], int, bool]:
        selected: List[Dict[str, str]] = []
        remaining = max(0, token_budget)
        truncated = False
        source_messages = list(messages)
        for index, message in enumerate(source_messages):
            normalized = self._normalize(message)
            fitted = self._fit_message(normalized, remaining)
            if fitted is None:
                truncated = True
                break
            selected.append(fitted)
            used = estimate_message_tokens(fitted)
            remaining -= used
            if fitted["content"] != normalized["content"]:
                truncated = True
                break
            if index < len(source_messages) - 1 and remaining <= MESSAGE_OVERHEAD_TOKENS:
                truncated = True
                break
        return selected, remaining, truncated

    def _select_recent(
        self,
        messages: List[Dict[str, str]],
        token_budget: int,
    ) -> tuple[List[Dict[str, str]], bool]:
        selected_reversed: List[Dict[str, str]] = []
        remaining = max(0, token_budget)
        truncated = False
        for message in reversed(messages):
            message_tokens = estimate_message_tokens(message)
            if message_tokens <= remaining:
                selected_reversed.append(message)
                remaining -= message_tokens
                continue
            if not selected_reversed:
                fitted = self._fit_message(message, remaining)
                if fitted is not None:
                    selected_reversed.append(fitted)
                    truncated = fitted["content"] != message["content"]
            break
        return list(reversed(selected_reversed)), truncated

    @staticmethod
    def _fit_message(
        message: Dict[str, str],
        token_budget: int,
    ) -> Optional[Dict[str, str]]:
        normalized = ContextAssembler._normalize(message)
        overhead = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(normalized["role"])
        if token_budget <= overhead:
            return None
        content = truncate_text(normalized["content"], token_budget - overhead)
        return {"role": normalized["role"], "content": content}

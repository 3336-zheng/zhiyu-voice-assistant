"""检索证据充分性评估，不让模型自行决定是否有证据。"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from ..core.config import settings


@dataclass(frozen=True)
class EvidenceAssessment:
    """一次检索结果的结构化证据评估。"""

    status: str
    score: Optional[float]
    source_count: int
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "evidence_status": self.status,
            "evidence_score": self.score,
            "evidence_source_count": self.source_count,
            "evidence_reason": self.reason,
        }


def assess_evidence(results: Iterable[Dict[str, Any]] | None) -> EvidenceAssessment:
    """基于可验证的检索结果判断证据是否足够。"""
    items = [item for item in (results or []) if item.get("content", "").strip()]
    source_keys = {
        item.get("page_id") or item.get("chunk_id") or item.get("id")
        for item in items
    }
    source_keys.discard(None)
    scores = []
    for item in items:
        value = item.get("rerank_score")
        if value is None:
            value = item.get("score")
        try:
            if value is not None:
                scores.append(float(value))
        except (TypeError, ValueError):
            continue

    best_score = max(scores) if scores else None
    source_count = len(source_keys) or len(items)
    if not items:
        return EvidenceAssessment("insufficient", None, 0, "没有检索到可引用的内容")
    if source_count < settings.evidence_min_sources:
        return EvidenceAssessment("insufficient", best_score, source_count, "可引用来源数量不足")
    if best_score is not None and best_score < settings.evidence_min_score:
        return EvidenceAssessment(
            "insufficient",
            best_score,
            source_count,
            f"最高相关性分数 {best_score:.3f} 低于阈值 {settings.evidence_min_score:.3f}",
        )
    return EvidenceAssessment("sufficient", best_score, source_count, "存在满足阈值的可引用内容")

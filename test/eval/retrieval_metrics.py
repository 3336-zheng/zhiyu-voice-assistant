"""与检索实现无关的标准排名指标。"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _relevant_ids(relevance: Mapping[str, float]) -> set[str]:
    return {doc_id for doc_id, grade in relevance.items() if grade > 0}


def _unique_top_k(retrieved_ids: Sequence[str], k: int) -> list[str]:
    """按原排名去重，防止重复文档虚增指标。"""
    unique = []
    seen = set()
    for doc_id in retrieved_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)
        if len(unique) >= k:
            break
    return unique


def hit_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """前 K 条结果是否至少命中一个相关文档。"""
    relevant = _relevant_ids(relevance)
    return float(any(doc_id in relevant for doc_id in _unique_top_k(retrieved_ids, k)))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """前 K 条结果中的相关文档比例，分母固定为 K。"""
    if k <= 0:
        return 0.0
    relevant = _relevant_ids(relevance)
    hits = sum(doc_id in relevant for doc_id in _unique_top_k(retrieved_ids, k))
    return hits / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """前 K 条结果覆盖的相关文档比例。"""
    relevant = _relevant_ids(relevance)
    if not relevant:
        return 0.0
    hits = len(set(_unique_top_k(retrieved_ids, k)) & relevant)
    return hits / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float],
) -> float:
    """首个相关结果排名的倒数。"""
    relevant = _relevant_ids(relevance)
    for rank, doc_id in enumerate(_unique_top_k(retrieved_ids, len(retrieved_ids)), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """支持分级相关性的 NDCG@K，二值标签同样适用。"""

    def gain(grade: float, rank: int) -> float:
        return (2.0**grade - 1.0) / math.log2(rank + 1)

    dcg = sum(
        gain(float(relevance.get(doc_id, 0.0)), rank)
        for rank, doc_id in enumerate(_unique_top_k(retrieved_ids, k), start=1)
        if relevance.get(doc_id, 0.0) > 0
    )
    ideal_grades = sorted(
        (float(grade) for grade in relevance.values() if grade > 0),
        reverse=True,
    )[:k]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal_grades, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_query(
    retrieved_ids: Sequence[str],
    relevance: Mapping[str, float],
    k_values: Iterable[int],
) -> dict[str, float]:
    """计算单条查询的全部指标。"""
    metrics = {"mrr": reciprocal_rank(retrieved_ids, relevance)}
    for k in sorted(set(k_values)):
        metrics[f"hit@{k}"] = hit_at_k(retrieved_ids, relevance, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevance, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevance, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevance, k)
    return metrics


def aggregate_metrics(
    query_results: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """对逐查询指标做宏平均，失败查询按零分保留。"""
    values: dict[str, list[float]] = defaultdict(list)
    for result in query_results:
        metrics = result.get("metrics", {})
        if isinstance(metrics, Mapping):
            for name, value in metrics.items():
                if isinstance(name, str) and isinstance(value, (int, float)):
                    values[name].append(float(value))
    return {
        name: statistics.fmean(metric_values) if metric_values else 0.0
        for name, metric_values in sorted(values.items())
    }


def aggregate_by_category(
    query_results: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    """按数据集自定义 category 输出宏平均指标。"""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for result in query_results:
        category = result.get("category", "default")
        grouped[str(category)].append(result)
    return {
        category: aggregate_metrics(results)
        for category, results in sorted(grouped.items())
    }
